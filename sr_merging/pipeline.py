"""Frozen encoder + offline references + shared-vocabulary Qwen inference."""
import json
from pathlib import Path
import threading

import numpy as np
import torch
from model2vec import StaticModel
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .composition import SparseComposition
from .routing import ReferenceBank, SRR, TCSM


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def routing_text(record: dict) -> str:
    if "text" in record:
        return str(record["text"]).strip()
    return (str(record.get("instruction", "")).strip() + "\n" + str(record["input"]).strip()).strip()


def messages(record: dict) -> list[dict]:
    if "text" in record:
        return [{"role": "user", "content": routing_text(record)}]
    result = []
    if record.get("instruction"):
        result.append({"role": "system", "content": record["instruction"]})
    return result + [{"role": "user", "content": record["input"]}]


def check_adapter(config: PeftConfig, expected: dict) -> None:
    for key in ["r", "lora_alpha", "lora_dropout", "bias"]:
        if getattr(config, key, None) != expected[key]:
            raise ValueError(f"Adapter {key} differs from paper configuration: {getattr(config, key, None)} != {expected[key]}")
    if set(config.target_modules or []) != set(expected["target_modules"]):
        raise ValueError("Adapter target_modules differ from paper configuration")
    if (config.modules_to_save or getattr(config, "use_dora", False)
            or getattr(config, "use_rslora", False) or getattr(config, "rank_pattern", {})
            or getattr(config, "alpha_pattern", {}) or getattr(config, "lora_bias", False)):
        raise ValueError("Use standard uniform-rank LoRA with a shared frozen output head")


class SRMerging:
    def __init__(self, config: dict):
        if config["dtype"] != "float32":
            raise ValueError("The paper configuration uses FP32")
        torch.set_num_threads(config.get("cpu_threads", 8))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = torch.device(config["device"])
        self.names = [e["name"] for e in config["experts"]]
        if len(set(self.names)) != len(self.names):
            raise ValueError("Expert names must be unique")
        bank = ReferenceBank.load(config["reference_bank"])
        if bank.names != tuple(self.names):
            raise ValueError("Expert order differs from offline reference bank")
        if bank.metadata.get("encoder") != config["routing_encoder"]:
            raise ValueError("Use the same frozen encoder for references and queries")
        if bank.metadata.get("text_format") != "text-or-instruction-newline-input-v1":
            raise ValueError("Unsupported reference text format")
        expected_count = config.get("references_per_expert")
        if expected_count is not None and any(
            bank.metadata.get("sources", {}).get(name, {}).get("count") != expected_count
            for name in self.names
        ):
            raise ValueError("Reference counts differ from configuration; rebuild the offline bank")
        self.srr, self.tcsm = SRR(bank), TCSM(config["top_k"], config["temperature"])
        if config["top_k"] > len(self.names):
            raise ValueError("K must not exceed the number of experts")
        # Fail before loading large checkpoints when the adapter configuration differs.
        for expert in config["experts"]:
            check_adapter(PeftConfig.from_pretrained(expert["adapter"]), config["lora"])
        self.encoder = StaticModel.from_pretrained(config["routing_encoder"])
        self.tokenizer = AutoTokenizer.from_pretrained(config["base_model"], padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(config["base_model"], dtype=torch.float32)
        first, *others = config["experts"]
        peft = PeftModel.from_pretrained(base, first["adapter"], adapter_name=first["name"], is_trainable=False)
        for expert in others:
            peft.load_adapter(expert["adapter"], adapter_name=expert["name"], is_trainable=False)
        # Use the underlying CausalLM generation API, without PEFT's active-adapter state.
        self.model = peft.get_base_model().to(device=self.device, dtype=torch.float32).eval().requires_grad_(False)
        self.composition = SparseComposition(self.model, self.names)
        self._lock = threading.Lock()

    def _route(self, records):
        embeddings = torch.from_numpy(np.asarray(self.encoder.encode([routing_text(r) for r in records]), dtype=np.float32))
        return self.tcsm(self.srr(embeddings))

    @torch.inference_mode()
    def classify(self, records: list[dict], candidates: list[str]) -> list[dict]:
        """Mean token log-likelihood over ONE candidate space shared by all tasks."""
        if not records:
            return []
        if not candidates or len(set(candidates)) != len(candidates):
            raise ValueError("Provide a nonempty, unique shared candidate list")
        with self._lock:
            indices, weights = self._route(records)
            prompts = [self.tokenizer.apply_chat_template(messages(r), tokenize=True, add_generation_prompt=True) for r in records]
            scores = []
            # Route only the original input, never its candidate continuation.
            with self.composition.use(indices, weights, self.device):
                for candidate in candidates:
                    target = self.tokenizer.encode(candidate, add_special_tokens=False)
                    if not target:
                        raise ValueError("A candidate must contain at least one token")
                    limit = getattr(self.model.config, "max_position_embeddings", None)
                    if limit and max(map(len, prompts)) + len(target) > limit:
                        raise ValueError("Prompt plus candidate exceeds model context")
                    batch = self.tokenizer.pad({"input_ids": [p + target for p in prompts]}, padding=True, return_tensors="pt").to(self.device)
                    positions = batch.attention_mask.cumsum(-1).sub(1).clamp_min(0)
                    logits = self.model(**batch, position_ids=positions, use_cache=False,
                                        logits_to_keep=len(target) + 1).logits[:, :-1]
                    labels = torch.tensor(target, device=self.device)[None, :, None].expand(len(records), -1, -1)
                    scores.append(logits.float().log_softmax(-1).gather(-1, labels).squeeze(-1).mean(-1).cpu())
            scores = torch.stack(scores, dim=1)
            return [{"label": candidates[best], "candidate_scores": values,
                     "experts": [self.names[i] for i in ids], "weights": ws}
                    for best, values, ids, ws in zip(scores.argmax(-1).tolist(), scores.tolist(), indices.tolist(), weights.tolist())]

    @torch.inference_mode()
    def generate(self, records: list[dict], *, max_new_tokens: int) -> list[dict]:
        """Greedy generation, one output per input; input-specific routes stay fixed."""
        if not records:
            return []
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        with self._lock:
            indices, weights = self._route(records)
            prompts = [self.tokenizer.apply_chat_template(messages(r), tokenize=False, add_generation_prompt=True) for r in records]
            tokens = self.tokenizer(prompts, padding=True, add_special_tokens=False, return_token_type_ids=False, return_tensors="pt").to(self.device)
            limit = getattr(self.model.config, "max_position_embeddings", None)
            if limit and tokens.input_ids.shape[1] + max_new_tokens > limit:
                raise ValueError("Prompt plus output budget exceeds the model context; no silent truncation")
            with self.composition.use(indices, weights, self.device):
                output = self.model.generate(**tokens, max_new_tokens=max_new_tokens,
                    do_sample=False, num_beams=1, num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id, use_cache=True)
            texts = self.tokenizer.batch_decode(output[:, tokens.input_ids.shape[1]:], skip_special_tokens=True)
            return [{"text": text, "experts": [self.names[i] for i in ids], "weights": ws}
                    for text, ids, ws in zip(texts, indices.tolist(), weights.tolist())]
