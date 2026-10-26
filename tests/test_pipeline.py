"""Exercise checkpoint loading and shared-space classification with a tiny Qwen."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2ForCausalLM

from test_core import tiny_experts
from sr_merging.pipeline import SRMerging, messages
from sr_merging.routing import ReferenceBank
from sr_merging.__main__ import sample_references


class PipelineTests(unittest.TestCase):
    def test_sampling_is_repeatable_and_without_replacement(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"source.jsonl"
            path.write_text("\n".join(json.dumps({"text": f"sample {i}"}) for i in range(30)))
            a, checksum = sample_references(str(path), 10, 42)
            b, other = sample_references(str(path), 10, 42)
            self.assertEqual((a, checksum), (b, other))
            self.assertEqual(len({i for i,_ in a}), 10)

    def test_loaded_pipeline_classifies_and_generates(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            peft, names = tiny_experts()
            torch.manual_seed(7)
            Qwen2ForCausalLM(peft.get_base_model().config).save_pretrained(root/"base")
            tok = Tokenizer(models.WordLevel({"<pad>":0,"<s>":1,"</s>":2,"<unk>":3,
                        "good":4,"bad":5,"movie":6,"very":7,"positive":8,"negative":9}, unk_token="<unk>"))
            tok.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>",
                        bos_token="<s>", eos_token="</s>", unk_token="<unk>")
            tokenizer.chat_template = "{% for m in messages %}{{ m['content'] + ' ' }}{% endfor %}"
            tokenizer.save_pretrained(root/"base")
            peft.save_pretrained(root/"adapters", save_embedding_layers=False)
            bank = ReferenceBank(tuple(names), torch.eye(3,4),
                dict(encoder="test-encoder", text_format="text-or-instruction-newline-input-v1"))
            bank.save(root/"bank.npz")
            config = dict(base_model=str(root/"base"), routing_encoder="test-encoder",
                reference_bank=str(root/"bank.npz"), device="cpu", dtype="float32",
                top_k=3,temperature=.8,lora=dict(r=8,lora_alpha=32,lora_dropout=.1,
                    bias="none",target_modules=["gate_proj","up_proj","down_proj"]),
                experts=[dict(name=n,adapter=str(root/"adapters"/n)) for n in names])
            class Encoder:
                calls = 0
                def encode(self, texts):
                    self.calls += 1
                    return np.asarray([[len(t),1,t.count(' ')+1,.5] for t in texts],dtype=np.float32)
            encoder = Encoder()
            with patch("sr_merging.pipeline.StaticModel.from_pretrained", return_value=encoder):
                pipeline = SRMerging(config)
            records = [{"text":"good movie"},{"text":"very bad movie"}]
            classified = pipeline.classify(records,["positive","very negative"])
            self.assertEqual(encoder.calls,1)  # Not re-routed for every candidate.
            self.assertEqual(len(classified),2)
            for row in classified:
                self.assertEqual(len(row['candidate_scores']),2)
                self.assertAlmostEqual(sum(row['weights']),1.,places=6)
            # Verify candidate alignment against full teacher-forced logits.
            prompt = pipeline.tokenizer.apply_chat_template(messages(records[0]),tokenize=True,add_generation_prompt=True)
            target = pipeline.tokenizer.encode('very negative',add_special_tokens=False)
            ids = torch.tensor([[names.index(n) for n in classified[0]['experts']]])
            weights = torch.tensor([classified[0]['weights']])
            with torch.inference_mode(), pipeline.composition.use(ids,weights,torch.device('cpu')):
                full = pipeline.model(input_ids=torch.tensor([prompt+target]),use_cache=False).logits
                expected = full[0,len(prompt)-1:len(prompt)+len(target)-1].log_softmax(-1)
                expected = expected.gather(1,torch.tensor(target)[:,None]).mean().item()
            self.assertAlmostEqual(classified[0]['candidate_scores'][1],expected,places=5)
            # Batched left-padded scoring must equal separate per-input scoring.
            for i, record in enumerate(records):
                single = pipeline.classify([record],["positive","very negative"])[0]
                np.testing.assert_allclose(classified[i]['candidate_scores'],single['candidate_scores'],atol=1e-5)
            generated = pipeline.generate(records,max_new_tokens=3)
            self.assertEqual(len(generated),2)
            self.assertTrue(all('text' in r and len(r['experts'])==3 for r in generated))


if __name__ == "__main__":
    unittest.main()
