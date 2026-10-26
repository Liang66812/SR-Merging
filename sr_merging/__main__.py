"""python -m sr_merging build-bank|infer --config configs/paper.json ..."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from model2vec import StaticModel

from .pipeline import SRMerging, load_config, routing_text
from .routing import ERC, ReferenceBank


def sample_references(path: str, count: int, seed: int):
    """Uniform reservoir sampling without replacement; record source line IDs."""
    rng, samples, digest, seen = random.Random(seed), [], hashlib.sha256(), 0
    with open(path, "rb") as stream:
        for line_id, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            text = routing_text(row)
            if not text:
                raise ValueError(f"Empty reference at {path}:{line_id}")
            seen += 1
            item = (line_id, text)
            if len(samples) < count:
                samples.append(item)
            else:
                slot = rng.randrange(seen)
                if slot < count:
                    samples[slot] = item
    if len(samples) < count:
        raise ValueError(f"{path} contains fewer than {count} reference inputs")
    return samples, digest.hexdigest()


def build_bank(config):
    encoder = StaticModel.from_pretrained(config["routing_encoder"])
    vectors, sources = [], {}
    count, batch_size = config["references_per_expert"], config["batch_size"]
    if count < 1:
        raise ValueError("references_per_expert must be positive")
    for offset, expert in enumerate(config["experts"]):
        selected, checksum = sample_references(expert["references"], count, config["seed"] + offset)
        texts = [text for _, text in selected]
        embeddings = np.concatenate([encoder.encode(texts[start:start + batch_size])
                                     for start in range(0, count, batch_size)])
        vectors.append(ERC.compress(torch.from_numpy(embeddings)))
        sources[expert["name"]] = dict(count=count, sha256=checksum, line_ids=[i for i, _ in selected])
    metadata = dict(encoder=config["routing_encoder"], seed=config["seed"],
                    text_format="text-or-instruction-newline-input-v1", sources=sources)
    names = tuple(e["name"] for e in config["experts"])
    if not names or len(set(names)) != len(names):
        raise ValueError("Need uniquely named experts")
    ReferenceBank(names, torch.stack(vectors), metadata).save(config["reference_bank"])
    print(f"Saved {len(names)} x {vectors[0].numel()} FP32 reference values to {config['reference_bank']}")


def infer(config, input_path, output_path, max_new_tokens, candidates=None):
    pipeline = SRMerging(config)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, encoding="utf-8") as src, open(output_path, "w", encoding="utf-8") as dst:
        batch = []
        def write_batch():
            results = (pipeline.classify(batch, candidates) if candidates is not None
                       else pipeline.generate(batch, max_new_tokens=max_new_tokens))
            for row, result in zip(batch, results):
                if "id" in row:
                    result["id"] = row["id"]
                dst.write(json.dumps(result, ensure_ascii=False) + "\n")
            dst.flush()
        for line in src:
            if line.strip():
                batch.append(json.loads(line))
            if len(batch) == config["batch_size"]:
                write_batch()
                batch.clear()
        if batch:
            write_batch()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build-bank", "infer"])
    parser.add_argument("--config", default="configs/paper.json")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--max-new-tokens", type=int, help="Explicit task-specific generation budget")
    parser.add_argument("--candidates", help="JSON list: one shared candidate space for ALL inputs")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["batch_size"] < 1:
        parser.error("batch_size must be positive")
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == "build-bank":
        build_bank(config)
    else:
        if not args.input or not args.output or ((args.max_new_tokens is None) == (args.candidates is None)):
            parser.error("infer requires --input, --output and exactly one of --max-new-tokens / --candidates")
        if Path(args.input).resolve() == Path(args.output).resolve():
            parser.error("Input and output paths must differ")
        candidates = json.loads(Path(args.candidates).read_text()) if args.candidates else None
        if candidates is not None and (not isinstance(candidates, list) or not all(isinstance(x, str) for x in candidates)):
            parser.error("--candidates must contain a JSON list of strings")
        infer(config, args.input, args.output, args.max_new_tokens, candidates)


if __name__ == "__main__":
    main()
