# SR-Merging — NLP inference

A compact implementation of **ERC → SRR → TCSM** for frozen Qwen LoRA experts.
All experts are loaded once. Each input gets its own sparse mixture; the same
mixture is reused across adapted layers and throughout autoregressive generation.

## Method

- **ERC (offline):** normalize each reference embedding, then average per expert.
  **Do not normalize the mean again.** Store one vector per expert.
- **SRR (online):** encode the query once and compute `scores = query @ C.T`.
- **TCSM (online):** select top-3 scores and apply softmax with temperature 0.8.
  Evaluate `W_base h + sum_i weight_i * (alpha_i / rank_i) * B_i(A_i h)`.

The implementation combines effective updates, not independently averaged
LoRA factors. A batch can contain inputs selecting different experts. Only
selected rows enter each expert's LoRA branch. No `set_adapter`, reference
reconstruction, or dense merged weight allocation occurs per request.

## Configuration

`configs/paper.json` contains the defaults:

| Setting | Value |
|---|---|
| Backbone | Qwen2.5-0.5B-Instruct |
| Frozen routing encoder | Potion-base-2M, CPU |
| Selected experts | `top_k = 3` |
| Routing temperature | `temperature = 0.8` |
| Precision / inference batch | FP32 / 32 |
| LoRA rank / alpha | 8 / 32 |
| LoRA target modules | `gate_proj`, `up_proj`, `down_proj` |
| Training-time LoRA dropout | 0.1 (disabled by `eval()` at inference) |
| LoRA bias | `none` |
| Reference inputs per expert | 200, matching the recent NLP cost setup |
| Reference sampling / seed | Uniform without replacement / 42 + expert index |
| CPU threads | 8 |

The bank builder records the selected line IDs and source-file SHA-256 hashes.
This explicit sampler is not the historical `reservoir(400)[:200]` sampler;
use the recorded indices to reproduce this bank. Parameters `K` and `tau` are
fixed before inference. Temperature here is **not** a generation temperature.

## Installation

Python 3.10+ is required. From this repository:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On an NVIDIA server, install the CUDA build of PyTorch 2.5.1 appropriate for
the environment before `pip install -e .`. Dependencies are pinned in
`pyproject.toml`; this package does not modify CUDA or GPU drivers.

## Prepare checkpoints and references

Update paths in `configs/paper.json`. All paths resolve from the working
directory. The eight adapters must derive from the **same exact** base checkpoint.
Each adapter folder must contain its PEFT configuration and weights. Loading
checks rank, alpha, dropout, target layers and shared-head compatibility;
incompatible old checkpoints fail explicitly rather than being silently changed.

```text
models/Qwen2.5-0.5B-Instruct/
models/potion-base-2M/
experts/{cola,mnli,mrpc,qnli,qqp,rte,sst2,stsb}/
data/{task}.references.jsonl
```

Reference files contain source-task inputs, one JSON object per line. Either
format is supported for references and queries:

```json
{"text": "The complete input to encode."}
{"instruction": "Judge grammatical acceptability.", "input": "The sentence to classify."}
```

For the second format, routing encodes `instruction + "\n" + input`; Qwen
receives the instruction as a system message and the input as a user message.
`output`, labels and query task IDs are not read. Use the same formatting for
references and queries. Reference files must contain training/source inputs,
not held-out evaluation examples. The two records above illustrate alternative
formats, not a runnable reference dataset.

Build the bank **once**, before inference:

```bash
python -m sr_merging build-bank --config configs/paper.json
```

Only the compact matrix and metadata are stored in `artifacts/nlp_references.npz`.
The model/encoder checkpoints, reference corpus and generated bank are not
included in this code release.

## Inference

### Classification in a shared output space

Provide one JSON list of candidate strings for **all** inputs. The same candidates
are scored for every input; no task ID selects a head or candidate subset. Scores
are mean candidate-token log-likelihoods. Routing is computed from the original
input only and reused while scoring every candidate.

```bash
python -m sr_merging infer --config configs/paper.json \
  --input examples/queries.jsonl --output predictions.jsonl \
  --candidates examples/candidates.json
```

`examples/candidates.json` is a four-label usage example, not a claimed full
benchmark label mapping. Replace it with the shared candidate space used by
your evaluation protocol before reporting classification accuracy.

### Generation

```bash
python -m sr_merging infer --config configs/paper.json \
  --input examples/queries.jsonl --output predictions.jsonl \
  --max-new-tokens 64
```

Generation uses the shared Qwen vocabulary/output head, greedy decoding and one
output per input. `64` is an example output budget, not a claimed paper setting;
set the task's actual budget explicitly. There is no silent prompt truncation.
Beam search and multiple returned sequences are outside this compact implementation.

Outputs include the prediction, selected expert names and composition weights.
The model instance serializes requests with a lock; use separate workers for
concurrent serving. Quantized models, DoRA, trainable expert heads and sharded
`device_map` deployment are outside this FP32 single-device implementation.

## Code map and checks

```text
sr_merging/routing.py       # ERC, SRR, TCSM and bank I/O
sr_merging/composition.py   # resident per-input weighted LoRA branches
sr_merging/pipeline.py      # checkpoint validation, classification/generation
sr_merging/__main__.py      # offline build-bank and online infer commands
configs/paper.json          # central configuration
tests/test_core.py          # numerical and tiny-Qwen correctness tests
```

```bash
python -m unittest discover -s tests -v
```

This is the current **mean-cosine ERC/SRR/TCSM** implementation. It does not use
the historical nearest-neighbor/schema-weighted router. Code correctness tests
are separate from benchmark accuracy; this release does not claim to reproduce
historical accuracy numbers without matching checkpoints, inputs and evaluation.
