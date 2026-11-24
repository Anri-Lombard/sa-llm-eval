# South African Language Model Evaluation Suite

Lightweight evaluation kit for any Hugging Face LLM across South African languages (base/CPT models via `lm-eval` + perplexity, plus optional chat-template evaluation for instruct models).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set `HF_HOME`, and set `HF_TOKEN` if evaluating gated models (e.g. Gemma) or `facebook/flores`.

## Usage

### 1. Per-language perplexity and bits/UTF-8 byte

Scores all 11 official SA languages on `uctnlp/mzansi-text-deduplicated` by default (pass `--corpus` to use any other Hugging Face text dataset):

```bash
python scripts/eval_ppl.py \
  --model google/gemma-3-4b-pt \
  --split validation \
  --output-json results/gemma3/ppl_val.json
```

### 2. Downstream benchmarks (`lm-eval`)

Run validation-only splits while selecting checkpoints or mixtures (`val_discriminative`), or held-out test splits for final comparisons (`test_fast`, `test_discriminative`, `test_math`, `test_translation`, `test_all`):

```bash
# Validation-only (safe for checkpoint/mixture selection)
python scripts/run_lm_eval.py \
  --model google/gemma-3-4b-pt \
  --suite val_discriminative \
  --output-dir results/gemma3/val

# Full held-out test suite (discriminative 5-shot + math 8-shot + translation 5-shot)
python scripts/run_lm_eval.py \
  --model /path/to/checkpoint \
  --suite test_all \
  --output-dir results/my_model
```

Useful flags: `--languages zul,xho,sot,afr,eng`, `--limit 10` (smoke test), `--backend vllm`, `--dry-run`.

### 3. Re-score and compare models

```bash
python scripts/rescore.py results/base results/model_a results/model_b \
  --output-json results/summary.json
```

### Optional: instruct / chat models

For chat-tuned checkpoints (e.g. `google/gemma-3-4b-it`), evaluate with chat templates instead of raw continuation:

```bash
python scripts/run_chat_afrobench.py \
  --model google/gemma-3-4b-it \
  --output-dir results/gemma3-it/chat
```

## Benchmarks Included

| Task | Splits | SA Languages |
| --- | --- | --- |
| Held-out PPL / bits per UTF-8 byte (`eval_ppl.py`) | val, test | `afr, eng, nbl, nso, sot, ssw, tsn, tso, ven, xho, zul` (all 11) |
| Belebele | test | `afr, eng, nso*, sot, ssw, tsn, tso, xho, zul` (9) |
| SIB-200 | val, test | `afr, eng, nso, sot, ssw, tsn*, tso, xho, zul` (9) |
| AfriXNLI | val, test | `eng, sot, xho, zul` (4) |
| AfriMMLU | val, test | `eng, sot, xho, zul` (4) |
| Uhura-ARC-Easy | test | `eng, nso, zul` (3) |
| InjongoIntent | val, test | `eng, sot, xho, zul` (4) |
| AfriMGSM CoT | test | `eng, sot, xho, zul` (4) |
| FLORES-200 (`eng <-> SA`) | dev, devtest | `afr, nso, sot, ssw, tsn, tso, xho, zul` (8 pairs) |
| NTREX-128 (`eng <-> SA`) | test | `afr, nso, ssw, tsn, ven, xho, zul` (7 pairs) |

`*` `belebele_nso` and `sib_tsn` are bundled in `custom_tasks/` because upstream AfroBench omits them. Validation-only YAMLs (with no `test_split`) live under `custom_tasks/validation_only/`.

## Notes

* **Always score math and translation with `scripts/rescore.py`:** upstream `lm-eval` omits stop sequences on `afrimgsm_cot` (so models keep inventing follow-up questions) and `flores_eng_Latn-*` (no newline stop). `rescore.py` fixes both and reports prompt-averaged accuracy with 95% bootstrap CIs.
* **`afrixnli_*_prompt_1` is excluded** because `lm-eval <= 0.4.12` leaves its `{premise}` / `{hypothesis}` placeholders unrendered.
* **Gemma 2/3:** use `bfloat16` (the default; `float16` can overflow), `lm-eval>=0.4.12`, and `transformers>=4.50`. `eval_ppl.py` and `run_chat_afrobench.py` automatically load Gemma 3 multimodal checkpoints via `AutoModelForImageTextToText` and prepend `<bos>`.
* **Tests:** run `pytest tests/test_eval_kit.py` offline to check task resolution and scoring logic.
