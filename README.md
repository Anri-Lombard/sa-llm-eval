# sa-llm-eval

Evaluation suite for Hugging Face language models across South African languages (all 11 official languages for perplexity, plus AfroBench classification, QA, math, and translation).

## 1. Install

```bash
git clone https://github.com/Anri-Lombard/sa-llm-eval.git
cd sa-llm-eval
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set `HF_HOME` to a directory with enough disk space, and set `HF_TOKEN` if you use gated models (e.g. Gemma) or `facebook/flores`.

## 2. Quick smoke test

Before launching a full evaluation, run a tiny 5-example check to confirm your GPU, `HF_TOKEN`, and model loader all work:

```bash
python scripts/eval_ppl.py \
  --model google/gemma-3-4b-pt \
  --languages zul,eng \
  --max-docs 5 \
  --output-json results/smoke/ppl.json

python scripts/run_lm_eval.py \
  --model google/gemma-3-4b-pt \
  --suite val_discriminative \
  --languages zul \
  --limit 5 \
  --output-dir results/smoke/val

python scripts/rescore.py results/smoke/val
```

## 3. How to run evaluations

Every script accepts either a Hugging Face model ID (e.g. `google/gemma-3-4b-pt`) or a local checkpoint directory (e.g. `/path/to/checkpoint`). Save each model under its own folder inside `results/` so `rescore.py` can compare them side by side.

### Step A: Compare checkpoints or data mixtures (validation splits only)

Use `val_discriminative` and `--split validation` while iterating so held-out test splits stay untouched:

```bash
# Per-language perplexity and bits/UTF-8 byte across all 11 SA languages
python scripts/eval_ppl.py \
  --model /path/to/checkpoint \
  --split validation \
  --output-json results/run_a/ppl_val.json

# Validation-only downstream tasks
python scripts/run_lm_eval.py \
  --model /path/to/checkpoint \
  --suite val_discriminative \
  --output-dir results/run_a/val
```

### Step B: Final evaluation (held-out test splits)

Once you pick shortlisted checkpoints, run `test_all` (discriminative 5-shot + math 8-shot + translation 5-shot), or `test_fast` for a quicker 1-prompt-per-task sweep:

```bash
python scripts/run_lm_eval.py \
  --model /path/to/checkpoint \
  --suite test_all \
  --output-dir results/run_a
```

`run_lm_eval.py` writes `lm-eval` outputs and `samples_*.jsonl` logs into `results/run_a/<phase>/`.

### Step C: Score and compare models

Pass one or more model folders to `rescore.py`. It prints Markdown comparison tables and saves a JSON summary:

```bash
python scripts/rescore.py results/base results/run_a results/run_b \
  --output-json results/summary.json
```

### Optional: chat / instruct models

For instruction-tuned checkpoints (e.g. `google/gemma-3-4b-it`), use the chat-template evaluator instead of base continuation:

```bash
python scripts/run_chat_afrobench.py \
  --model google/gemma-3-4b-it \
  --output-dir results/gemma3-it/chat
```

## Suites and useful options

| Suite (`--suite`) | Split | Default shots | What it runs |
| --- | --- | ---: | --- |
| `val_discriminative` | validation | 0 | SIB-200, AfriXNLI, AfriMMLU, InjongoIntent |
| `test_fast` | test | 5 | 1 prompt per discriminative task (quick ablation) |
| `test_discriminative` | test | 5 | All prompts for Belebele, SIB-200, AfriXNLI, AfriMMLU, Uhura-ARC-Easy, InjongoIntent |
| `test_math` | test | 8 | AfriMGSM CoT |
| `test_translation` | test | 5 | FLORES-200 and NTREX-128 (`eng <-> SA`) |
| `test_all` | test | 5 / 8 / 5 | Runs `test_discriminative`, `test_math`, and `test_translation` in sequence |

Common flags:
* `--languages zul,xho,sot,afr,eng` (all scripts): evaluate a subset of languages.
* `--batch-size 4` (or `auto:4`): lower if you hit GPU OOM.
* `--backend vllm` (`run_lm_eval.py`): faster generation for math and translation.
* `--dry-run` (`run_lm_eval.py`): print the exact `lm-eval` command without running it.
* `--corpus <hf-dataset> --text-column text --lang-column lang` (`eval_ppl.py`): evaluate perplexity on any Hugging Face text dataset instead of the default `uctnlp/mzansi-text-deduplicated` (pass `--lang-column ""` for single-language datasets).

## Language coverage

| Benchmark | SA Languages |
| --- | --- |
| Perplexity / bits per UTF-8 byte (`eval_ppl.py`) | `afr, eng, nbl, nso, sot, ssw, tsn, tso, ven, xho, zul` (all 11) |
| Belebele | `afr, eng, nso*, sot, ssw, tsn, tso, xho, zul` (9) |
| SIB-200 | `afr, eng, nso, sot, ssw, tsn*, tso, xho, zul` (9) |
| AfriXNLI, AfriMMLU, InjongoIntent, AfriMGSM | `eng, sot, xho, zul` (4) |
| Uhura-ARC-Easy | `eng, nso, zul` (3) |
| FLORES-200 (`eng <-> SA`) | `afr, nso, sot, ssw, tsn, tso, xho, zul` (8 pairs) |
| NTREX-128 (`eng <-> SA`) | `afr, nso, ssw, tsn, ven, xho, zul` (7 pairs) |

`*` `belebele_nso` and `sib_tsn` are included in `custom_tasks/` because upstream AfroBench omits them.

## Notes

* **Always run `scripts/rescore.py` after `test_math` and `test_translation`:** upstream `lm-eval` lacks stop sequences on `afrimgsm_cot` and `flores_eng_Latn-*`, which lowers raw scores. `rescore.py` extracts the first answer / first line and computes prompt-averaged accuracy with 95% bootstrap CIs.
* **`afrixnli_*_prompt_1` is skipped** because `lm-eval <= 0.4.12` leaves its `{premise}` / `{hypothesis}` placeholders unrendered.
* **Gemma 2/3:** keep `bfloat16` (default; `float16` can overflow), `lm-eval>=0.4.12`, and `transformers>=4.50`. `eval_ppl.py` and `run_chat_afrobench.py` automatically load Gemma 3 multimodal checkpoints via `AutoModelForImageTextToText` and prepend `<bos>`.
* **Offline check:** run `pytest tests/test_eval_kit.py` to verify task definitions and rescoring without a GPU.
