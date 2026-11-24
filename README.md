# South African Language CPT Evaluation Kit

Standalone evaluation suite for continual pretraining (CPT) experiments on South African languages (built for Gemma 2/3, Qwen, Llama, and other Hugging Face checkpoints).

It combines:
1. **Per-language perplexity and bits per UTF-8 byte** on `uctnlp/mzansi-text-deduplicated` across all 11 official SA languages (`scripts/eval_mzansi_ppl.py`).
2. **Multi-language AfroBench `lm-eval` suites** with custom tasks for missing SA languages (`belebele_nso`, `sib_tsn`) and strict validation-only splits for mixture selection (`suites.yaml`, `custom_tasks/`, `scripts/run_lm_eval.py`).
3. **Post-hoc rescoring and bootstrap summaries** fixing two upstream `lm-eval` AfroBench bugs (`scripts/rescore.py`).
4. **Optional chat-template AfroBench evaluation** for instruct-tuned checkpoints (`scripts/run_chat_afrobench.py`).

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set `HF_HOME` (and `HF_TOKEN` for gated Gemma checkpoints and `facebook/flores`) before running.

### 1. Fast mixture selection (validation splits only)

Keep held-out test splits untouched while comparing CPT data mixtures or intermediate checkpoints:

```bash
# Per-language perplexity and bits/UTF-8 byte across all 11 SA languages (validation split)
python scripts/eval_mzansi_ppl.py \
  --model google/gemma-3-4b-pt \
  --split validation \
  --output-json results/gemma3-4b-pt/mzansi_val_ppl.json

# Validation-only discriminative tasks (SIB-200 9 langs, AfriXNLI, AfriMMLU, InjongoIntent)
python scripts/run_lm_eval.py \
  --model google/gemma-3-4b-pt \
  --suite val_discriminative \
  --output-dir results/gemma3-4b-pt/val_discriminative

python scripts/rescore.py results/gemma3-4b-pt/val_discriminative
```

Use `--max-docs 20` on `eval_mzansi_ppl.py` and `--limit 10` on `run_lm_eval.py` for a quick smoke test.

### 2. Full held-out test suite (final checkpoint comparison)

```bash
# Runs test_discriminative (5-shot), test_math (8-shot), and test_translation (5-shot)
python scripts/run_lm_eval.py \
  --model /path/to/checkpoint \
  --suite test_all \
  --output-dir results/mix_a

# Compare multiple checkpoints side by side (applies AfriMGSM + FLORES fixes automatically)
python scripts/rescore.py results/base results/mix_a results/mix_b \
  --output-json results/summary.json
```

For faster ablations on the test split (1 prompt per benchmark instead of 5), use `--suite test_fast`.
Filter to a subset of languages on any script with `--languages zul,xho,sot,afr,eng`.
For faster math and translation generation on GPU, pass `--backend vllm` to `run_lm_eval.py`.

### 3. Optional: instruct / chat checkpoints

For Gemma IT or SFT checkpoints (`google/gemma-3-4b-it`, etc.), evaluate with chat templates instead of raw base continuation:

```bash
python scripts/run_chat_afrobench.py \
  --model google/gemma-3-4b-it \
  --split test \
  --output-dir results/gemma3-4b-it/chat_test
```

## SA Language Coverage

| Benchmark | Split(s) | Metric | SA Languages Covered | Notes |
| --- | --- | --- | --- | --- |
| MzansiText (`uctnlp/mzansi-text-deduplicated`) | val, test | PPL, bits/UTF-8 byte | `afr, eng, nbl, nso, sot, ssw, tsn, tso, ven, xho, zul` (all 11) | Use bits/UTF-8 byte when comparing across tokenizers |
| Belebele | test only | Acc (5-prompt mean) | `afr, eng, nso*, sot, ssw, tsn, tso, xho, zul` (9) | `*nso` added via `custom_tasks/belebele_nso/` |
| SIB-200 | val, test | Acc (5-prompt mean) | `afr, eng, nso, sot, ssw, tsn*, tso, xho, zul` (9) | `*tsn` added via `custom_tasks/sib_tsn/` |
| AfriXNLI | val, test | Acc (4-prompt mean) | `eng, sot, xho, zul` (4) | `prompt_1` excluded (broken in `lm-eval <= 0.4.12`) |
| AfriMMLU | val, test | Acc (5-prompt mean) | `eng, sot, xho, zul` (4) | Validation uses custom `sa_val_afrimmlu_*` |
| Uhura-ARC-Easy | test only | Acc (5-prompt mean) | `eng, nso, zul` (3) | Task codes use `en`, `nso`, `zu` |
| InjongoIntent | val, test | Acc (prompt 1) | `eng, sot, xho, zul` (4) | 40-way intent classification |
| AfriMGSM CoT | test only | First-answer EM (8-shot) | `eng, sot, xho, zul` (4) | Re-score with `scripts/rescore.py` |
| FLORES-200 (`eng <-> SA`) | dev, devtest | First-line chrF / chrF++ | `afr, nso, sot, ssw, tsn, tso, xho, zul` (8 pairs) | Re-score with `scripts/rescore.py` |
| NTREX-128 (`eng <-> SA`) | test only | First-line chrF / chrF++ | `afr, nso, ssw, tsn, ven, xho, zul` (7 pairs) | Covers Tshivenda (`ven`) |

## Known Upstream `lm-eval` and Gemma Gotchas

1. **Gemma 3 multimodal model class (`4B / 12B / 27B`):** `google/gemma-3-4b-pt` and larger checkpoints use `Gemma3ForConditionalGeneration` rather than a text-only `CausalLM`. `eval_mzansi_ppl.py` and `run_chat_afrobench.py` detect this automatically and load via `AutoModelForImageTextToText`. For `lm-eval`, use `lm-eval>=0.4.12` and `transformers>=4.50`.
2. **Gemma BOS token and dtype:** Gemma tokenizers expect `<bos>` at sequence start, and Gemma 2/3 can overflow in `float16`. All scripts default to `bfloat16`, and `eval_mzansi_ppl.py` prepends `bos_token_id` automatically whenever defined. On `lm-eval>=0.4.12`, `--model hf` respects the tokenizer default; if running an older `lm-eval`, pass `--add-bos-token true` to `run_lm_eval.py`.
3. **AfriXNLI `prompt_1` placeholder bug:** In `lm-eval <= 0.4.12`, `afrixnli_*_prompt_1` uses single braces `{premise}` / `{hypothesis}` so the inputs are never rendered. All suites in `suites.yaml` skip `prompt_1` and average prompts 2 to 5.
4. **AfriMGSM CoT missing stop sequence:** Upstream `afrimgsm_cot_*_prompt_1` omits the `Question:` prefix on few-shot examples while stopping on `Question:`. Models answer and then invent follow-up questions, causing `flexible-extract` to score the wrong number. Always score `samples_afrimgsm*.jsonl` with `scripts/rescore.py`, which takes the last number before the first blank line (`\n\n`).
5. **FLORES-200 `eng -> SA` missing newline stop:** Upstream AfroBench `flores_eng_Latn-*` sets `until: ["**", "</s>"]` instead of `"\n"`, so base models keep generating extra translation pairs. `scripts/rescore.py` truncates each generation to its first line before computing `sacrebleu` chrF / chrF++.
6. **Dataset loader compatibility:** Use `datasets<5.0` (tested with `datasets==4.8.5` and `lm-eval>=0.4.9.2`) so all Hugging Face datasets in the AfroBench suite load cleanly.

## Repository Layout

```text
sa-cpt-eval/
├── README.md
├── requirements.txt
├── suites.yaml                  # Single source of truth for suites and SA language mappings
├── custom_tasks/
│   ├── belebele_nso/            # Sepedi Belebele (prompts 1-5)
│   ├── sib_tsn/                 # Setswana SIB-200 (prompts 1-5)
│   └── validation_only/         # Validation-only tasks (no test_split)
├── scripts/
│   ├── eval_mzansi_ppl.py       # All-11-language MzansiText PPL and bits/UTF-8 byte
│   ├── run_lm_eval.py           # CLI wrapper over lm-eval run
│   ├── rescore.py               # Post-hoc AfriMGSM/FLORES fixes + bootstrap CI summary
│   └── run_chat_afrobench.py    # Optional chat-template evaluator for IT checkpoints
└── tests/
    └── test_eval_kit.py
```

## Verification

Run the offline verification tests (no GPU or downloads needed):

```bash
pytest tests/test_eval_kit.py
```
