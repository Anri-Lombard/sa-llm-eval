from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import eval_ppl as ppl  # noqa: E402
import rescore  # noqa: E402
import run_chat_afrobench as chat  # noqa: E402
import run_lm_eval as runner  # noqa: E402


def test_suites_skip_broken_afrixnli_prompt_1():
    cfg = runner.load_suites()
    offending = []
    for suite_name, spec in cfg["suites"].items():
        for lang, tasks in spec.get("tasks_by_language", {}).items():
            for task in tasks:
                if "afrixnli" in task and task.endswith("_prompt_1"):
                    offending.append(f"{suite_name}/{lang}: {task}")
    assert offending == []


def test_validation_custom_tasks_never_define_test_split():
    val_dir = REPO_ROOT / "custom_tasks" / "validation_only"
    yamls = sorted(val_dir.glob("*.yaml"))
    assert len(yamls) > 0
    for path in yamls:
        text = path.read_text()
        assert "test_split:" not in text, f"{path.name} defines test_split"
        if path.name.startswith("_"):
            assert "validation_split: validation" in text


def test_all_suite_tasks_resolve_in_lm_eval_task_manager():
    lm_eval_tasks = pytest.importorskip("lm_eval.tasks")
    tm = lm_eval_tasks.TaskManager(include_path=str(REPO_ROOT / "custom_tasks"))
    available = set(tm.all_tasks)

    cfg = runner.load_suites()
    missing = []
    for suite_name, spec in cfg["suites"].items():
        for lang, tasks in spec.get("tasks_by_language", {}).items():
            for task in tasks:
                if task not in available:
                    missing.append(f"{suite_name}/{lang}: {task}")
    assert missing == [], f"Unresolved lm-eval tasks: {missing[:10]}"


def test_run_lm_eval_dry_run_all_suites(tmp_path):
    for suite in ("val_discriminative", "test_fast", "test_discriminative", "test_math", "test_translation", "test_all"):
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "run_lm_eval.py"),
                "--model",
                "google/gemma-3-4b-pt",
                "--suite",
                suite,
                "--languages",
                "zul,xho,nso,tsn,ven",
                "--output-dir",
                str(tmp_path / suite),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "--include_path" in proc.stdout
        assert "--log_samples" in proc.stdout


def test_afrimgsm_first_answer_extraction():
    invented_followup = (
        " 16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. Impendulo ithi 18.\n\n"
        "U-Adam wayenama-apula angu-10. Udadewabo wamnika amanye angu-5. Impendulo ithi 15."
    )
    assert rescore.extract_afrimgsm_answer([[invented_followup]]) == 18.0
    assert rescore.extract_afrimgsm_answer("Impendulo ithi 70,000.") == 70000.0
    assert rescore.extract_afrimgsm_answer("no numbers here") is None


def test_translation_first_line_and_summarize_model_dir(tmp_path):
    pytest.importorskip("sacrebleu")
    model_dir = tmp_path / "gemma_cpt"
    model_dir.mkdir()

    # Discriminative: 2 prompts x 2 docs
    p1 = [
        {"doc_id": 0, "acc": 1.0},
        {"doc_id": 1, "acc": 0.0},
    ]
    p2 = [
        {"doc_id": 0, "acc": 1.0},
        {"doc_id": 1, "acc": 1.0},
    ]
    (model_dir / "samples_belebele_zul_prompt_1_2026-11-24T00-00-00.jsonl").write_text(
        "\n".join(json.dumps(r) for r in p1) + "\n"
    )
    (model_dir / "samples_belebele_zul_prompt_2_2026-11-24T00-00-00.jsonl").write_text(
        "\n".join(json.dumps(r) for r in p2) + "\n"
    )

    # AfriMGSM: invented continuation should be ignored
    gsm = [
        {"doc_id": 0, "target": "18", "resps": [["Step 1... 18.\n\nQuestion: 5+5=10"]]},
        {"doc_id": 1, "target": "42", "resps": [["Answer is 7."]]},
    ]
    (model_dir / "samples_afrimgsm_cot_zul_prompt_1_2026-11-24T00-00-00.jsonl").write_text(
        "\n".join(json.dumps(r) for r in gsm) + "\n"
    )

    # FLORES: trailing hallucinated lines should be stripped before chrF
    flores = [
        {"doc_id": 0, "target": "Sawubona mhlaba.", "resps": [["Sawubona mhlaba.\nEnglish: Extra\nZulu: Garbage"]]},
    ]
    (model_dir / "samples_flores_eng_Latn-zul_Latn_prompt_1_2026-11-24T00-00-00.jsonl").write_text(
        "\n".join(json.dumps(r) for r in flores) + "\n"
    )

    summary = rescore.summarize_model_dir(model_dir, resamples=200)
    assert summary["discriminative"]["belebele_zul"]["accuracy"] == pytest.approx(75.0)
    assert summary["discriminative"]["belebele_zul"]["prompts"] == 2
    assert summary["math"]["afrimgsm_cot_zul"]["accuracy"] == pytest.approx(50.0)
    assert summary["translation"]["flores_eng_Latn-zul_Latn"]["chrf"] == pytest.approx(100.0)


def test_ppl_windowing_and_bpb_math():
    windows = list(ppl.window_document_tokens([10, 11, 12, 13, 14], prefix_token_id=2, sequence_length=4))
    assert windows == [
        {"input_ids": [2, 10, 11, 12], "labels": [-100, 10, 11, 12]},
        {"input_ids": [12, 13, 14], "labels": [-100, 13, 14]},
    ]
    m = ppl.aggregate_language_metrics(
        total_nll=10.0 * math.log(2), predicted_tokens=5, utf8_bytes=10, documents=1, windows=2
    )
    assert m["bits_per_utf8_byte"] == pytest.approx(1.0)
    assert m["perplexity"] == pytest.approx(4.0)


def test_chat_parsers():
    assert chat.predicted_letter("The answer is C.", 4) == "C"
    assert chat.predicted_letter("E", 4) is None
    assert chat.predicted_number("Impendulo ngu-\\boxed{1,250}.") == 1250.0
    assert chat.score_item("label", "book_flight.\nextra", "book_flight", 40) == 1.0
