#!/usr/bin/env python3
"""Re-score and summarize lm-eval sample logs across South African benchmarks.

Applies two required post-hoc fixes for upstream lm-eval AfroBench bugs:
1. AfriMGSM CoT: extracts the last number before the first blank line (`\\n\\n`), because
   upstream few-shot prompts omit `Question:` so models keep inventing follow-up questions.
2. FLORES-200 / NTREX-128: truncates each generation at the first newline before computing
   corpus chrF and chrF++, because upstream FLORES eng->African tasks lack a newline stop.

For discriminative tasks, computes prompt-averaged accuracy per (benchmark, language) with
document-level 95% bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

NUMBER = re.compile(r"-?[\d.,]*\d")
SAMPLE_RE = re.compile(r"^samples_(.+)_\d{4}-\d{2}-\d{2}T")


def unwrap_response(resp) -> str:
    while isinstance(resp, list):
        if not resp:
            return ""
        resp = resp[0]
    return str(resp)


def extract_afrimgsm_answer(response) -> float | None:
    first_block = unwrap_response(response).split("\n\n")[0]
    numbers = NUMBER.findall(first_block)
    if not numbers:
        return None
    try:
        return float(numbers[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def first_line(response) -> str:
    text = unwrap_response(response).strip()
    return text.split("\n")[0].strip() if text else ""


def bootstrap_ci(values: list[float], *, resamples: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    if not values:
        raise ValueError("bootstrap_ci requires at least one value")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    mean = sum(values) / n
    return mean, means[int(0.025 * resamples)], means[int(0.975 * resamples)]


def strip_prompt_suffix(task: str) -> str:
    return re.sub(r"_prompt_\d+$", "", task)


def task_name_from_sample_path(path: Path) -> str | None:
    match = SAMPLE_RE.match(path.name)
    return match.group(1) if match else None


def score_afrimgsm_file(path: Path) -> dict[str, float | int]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return {"accuracy": 0.0, "n": 0}
    hits = []
    for row in rows:
        gold = float(str(row["target"]).replace(",", "").strip())
        pred = extract_afrimgsm_answer(row["resps"])
        hits.append(1.0 if pred == gold else 0.0)
    mean, lo, hi = bootstrap_ci(hits)
    return {"accuracy": 100.0 * mean, "ci_low": 100.0 * lo, "ci_high": 100.0 * hi, "n": len(rows)}


def score_translation_file(path: Path) -> dict[str, float | int]:
    from sacrebleu.metrics import CHRF

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return {"chrf": 0.0, "chrf_pp": 0.0, "n": 0}
    hypotheses = [first_line(row["resps"]) for row in rows]
    references = [str(row["target"]).strip() for row in rows]
    chrf = CHRF(word_order=0).corpus_score(hypotheses, [references]).score
    chrf_pp = CHRF(word_order=2).corpus_score(hypotheses, [references]).score
    return {"chrf": chrf, "chrf_pp": chrf_pp, "n": len(rows)}


def summarize_model_dir(model_dir: Path, *, resamples: int = 2000) -> dict[str, dict]:
    # Group sample files by task name; if a task was run more than once, keep the newest file.
    latest_by_task: dict[str, Path] = {}
    for path in sorted(model_dir.rglob("samples_*.jsonl")):
        task = task_name_from_sample_path(path)
        if task:
            latest_by_task[task] = path

    disc_docs: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    math_scores: dict[str, dict] = {}
    mt_scores: dict[str, dict] = {}

    for task, path in sorted(latest_by_task.items()):
        base = strip_prompt_suffix(task)
        if base.startswith("afrimgsm"):
            math_scores[base] = score_afrimgsm_file(path)
        elif base.startswith(("flores_", "ntrex_")):
            mt_scores[base] = score_translation_file(path)
        else:
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if "acc" in row:
                    disc_docs[base][int(row["doc_id"])].append(float(row["acc"]))

    disc_scores: dict[str, dict] = {}
    for base, docs in sorted(disc_docs.items()):
        per_doc = [sum(v) / len(v) for v in docs.values()]
        mean, lo, hi = bootstrap_ci(per_doc, resamples=resamples)
        prompts = max(len(v) for v in docs.values())
        disc_scores[base] = {
            "accuracy": 100.0 * mean,
            "ci_low": 100.0 * lo,
            "ci_high": 100.0 * hi,
            "n": len(per_doc),
            "prompts": prompts,
        }

    return {
        "discriminative": disc_scores,
        "math": math_scores,
        "translation": mt_scores,
    }


def _render_table(
    title: str,
    model_names: list[str],
    rows_by_model: dict[str, dict[str, dict]],
    fmt,
) -> str:
    tasks = sorted({t for d in rows_by_model.values() for t in d})
    if not tasks:
        return ""
    lines = [
        f"## {title}",
        "",
        "| Task | " + " | ".join(model_names) + " |",
        "| --- | " + " | ".join("---:" for _ in model_names) + " |",
    ]
    for task in tasks:
        cells = [fmt(rows_by_model[m][task]) if task in rows_by_model[m] else "-" for m in model_names]
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-score and summarize lm-eval SA results.")
    parser.add_argument(
        "model_dirs",
        nargs="+",
        type=Path,
        help="One or more model output directories containing lm-eval samples_*.jsonl files.",
    )
    parser.add_argument("--resamples", type=int, default=2000, help="Bootstrap resamples (default: 2000).")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON summary output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries: dict[str, dict] = {}
    for model_dir in args.model_dirs:
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Not a directory: {model_dir}")
        summaries[model_dir.name] = summarize_model_dir(model_dir, resamples=args.resamples)

    names = list(summaries.keys())
    sections = [
        _render_table(
            "Discriminative (prompt-averaged accuracy %, 95% bootstrap CI)",
            names,
            {m: summaries[m]["discriminative"] for m in names},
            lambda s: f"{s['accuracy']:.1f} [{s['ci_low']:.1f}, {s['ci_high']:.1f}]",
        ),
        _render_table(
            "AfriMGSM CoT (first-answer accuracy %, 95% bootstrap CI)",
            names,
            {m: summaries[m]["math"] for m in names},
            lambda s: f"{s['accuracy']:.1f} [{s['ci_low']:.1f}, {s['ci_high']:.1f}]",
        ),
        _render_table(
            "Translation (first-line chrF / chrF++)",
            names,
            {m: summaries[m]["translation"] for m in names},
            lambda s: f"{s['chrf']:.1f} / {s['chrf_pp']:.1f}",
        ),
    ]
    output = "\n\n".join(s for s in sections if s)
    if not output:
        print("No samples_*.jsonl files found in the supplied directories.")
    else:
        print(output)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summaries, indent=2) + "\n")
        print(f"\nSaved summary JSON to {args.output_json}")


if __name__ == "__main__":
    main()
