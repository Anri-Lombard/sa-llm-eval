#!/usr/bin/env python3
"""Run South African lm-eval suites defined in suites.yaml."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SUITES_PATH = REPO_ROOT / "suites.yaml"
CUSTOM_TASKS_DIR = REPO_ROOT / "custom_tasks"


def load_suites(path: Path = SUITES_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def resolve_phases(
    config: dict, suite_name: str, languages: list[str] | None = None
) -> list[tuple[str, dict, list[str]]]:
    suites = config["suites"]
    if suite_name not in suites:
        raise ValueError(f"Unknown suite '{suite_name}'. Available: {sorted(suites)}")
    spec = suites[suite_name]
    phase_names = spec.get("phases", [suite_name])

    resolved: list[tuple[str, dict, list[str]]] = []
    for phase in phase_names:
        phase_spec = suites[phase]
        tasks_by_lang: dict[str, list[str]] = phase_spec["tasks_by_language"]
        selected_langs = list(tasks_by_lang.keys()) if not languages else [
            lang for lang in languages if lang in tasks_by_lang
        ]
        tasks = [task for lang in selected_langs for task in tasks_by_lang[lang]]
        if tasks:
            resolved.append((phase, phase_spec, tasks))
    if not resolved:
        raise ValueError(f"No tasks matched suite='{suite_name}' with languages={languages}")
    return resolved


def build_lm_eval_command(
    *,
    lm_eval_bin: str,
    model: str,
    tasks: list[str],
    output_path: Path,
    num_fewshot: int,
    backend: str,
    dtype: str,
    device: str,
    batch_size: str,
    seed: int,
    apply_chat_template: bool,
    add_bos_token: str | None,
    limit: float | None,
    trust_remote_code: bool,
    extra_model_args: str | None,
) -> list[str]:
    model_args = [f"pretrained={model}", f"dtype={dtype}"]
    if trust_remote_code:
        model_args.append("trust_remote_code=True")
    if add_bos_token is not None:
        model_args.append(f"add_bos_token={add_bos_token}")
    if extra_model_args:
        model_args.append(extra_model_args)

    cmd = [
        lm_eval_bin,
        "run",
        "--model",
        backend,
        "--model_args",
        ",".join(model_args),
        "--tasks",
        ",".join(tasks),
        "--include_path",
        str(CUSTOM_TASKS_DIR),
        "--num_fewshot",
        str(num_fewshot),
        "--batch_size",
        str(batch_size),
        "--seed",
        str(seed),
        "--output_path",
        str(output_path),
        "--log_samples",
    ]
    if backend == "hf":
        cmd += ["--device", device]
    if apply_chat_template:
        cmd.append("--apply_chat_template")
    if limit is not None:
        cmd += ["--limit", str(limit)]
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SA language lm-eval suites.")
    parser.add_argument("--model", required=True, help="HF model ID or local checkpoint directory.")
    parser.add_argument(
        "--suite",
        default="val_discriminative",
        help="Suite name from suites.yaml (val_discriminative, test_fast, test_discriminative, test_math, test_translation, test_all).",
    )
    parser.add_argument(
        "--languages",
        default=None,
        help="Optional comma-separated SA language filter (e.g. zul,xho,afr,sot,eng).",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for lm-eval outputs and samples.")
    parser.add_argument("--num-fewshot", type=int, default=None, help="Override suite default fewshot count.")
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default="auto:4")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument(
        "--add-bos-token",
        choices=("true", "false"),
        default=None,
        help="Explicitly override tokenizer BOS behavior (useful on lm-eval < 0.4.12 with Gemma).",
    )
    parser.add_argument("--limit", type=float, default=None, help="Optional per-task example cap for smoke tests.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--extra-model-args", default=None, help="Extra comma-separated --model_args appended verbatim.")
    parser.add_argument("--lm-eval-bin", default=None, help="Path to lm-eval executable (defaults to PATH or current venv).")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved lm-eval command(s) without running.")
    return parser.parse_args()


def _find_lm_eval_bin(override: str | None) -> str:
    if override:
        return override
    sibling = Path(sys.executable).parent / "lm-eval"
    if sibling.exists():
        return str(sibling)
    found = shutil.which("lm-eval") or shutil.which("lm_eval")
    return found or "lm-eval"


def main() -> None:
    args = parse_args()
    config = load_suites()
    languages = [c.strip() for c in args.languages.split(",") if c.strip()] if args.languages else None
    phases = resolve_phases(config, args.suite, languages)
    lm_eval_bin = _find_lm_eval_bin(args.lm_eval_bin)

    for phase_name, phase_spec, tasks in phases:
        fewshot = args.num_fewshot if args.num_fewshot is not None else int(phase_spec["default_fewshot"])
        phase_out = args.output_dir / phase_name if len(phases) > 1 else args.output_dir
        cmd = build_lm_eval_command(
            lm_eval_bin=lm_eval_bin,
            model=args.model,
            tasks=tasks,
            output_path=phase_out,
            num_fewshot=fewshot,
            backend=args.backend,
            dtype=args.dtype,
            device=args.device,
            batch_size=args.batch_size,
            seed=args.seed,
            apply_chat_template=args.apply_chat_template,
            add_bos_token=args.add_bos_token,
            limit=args.limit,
            trust_remote_code=args.trust_remote_code,
            extra_model_args=args.extra_model_args,
        )
        print(f"# Phase: {phase_name} ({len(tasks)} tasks, {fewshot}-shot)")
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            phase_out.mkdir(parents=True, exist_ok=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
