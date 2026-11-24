#!/usr/bin/env python3
"""Optional chat-template AfroBench evaluator for instruct-tuned (IT) checkpoints.

Use this only when evaluating chat/instruct models (e.g. gemma-3-*-it or SFT checkpoints).
For base/CPT checkpoints, use scripts/run_lm_eval.py and scripts/eval_ppl.py instead.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

from eval_ppl import _load_model, _resolve_dtype
from rescore import bootstrap_ci

LETTERS = "ABCDEFGH"
MC_INSTRUCTION = (
    "Answer the multiple-choice question by replying with ONLY the letter ({letters}) of the correct answer."
)
NUMBER = re.compile(r"-?[\d.,]*\d")

LANGUAGE_NAMES = {
    "afr": "Afrikaans",
    "eng": "English",
    "nso": "Sepedi",
    "sot": "Sesotho",
    "ssw": "siSwati",
    "tsn": "Setswana",
    "tso": "Xitsonga",
    "ven": "Tshivenda",
    "xho": "isiXhosa",
    "zul": "isiZulu",
}

BENCHMARK_SPECS = {
    "belebele": {
        "languages": ("afr", "eng", "nso", "sot", "ssw", "tsn", "tso", "xho", "zul"),
        "max_new_tokens": 16,
    },
    "sib": {
        "languages": ("afr", "eng", "nso", "sot", "ssw", "tsn", "tso", "xho", "zul"),
        "max_new_tokens": 16,
    },
    "afrixnli": {
        "languages": ("eng", "sot", "xho", "zul"),
        "max_new_tokens": 16,
    },
    "afrimmlu": {
        "languages": ("eng", "sot", "xho", "zul"),
        "max_new_tokens": 16,
    },
    "uhura": {
        "languages": ("eng", "nso", "zul"),
        "max_new_tokens": 16,
    },
    "injongo": {
        "languages": ("eng", "sot", "xho", "zul"),
        "max_new_tokens": 24,
    },
    "afrimgsm": {
        "languages": ("eng", "sot", "xho", "zul"),
        "max_new_tokens": 1024,
    },
    "flores_en2sa": {
        "languages": ("afr", "nso", "sot", "ssw", "tsn", "tso", "xho", "zul"),
        "max_new_tokens": 256,
    },
    "flores_sa2en": {
        "languages": ("afr", "nso", "sot", "ssw", "tsn", "tso", "xho", "zul"),
        "max_new_tokens": 256,
    },
}

SIB_LABELS = [
    "science/technology",
    "travel",
    "politics",
    "sports",
    "health",
    "entertainment",
    "geography",
]
XNLI_LABELS = ["entailment", "neutral", "contradiction"]
UHURA_CONFIG = {"eng": "en_multiple_choice", "nso": "nso_multiple_choice", "zul": "zu_multiple_choice"}


def format_mc(question: str, options: list[str]) -> str:
    letters = ", ".join(LETTERS[: len(options)])
    lines = [MC_INSTRUCTION.format(letters=letters), "", f"Question: {question}", ""]
    lines += [f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options)]
    return "\n".join(lines + ["", "Answer:"])


def predicted_letter(text: str, count: int) -> str | None:
    match = re.search(rf"\b([{LETTERS[:count]}])\b", text.strip())
    return match.group(1) if match else None


def predicted_number(text: str) -> float | None:
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    numbers = NUMBER.findall(boxed[-1] if boxed else text)
    if not numbers:
        return None
    try:
        return float(numbers[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def load_items(benchmark: str, lang: str, split: str):
    from datasets import load_dataset

    lang_name = LANGUAGE_NAMES[lang]
    if benchmark == "belebele":
        if split != "test":
            return []
        ds = load_dataset("facebook/belebele", f"{lang}_Latn", split="test")
        return [
            (
                [{"role": "user", "content": format_mc(f"{d['flores_passage']}\n\n{d['question']}", [d[f"mc_answer{i}"] for i in range(1, 5)])}],
                LETTERS[int(d["correct_answer_num"]) - 1],
                "letter",
                4,
            )
            for d in ds
        ]
    if benchmark == "sib":
        ds = load_dataset("Davlan/sib200", f"{lang}_Latn", split=split)
        return [
            (
                [{"role": "user", "content": format_mc(f"What is the topic of this text?\n\n{d['text']}", SIB_LABELS)}],
                LETTERS[SIB_LABELS.index(d["category"])],
                "letter",
                len(SIB_LABELS),
            )
            for d in ds
        ]
    if benchmark == "afrixnli":
        ds = load_dataset("masakhane/afrixnli", lang, split=split)
        return [
            (
                [
                    {
                        "role": "user",
                        "content": format_mc(
                            f"Premise: {d['premise']}\nHypothesis: {d['hypothesis']}\n"
                            "Does the premise entail, contradict, or say nothing about the hypothesis?",
                            XNLI_LABELS,
                        ),
                    }
                ],
                LETTERS[int(d["label"])],
                "letter",
                3,
            )
            for d in ds
        ]
    if benchmark == "afrimmlu":
        ds = load_dataset("masakhane/afrimmlu", lang, split=split)
        items = []
        for d in ds:
            opts = ast.literal_eval(d["choices"])
            items.append(
                ([{"role": "user", "content": format_mc(d["question"], opts)}], d["answer"].strip(), "letter", len(opts))
            )
        return items
    if benchmark == "uhura":
        if split != "test":
            return []
        ds = load_dataset("masakhane/uhura-arc-easy", UHURA_CONFIG[lang], split="test")
        items = []
        for d in ds:
            opts, labels = d["choices"]["text"], d["choices"]["label"]
            items.append(
                ([{"role": "user", "content": format_mc(d["question"], opts)}], LETTERS[labels.index(d["answerKey"])], "letter", len(opts))
            )
        return items
    if benchmark == "injongo":
        ds = load_dataset("masakhane/InjongoIntent", lang, split=split)
        intents = sorted({d["intent"] for d in ds})
        return [
            (
                [
                    {
                        "role": "user",
                        "content": (
                            f"Classify the intent of this {lang_name} text. "
                            f"Reply with ONLY one intent from this list: {', '.join(intents)}.\n\n"
                            f"Text: {d['text']}\nIntent:"
                        ),
                    }
                ],
                d["intent"],
                "label",
                len(intents),
            )
            for d in ds
        ]
    if benchmark == "afrimgsm":
        if split != "test":
            return []
        ds = load_dataset("masakhane/afrimgsm", lang, split="test")
        math_prompt = (
            f"You are a helpful assistant that solves math word problems. "
            f"Reason step by step in {lang_name}, and put your final numerical answer inside \\boxed{{}}."
        )
        return [
            (
                [{"role": "user", "content": f"{math_prompt}\n\n{d['question']}"}],
                float(d["answer_number"]),
                "number",
                0,
            )
            for d in ds
        ]
    if benchmark in ("flores_en2sa", "flores_sa2en"):
        hf_split = "dev" if split == "validation" else "devtest"
        ds = load_dataset("facebook/flores", f"eng_Latn-{lang}_Latn", split=hf_split)
        en2sa = benchmark == "flores_en2sa"
        src_name, tgt_name = ("English", lang_name) if en2sa else (lang_name, "English")
        src_key, tgt_key = ("sentence_eng_Latn", f"sentence_{lang}_Latn") if en2sa else (f"sentence_{lang}_Latn", "sentence_eng_Latn")
        return [
            (
                [
                    {
                        "role": "user",
                        "content": f"Translate the following sentence from {src_name} to {tgt_name}. Reply with ONLY the translation.\n\n{src_name}: {d[src_key]}\n{tgt_name}:",
                    }
                ],
                d[tgt_key],
                "translation",
                0,
            )
            for d in ds
        ]
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def score_item(kind: str, output: str, gold, option_count: int) -> float:
    if kind == "letter":
        return 1.0 if predicted_letter(output, option_count) == gold else 0.0
    if kind == "label":
        pred = output.strip().split("\n")[0].strip().strip(".").lower()
        return 1.0 if pred == str(gold).lower() else 0.0
    if kind == "number":
        return 1.0 if predicted_number(output) == gold else 0.0
    raise ValueError(kind)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat-template AfroBench eval for instruct Gemma/LLM checkpoints.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tasks", default="belebele,sib,afrixnli,afrimmlu,uhura,injongo,afrimgsm,flores_en2sa,flores_sa2en")
    parser.add_argument("--languages", default="afr,eng,nso,sot,ssw,tsn,tso,xho,zul")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    langs = [c.strip() for c in args.languages.split(",") if c.strip()]

    import torch
    from sacrebleu.metrics import CHRF
    from transformers import AutoTokenizer

    dtype = _resolve_dtype(args.dtype, torch)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=args.trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = _load_model(
        args.model,
        revision=args.revision,
        dtype=dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    for task in tasks:
        if task not in BENCHMARK_SPECS:
            raise ValueError(f"Unknown chat task '{task}'. Valid: {sorted(BENCHMARK_SPECS)}")
        spec = BENCHMARK_SPECS[task]
        max_new_tokens = spec["max_new_tokens"]
        for lang in langs:
            if lang not in spec["languages"]:
                continue
            items = load_items(task, lang, args.split)
            if not items:
                continue
            if args.limit is not None:
                items = items[: args.limit]

            prompts = [
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                for msgs, _, _, _ in items
            ]
            outputs: list[str] = []
            with torch.inference_mode():
                for start in range(0, len(prompts), args.batch_size):
                    batch_prompts = prompts[start : start + args.batch_size]
                    enc = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(args.device)
                    gen = model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    prompt_len = enc["input_ids"].shape[1]
                    decoded = tokenizer.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)
                    outputs.extend(decoded)

            key = f"{task}_{lang}"
            kind = items[0][2]
            records = []
            if kind == "translation":
                hyps = [o.strip().split("\n")[0].strip() for o in outputs]
                refs = [str(gold).strip() for _, gold, _, _ in items]
                chrf = CHRF(word_order=0).corpus_score(hyps, [refs]).score
                chrf_pp = CHRF(word_order=2).corpus_score(hyps, [refs]).score
                summary[key] = {"chrf": chrf, "chrf_pp": chrf_pp, "n": len(items)}
                print(f"[{key}] n={len(items)} chrF={chrf:.1f} chrF++={chrf_pp:.1f}", flush=True)
                for (_, gold, _, _), out, hyp in zip(items, outputs, hyps):
                    records.append({"gold": gold, "output": out, "first_line": hyp})
            else:
                hits = [
                    score_item(kind, out, gold, opt_count)
                    for (_, gold, _, opt_count), out in zip(items, outputs)
                ]
                mean, lo, hi = bootstrap_ci(hits)
                summary[key] = {"accuracy": 100.0 * mean, "ci_low": 100.0 * lo, "ci_high": 100.0 * hi, "n": len(items)}
                print(f"[{key}] n={len(items)} acc={100.0 * mean:.1f} [{100.0 * lo:.1f}, {100.0 * hi:.1f}]", flush=True)
                for (_, gold, _, _), out, hit in zip(items, outputs, hits):
                    records.append({"gold": gold, "output": out, "correct": bool(hit)})

            (args.output_dir / f"samples_{key}.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
            )

    (args.output_dir / "chat_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSaved chat summary to {args.output_dir / 'chat_summary.json'}")


if __name__ == "__main__":
    main()
