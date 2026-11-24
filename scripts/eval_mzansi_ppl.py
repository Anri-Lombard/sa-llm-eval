#!/usr/bin/env python3
"""Evaluate perplexity and bits per UTF-8 byte on uctnlp/mzansi-text-deduplicated."""

from __future__ import annotations

import argparse
import json
import math
import time
import unicodedata
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path

ALL_SA_LANGUAGES = (
    "afr",
    "eng",
    "nbl",
    "nso",
    "sot",
    "ssw",
    "tsn",
    "tso",
    "ven",
    "xho",
    "zul",
)
DEFAULT_CORPUS = "uctnlp/mzansi-text-deduplicated"
DEFAULT_CORPUS_REVISION = "08bee62b1cee62a096628b53535145c5ce617421"


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def window_document_tokens(
    token_ids: list[int], *, prefix_token_id: int, sequence_length: int
):
    """Split one document into non-overlapping prediction windows with 1-token overlap context."""
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if not token_ids:
        return
    stride = sequence_length - 1
    for start in range(0, len(token_ids), stride):
        content = token_ids[start : start + stride]
        context = prefix_token_id if start == 0 else token_ids[start - 1]
        yield {
            "input_ids": [context, *content],
            "labels": [-100, *content],
        }


def aggregate_language_metrics(
    *, total_nll: float, predicted_tokens: int, utf8_bytes: int, documents: int, windows: int
) -> dict[str, float | int]:
    if predicted_tokens < 1 or utf8_bytes < 1:
        raise ValueError("predicted_tokens and utf8_bytes must be positive")
    mean_nll = total_nll / predicted_tokens
    return {
        "documents": documents,
        "windows": windows,
        "predicted_tokens": predicted_tokens,
        "utf8_bytes": utf8_bytes,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "bits_per_utf8_byte": total_nll / (utf8_bytes * math.log(2)),
        "tokens_per_1000_utf8_bytes": predicted_tokens * 1000.0 / utf8_bytes,
    }


def _batched(iterator, size: int):
    while batch := list(islice(iterator, size)):
        yield batch


def _resolve_dtype(name: str, torch):
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def _load_model(model_id: str, *, revision: str | None, dtype, device: str, trust_remote_code: bool):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        model_id, revision=revision, trust_remote_code=trust_remote_code
    )
    # Gemma 3 4B/12B/27B checkpoints use Gemma3ForConditionalGeneration (image-text-to-text)
    # rather than a pure text CausalLM class.
    architectures = getattr(config, "architectures", None) or []
    is_multimodal = hasattr(config, "vision_config") or any(
        "ConditionalGeneration" in arch for arch in architectures
    )
    kwargs = dict(
        revision=revision,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    if is_multimodal:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model.to(device).eval()


def evaluate_language(
    *,
    model,
    tokenizer,
    texts: list[str],
    prefix_token_id: int,
    pad_token_id: int,
    sequence_length: int,
    batch_size: int,
    device: str,
    dtype,
    torch,
) -> dict[str, float | int]:
    import torch.nn.functional as F

    cleaned = [normalize_text(t) for t in texts]
    cleaned = [t for t in cleaned if t]
    windows = (
        window
        for text in cleaned
        for window in window_document_tokens(
            tokenizer.encode(text, add_special_tokens=False),
            prefix_token_id=prefix_token_id,
            sequence_length=sequence_length,
        )
    )

    total_nll = 0.0
    predicted_tokens = 0
    window_count = 0
    device_type = device.split(":")[0]
    use_autocast = device_type == "cuda" and dtype in (torch.bfloat16, torch.float16)

    with torch.inference_mode():
        for batch in _batched(windows, batch_size):
            max_len = max(len(item["input_ids"]) for item in batch)
            input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros_like(input_ids)
            labels = torch.full_like(input_ids, -100)
            for idx, item in enumerate(batch):
                length = len(item["input_ids"])
                input_ids[idx, :length] = torch.tensor(item["input_ids"], dtype=torch.long, device=device)
                attention_mask[idx, :length] = 1
                labels[idx, :length] = torch.tensor(item["labels"], dtype=torch.long, device=device)

            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

            shifted_labels = labels[:, 1:]
            total_nll += float(
                F.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                    shifted_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
            )
            predicted_tokens += int((shifted_labels != -100).sum())
            window_count += len(batch)

    utf8_bytes = sum(len(text.encode("utf-8")) for text in cleaned)
    return aggregate_language_metrics(
        total_nll=total_nll,
        predicted_tokens=predicted_tokens,
        utf8_bytes=utf8_bytes,
        documents=len(cleaned),
        windows=window_count,
    )


def format_markdown_table(per_language: dict[str, dict[str, float | int]]) -> str:
    lines = [
        "| Language | Docs | Tokens | PPL | Bits/UTF-8 byte | Tok/1k bytes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lang, m in per_language.items():
        lines.append(
            f"| {lang} | {m['documents']} | {m['predicted_tokens']:,} | "
            f"{m['perplexity']:.3f} | {m['bits_per_utf8_byte']:.4f} | "
            f"{m['tokens_per_1000_utf8_bytes']:.1f} |"
        )
    sa_only = [m for lang, m in per_language.items() if lang != "eng"]
    if sa_only:
        mean_bpb = sum(m["bits_per_utf8_byte"] for m in sa_only) / len(sa_only)
        geo_ppl = math.exp(sum(math.log(m["perplexity"]) for m in sa_only) / len(sa_only))
        lines.append(f"| **SA macro (excl. eng)** | - | - | **{geo_ppl:.3f}** | **{mean_bpb:.4f}** | - |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure per-language perplexity and bits/UTF-8 byte on MzansiText."
    )
    parser.add_argument("--model", required=True, help="HF model ID or local checkpoint path.")
    parser.add_argument("--revision", default=None, help="Optional HF model revision.")
    parser.add_argument(
        "--languages",
        default=",".join(ALL_SA_LANGUAGES),
        help="Comma-separated language codes (default: all 11 SA languages).",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Dataset split (default: validation; keep test held out during CPT selection).",
    )
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--corpus-revision", default=DEFAULT_CORPUS_REVISION)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-docs", type=int, default=None, help="Optional per-language doc cap for smoke tests.")
    parser.add_argument("--dtype", default="bfloat16", help="bfloat16, float16, or float32.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    languages = [code.strip() for code in args.languages.split(",") if code.strip()]
    unknown = sorted(set(languages) - set(ALL_SA_LANGUAGES))
    if unknown:
        raise ValueError(f"Unknown language codes: {unknown}. Valid: {ALL_SA_LANGUAGES}")

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    dtype = _resolve_dtype(args.dtype, torch)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=args.trust_remote_code
    )
    # Gemma tokenizers define bos_token_id; Qwen tokenizers omit it and use eos_token_id.
    prefix_token_id = (
        tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    )
    if prefix_token_id is None:
        raise ValueError("Tokenizer has neither bos_token_id nor eos_token_id")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else prefix_token_id

    corpus = load_dataset(args.corpus, revision=args.corpus_revision, split=args.split)
    by_lang: dict[str, list[str]] = {lang: [] for lang in languages}
    wanted = set(languages)
    for row in corpus:
        lang = row["lang"]
        if lang in wanted:
            if args.max_docs is None or len(by_lang[lang]) < args.max_docs:
                by_lang[lang].append(row["text"])

    started = time.monotonic()
    model = _load_model(
        args.model,
        revision=args.revision,
        dtype=dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )

    per_language: dict[str, dict[str, float | int]] = {}
    for lang in languages:
        texts = by_lang[lang]
        if not texts:
            raise RuntimeError(f"No documents found for language={lang} in {args.corpus}:{args.split}")
        per_language[lang] = evaluate_language(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            prefix_token_id=prefix_token_id,
            pad_token_id=pad_token_id,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            device=args.device,
            dtype=dtype,
            torch=torch,
        )
        m = per_language[lang]
        print(
            f"[{lang}] docs={m['documents']} tokens={m['predicted_tokens']:,} "
            f"ppl={m['perplexity']:.3f} bpb={m['bits_per_utf8_byte']:.4f}",
            flush=True,
        )

    elapsed = time.monotonic() - started
    payload = {
        "model": args.model,
        "revision": args.revision,
        "corpus": args.corpus,
        "corpus_revision": args.corpus_revision,
        "split": args.split,
        "held_out_test_accessed": args.split == "test",
        "sequence_length": args.sequence_length,
        "prefix_token_id": prefix_token_id,
        "dtype": args.dtype,
        "max_docs": args.max_docs,
        "elapsed_seconds": round(elapsed, 2),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "per_language": per_language,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")

    print("\n" + format_markdown_table(per_language))
    print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
