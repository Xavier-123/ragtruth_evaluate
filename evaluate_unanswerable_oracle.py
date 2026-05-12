from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
import string
from pathlib import Path
from typing import Any


DEFAULT_REFUSAL_PHRASES = [
    "unanswerable",
    "unknown",
    "no answer",
    "no information",
    "not enough information",
    "insufficient information",
    "cannot be determined",
    "can't be determined",
    "cannot determine",
    "can't determine",
    "not possible to determine",
    "unclear",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FaithEval Unanswerable oracle-context JSONL outputs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSONL produced by evaluation/faitheval/run_unanswerable_oracle.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for metrics. Defaults to <input_stem>_eval.json beside input.",
    )
    parser.add_argument(
        "--refusal-phrases",
        nargs="*",
        default=DEFAULT_REFUSAL_PHRASES,
        help="Accepted refusal phrases. Empty answers are always counted as refusal.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum concurrent record evaluation workers. Defaults to 1.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    lowered = text.lower()
    punctuation = set(string.punctuation)
    without_punc = "".join(" " if ch in punctuation else ch for ch in lowered)
    return " ".join(without_punc.split())


def is_refusal(answer: str, refusal_phrases: list[str]) -> bool:
    if not isinstance(answer, str):
        answer = str(answer)
    if not answer.strip():
        return True
    normalized = normalize_text(answer)
    for phrase in refusal_phrases:
        normalized_phrase = normalize_text(phrase)
        if normalized_phrase and re.search(rf"\b{re.escape(normalized_phrase)}\b", normalized):
            return True
    return False


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            result = payload.get("result")
            if isinstance(result, dict):
                records.append(result)
    return records


def get_first_node_answer(record: dict[str, Any]) -> str:
    node_records = record.get("node_records", [])
    if not isinstance(node_records, list):
        return ""
    for node_record in node_records:
        if not isinstance(node_record, dict):
            continue
        answer = node_record.get("answer", "")
        return answer if isinstance(answer, str) else str(answer)
    return ""


def get_first_node_critique(record: dict[str, Any]) -> str:
    node_records = record.get("node_records", [])
    if not isinstance(node_records, list):
        return ""
    for node_record in node_records:
        if not isinstance(node_record, dict):
            continue
        critique = node_record.get("critique", "")
        return critique if isinstance(critique, str) else str(critique)
    return ""


def evaluate_record(record: dict[str, Any], refusal_phrases: list[str]) -> dict[str, Any]:
    is_ok = record.get("status") == "ok"
    final_answer = record.get("predicted_answer", "")
    if not isinstance(final_answer, str):
        final_answer = str(final_answer)
    node_answer = get_first_node_answer(record)
    node_critique = get_first_node_critique(record)
    final_is_refusal = is_refusal(final_answer, refusal_phrases) if is_ok else False
    node_is_refusal = is_refusal(node_answer, refusal_phrases) if is_ok else False
    return {
        "is_ok": is_ok,
        "final_answer": final_answer,
        "node_answer": node_answer,
        "node_critique": node_critique,
        "final_is_refusal": final_is_refusal,
        "node_is_refusal": node_is_refusal,
        "final_is_empty": not final_answer.strip() if is_ok else False,
        "node_is_empty": not node_answer.strip() if is_ok else False,
        "critic_rejection_observed": bool(
            is_ok
            and record.get("enable_answer_critique")
            and node_is_refusal
            and node_critique.strip()
        ),
        "hallucination_example": (
            {
                "index": record.get("index"),
                "qid": record.get("qid") or record.get("sample_id"),
                "question": record.get("question", ""),
                "predicted_answer": final_answer,
                "node_answer": node_answer,
                "node_critique": node_critique,
                "justification": record.get("justification", ""),
            }
            if is_ok and not final_is_refusal
            else None
        ),
    }


def evaluate_records(
    records: list[dict[str, Any]],
    refusal_phrases: list[str],
    *,
    max_workers: int,
) -> list[dict[str, Any]]:
    if max_workers == 1:
        return [evaluate_record(record, refusal_phrases) for record in records]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(lambda record: evaluate_record(record, refusal_phrases), records))


def compute_metrics(
    records: list[dict[str, Any]],
    refusal_phrases: list[str],
    *,
    max_workers: int = 1,
) -> dict[str, Any]:
    total = len(records)
    evaluations = evaluate_records(records, refusal_phrases, max_workers=max_workers)
    ok = sum(1 for item in evaluations if item["is_ok"])
    errors = total - ok
    if ok == 0:
        return {
            "total": total,
            "ok": ok,
            "errors": errors,
            "error_rate": round(errors / total, 6) if total else 0.0,
            "enable_answer_critique_values": sorted(
                {
                    record.get("enable_answer_critique")
                    for record in records
                    if "enable_answer_critique" in record
                },
                key=lambda value: str(value),
            ),
            "final_refusal_accuracy": 0.0,
            "final_hallucination_rate": 0.0,
            "final_empty_answer_rate": 0.0,
            "node_refusal_accuracy": 0.0,
            "node_hallucination_rate": 0.0,
            "node_empty_answer_rate": 0.0,
            "critic_rejection_observed_rate": 0.0,
            "refusal_phrases": refusal_phrases,
            "final_hallucination_examples": [],
        }

    final_refusals = 0
    final_empty = 0
    node_refusals = 0
    node_empty = 0
    critic_rejections = 0
    final_hallucination_examples: list[dict[str, Any]] = []

    for item in evaluations:
        if not item["is_ok"]:
            continue
        if item["final_is_refusal"]:
            final_refusals += 1
        if item["final_is_empty"]:
            final_empty += 1
        if item["node_is_refusal"]:
            node_refusals += 1
        if item["node_is_empty"]:
            node_empty += 1
        if item["critic_rejection_observed"]:
            critic_rejections += 1
        if item["hallucination_example"] is not None and len(final_hallucination_examples) < 20:
            final_hallucination_examples.append(item["hallucination_example"])

    denominator = ok
    return {
        "total": total,
        "ok": ok,
        "errors": errors,
        "error_rate": round(errors / total, 6) if total else 0.0,
        "enable_answer_critique_values": sorted(
            {
                record.get("enable_answer_critique")
                for record in records
                if "enable_answer_critique" in record
            },
            key=lambda value: str(value),
        ),
        "final_refusal_accuracy": round(final_refusals / denominator, 6),
        "final_hallucination_rate": round(1.0 - final_refusals / denominator, 6),
        "final_empty_answer_rate": round(final_empty / denominator, 6),
        "node_refusal_accuracy": round(node_refusals / denominator, 6),
        "node_hallucination_rate": round(1.0 - node_refusals / denominator, 6),
        "node_empty_answer_rate": round(node_empty / denominator, 6),
        "critic_rejection_observed_rate": round(critic_rejections / denominator, 6),
        "refusal_phrases": refusal_phrases,
        "final_hallucination_examples": final_hallucination_examples,
    }


def main() -> int:
    args = parse_args()
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be >= 1")
    records = load_records(args.input)
    metrics = compute_metrics(records, args.refusal_phrases, max_workers=args.max_workers)
    output_path = args.output or args.input.with_name(f"{args.input.stem}_eval.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote evaluation to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
