from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
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

POSITIVE_LABELS = {
    "1",
    "true",
    "yes",
    "hallucination",
    "hallucinated",
    "hallucinating",
    "hallucinated_answer",
}
NEGATIVE_LABELS = {"0", "false", "no", "none", "non-hallucination", "non_hallucination"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RAGTruth-processed oracle-context JSONL outputs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSONL produced by run_ragtruth_processed_oracle.py.",
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


@lru_cache(maxsize=64)
def compile_refusal_patterns(refusal_phrases_key: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    patterns: list[re.Pattern[str]] = []
    for phrase in refusal_phrases_key:
        normalized_phrase = normalize_text(phrase)
        if not normalized_phrase:
            continue
        patterns.append(re.compile(rf"\b{re.escape(normalized_phrase)}\b"))
    return tuple(patterns)


def is_refusal(answer: str, refusal_phrases: list[str]) -> bool:
    if not isinstance(answer, str):
        answer = str(answer)
    if not answer.strip():
        return True
    normalized = normalize_text(answer)
    patterns = compile_refusal_patterns(tuple(refusal_phrases))
    for pattern in patterns:
        if pattern.search(normalized):
            return True
    return False


def normalize_label_token(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return "1" if value > 0 else "0"
    return normalize_text(str(value))


def parse_label_value(value: Any) -> bool | None:
    token = normalize_label_token(value)
    if token in POSITIVE_LABELS:
        return True
    if token in NEGATIVE_LABELS:
        return False
    return None


def parse_label_collection(values: list[Any]) -> bool | None:
    parsed = [parse_label_value(value) for value in values]
    positives = any(item is True for item in parsed)
    negatives = any(item is False for item in parsed)
    if positives:
        return True
    if negatives:
        return False
    return None


def parse_ground_truth_hallucination(record: dict[str, Any]) -> bool | None:
    value = record.get("hallucination_labels_processed")
    if value in (None, "", []):
        value = record.get("hallucination_labels")
    if value in (None, "", []):
        return None

    if isinstance(value, list):
        return parse_label_collection(value)
    if isinstance(value, tuple):
        return parse_label_collection(list(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed_json = json.loads(text)
        except json.JSONDecodeError:
            parsed_json = None
        if isinstance(parsed_json, list):
            return parse_label_collection(parsed_json)
        single = parse_label_value(text)
        if single is not None:
            return single
        lowered = normalize_text(text)
        if any(pos in lowered for pos in POSITIVE_LABELS):
            return True
        if any(neg in lowered for neg in NEGATIVE_LABELS):
            return False
        return None
    return parse_label_value(value)


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


def evaluate_record(record: dict[str, Any], refusal_phrases: list[str]) -> dict[str, Any]:
    is_ok = record.get("status") == "ok"
    final_answer = record.get("predicted_answer", "")
    if not isinstance(final_answer, str):
        final_answer = str(final_answer)
    node_answer = get_first_node_answer(record)
    final_is_refusal = is_refusal(final_answer, refusal_phrases) if is_ok else False
    node_is_refusal = is_refusal(node_answer, refusal_phrases) if is_ok else False
    gt_hallucination = parse_ground_truth_hallucination(record)
    final_pred_hallucination = (not final_is_refusal) if is_ok else None
    node_pred_hallucination = (not node_is_refusal) if is_ok else None
    return {
        "is_ok": is_ok,
        "final_is_refusal": final_is_refusal,
        "node_is_refusal": node_is_refusal,
        "final_is_empty": not final_answer.strip() if is_ok else False,
        "node_is_empty": not node_answer.strip() if is_ok else False,
        "gt_hallucination": gt_hallucination,
        "final_pred_hallucination": final_pred_hallucination,
        "node_pred_hallucination": node_pred_hallucination,
        "label_mismatch_example": (
            {
                "index": record.get("index"),
                "id": record.get("id") or record.get("sample_id"),
                "query": record.get("query") or record.get("question", ""),
                "predicted_answer": final_answer,
                "reference_output": record.get("reference_output") or record.get("output", ""),
                "gt_hallucination": gt_hallucination,
                "final_pred_hallucination": final_pred_hallucination,
            }
            if is_ok
            and gt_hallucination is not None
            and final_pred_hallucination is not None
            and final_pred_hallucination != gt_hallucination
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


def compute_binary_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "accuracy": round(accuracy, 6),
    }


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

    final_refusals = 0
    final_empty = 0
    node_refusals = 0
    node_empty = 0
    label_mismatch_examples: list[dict[str, Any]] = []

    final_tp = final_tn = final_fp = final_fn = 0
    node_tp = node_tn = node_fp = node_fn = 0
    labeled_count = 0

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

        gt = item["gt_hallucination"]
        final_pred = item["final_pred_hallucination"]
        node_pred = item["node_pred_hallucination"]
        if gt is not None and final_pred is not None and node_pred is not None:
            labeled_count += 1
            if final_pred and gt:
                final_tp += 1
            elif final_pred and not gt:
                final_fp += 1
            elif (not final_pred) and gt:
                final_fn += 1
            else:
                final_tn += 1

            if node_pred and gt:
                node_tp += 1
            elif node_pred and not gt:
                node_fp += 1
            elif (not node_pred) and gt:
                node_fn += 1
            else:
                node_tn += 1

            if item["label_mismatch_example"] is not None and len(label_mismatch_examples) < 20:
                label_mismatch_examples.append(item["label_mismatch_example"])

    if ok == 0:
        return {
            "total": total,
            "ok": 0,
            "errors": errors,
            "error_rate": round(errors / total, 6) if total else 0.0,
            "final_refusal_accuracy": 0.0,
            "final_hallucination_rate_proxy": 0.0,
            "final_empty_answer_rate": 0.0,
            "node_refusal_accuracy": 0.0,
            "node_hallucination_rate_proxy": 0.0,
            "node_empty_answer_rate": 0.0,
            "labeled_count": 0,
            "final_label_metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0},
            "node_label_metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0},
            "refusal_phrases": refusal_phrases,
            "label_mismatch_examples": [],
        }

    final_label_metrics = compute_binary_metrics(final_tp, final_tn, final_fp, final_fn)
    node_label_metrics = compute_binary_metrics(node_tp, node_tn, node_fp, node_fn)
    denominator = ok

    return {
        "total": total,
        "ok": ok,
        "errors": errors,
        "error_rate": round(errors / total, 6) if total else 0.0,
        "final_refusal_accuracy": round(final_refusals / denominator, 6),
        "final_hallucination_rate_proxy": round(1.0 - final_refusals / denominator, 6),
        "final_empty_answer_rate": round(final_empty / denominator, 6),
        "node_refusal_accuracy": round(node_refusals / denominator, 6),
        "node_hallucination_rate_proxy": round(1.0 - node_refusals / denominator, 6),
        "node_empty_answer_rate": round(node_empty / denominator, 6),
        "labeled_count": labeled_count,
        "final_label_counts": {
            "tp": final_tp,
            "tn": final_tn,
            "fp": final_fp,
            "fn": final_fn,
        },
        "node_label_counts": {
            "tp": node_tp,
            "tn": node_tn,
            "fp": node_fp,
            "fn": node_fn,
        },
        "final_label_metrics": final_label_metrics,
        "node_label_metrics": node_label_metrics,
        "refusal_phrases": refusal_phrases,
        "label_mismatch_examples": label_mismatch_examples,
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
