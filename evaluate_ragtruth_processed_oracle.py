from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
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
    """Derive a boolean hallucination ground-truth from the RAGTruth-processed fields.

    Priority order:
    1. ``hallucination_labels_processed`` – a count dict such as
       ``{"evident_conflict": 0, "baseless_info": 1}``.
       Hallucinated when the sum of all label counts is > 0.
    2. ``hallucination_labels`` – a list of span-annotation dicts (each entry
       represents one annotated hallucination span).
       Hallucinated when the list is non-empty.

    Returns ``True`` (hallucinated), ``False`` (clean), or ``None`` (no usable label).
    """
    processed = record.get("hallucination_labels_processed")
    if isinstance(processed, dict) and processed:
        # Any label count > 0 indicates at least one hallucination
        return sum(v for v in processed.values() if isinstance(v, (int, float))) > 0

    labels = record.get("hallucination_labels")
    if isinstance(labels, list):
        # A non-empty list means at least one span was annotated as hallucination
        return len(labels) > 0

    return None


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
        "task_type": record.get("task_type", ""),
        "quality": record.get("quality", ""),
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
                "id": record.get("id"),
                "query": record.get("query", ""),
                "predicted_answer": final_answer,
                "reference_output": record.get("reference_output", ""),
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
        return list(executor.map(partial(evaluate_record, refusal_phrases=refusal_phrases), records))


def compute_binary_metrics(*, tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
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


def _empty_counts() -> dict[str, int]:
    return {"tp": 0, "tn": 0, "fp": 0, "fn": 0}



    """Return a fresh per-task-type accumulator dict."""
    return {
        "ok": 0,
        "total": 0,
        "final": _empty_counts(),
        "node": _empty_counts(),
        "final_labeled": 0,
        "node_labeled": 0,
    }


def _accumulate_confusion(counts: dict[str, int], pred: bool, gt: bool) -> None:
    if pred and gt:
        counts["tp"] += 1
    elif pred and not gt:
        counts["fp"] += 1
    elif not pred and gt:
        counts["fn"] += 1
    else:
        counts["tn"] += 1


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

    # Overall confusion matrices
    final_counts = _empty_counts()
    node_counts = _empty_counts()
    # Track separately so each can be counted independently (not AND-gated)
    final_labeled = 0
    node_labeled = 0

    # Per-task-type accumulators: task_type -> {"final": counts, "node": counts, "ok": int, "total": int}
    task_stats: dict[str, dict[str, Any]] = defaultdict(_empty_task_stats)
    # Per-quality sample counts (e.g. good / truncated / incorrect_refusal)
    quality_counts: dict[str, int] = defaultdict(int)

    for item, record in zip(evaluations, records, strict=True):
        task_type = item.get("task_type", "") or "unknown"
        quality = item.get("quality", "") or "unknown"
        task_stats[task_type]["total"] += 1
        quality_counts[quality] += 1

        if not item["is_ok"]:
            continue

        task_stats[task_type]["ok"] += 1

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

        # Update final metrics independently from node metrics
        if gt is not None and final_pred is not None:
            final_labeled += 1
            task_stats[task_type]["final_labeled"] += 1
            _accumulate_confusion(final_counts, final_pred, gt)
            _accumulate_confusion(task_stats[task_type]["final"], final_pred, gt)
            if item["label_mismatch_example"] is not None and len(label_mismatch_examples) < 20:
                label_mismatch_examples.append(item["label_mismatch_example"])

        if gt is not None and node_pred is not None:
            node_labeled += 1
            task_stats[task_type]["node_labeled"] += 1
            _accumulate_confusion(node_counts, node_pred, gt)
            _accumulate_confusion(task_stats[task_type]["node"], node_pred, gt)

    if ok == 0:
        zero_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
        return {
            "total": total,
            "ok": 0,
            "errors": errors,
            "error_rate": round(errors / total, 6) if total else 0.0,
            "quality_counts": dict(quality_counts),
            "final_refusal_rate": 0.0,
            "final_hallucination_rate_proxy": 0.0,
            "final_empty_answer_rate": 0.0,
            "node_refusal_rate": 0.0,
            "node_hallucination_rate_proxy": 0.0,
            "node_empty_answer_rate": 0.0,
            "final_labeled_count": 0,
            "node_labeled_count": 0,
            "final_label_counts": _empty_counts(),
            "final_label_metrics": zero_metrics,
            "node_label_counts": _empty_counts(),
            "node_label_metrics": zero_metrics,
            "by_task_type": {},
            "refusal_phrases": refusal_phrases,
            "label_mismatch_examples": [],
        }

    final_label_metrics = compute_binary_metrics(**final_counts)
    node_label_metrics = compute_binary_metrics(**node_counts)

    # Build per-task-type summary
    by_task_type: dict[str, Any] = {}
    for ttype, ts in task_stats.items():
        t_ok = ts["ok"]
        by_task_type[ttype] = {
            "total": ts["total"],
            "ok": t_ok,
            "final_labeled_count": ts["final_labeled"],
            "node_labeled_count": ts["node_labeled"],
            "final_label_counts": ts["final"],
            "final_label_metrics": compute_binary_metrics(**ts["final"]),
            "node_label_counts": ts["node"],
            "node_label_metrics": compute_binary_metrics(**ts["node"]),
        }

    return {
        "total": total,
        "ok": ok,
        "errors": errors,
        "error_rate": round(errors / total, 6) if total else 0.0,
        "quality_counts": dict(quality_counts),
        "final_refusal_rate": round(final_refusals / ok, 6),
        "final_hallucination_rate_proxy": round(1.0 - final_refusals / ok, 6),
        "final_empty_answer_rate": round(final_empty / ok, 6),
        "node_refusal_rate": round(node_refusals / ok, 6),
        "node_hallucination_rate_proxy": round(1.0 - node_refusals / ok, 6),
        "node_empty_answer_rate": round(node_empty / ok, 6),
        "final_labeled_count": final_labeled,
        "node_labeled_count": node_labeled,
        "final_label_counts": final_counts,
        "final_label_metrics": final_label_metrics,
        "node_label_counts": node_counts,
        "node_label_metrics": node_label_metrics,
        "by_task_type": by_task_type,
        "refusal_phrases": refusal_phrases,
        "label_mismatch_examples": label_mismatch_examples,
    }


def main() -> int:
    args = parse_args()
    if args.max_workers < 1:
        raise SystemExit(f"--max-workers must be >= 1, got {args.max_workers}")
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
