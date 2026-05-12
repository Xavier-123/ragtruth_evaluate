from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Literal

import pandas as pd


def infer_root() -> Path:
    here = Path(__file__).resolve().parent
    if (here / "src").exists():
        return here
    for parent in Path(__file__).resolve().parents:
        if (parent / "src").exists():
            return parent
    return here


ROOT = infer_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lhrag.graph import AgenticRAGWorkflow, build_agentic_rag_graph
from lhrag.config import (
    EvidenceRefinerConfig,
    OrchestratorConfig,
    QueryProcessorConfig,
    ReflectiveReasonerConfig,
)
from lhrag.llm import OpenAIClient
from lhrag.modules.evidence_refiner import EvidenceRefiner
from lhrag.modules.orchestrator import Orchestrator
from lhrag.modules.query_processor import QueryProcessor
from lhrag.modules.reflective_reasoner import ReflectiveReasoner
from lhrag.schema import EvidenceItem, FollowupPlan, RetrievalOutput, TaskExecutionState
from lhrag.state import AgenticRAGState, create_initial_state
from lhrag.utils import configure_logging


DEFAULT_DATASET_PATH = (
    ROOT / "data" / "ragtruth-processed" / "test-00000-of-00001.parquet"
)
DEFAULT_OUTPUT_DIR = ROOT / "output" / "ragtruth"
DEFAULT_LOG_DIR = ROOT / "logs" / "ragtruth"
RunMode = Literal["agentic", "naive"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run agentic RAG on RAGTruth-processed with oracle context. "
            "This script is isolated and does not modify project runtime code."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"RAGTruth-processed parquet path. Defaults to {DEFAULT_DATASET_PATH}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL path. Defaults to "
            "output/ragtruth/ragtruth_processed_oracle_<mode>_<timestamp>.jsonl."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N samples after --start.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start offset in the parquet file. Defaults to 0.",
    )
    parser.add_argument(
        "--enable-answer-critique",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable or disable ReflectiveReasoner answer critique for agentic mode. "
            "Naive mode does not run the reflective reasoner."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("agentic", "naive"),
        default="agentic",
        help="Workflow mode. Defaults to agentic.",
    )
    parser.add_argument(
        "--enable-decomposition",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable query decomposition for agentic mode. Defaults to disabled.",
    )
    parser.add_argument(
        "--llm-config",
        type=Path,
        default=None,
        help="Optional LLM config path used by reasoner, critic, and final answer generation.",
    )
    parser.add_argument(
        "--critic-llm-config",
        type=Path,
        default=None,
        help="Optional separate LLM config for the answer critic. Defaults to --llm-config/client.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Workflow max attempts.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum concurrent sample workers. Defaults to 1.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Optional log path. Defaults to logs/ragtruth/ragtruth_processed_oracle_<timestamp>.log.",
    )
    return parser.parse_args()


def dataclass_to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [dataclass_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: dataclass_to_jsonable(item) for key, item in value.items()}
    return value


def normalize_cell(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def load_samples(path: Path, *, start: int, limit: int | None) -> list[dict[str, Any]]:
    if start < 0:
        raise ValueError("--start must be >= 0")
    if limit is not None and limit < 0:
        raise ValueError("--limit must be >= 0")

    df = pd.read_parquet(path)
    if start:
        df = df.iloc[start:]
    if limit is not None:
        df = df.head(limit)

    records: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        records.append({key: normalize_cell(value) for key, value in record.items()})
    return records


class OracleRetriever:
    """Retriever-compatible oracle that returns the current sample context."""

    def __init__(self, sample: dict[str, Any]) -> None:
        self.sample = sample

    def run(self, state: AgenticRAGState) -> dict[str, Any]:
        query_graph = state.get("query_graph")
        node_id = state.get("active_node_id")
        if not node_id:
            raise ValueError("OracleRetriever requires active_node_id.")

        question = state.get("query", "")
        retrieval_query = question
        if query_graph is not None and node_id in query_graph.nodes:
            node = query_graph.nodes[node_id]
            question = node.question
            retrieval_query = node.retrieval_query or node.question

        sample_id = str(self.sample.get("id") or "")
        context = str(self.sample.get("context") or "").strip()
        evidences: list[EvidenceItem] = []
        if context:
            evidences.append(
                EvidenceItem(
                    evidence_id=f"oracle:{sample_id or node_id}",
                    content=context,
                    source_type="text",
                    source_id="RAGTruth-processed",
                    metadata={
                        "sample_id": sample_id,
                        "node_id": node_id,
                        "oracle_context": True,
                    },
                    score=1.0,
                )
            )

        output = RetrievalOutput(
            evidences=evidences,
            retrieval_logs=[
                "OracleRetriever returned the sample context.",
                f"question={question}",
                f"retrieval_query={retrieval_query}",
                f"context_chars={len(context)}",
            ],
        )

        node_results = dict(state.get("node_results", {}))
        task_state = node_results.get(node_id, TaskExecutionState(node_id=node_id))
        task_state.retrieval_output = output
        task_state.accumulated_evidence = list(evidences)
        node_results[node_id] = task_state

        return {
            "raw_evidence": evidences,
            "retrieval_output": output,
            "node_results": node_results,
            "logs": [*state.get("logs", []), "oracle_retriever completed"],
        }


class SinglePassReflectiveReasoner(ReflectiveReasoner):
    """Experiment-only reasoner that prevents follow-up expansion/retrieval loops."""

    def build_followup_plan(
        self,
        query_graph,
        node_id,
        *,
        log_context=None,
    ) -> FollowupPlan:
        return FollowupPlan(
            action="final",
            target_node_id=node_id,
            followup_queries=[],
            supplemental_queries=[],
            rationale="Single-pass oracle experiment disables follow-up planning.",
        )


def create_workflow(
    sample: dict[str, Any],
    *,
    mode: RunMode,
    enable_answer_critique: bool,
    enable_decomposition: bool,
    llm_config_path: Path | None,
    critic_llm_config_path: Path | None,
) -> AgenticRAGWorkflow:
    llm_client = OpenAIClient.from_config(config_path=llm_config_path)
    critic_client = (
        OpenAIClient.from_config(config_path=critic_llm_config_path)
        if critic_llm_config_path is not None
        else llm_client
    )
    query_processor = QueryProcessor(
        config=QueryProcessorConfig(
            enable_decomposition=enable_decomposition if mode == "agentic" else False,
            enable_entity_extraction=False,
            enable_relation_extraction=False,
        ),
        llm_client=llm_client,
        mode=mode,
    )
    reasoner = SinglePassReflectiveReasoner(
        config=ReflectiveReasonerConfig(
            default_confidence=0.0,
            enable_answer_critique=enable_answer_critique,
            enable_followup_queries=False,
            enable_followup_query_llm_validation=False,
            enable_followup_query_critic=False,
        ),
        llm_client=llm_client,
    )
    reasoner._critic_llm_client = critic_client
    return AgenticRAGWorkflow(
        orchestrator=Orchestrator(
            config=OrchestratorConfig(default_max_attempts=3, default_should_retrieve=True),
            query_processor=query_processor,
            llm_client=llm_client,
            mode=mode,
        ),
        query_processor=query_processor,
        retriever=OracleRetriever(sample),
        evidence_refiner=EvidenceRefiner(config=EvidenceRefinerConfig(min_score_threshold=0.0)),
        reflective_reasoner=reasoner,
    )


def extract_node_records(run_result: dict[str, Any]) -> list[dict[str, Any]]:
    query_graph = run_result.get("query_graph") or {}
    graph_nodes = query_graph.get("nodes") if isinstance(query_graph, dict) else {}
    node_results = run_result.get("node_results") or {}
    records: list[dict[str, Any]] = []
    if not isinstance(node_results, dict):
        return records
    for node_id, task_state in node_results.items():
        if not isinstance(task_state, dict):
            continue
        reasoning = task_state.get("reasoning_output") or {}
        refiner_output = task_state.get("refiner_output") or {}
        retrieval_output = task_state.get("retrieval_output") or {}
        graph_node = graph_nodes.get(node_id, {}) if isinstance(graph_nodes, dict) else {}
        records.append(
            {
                "node_id": node_id,
                "question": graph_node.get("question", ""),
                "answer": reasoning.get("answer", "") if isinstance(reasoning, dict) else "",
                "known_information": (
                    reasoning.get("known_information", []) if isinstance(reasoning, dict) else []
                ),
                "critique": reasoning.get("critique", "") if isinstance(reasoning, dict) else "",
                "supporting_evidence_count": len(
                    reasoning.get("supporting_evidence", [])
                    if isinstance(reasoning, dict)
                    and isinstance(reasoning.get("supporting_evidence", []), list)
                    else []
                ),
                "refined_evidence_count": len(
                    refiner_output.get("refined_evidences", [])
                    if isinstance(refiner_output, dict)
                    and isinstance(refiner_output.get("refined_evidences", []), list)
                    else []
                ),
                "retrieved_evidence_count": len(
                    retrieval_output.get("evidences", [])
                    if isinstance(retrieval_output, dict)
                    and isinstance(retrieval_output.get("evidences", []), list)
                    else []
                ),
            }
        )
    return records


def build_record(
    *,
    sample_index: int,
    sample: dict[str, Any],
    run_result: dict[str, Any],
    status: str,
    error: str | None,
    mode: RunMode,
    enable_answer_critique: bool,
    enable_decomposition: bool,
) -> dict[str, Any]:
    final_answer = run_result.get("final_answer") or {}
    if not isinstance(final_answer, dict):
        final_answer = {}
    return {
        "index": sample_index,
        "sample_id": sample.get("id", ""),
        "id": sample.get("id", ""),
        "query": sample.get("query", ""),
        "question": sample.get("query", ""),
        "reference_output": sample.get("output", ""),
        "task_type": sample.get("task_type", ""),
        "quality": sample.get("quality", ""),
        "model": sample.get("model", ""),
        "temperature": sample.get("temperature", ""),
        "hallucination_labels": sample.get("hallucination_labels", []),
        "hallucination_labels_processed": sample.get("hallucination_labels_processed", []),
        "input_str": sample.get("input_str", ""),
        "context_chars": len(str(sample.get("context") or "")),
        "mode": mode,
        "oracle_context": True,
        "single_pass": mode == "agentic",
        "enable_answer_critique": enable_answer_critique,
        "enable_decomposition": enable_decomposition if mode == "agentic" else False,
        "predicted_answer": final_answer.get("answer", ""),
        "predicted_evidence": final_answer.get("evidence", []),
        "reasoning_summary": final_answer.get("reasoning_summary", ""),
        "node_records": extract_node_records(run_result),
        "logs": run_result.get("logs", []),
        "status": status,
        **({"error": error} if error else {}),
    }


def build_output_path(
    explicit_path: Path | None,
    *,
    mode: RunMode,
    enable_answer_critique: bool,
    run_id: str,
) -> Path:
    if explicit_path is not None:
        return explicit_path
    if mode == "naive":
        return DEFAULT_OUTPUT_DIR / f"ragtruth_processed_oracle_naive_{run_id}.jsonl"
    critic_label = "critic_on" if enable_answer_critique else "critic_off"
    return DEFAULT_OUTPUT_DIR / f"ragtruth_processed_oracle_agentic_{critic_label}_{run_id}.jsonl"


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


def run_sample(
    *,
    sample_index: int,
    sample: dict[str, Any],
    mode: RunMode,
    enable_answer_critique: bool,
    enable_decomposition: bool,
    llm_config_path: Path | None,
    critic_llm_config_path: Path | None,
    max_attempts: int,
) -> dict[str, Any]:
    query = str(sample.get("query") or "")
    try:
        workflow = create_workflow(
            sample,
            mode=mode,
            enable_answer_critique=enable_answer_critique,
            enable_decomposition=enable_decomposition,
            llm_config_path=llm_config_path,
            critic_llm_config_path=critic_llm_config_path,
        )
        app = build_agentic_rag_graph(workflow=workflow, mode=mode)
        initial_state = create_initial_state(query=query, max_attempts=max_attempts)
        run_result = dataclass_to_jsonable(app.invoke(initial_state))
        status = "ok"
        error = None
    except Exception as exc:  # noqa: BLE001
        run_result = {}
        status = "error"
        error = str(exc)

    return build_record(
        sample_index=sample_index,
        sample=sample,
        run_result=run_result,
        status=status,
        error=error,
        mode=mode,
        enable_answer_critique=enable_answer_critique,
        enable_decomposition=enable_decomposition,
    )


def build_payload(
    *,
    dataset_path: Path,
    record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": "RAGTruth-processed",
        "dataset_path": str(dataset_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": record,
    }


def main() -> int:
    args = parse_args()
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be >= 1")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = args.log or DEFAULT_LOG_DIR / f"ragtruth_processed_oracle_{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(log_path=log_path)

    samples = load_samples(args.dataset, start=args.start, limit=args.limit)
    output_path = build_output_path(
        args.output,
        mode=args.mode,
        enable_answer_critique=args.enable_answer_critique,
        run_id=run_id,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.touch(exist_ok=True)

    indexed_samples = list(enumerate(samples, start=args.start))
    total = len(indexed_samples)

    if args.max_workers == 1:
        for ordinal, (offset, sample) in enumerate(indexed_samples, start=1):
            query = str(sample.get("query") or "")
            print(
                f"[{ordinal}/{total}] id={sample.get('id', '')} "
                f"mode={args.mode} critic={args.enable_answer_critique} {query}",
                flush=True,
            )
            record = run_sample(
                sample_index=offset,
                sample=sample,
                mode=args.mode,
                enable_answer_critique=args.enable_answer_critique,
                enable_decomposition=args.enable_decomposition,
                llm_config_path=args.llm_config,
                critic_llm_config_path=args.critic_llm_config,
                max_attempts=args.max_attempts,
            )
            append_jsonl(
                output_path,
                build_payload(dataset_path=args.dataset, record=record),
            )
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_sample = {
                executor.submit(
                    run_sample,
                    sample_index=offset,
                    sample=sample,
                    mode=args.mode,
                    enable_answer_critique=args.enable_answer_critique,
                    enable_decomposition=args.enable_decomposition,
                    llm_config_path=args.llm_config,
                    critic_llm_config_path=args.critic_llm_config,
                    max_attempts=args.max_attempts,
                ): (ordinal, offset, sample)
                for ordinal, (offset, sample) in enumerate(indexed_samples, start=1)
            }

            completed = 0
            for future in as_completed(future_to_sample):
                ordinal, _, sample = future_to_sample[future]
                record = future.result()
                completed += 1
                print(
                    f"[{completed}/{total}] completed original_order={ordinal} "
                    f"id={sample.get('id', '')} mode={args.mode} "
                    f"critic={args.enable_answer_critique} status={record.get('status')}",
                    flush=True,
                )
                append_jsonl(
                    output_path,
                    build_payload(dataset_path=args.dataset, record=record),
                )

    print(f"Wrote {len(samples)} records to {output_path}")
    print(f"Log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
