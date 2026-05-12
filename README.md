# FaithEval Oracle-Context Evaluation

This folder contains isolated evaluation scripts for FaithEval. The scripts do
not modify the project runtime code under `src/`.

## Task

The current scripts target `FaithEval-unanswerable-v1.0` with oracle context.
Each sample's official `context` is injected directly as retrieved evidence, so
the experiment removes retrieval quality from the comparison and focuses on
whether the answer generation path refuses unanswerable questions.

Default dataset:

```bash
data/faitheval/unanswerable-v1.0/data/test-00000-of-00001.parquet
```

## Run Answer Generation

Use `run_unanswerable_oracle.py` to generate JSONL outputs.

### Agentic, Critic On

```bash
python evaluation/faitheval/run_unanswerable_oracle.py \
  --mode agentic \
  --enable-answer-critique \
  --llm-config config/llm.json \
  --critic-llm-config config/critic_llm.json \
  --limit 10 \
  --max-workers 2 \
  --output output/faitheval/unanswerable_oracle_agentic_critic_on_10.jsonl
```

### Agentic, Critic Off

```bash
python evaluation/faitheval/run_unanswerable_oracle.py \
  --mode agentic \
  --no-enable-answer-critique \
  --llm-config config/llm.json \
  --limit 10 \
  --max-workers 2 \
  --output output/faitheval/unanswerable_oracle_agentic_critic_off_10.jsonl
```

### Naive RAG

```bash
python evaluation/faitheval/run_unanswerable_oracle.py \
  --mode naive \
  --llm-config config/llm.json \
  --limit 10 \
  --max-workers 2 \
  --output output/faitheval/unanswerable_oracle_naive_10.jsonl
```

## Runner Notes

- `--start` is optional. It skips the first `N` samples in the parquet file
  before any `--limit` is applied.
- `--limit` is optional. If set, the runner only processes the first `N`
  samples after applying `--start`.
- `--max-workers` controls concurrent sample execution. Start with `2` and
  increase only if the configured LLM endpoint is stable.
- `--critic-llm-config` is optional. If omitted, the answer critic reuses the
  client created from `--llm-config`.
- `--mode naive` uses the same oracle evidence but follows the project's naive
  final-answer path. It does not run `ReflectiveReasoner` or answer critique.
- `--mode agentic` uses a single-pass experiment reasoner to prevent follow-up
  query expansion and retrieval loops. This keeps the critic on/off comparison
  focused.
- `--enable-decomposition` is disabled by default. Keep it disabled when
  isolating the answer-critique effect.
- Output order may differ from dataset order when `--max-workers > 1`; use the
  `index` field in each result to recover the original order.

## Evaluate Outputs

Use `evaluate_unanswerable_oracle.py`:

```bash
python evaluation/faitheval/evaluate_unanswerable_oracle.py \
  --input output/faitheval/unanswerable_oracle_agentic_critic_on_100.jsonl \
  --max-workers 4
```

The evaluator treats a prediction as correct refusal when the final or node
answer is empty or contains an accepted refusal phrase such as `unknown`,
`no answer`, `no information`, `insufficient information`, or `unclear`.

## Metrics

The evaluator reports:

- `final_refusal_accuracy`: fraction of successful samples where the final
  answer refuses to answer.
- `final_hallucination_rate`: `1 - final_refusal_accuracy`.
- `final_empty_answer_rate`: fraction of successful samples with an empty final
  answer.
- `node_refusal_accuracy`: same refusal metric for the first node-level answer.
- `node_hallucination_rate`: `1 - node_refusal_accuracy`.
- `node_empty_answer_rate`: fraction of successful samples with an empty
  node-level answer.
- `critic_rejection_observed_rate`: approximate signal that critic was involved,
  based on a non-empty node critique and a refused node answer.
- `error_rate`: fraction of records with non-`ok` status.
- `final_hallucination_examples`: up to 20 final non-refusal examples for manual
  inspection.

For user-visible behavior, compare `final_refusal_accuracy` and
`final_hallucination_rate`. For critic-specific behavior, compare
`agentic critic on` against `agentic critic off`, and inspect node-level metrics.

## Recommended Comparison

Run and evaluate these three outputs on the same sample range:

1. `agentic + critic on`
2. `agentic + critic off`
3. `naive`

The critic effect should be attributed to:

```text
agentic critic on - agentic critic off
```

The broader system comparison can use:

```text
agentic critic on - naive
```

## RAGTruth-processed

The repository also includes an oracle-context pair for `RAGTruth-processed`:

- `run_ragtruth_processed_oracle.py`
- `evaluate_ragtruth_processed_oracle.py`

Default dataset:

```bash
data/ragtruth-processed/test-00000-of-00001.parquet
```

### Run Answer Generation (RAGTruth-processed)

```bash
python run_ragtruth_processed_oracle.py \
  --mode agentic \
  --enable-answer-critique \
  --llm-config config/llm.json \
  --critic-llm-config config/critic_llm.json \
  --max-workers 2 \
  --output output/ragtruth/ragtruth_processed_oracle_agentic_critic_on.jsonl
```

### Evaluate Outputs (RAGTruth-processed)

```bash
python evaluate_ragtruth_processed_oracle.py \
  --input output/ragtruth/ragtruth_processed_oracle_agentic_critic_on.jsonl \
  --max-workers 4
```
