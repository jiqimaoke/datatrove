"""Run shared CPU text-quality rules with DataTrove's local executor."""

import argparse

import ray

from datatrove.executor import RayPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter

from custom_filters import (
    DecodedBase64Filter,
    FastTextQualityThresholdFilter,
    FastTextLanguageFilter,
    KenLMPerplexityFilter,
    HFTokenCountFilter,
    MaxSentenceLengthFilter,
    RealTextFilter,
    SentenceRepeatRatioFilter,
    XGBoostQualityFilter,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input directory containing JSONL shards")
    parser.add_argument("--output", required=True)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--fasttext-model")
    parser.add_argument("--fasttext-label", default="high_quality")
    parser.add_argument("--fasttext-threshold", type=float, default=0.5)
    parser.add_argument("--language-model")
    parser.add_argument("--language-threshold", type=float, default=0.65)
    parser.add_argument("--kenlm-model")
    parser.add_argument("--sentencepiece-model")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--xgboost-model")
    parser.add_argument("--xgboost-threshold", type=float, default=0.5)
    parser.add_argument("--rules", default="all", help="comma-separated rules or all")
    args = parser.parse_args()

    selected = {x.strip() for x in args.rules.split(",")}
    all_rules = args.rules == "all"
    pipeline = [JsonlReader(args.input, text_key="text", id_key="case_id")]
    if all_rules or "real_text" in selected: pipeline.append(RealTextFilter())
    if all_rules or "base64" in selected: pipeline.append(DecodedBase64Filter())
    if all_rules or "max_sentence_length" in selected: pipeline.append(MaxSentenceLengthFilter(max_len=300))
    if all_rules or "sentence_repeat" in selected: pipeline.append(SentenceRepeatRatioFilter(max_ratio=0.3))
    if args.language_model and (all_rules or "language" in selected):
        pipeline.append(FastTextLanguageFilter(args.language_model, min_score=args.language_threshold))
    if args.fasttext_model and (all_rules or "quality" in selected):
        pipeline.append(
            FastTextQualityThresholdFilter(
                args.fasttext_model,
                args.fasttext_label,
                args.fasttext_threshold,
            )
        )
    if args.kenlm_model and (all_rules or "ppl" in selected):
        pipeline.append(KenLMPerplexityFilter(args.kenlm_model, args.sentencepiece_model))
    if args.tokenizer_path and (all_rules or "token_count" in selected): pipeline.append(HFTokenCountFilter(args.tokenizer_path))
    if args.xgboost_model:
        pipeline.append(XGBoostQualityFilter(args.xgboost_model, args.xgboost_threshold))
    pipeline.append(JsonlWriter(args.output, compression=None))

    ray.init(address="local", include_dashboard=False)
    executor = RayPipelineExecutor(
        pipeline=pipeline,
        tasks=args.tasks,
        workers=args.workers,
        cpus_per_task=1,
        mem_per_cpu_gb=1,
        nodes_per_task=1,
        tasks_per_job=1,
        logging_dir=args.logs,
        skip_completed=False,
    )
    try:
        executor.run()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
