"""
Benchmark test plans — inductor / dynamo benchmarks.
Each model is a separate TestStep so individual failures can be reproduced.
"""

from __future__ import annotations

import csv
import logging
import os

from cli.lib.core.pytorch.pytorch_test_library import (
    BenchmarkTestPlan,
    is_cuda,
    is_rocm,
    is_xpu,
)


logger = logging.getLogger(__name__)


def _merge_csv_results(output_paths: list[str], merged_path: str) -> None:
    """Concatenate per-model CSVs into a single result file."""
    rows = []
    header = None
    for path in output_paths:
        if not os.path.exists(path):
            logger.warning("missing result file: %s", path)
            continue
        with open(path) as f:
            reader = csv.DictReader(f)
            if header is None:
                header = reader.fieldnames
            rows.extend(reader)

    if not rows:
        logger.warning("no benchmark results to merge")
        return

    os.makedirs(os.path.dirname(merged_path), exist_ok=True)
    with open(merged_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("merged %d model results → %s", len(rows), merged_path)


BENCHMARK_TEST_PLANS: dict[str, BenchmarkTestPlan] = {
    "pytorch_inductor_smoketest": BenchmarkTestPlan(
        group_id="pytorch_inductor_smoketest",
        title="Inductor Torchbench Smoketest",
        run_on=[is_cuda, is_rocm],
        device={is_xpu: "xpu", "cpu": "cpu", is_cuda: "cuda", is_rocm: "cuda"},
        backend="inductor",
        suite="torchbench",
        modes={is_rocm: ["training", "inference"], is_cuda: ["training"]},
        dtype={is_rocm: "amp", is_cuda: ["float16", "amp"]},
        models=[
            "BERT_pytorch",
            "resnet50",
            "yolov3",
        ],
        join_results_fn=lambda paths: _merge_csv_results(
            paths,
            merged_path="test/test-reports/inductor_smoketest_merged.csv",
        ),
        steps=[],
    ),
}
