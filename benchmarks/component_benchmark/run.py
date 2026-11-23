"""Run: python benchmarks/component_benchmark/run.py --model deepfm --device cuda."""

import argparse
from pathlib import Path

from models import make_workload

from torch.utils.component_benchmark import (
    AccuracyCheckPlugin,
    component_loss_and_bwd,
    ComponentCapturePlugin,
    CoreForwardBenchmarkPlugin,
    FlopsCounterPlugin,
    MemoryPlugin,
    ProfilerPlugin,
    ReportPlugin,
    run_cb,
)


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical component benchmark examples"
    )
    parser.add_argument("--model", choices=("deepfm", "transformer"), default="deepfm")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--output", type=Path, default=Path("component_benchmark_results")
    )
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    model, inputs = make_workload(args.model, args.device)
    backward = component_loss_and_bwd if args.backward else None
    plugins = [
        CoreForwardBenchmarkPlugin(loss_and_bwd=backward),
        FlopsCounterPlugin(loss_and_bwd=backward),
        MemoryPlugin(args.output / "memory", loss_and_bwd=backward),
        ProfilerPlugin(args.output / "traces", loss_and_bwd=backward),
        ComponentCapturePlugin(args.output / "components"),
    ]
    if args.compile:
        plugins.extend(
            [
                CoreForwardBenchmarkPlugin(torch_compile=True, loss_and_bwd=backward),
                AccuracyCheckPlugin(loss_and_bwd=backward),
            ]
        )
    result = run_cb(
        model,
        inputs,
        custom_plugins=plugins,
        result_plugins=[ReportPlugin(args.output)],
        max_depth=args.max_depth,
    )
    print(f"Report: {(args.output / 'report.html').resolve()}")
    print(f"Components: {len(result.result)}; failures: {len(result.errors)}")


if __name__ == "__main__":
    main()
