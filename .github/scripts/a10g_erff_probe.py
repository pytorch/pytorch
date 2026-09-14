#!/usr/bin/env python3
"""Compare Triton erff/GELU output with eager CUDA and an FP64 CPU reference."""

import argparse
import json
import math
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def erff_probe_kernel(x, out_erf, out_gelu, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    values = tl.load(x + offsets, mask=mask)
    erf_values = libdevice.erf(values)
    gelu_erf_values = libdevice.erf(values * 0.7071067811865476)
    tl.store(out_erf + offsets, erf_values, mask=mask)
    tl.store(
        out_gelu + offsets,
        0.5 * values * (1.0 + gelu_erf_values),
        mask=mask,
    )


def ordered_float32_bits(values: torch.Tensor) -> torch.Tensor:
    bits = values.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    negative = (bits & 0x80000000) != 0
    return torch.where(negative, (~bits) & 0xFFFFFFFF, bits | 0x80000000)


def summarize(output: torch.Tensor, reference64: torch.Tensor) -> dict[str, object]:
    reference32 = reference64.to(torch.float32)
    ulp = (ordered_float32_bits(output) - ordered_float32_bits(reference32)).abs()
    absolute = (output.to(torch.float64) - reference64).abs()
    return {
        "elements": output.numel(),
        "exactly_correctly_rounded": int((output == reference32).sum().item()),
        "differing_from_correctly_rounded": int((output != reference32).sum().item()),
        "max_ulp_from_correctly_rounded": int(ulp.max().item()),
        "over_1_ulp": int((ulp > 1).sum().item()),
        "over_2_ulp": int((ulp > 2).sum().item()),
        "max_absolute_error": float(absolute.max().item()),
        "rmse": float(torch.sqrt(torch.mean(absolute.square())).item()),
    }


def summarize_pair(left: torch.Tensor, right: torch.Tensor) -> dict[str, object]:
    ulp = (ordered_float32_bits(left) - ordered_float32_bits(right)).abs()
    absolute = (left.to(torch.float64) - right.to(torch.float64)).abs()
    return {
        "differing": int((left != right).sum().item()),
        "max_ulp": int(ulp.max().item()),
        "over_1_ulp": int((ulp > 1).sum().item()),
        "max_absolute_difference": float(absolute.max().item()),
    }


def run_probe(label: str, output_path: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    # The dense interval covers the region where erf transitions and the wider
    # range in which exact GELU uses erf(x / sqrt(2)). Inputs are first rounded
    # to float32 because that is what the affected Triton kernels consume.
    inputs = torch.linspace(-10.0, 10.0, 1_000_001, dtype=torch.float64)
    inputs = torch.unique_consecutive(inputs.to(torch.float32))
    inputs_cuda = inputs.cuda()
    output_erf = torch.empty_like(inputs_cuda)
    output_gelu = torch.empty_like(inputs_cuda)
    block = 256
    erff_probe_kernel[(triton.cdiv(inputs.numel(), block),)](
        inputs_cuda,
        output_erf,
        output_gelu,
        n=inputs.numel(),
        BLOCK=block,
    )
    torch.cuda.synchronize()

    eager_erf = torch.erf(inputs_cuda)
    eager_gelu = torch.nn.functional.gelu(inputs_cuda, approximate="none")
    input64 = inputs.to(torch.float64)
    reference_erf = torch.erf(input64)
    reference_gelu = 0.5 * input64 * (1.0 + torch.erf(input64 / math.sqrt(2.0)))
    output_erf = output_erf.cpu()
    output_gelu = output_gelu.cpu()
    eager_erf = eager_erf.cpu()
    eager_gelu = eager_gelu.cpu()

    report = {
        "label": label,
        "gpu": torch.cuda.get_device_name(0),
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "erf_vs_fp64": summarize(output_erf, reference_erf),
        "erf_vs_eager_cuda": summarize_pair(output_erf, eager_erf),
        "gelu_vs_fp64": summarize(output_gelu, reference_gelu),
        "gelu_vs_eager_cuda": summarize_pair(output_gelu, eager_gelu),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "label": label,
            "inputs": inputs,
            "erf": output_erf,
            "gelu": output_gelu,
            "eager_erf": eager_erf,
            "eager_gelu": eager_gelu,
            "report": report,
        },
        output_path,
    )
    output_path.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


def compare(outputs: list[Path], report_path: Path) -> None:
    loaded = [
        torch.load(path, map_location="cpu", weights_only=True) for path in outputs
    ]
    baseline = loaded[0]
    comparisons: dict[str, object] = {}
    for candidate in loaded[1:]:
        if not torch.equal(baseline["inputs"], candidate["inputs"]):
            raise RuntimeError("Probe inputs differ")
        comparisons[candidate["label"]] = {
            "erf": summarize_pair(baseline["erf"], candidate["erf"]),
            "gelu": summarize_pair(baseline["gelu"], candidate["gelu"]),
        }
    report = {"baseline": baseline["label"], "comparisons": comparisons}
    print(json.dumps(report, indent=2, sort_keys=True))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--label", required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "run":
        run_probe(args.label, args.output)
    else:
        compare(args.inputs, args.output)


if __name__ == "__main__":
    main()
