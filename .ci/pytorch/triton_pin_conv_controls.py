"""Isolate the unchanged MI350 convolution test from Triton code generation."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import triton_pin_mi350_controls as controls


ROOT = controls.ROOT
REPORT = ROOT / "test/test-reports/triton-pin-conv"
DEBUG = ROOT / "test/debug/triton-pin-conv"
WORK = ROOT / "build/triton-pin-conv"
TORCH_HEAD = "aaa1b73d62015d15004355001092982aa1bf53bf"
NODE = "inductor/test_torchinductor.py::GPUTests::test_conv_with_as_strided_cuda"


def statistics(actual, reference):
    import torch

    actual = actual.detach().cpu().double()
    reference = reference.detach().cpu().double()
    absolute = (actual - reference).abs()
    relative = torch.where(
        reference != 0,
        absolute / reference.abs(),
        torch.where(absolute == 0, 0.0, float("inf")),
    )
    return {
        "numel": actual.numel(),
        "bitwise_equal": torch.equal(actual, reference),
        "mismatches_at_original_tolerance": int(
            (absolute > 2e-3 + 2e-3 * reference.abs()).sum()
        ),
        "max_absolute": absolute.max().item(),
        "max_relative": relative.max().item(),
    }


class CaptureOriginal:
    def __init__(self, label, aten):
        self.label = label
        self.aten = aten
        self.captures = []
        self.actual = self.reference = None

    def pytest_runtest_call(self, item):
        if item.name != "test_conv_with_as_strided_cuda":
            return
        import torch

        policy = {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "matmul_allow_fp16_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "matmul_allow_bf16_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "compiler_debug": os.environ.get("TORCH_COMPILE_DEBUG"),
            "disable_fresh_cache": os.environ.get("INDUCTOR_TEST_DISABLE_FRESH_CACHE"),
            "inductor_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "triton_cache": os.environ.get("TRITON_CACHE_DIR"),
            "environment": {
                k: v
                for k, v in os.environ.items()
                if k.startswith(("MIOPEN_", "ROCBLAS_", "HIPBLAS_"))
            },
        }
        (REPORT / f"{self.label}-precision-policy.json").write_text(
            json.dumps(policy, indent=2)
        )
        module = item.module
        original_check = module.check_model
        original_equal = module._assert_equal_with_significant_stride_check

        def capture_equal(test, assertion, actual, reference, **kwargs):
            if (
                isinstance(actual, torch.Tensor)
                and actual.dtype == torch.float16
                and tuple(actual.shape) == (8, 384, 2, 2, 12, 12)
            ):
                self.actual = actual.detach().cpu().clone()
                self.reference = reference.detach().cpu().clone()
            return original_equal(test, assertion, actual, reference, **kwargs)

        def capture_check(test, model, inputs, *args, **kwargs):
            lowp = inputs[0].dtype == torch.float16
            if lowp:
                self.actual = self.reference = None
                saved_input = inputs[0].detach().clone()
                saved_model = copy.deepcopy(model)
            try:
                return original_check(test, model, inputs, *args, **kwargs)
            finally:
                if lowp and self.actual is not None:
                    with torch.no_grad():
                        # The original DeterministicGuard is still active here.
                        eager = saved_model(saved_input).detach().cpu()
                    cpu_input = saved_input.detach().cpu()
                    cpu_weight = saved_model.kv.weight.detach().cpu()
                    n, _, height, width = cpu_input.shape
                    convolution64 = (
                        (
                            cpu_weight[:, :, 0, 0].double()
                            @ cpu_input.double().permute(1, 0, 2, 3).reshape(256, -1)
                        )
                        .reshape(384, n, height, width)
                        .permute(1, 0, 2, 3)
                        .contiguous()
                    )
                    padded = torch.nn.functional.pad(convolution64, (2, 2, 2, 2))
                    view = torch.as_strided(
                        padded, (8, 384, 2, 20, 12), (153600, 400, 160, 1, 20)
                    )
                    reference64 = torch.as_strided(
                        view,
                        (8, 384, 2, 2, 12, 12),
                        (153600, 400, 160, 8, 20, 1),
                    ).clone(memory_format=torch.contiguous_format)
                    data = {
                        "actual": self.actual,
                        "original_fp32_reference": self.reference,
                        "eager_fp16": eager,
                        "fp64_reference": reference64,
                        "input_fp16": cpu_input,
                        "weight_fp16": cpu_weight,
                    }
                    torch.save(data, DEBUG / f"{self.label}-operands.pt")
                    result = {
                        "actual_vs_original_reference": statistics(
                            self.actual, self.reference
                        ),
                        "eager_vs_original_reference": statistics(
                            eager, self.reference
                        ),
                        "actual_vs_eager": statistics(self.actual, eager),
                        "actual_vs_fp64": statistics(self.actual, reference64),
                        "original_reference_vs_fp64": statistics(
                            self.reference, reference64
                        ),
                    }
                    self.captures.append(result)
                    (REPORT / f"{self.label}-numerics.json").write_text(
                        json.dumps(result, indent=2)
                    )
                    print("CONV_NUMERICS", self.label, json.dumps(result), flush=True)
                    from torch._inductor.runtime.cache_dir_utils import (
                        cache_dir,
                        triton_cache_dir,
                    )

                    # Copy while the original test's fresh_cache is still active.
                    with zipfile.ZipFile(
                        DEBUG / f"{self.label}-generated-kernels.zip",
                        "w",
                        zipfile.ZIP_DEFLATED,
                    ) as archive:
                        for kind, directory in (
                            ("inductor", cache_dir()),
                            ("triton", triton_cache_dir(torch.cuda.current_device())),
                        ):
                            base = Path(directory)
                            if base.exists():
                                for path in base.rglob("*"):
                                    if path.is_file() and path.suffix in {
                                        ".py",
                                        ".json",
                                        ".ttir",
                                        ".ttgir",
                                        ".ll",
                                        ".amdgcn",
                                    }:
                                        archive.write(
                                            path, Path(kind) / path.relative_to(base)
                                        )

        module.check_model = capture_check
        module._assert_equal_with_significant_stride_check = capture_equal
        if self.aten:

            def aten_graph(gm, inputs, **kwargs):
                path = (
                    REPORT
                    / f"{self.label}-aten-graph-{len(list(REPORT.glob(self.label + '-aten-graph-*')))}.py"
                )
                path.write_text(gm.code)
                print("ATEN_GRAPH_CONTROL_NO_INDUCTOR", gm.code, flush=True)
                return gm.forward

            module.compile_fx = aten_graph


def child(label, aten, prefix):
    import pytest
    import triton

    import torch

    prop = torch.cuda.get_device_properties(0)
    if not torch.version.hip or "gfx950" not in prop.gcnArchName:
        raise AssertionError(f"Expected native MI350, got {prop}")
    if torch.version.git_version != TORCH_HEAD:
        raise AssertionError(f"Unexpected Torch head: {torch.version.git_version}")
    distributions = [
        {
            "name": d.metadata["Name"],
            "version": d.version,
            "location": str(d.locate_file("")),
        }
        for d in importlib.metadata.distributions()
        if "triton" in d.metadata.get("Name", "").lower()
    ]
    if not triton.__file__ or not hasattr(triton, "jit"):
        raise AssertionError("Expected an importable native Triton package")
    probe = {
        "torch_sha": torch.version.git_version,
        "torch_config": torch.__config__.show(),
        "triton_file": triton.__file__,
        "triton_distributions": distributions,
        "triton_distribution": importlib.metadata.version("triton"),
        "device": str(prop),
        "hip": torch.version.hip,
        "aten_graph_control": aten,
    }
    (REPORT / f"{label}-runtime-provenance.json").write_text(
        json.dumps(probe, indent=2)
    )
    print("CONV_NATIVE_PROVENANCE", json.dumps(probe), flush=True)
    plugin = CaptureOriginal(label, aten)
    nodes = (
        json.loads((ROOT / ".ci/pytorch/triton_pin_conv_ci_prefix.json").read_text())[
            "nodes"
        ]
        if prefix
        else [NODE]
    )
    code = pytest.main(
        [
            "-s",
            "--tb=short",
            "--reruns=0",
            f"--junitxml={REPORT / (label + '.xml')}",
            "-p",
            "no:xdist",
            *nodes,
        ],
        plugins=[plugin],
    )
    if code not in (0, 1) or not plugin.captures:
        raise AssertionError((code, plugin.captures))
    sys.exit(code)


def phase(label, *, site=None, sha=None, aten=False, prefix=False):
    env = os.environ.copy()
    controls.configure_fusion(env, None)
    if site:
        env["PYTHONPATH"] = str(site)
    cache = WORK / f"cache-{label}"
    env["TRITON_CACHE_DIR"] = str(cache / "triton")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
    if prefix:
        # Match the original CI cache policy. Fresh per-phase directories isolate
        # attempts without changing tests that compare generated source text.
        env.pop("TORCHINDUCTOR_FORCE_DISABLE_CACHES", None)
        env.pop("TORCH_COMPILE_DEBUG", None)
        env.pop("INDUCTOR_TEST_DISABLE_FRESH_CACHE", None)
    else:
        env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
        env["TORCH_COMPILE_DEBUG"] = "1"
    env["MIOPEN_ENABLE_LOGGING"] = "1"
    env["MIOPEN_LOG_LEVEL"] = "6"
    if sha:
        controls.run(
            [
                sys.executable,
                "-c",
                "import triton,importlib.metadata,pathlib,sys; "
                "print(triton.__file__); "
                f"valid = '{sha[:8]}' in importlib.metadata.version('triton') and "
                f"pathlib.Path(triton.__file__).resolve().is_relative_to({str(site)!r}); "
                "sys.exit(0 if valid else 1)",
            ],
            cwd=ROOT / "test",
            env=env,
            name=f"{label}-full-native-pin",
        )
    command = [sys.executable, str(Path(__file__).resolve()), "--child", label]
    if aten:
        command.append("--aten")
    if prefix:
        command.append("--ci-prefix")
    code = controls.run(
        command, cwd=ROOT / "test", env=env, name=label, check=False, timeout=1200
    )
    log = (REPORT / f"{label}.log").read_text(errors="replace")
    if code not in (0, 1) or "CONV_NUMERICS" not in log:
        raise AssertionError(f"Missing convolution capture, exit {code}")
    if "passed" not in log and "1 failed" not in log:
        raise AssertionError("Missing convolution pytest result")
    if not prefix and "RuntimeError:" in log:
        raise AssertionError("Standalone convolution raised RuntimeError")
    import xml.etree.ElementTree as ET

    cases = ET.parse(REPORT / f"{label}.xml").getroot().iter("testcase")
    cases = list(cases)
    target = [c for c in cases if c.get("name") == "test_conv_with_as_strided_cuda"]
    if len(target) != 1 or target[0].find("skipped") is not None:
        raise AssertionError("Original convolution must execute exactly once")
    test_results = {
        "original_case": {
            "name": target[0].get("name"),
            "failed": target[0].find("failure") is not None,
            "error": target[0].find("error") is not None,
        },
        "executed_cases": len(cases),
        "all_failures": [
            {"name": c.get("name"), "classname": c.get("classname")}
            for c in cases
            if c.find("failure") is not None or c.find("error") is not None
        ],
    }
    return {
        "exit": code,
        **test_results,
        **json.loads((REPORT / f"{label}-numerics.json").read_text()),
    }


def replay_capture(label, full_accumulation):
    import torch
    from torch.testing._internal.common_utils import DeterministicGuard, TestCase

    data = torch.load(DEBUG / "ci-prefix-operands.pt", weights_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
        not full_accumulation
    )
    with DeterministicGuard(True, warn_only=True), torch.no_grad():
        convolution = torch.nn.functional.conv2d(
            data["input_fp16"].cuda(), data["weight_fp16"].cuda()
        )
        padded = torch.nn.functional.pad(convolution, (2, 2, 2, 2))
        view = torch.as_strided(padded, (8, 384, 2, 20, 12), (153600, 400, 160, 1, 20))
        actual = (
            torch.as_strided(view, (8, 384, 2, 2, 12, 12), (153600, 400, 160, 8, 20, 1))
            .clone(memory_format=torch.contiguous_format)
            .cpu()
        )
    result = {
        "solver": os.environ["MIOPEN_DEBUG_FIND_ONLY_SOLVER"],
        "allow_fp16_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "actual_vs_original_reference": statistics(
            actual, data["original_fp32_reference"]
        ),
        "actual_vs_fp64": statistics(actual, data["fp64_reference"]),
        "actual_vs_failed_capture": statistics(actual, data["actual"]),
    }
    torch.save(actual, DEBUG / f"{label}-actual.pt")
    (REPORT / f"{label}-numerics.json").write_text(json.dumps(result, indent=2))
    print("CONV_CAPTURE_REPLAY", label, json.dumps(result), flush=True)
    TestCase().assertEqual(
        actual, data["original_fp32_reference"], atol=2e-3, rtol=2e-3
    )


def solver_control(label, solver, full_accumulation=False):
    env = os.environ.copy()
    env["MIOPEN_DEBUG_FIND_ONLY_SOLVER"] = solver
    env["MIOPEN_FIND_MODE"] = "normal"
    env["MIOPEN_FIND_ENFORCE"] = "4"
    env["MIOPEN_USER_DB_PATH"] = str(WORK / f"solver-db-{label}")
    env["MIOPEN_ENABLE_LOGGING"] = "1"
    env["MIOPEN_LOG_LEVEL"] = "6"
    Path(env["MIOPEN_USER_DB_PATH"]).mkdir(parents=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--replay-capture", label]
    if full_accumulation:
        command.append("--full-accumulation")
    code = controls.run(command, env=env, name=label, check=False, timeout=300)
    log = (REPORT / f"{label}.log").read_text(errors="replace")
    if code not in (0, 1) or "CONV_CAPTURE_REPLAY" not in log:
        raise AssertionError(f"Captured-operand replay did not complete: {label}")
    return {
        "exit_code": code,
        "numerics": json.loads((REPORT / f"{label}-numerics.json").read_text()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child")
    parser.add_argument("--aten", action="store_true")
    parser.add_argument("--ci-prefix", action="store_true")
    parser.add_argument("--full-accumulation", action="store_true")
    parser.add_argument("--replay-capture")
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    DEBUG.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if args.replay_capture:
        replay_capture(args.replay_capture, args.full_accumulation)
        return
    if args.child:
        child(args.child, args.aten, args.ci_prefix)
        return
    controls.REPORT, controls.WORK = REPORT, WORK
    provenance = {
        "expected_torch_head": TORCH_HEAD,
        "native_artifact_run_id": 40915231984,
        "diagnostic_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "test_source_sha256": hashlib.sha256(
            (ROOT / "test/inductor/test_torchinductor.py").read_bytes()
        ).hexdigest(),
    }
    (REPORT / "source-provenance.json").write_text(json.dumps(provenance, indent=2))
    if args.ci_prefix:
        result = {
            "mode": "original-ci-prefix",
            "controls": {
                "ci-prefix": phase("ci-prefix", prefix=True),
                "ci-prefix-aten": phase("ci-prefix-aten", prefix=True, aten=True),
                "capture-gemm": solver_control("capture-gemm", "GemmFwd1x1_0_1"),
                "capture-winograd": solver_control(
                    "capture-winograd", "ConvWinoRageRxS<2-3>"
                ),
                "capture-winograd-full-accumulation": solver_control(
                    "capture-winograd-full-accumulation",
                    "ConvWinoRageRxS<2-3>",
                    full_accumulation=True,
                ),
            },
        }
        (REPORT / "results.json").write_text(json.dumps(result, indent=2))
        print("CONV_CI_PREFIX_RESULTS", json.dumps(result), flush=True)
        return
    results = {
        "aten": phase("aten", aten=True),
        "current_image": phase("current-image"),
    }
    current = results["current_image"]
    isolated = (
        results["aten"]["exit"] == 1
        and current["exit"] == 1
        and current["actual_vs_eager"]["bitwise_equal"]
        and current["eager_vs_original_reference"]["mismatches_at_original_tolerance"]
        > 0
    )
    if not isolated:
        for label, (repository, sha) in controls.PINS.items():
            site = controls.build_pin(label, repository, sha)
            results[label] = phase(label, site=site, sha=sha)
    result = {"aten_convolution_precision_isolated": isolated, "controls": results}
    (REPORT / "results.json").write_text(json.dumps(result, indent=2))
    print("CONV_CONTROL_RESULTS", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
