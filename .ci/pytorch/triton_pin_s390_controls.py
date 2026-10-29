"""Capture the unchanged native s390x grid-sampler VJP failure and layout controls."""

import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import sys

import pytest
import torch
from torch.testing._internal.common_utils import TestCase


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "test/test-reports/triton-pin-s390-controls"
DEBUG = ROOT / "test/debug/triton-pin-s390-controls"


def main():
    assert platform.machine() == "s390x", platform.machine()
    assert torch.version.git_version == "914c15e40b4d5818fb0e85fe7b94d77b5f00c332"
    assert not torch.cuda.is_available()
    REPORT.mkdir(parents=True, exist_ok=True)
    DEBUG.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"machine": platform.machine(), "torch": torch.__version__,
                      "torch_sha": torch.version.git_version, "triton_spec": str(importlib.util.find_spec("triton")),
                      "torch_config": torch.__config__.show()}, indent=2), flush=True)
    original = TestCase.assertEqual
    captures = []

    def capture(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except AssertionError:
            frame = inspect.currentframe().f_back
            while frame is not None:
                local = frame.f_locals
                if frame.f_code.co_name == "_test" and "noncontig_result_vjps" in local:
                    names = ["primals", "cotangents", "noncontig_primals", "noncontig_cotangents",
                             "noncontig_result_vjps", "expected_vjps"]
                    data = {key: local[key] for key in names}
                    data["sample_args"] = local["sample"].args
                    data["sample_input"] = local["sample"].input
                    torch.save(data, DEBUG / "original-layout-repro.pt")
                    captures.append(data)
                    break
                frame = frame.f_back
            raise

    TestCase.assertEqual = capture
    os.chdir(ROOT / "test")
    sys.path.insert(0, str(ROOT / "test"))
    result = pytest.main(["-s", "--tb=short", "--reruns=0",
                          f"--junitxml={REPORT / 'original.xml'}",
                          "functorch/test_ops.py::TestOperatorsDeviceCPU::test_vjp_grid_sampler_2d_cpu_float32"])
    TestCase.assertEqual = original
    assert result in (0, 1), result
    summary = {"original_exit": int(result), "captured_layout_failure": bool(captures)}
    if captures:
        data = captures[-1]
        inp, grid = data["primals"]
        cotangent = data["cotangents"]
        ni, ng = data["noncontig_primals"]
        nc = data["noncontig_cotangents"]
        mode, padding, align = data["sample_args"][1:]
        truth = torch.ops.aten.grid_sampler_2d_backward(cotangent.double(), inp.double(), grid.double(),
                                                       mode, padding, align, [True, True])
        summary["layouts"] = []
        for il, iv in [("contiguous", inp), ("noncontiguous", ni)]:
            for gl, gv in [("contiguous", grid), ("noncontiguous", ng)]:
                for cl, cv in [("contiguous", cotangent), ("noncontiguous", nc)]:
                    grads = torch.ops.aten.grid_sampler_2d_backward(cv, iv, gv, mode, padding, align, [True, True])
                    summary["layouts"].append({"input": il, "grid": gl, "cotangent": cl,
                        "strides": [iv.stride(), gv.stride(), cv.stride()],
                        "max_error_to_float64": [(g.double() - t).abs().max().item() for g, t in zip(grads, truth)],
                        "grid_at_original_failure": grads[1][1, 1, 4, 1].item(),
                        "truth_at_original_failure": truth[1][1, 1, 4, 1].item()})
    (REPORT / "results.json").write_text(json.dumps(summary, indent=2))
    print("S390_LAYOUT_CONTROLS", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
