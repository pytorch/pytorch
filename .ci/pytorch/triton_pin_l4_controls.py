"""Temporary CUDA 13.4/L4 abs controls for draft PR #202417."""

import json
import os
import subprocess
import sys

import triton_pin_mi350_controls as controls


def main():
    controls.REPORT = controls.ROOT / "test/test-reports/triton-pin-l4-controls"
    controls.WORK = controls.ROOT / "build/triton-pin-l4-controls"
    controls.REPORT.mkdir(parents=True, exist_ok=True)
    controls.WORK.mkdir(parents=True, exist_ok=True)
    controls.run(
        [
            sys.executable,
            "-c",
            "import torch; p=torch.cuda.get_device_properties(0); "
            "assert 'L4' in p.name and torch.cuda.get_device_capability(0)==(8,9), str(p); "
            "assert torch.version.cuda=='13.4', torch.version.cuda; print(p)",
        ],
        cwd=controls.ROOT / "test",
        name="l4-preflight",
        timeout=180,
    )
    results = {}
    for label, (repository, sha) in controls.PINS.items():
        site = controls.build_pin(label, repository, sha)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(site) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        env["TRITON_CACHE_DIR"] = str(controls.WORK / f"cache-{label}/triton")
        env["TORCHINDUCTOR_CACHE_DIR"] = str(controls.WORK / f"cache-{label}/inductor")
        probe = (
            "import json,torch,triton,importlib.metadata,pathlib; "
            "v=importlib.metadata.version('triton'); "
            f"assert '{sha[:8]}' in v, v; "
            f"assert pathlib.Path(triton.__file__).resolve().is_relative_to({str(site)!r}), triton.__file__; "
            "x=torch.arange(65536,device='cuda',dtype=torch.int32).to(torch.int16).view(torch.bfloat16); "
            "e=x.abs(); c=torch.compile(torch.abs,options={'numerics':'strict','triton.autotune_pointwise':False})(x); "
            "bad=torch.nonzero(e.view(torch.int16)!=c.view(torch.int16)).flatten().cpu().tolist(); "
            "print(json.dumps(dict(torch_version=torch.__version__,torch_sha=torch.version.git_version,"
            "cuda=torch.version.cuda,device=str(torch.cuda.get_device_properties(0)),"
            "triton_version=triton.__version__,triton_distribution=v,triton_file=triton.__file__,"
            "mismatches=len(bad),examples=[dict(input=hex(i),eager=hex(e.view(torch.int16)[i].item()&65535),"
            "compiled=hex(c.view(torch.int16)[i].item()&65535)) for i in bad[:8]]),indent=2))"
        )
        controls.run(
            [sys.executable, "-c", probe],
            cwd=controls.ROOT / "test",
            env=env,
            name=f"{label}-bf16-all-encodings",
            timeout=1200,
        )
        node = "inductor/test_strict_numerics.py::PointwiseStrictNumericsTestCUDA::test_pointwise_bitwise_abs_cuda_bfloat16"
        results[label] = controls.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-s",
                "--tb=short",
                "--reruns=0",
                f"--junitxml={controls.REPORT / (label + '-abs.xml')}",
                node,
            ],
            cwd=controls.ROOT / "test",
            env=env,
            name=f"{label}-original-abs",
            check=False,
            timeout=1200,
        )
        log = (controls.REPORT / f"{label}-original-abs.log").read_text(
            errors="replace"
        )
        assert results[label] in (0, 1) and ("1 passed" in log or "forward now matches eager under strict numerics" in log), (
            f"Original abs case did not complete: {label}"
        )
        (controls.REPORT / "results.json").write_text(json.dumps(results, indent=2))
    print("L4 abs controls completed", json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
