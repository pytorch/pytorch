"""Native old/candidate controls for the original MI350 strict subtraction cases."""

import hashlib
import json
import os
import subprocess
import sys

import triton_pin_mi350_controls as controls


ROOT = controls.ROOT
REPORT = ROOT / "test/test-reports/triton-pin-sub-expectations"
WORK = ROOT / "build/triton-pin-sub-expectations"
TORCH_HEAD = "aaa1b73d62015d15004355001092982aa1bf53bf"
DTYPES = ("bfloat16", "float16", "float32")


def main():
    controls.REPORT = REPORT
    controls.WORK = WORK
    REPORT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    provenance = {
        "expected_torch_head": TORCH_HEAD,
        "native_artifact_run_id": 40915231984,
        "control_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "property_test_sha256": hashlib.sha256(
            (
                ROOT / "test/inductor/test_torchinductor_opinfo_properties.py"
            ).read_bytes()
        ).hexdigest(),
    }
    (REPORT / "source-provenance.json").write_text(json.dumps(provenance, indent=2))
    results = {}
    for pin, (repository, sha) in controls.PINS.items():
        site = controls.build_pin(pin, repository, sha)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(site) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        controls.configure_fusion(env, None)
        probe = (
            "import torch,triton,importlib.metadata,pathlib; "
            "p=torch.cuda.get_device_properties(0); "
            "assert torch.version.hip and 'gfx950' in p.gcnArchName, str(p); "
            f"assert torch.version.git_version == {TORCH_HEAD!r}; "
            f"assert '{sha[:8]}' in importlib.metadata.version('triton'); "
            f"assert pathlib.Path(triton.__file__).resolve().is_relative_to({str(site)!r}); "
            + controls.fusion_provenance(None)
            + "print(torch.version.git_version,p,triton.__file__)"
        )
        controls.run(
            [sys.executable, "-c", probe],
            cwd=ROOT / "test",
            env=env,
            name=f"{pin}-runtime-provenance",
        )
        results[pin] = {}
        for expectation in ("original", "corrected"):
            results[pin][expectation] = {}
            for dtype in DTYPES:
                name = f"{pin}-{expectation}-{dtype}"
                cache = WORK / f"cache-{name}"
                env["TRITON_CACHE_DIR"] = str(cache / "triton")
                env["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
                env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
                command = [
                    sys.executable,
                    str(ROOT / ".ci/pytorch/triton_pin_trace_property.py"),
                    "-s",
                    "--tb=short",
                    "--reruns=0",
                    f"--junitxml={REPORT / (name + '.xml')}",
                    "inductor/test_torchinductor_opinfo_properties.py::"
                    "TestOpInfoPropertiesCUDA::test_eager_equivalence_sub_"
                    f"backend_inductor_numerics_cuda_{dtype}",
                ]
                if expectation == "corrected":
                    command.append("--remove-rocm-sub-xfail")
                code = controls.run(
                    command,
                    cwd=ROOT / "test",
                    env=env,
                    name=name,
                    check=False,
                    timeout=1200,
                )
                log = (REPORT / f"{name}.log").read_text(errors="replace")
                assert code in (0, 1), f"Invalid execution: {name}"
                assert "RuntimeError:" not in log, f"Runtime failure: {name}"
                if "XPASS:" in log:
                    outcome = "numerics-pass-stale-expectation"
                elif "1 xfailed" in log:
                    assert "PIN_ORIGINAL_EXPECTED_EXCEPTION" in log
                    assert "AssertionError:" in log
                    outcome = "numerical-xfail"
                elif "1 passed" in log:
                    assert code == 0
                    outcome = "passed"
                else:
                    assert "1 failed" in log and "AssertionError:" in log
                    outcome = "failed-comparison"
                results[pin][expectation][dtype] = {"exit": code, "outcome": outcome}
                (REPORT / "results.json").write_text(json.dumps(results, indent=2))
    passed = all(
        result["outcome"] == "passed"
        for phase in results.values()
        for result in phase["corrected"].values()
    )
    print("SUB_EXPECTATION_CONTROLS", json.dumps(results), flush=True)
    print("ALL_CORRECTED_CASES_PASSED", passed, flush=True)


if __name__ == "__main__":
    main()
