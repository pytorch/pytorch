"""Current dd73676 native controls for the MI350 scalar-subtraction failure."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "test/test-reports/pin-scalar-controls"
DEBUG = ROOT / "test/debug/pin-scalar-controls"
NATIVE_SHA = "dd736768a3680bf63410e798da1a08b4f9d9a37a"
PINS = {"old": "b7ddcf86", "candidate": "1b9ee7ae"}


def child():
    import importlib.metadata

    import torch
    import triton
    from torch._inductor import config
    from torch._inductor.utils import run_and_get_code

    assert torch.version.git_version == NATIVE_SHA, torch.version.git_version
    assert torch.version.hip, torch.version.hip
    device = torch.cuda.get_device_properties(0)
    assert "gfx950" in device.gcnArchName, str(device)
    assert os.environ["PIN_TRITON_SHORT"] in importlib.metadata.version("triton")
    assert Path(triton.__file__).resolve().is_relative_to(os.environ["PIN_TRITON_SITE"])
    phase = Path(os.environ["PIN_SCALAR_PHASE_DIR"])
    values = []
    with config.patch(emulate_precision_casts=True):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for other_kind in ("python", "tensor", "cpu_scalar", "gpu_scalar"):
                for dynamic in (False, True):
                    for inplace in (False, True):
                        a = torch.full((256,), 2, dtype=dtype, device="cuda")
                        if other_kind == "tensor":
                            other = torch.full_like(a, 1 / 3)
                        elif other_kind == "cpu_scalar":
                            other = torch.tensor(1 / 3, dtype=torch.float64)
                        elif other_kind == "gpu_scalar":
                            other = torch.tensor(1 / 3, dtype=dtype, device="cuda")
                        else:
                            other = 1 / 3

                        def fn(a, other):
                            return a.sub_(other, alpha=6) if inplace else a.sub(other, alpha=6)

                        expected = fn(a.clone(), other)
                        torch._dynamo.reset()
                        actual, codes = run_and_get_code(torch.compile(fn, dynamic=dynamic), a, other)
                        name = f"{dtype}-{other_kind}-{dynamic}-{inplace}"
                        (phase / (name + ".py")).write_text("\n".join(codes))
                        item = {"dtype": str(dtype), "other_kind": other_kind,
                                "dynamic": dynamic, "inplace": inplace,
                                "eager": expected[0].item(), "compiled": actual[0].item(),
                                "mismatches": torch.count_nonzero(actual != expected).item(),
                                "explicit_fma": any("tl.fma" in code for code in codes)}
                        values.append(item)
                        print(json.dumps(item), flush=True)
    (phase / "values.json").write_text(json.dumps(values, indent=2) + "\n")
    if os.environ["PIN_SCALAR_MODE"] == "patched":
        assert all(item["mismatches"] == 0 for item in values), values
        for item in values:
            non_fused_scalar = item["dtype"] in ("torch.float16", "torch.bfloat16") and item["other_kind"] in ("python", "cpu_scalar")
            assert item["explicit_fma"] != non_fused_scalar, item
    sys.path.insert(0, str(ROOT / "test"))
    import pytest

    raise SystemExit(pytest.main(["-q", "--tb=short", "--reruns=0",
                                 str(phase / "test_scalar_suite.py") + "::TestAlphaFMACUDA"]))


if len(sys.argv) > 1 and sys.argv[1] == "child":
    child()
    raise AssertionError("child unexpectedly returned")


REPORT.mkdir(parents=True, exist_ok=True)
DEBUG.mkdir(parents=True, exist_ok=True)
source = (ROOT / "torch/_inductor/lowering.py").read_text()
assert hashlib.sha256(source.encode()).hexdigest() == "5879976a41b8029b19ef9ff2a1b0b9559de1b6ab058d18ede2c09b993298d981"
tests = (ROOT / "test/inductor/test_torchinductor_opinfo_properties.py").read_text()
scratch = DEBUG / "patched-source"
for name in ("torch/_inductor/lowering.py", "test/inductor/test_torchinductor_opinfo_properties.py"):
    target = scratch / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((ROOT / name).read_bytes())
patch = ROOT / ".ci/pytorch/triton_pin_scalar_fix.patch"
prefix = str(scratch.relative_to(ROOT))
subprocess.run(["git", "apply", "--check", "--directory=" + prefix, str(patch)], cwd=ROOT, check=True)
subprocess.run(["git", "apply", "--directory=" + prefix, str(patch)], cwd=ROOT, check=True)
patched = (scratch / "torch/_inductor/lowering.py").read_text()
patched_tests = (scratch / "test/inductor/test_torchinductor_opinfo_properties.py").read_text()
wheels = list((ROOT / "build/pin-scalar-native-packages").rglob("*.whl"))
results = []
for label, short in PINS.items():
    matches = [wheel for wheel in wheels if short in wheel.name]
    assert len(matches) == 1, (label, matches)
    wheel = matches[0]
    site = DEBUG / label / "site"
    site.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(site)
    for mode, code, suite in (("original", source, tests), ("patched", patched, patched_tests)):
        phase = DEBUG / f"{label}-{mode}"
        phase.mkdir(parents=True, exist_ok=True)
        overlay = phase / "overlay"
        overlay.mkdir(exist_ok=True)
        lowering = overlay / "lowering.py"
        lowering.write_text(code)
        (phase / "test_scalar_suite.py").write_text(suite)
        trace = phase / "workers"
        trace.mkdir(exist_ok=True)
        (overlay / "sitecustomize.py").write_text(
            "import hashlib, importlib.util, importlib.metadata, json, os, pathlib, sys\n"
            "import torch, torch._inductor, triton\n"
            f"assert torch.version.git_version == {NATIVE_SHA!r}\n"
            f"assert {short!r} in importlib.metadata.version('triton')\n"
            f"assert pathlib.Path(triton.__file__).resolve().is_relative_to({str(site)!r})\n"
            f"spec = importlib.util.spec_from_file_location('torch._inductor.lowering', {str(lowering)!r})\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "sys.modules[spec.name] = module\n"
            "spec.loader.exec_module(module)\n"
            "torch._inductor.lowering = module\n"
            "import triton.compiler, triton.compiler.compiler as compiler\n"
            "from triton._C import libtriton\n"
            "original_compile = compiler.compile\n"
            "def traced_compile(*args, **kwargs):\n"
            f"    pathlib.Path({str(trace)!r}, str(os.getpid()) + '-compile.json').write_text(json.dumps(dict("
            "triton=triton.__file__, native=libtriton.__file__, "
            "native_sha256=hashlib.sha256(pathlib.Path(libtriton.__file__).read_bytes()).hexdigest(), "
            "lowering=module.__file__, lowering_sha256=hashlib.sha256(pathlib.Path(module.__file__).read_bytes()).hexdigest())))\n"
            "    return original_compile(*args, **kwargs)\n"
            "compiler.compile = triton.compiler.compile = triton.compile = traced_compile\n"
            f"pathlib.Path({str(trace)!r}, str(os.getpid()) + '.json').write_text(json.dumps(dict("
            "torch_sha=torch.version.git_version, triton=triton.__file__, "
            "version=importlib.metadata.version('triton'), lowering=module.__file__, "
            "lowering_sha256=hashlib.sha256(pathlib.Path(module.__file__).read_bytes()).hexdigest())))\n"
        )
        env = os.environ.copy()
        for key in list(env):
            if key.startswith("TRITON_") or key in ("PYTHONPATH", "TORCH_CUSTOM_PYTHONPATH"):
                env.pop(key)
        env.update({"PYTHONPATH": os.pathsep.join((str(site), str(overlay))),
                    "TORCH_CUSTOM_PYTHONPATH": os.pathsep.join((str(site), str(overlay))),
                    "PIN_TRITON_SHORT": short, "PIN_TRITON_SITE": str(site),
                    "PIN_SCALAR_MODE": mode, "PIN_SCALAR_PHASE_DIR": str(phase),
                    "TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1",
                    "TRITON_CACHE_DIR": str(phase / "triton-cache"),
                    "TORCHINDUCTOR_CACHE_DIR": str(phase / "inductor-cache")})
        log = REPORT / f"{label}-{mode}.log"
        with log.open("w") as output:
            completed = subprocess.run([sys.executable, __file__, "child"], cwd=ROOT / "test",
                                       env=env, stdout=output, stderr=subprocess.STDOUT, timeout=1800)
        item = {"pin": label, "mode": mode, "exit": completed.returncode,
                "wheel": wheel.name,
                "wheel_sha256": hashlib.file_digest(wheel.open("rb"), "sha256").hexdigest(),
                "lowering_sha256": hashlib.sha256(code.encode()).hexdigest(),
                "log": str(log), "log_sha256": hashlib.file_digest(log.open("rb"), "sha256").hexdigest()}
        results.append(item)
        compile_traces = [json.loads(path.read_text()) for path in trace.glob('*-compile.json')]
        assert compile_traces, ('No actual compiler worker provenance', label, mode)
        assert all(Path(item['native']).is_relative_to(site) and
                   item['lowering_sha256'] == hashlib.sha256(code.encode()).hexdigest()
                   for item in compile_traces), compile_traces
        (REPORT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(item), flush=True)
        print(log.read_text()[-8000:], flush=True)
assert len(results) == 4
assert all(item["exit"] == 0 if item["mode"] == "patched" else item["exit"] != 0 for item in results)
