import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / "test/test-reports/logisticnormal-controls"
HEAD = "f8b01c494a199871937dbccee940ad064f11991f"
SITE = ROOT / "native-f8-site"
TEST = Path(os.environ["PIN_LOGISTICNORMAL_TEST"]) if "PIN_LOGISTICNORMAL_TEST" in os.environ else None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if len(sys.argv) > 1 and sys.argv[1] == "child":
    import importlib.metadata
    import runpy
    import unittest

    import torch
    import triton
    from triton._C import libtriton

    phase, mode = sys.argv[2:4]
    expected_pin = "b7ddcf86" if phase == "old" else "1b9ee7ae"
    version = importlib.metadata.version("triton")
    assert expected_pin in version, version
    assert torch.version.git_version == HEAD, torch.version.git_version
    assert Path(torch.__file__).resolve().is_relative_to(SITE)
    assert "COMMIT_SHA=" + HEAD in torch.__config__.show()
    assert torch.version.cuda == "13.2" and torch.cuda.is_available()
    assert "A10G" in torch.cuda.get_device_name() and "sm_86" in torch.cuda.get_arch_list()
    assert os.environ["PYTORCH_TEST_WITH_SLOW_GRADCHECK"] == "1"
    assert torch.backends.cudnn.version() is not None
    assert tuple(torch._C._cudnn.getRuntimeVersion()) == (9, 27, 0)
    provenance = {
        "phase": phase, "mode": mode, "native_torch": torch.version.git_version,
        "torch_file": torch.__file__, "native_cuda_sha256": digest(SITE / "torch/lib/libtorch_cuda.so"),
        "native_python_sha256": digest(SITE / "torch/lib/libtorch_python.so"),
        "triton_version": version, "triton_file": triton.__file__,
        "triton_native_file": libtriton.__file__, "triton_native_sha256": digest(libtriton.__file__),
        "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda, "native_arches": torch.cuda.get_arch_list(),
        "cudnn_runtime": torch._C._cudnn.getRuntimeVersion(),
        "cudnn_compiled": torch._C._cudnn.getCompileVersion(),
        "cudnn_sha256": digest(next(Path(line.split()[-1]) for line in Path("/proc/self/maps").read_text().splitlines() if "/libcudnn.so." in line)),
        "test_source_sha256": digest(TEST), "has_lapack": torch._C.has_lapack,
        "slow_gradcheck": os.environ.get("PYTORCH_TEST_WITH_SLOW_GRADCHECK"),
    }
    print(json.dumps(provenance), flush=True)
    compilation_calls = []
    original_compile = triton.compile

    def count_compile(*args, **kwargs):
        compilation_calls.append(True)
        return original_compile(*args, **kwargs)

    triton.compile = count_compile
    module = runpy.run_path(str(TEST), run_name="pin_logisticnormal_native_control")
    module["TestCase"]._default_dtype_check_enabled = True
    namespace = module["TestDistributionsDeviceCUDA"]._gradcheck_log_prob.__globals__
    original_gradcheck = namespace["gradcheck"]
    captured = []

    def capture(fn, inputs, **kwargs):
        tensors = [value.detach().cpu() if isinstance(value, torch.Tensor) else value
                   for value in inputs]
        requires_grad = [isinstance(value, torch.Tensor) and value.requires_grad for value in inputs]
        captured.append((fn, tensors, requires_grad))
        torch.save(tensors, ROOT / f"logisticnormal-{phase}-{mode}-inputs-{len(captured)}.pt")
        return original_gradcheck(fn, inputs, **kwargs)

    namespace["gradcheck"] = capture
    cases = [module["TestDistributionsDeviceCUDA"]("test_logisticnormal_cuda")]
    if mode == "proposed":
        cases.extend([module["TestDistributionsDeviceCPU"]("test_logisticnormal_cpu"),
                      module["TestDistributionsDeviceCUDA"]("test_logisticnormal_logprob_cuda"),
                      module["TestDistributionsDeviceCPU"]("test_logisticnormal_logprob_cpu")])
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(cases))
    assert result.testsRun == len(cases) and not result.skipped
    assert not compilation_calls, "Unexpected Triton compilation in eager test"
    summary = {**provenance, "tests": result.testsRun, "test_success": result.wasSuccessful(),
               "test_failures": len(result.failures), "test_errors": len(result.errors),
               "gradcheck_calls": len(captured), "triton_compile_calls": len(compilation_calls)}

    if mode == "original":
        import math
        import mpmath as mp

        assert len(result.errors) == 1 and "GradcheckError" in result.errors[0][1]
        assert captured
        fn, cpu_inputs, requires_grad = captured[0]
        inputs = [value.to("cuda").detach().requires_grad_(requires)
                  if isinstance(value, torch.Tensor) else value
                  for value, requires in zip(cpu_inputs, requires_grad)]
        sample, loc, scale = inputs
        assert sample.shape == (5, 6)
        row, coordinate = 2, 3
        analytical = torch.autograd.grad(fn(*inputs).sum(), sample)[0][row, coordinate].item()
        step = 1e-6
        plus, minus = sample.detach().clone(), sample.detach().clone()
        plus[row, coordinate] += step
        minus[row, coordinate] -= step
        numerical = ((fn(plus, loc, scale)[row] - fn(minus, loc, scale)[row]) / (2 * step)).item()
        mp.mp.dps = 80
        y = [mp.mpf(value) for value in sample[row].detach().cpu().tolist()]
        m = [mp.mpf(value) for value in loc[row].detach().cpu().tolist()]
        sd = [mp.mpf(value) for value in scale[row].detach().cpu().tolist()]

        def reference_density(value):
            coordinates = y.copy()
            coordinates[coordinate] = value
            total = mp.mpf(0)
            residual = mp.mpf(1)
            count = len(m)
            jacobian = mp.mpf(0)
            for index in range(count):
                residual -= coordinates[index]
                latent = mp.log(coordinates[index]) - mp.log(residual) + mp.log(count - index)
                total += -(latent - m[index]) ** 2 / (2 * sd[index] ** 2)
                total -= mp.log(sd[index]) + mp.log(2 * mp.pi) / 2
                jacobian += mp.log(coordinates[index])
            jacobian += mp.log(residual)
            return total - jacobian

        point = y[coordinate]
        reference_analytical = mp.diff(reference_density, point)
        differences = {}
        for epsilon in (1e-6, 5e-7, 1e-7):
            e = mp.mpf(epsilon)
            value = (reference_density(point + e) - reference_density(point - e)) / (2 * e)
            differences[str(epsilon)] = str(value)
        threshold = 1e-5 + 1e-3 * abs(float(reference_analytical))
        reference = {
            "sample_row": row, "coordinate": coordinate, "torch_analytical": analytical,
            "torch_central_1e_6": numerical, "mpmath_precision_digits": mp.mp.dps,
            "reference_analytical": str(reference_analytical),
            "reference_central_differences": differences,
            "strict_gradcheck_threshold": threshold,
            "native_analytical_reference_error": abs(analytical - float(reference_analytical)),
            "central_truncation_error": abs(float(differences["1e-06"]) - float(reference_analytical)),
            "minimum_scale": scale.min().item(), "minimum_sample_coordinate": sample.min().item(),
        }
        summary["reference"] = reference
        (ROOT / f"logisticnormal-{phase}-reference.json").write_text(json.dumps(reference, indent=2) + "\n")
        print(json.dumps(reference, indent=2), flush=True)
        assert reference["native_analytical_reference_error"] < 1e-8
        assert abs(numerical - float(differences["1e-06"])) < 1e-6
        assert reference["central_truncation_error"] > threshold
    else:
        assert result.wasSuccessful(), "Proposed strict test did not pass"
        assert len(captured) == 6 and all(values[0].shape == (5, 5) for _, values, _ in captured)
        assert all(flags[0] for _, _, flags in captured)
    (ROOT / f"logisticnormal-{phase}-{mode}-result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    raise SystemExit(0)


import zipfile

ROOT.mkdir(parents=True, exist_ok=True)
torch_wheels = list((WORKSPACE / "dist").glob("torch-*.whl"))
assert len(torch_wheels) == 1
torch_wheel = torch_wheels[0]
assert digest(torch_wheel) == "69781b9beaae571dd3be1249de415b662ab54bd5015c92bfb65949f60fe5ac03"
with zipfile.ZipFile(torch_wheel) as archive:
    archive.extractall(SITE)
expected_wheels = {
    "old": "86f90ad1eaa0e5db399095509a315d2015d3a581d6f3233d4727ffe8af03569f",
    "candidate": "56bfa69b791c38596d168a8f57417ec61d9c6898842c0a8817f1a260351fc38a",
}
packages = {}
for phase, wheel_sha in expected_wheels.items():
    wheels = list((WORKSPACE / "build/pin-logisticnormal-packages" / phase).rglob("*.whl"))
    assert len(wheels) == 1
    assert digest(wheels[0]) == wheel_sha
    target = ROOT / (phase + "-site")
    with zipfile.ZipFile(wheels[0]) as archive:
        archive.extractall(target)
    packages[phase] = target
results = []
for mode in ("original", "proposed"):
    test = WORKSPACE / f".ci/pytorch/triton_pin_logisticnormal_{mode}.py"
    for phase, package in packages.items():
        environment = dict(os.environ)
        for variable in ("PYTORCH_TESTING_DEVICE_ONLY_FOR", "PYTORCH_TESTING_DEVICE_EXCEPT_FOR", "PYTORCH_TESTING_DEVICE_FOR_CUSTOM"):
            environment.pop(variable, None)
        environment.update(PYTHONPATH=f"{package}:{SITE}", TORCH_CUSTOM_PYTHONPATH="",
                           PYTORCH_TEST_WITH_SLOW_GRADCHECK="1", PYTORCH_TEST_CUDA_MEM_LEAK_CHECK="1",
                           PIN_LOGISTICNORMAL_TEST=str(test))
        log = ROOT / f"logisticnormal-{phase}-{mode}.log"
        with log.open("w") as output:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "child", phase, mode],
                                    cwd=WORKSPACE, env=environment, stdout=output, stderr=subprocess.STDOUT)
        print(log.read_text(), flush=True)
        assert result.returncode == 0, (phase, mode, result.returncode)
        summary = json.loads((ROOT / f"logisticnormal-{phase}-{mode}-result.json").read_text())
        results.append(summary)
(ROOT / "logisticnormal-all-controls.json").write_text(json.dumps(results, indent=2) + "\n")
print("Complete exact-native old/candidate original/reference and strict proposed controls verified", flush=True)
