"""Compare unchanged YOLO memory measurements with complete native packages."""

import csv
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from importlib.metadata import Distribution
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
REPORT = WORKSPACE / "test/test-reports/a100-memory-controls"
DEBUG = WORKSPACE / "build/pin-a100-debug"
NATIVE = DEBUG / "native-site"
NATIVE_SHA = "f8b01c494a199871937dbccee940ad064f11991f"
PINS = {
    "old": {
        "short": "b7ddcf86",
        "wheel_sha256": "86f90ad1eaa0e5db399095509a315d2015d3a581d6f3233d4727ffe8af03569f",
        "native_sha256": "92d57fae9404795047c06fbf17a355778471a841b97fe8c34de8d46376a02bec",
    },
    "candidate": {
        "short": "1b9ee7ae",
        "wheel_sha256": "56bfa69b791c38596d168a8f57417ec61d9c6898842c0a8817f1a260351fc38a",
        "native_sha256": "adc86548b2ee251454558402fc7044bf0d54511eebe1847a7b7093e1ddb7513b",
    },
}


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


REPORT.mkdir(parents=True, exist_ok=True)
DEBUG.mkdir(parents=True, exist_ok=True)
native_wheels = list((WORKSPACE / "dist").rglob("*.whl"))
torch_wheels = [path for path in native_wheels if path.name.startswith("torch-")]
if len(torch_wheels) != 1 or "gitf8b01c4" not in torch_wheels[0].name:
    raise RuntimeError(f"Wrong native artifact: {torch_wheels}")
native_packages = []
for wheel in native_wheels:
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(NATIVE)
    native_packages.append({"wheel": wheel.name, "sha256": digest(wheel)})

sources = [
    "benchmarks/dynamo/common.py",
    "benchmarks/dynamo/torchbench.py",
    "benchmarks/dynamo/check_memory_compression_ratio.py",
    "benchmarks/dynamo/expected_ci_perf_inductor_torchbench.csv",
    "benchmarks/dynamo/torchbench_models_list.txt",
]
source_hashes = {name: digest(WORKSPACE / name) for name in sources}
torchbench_revision = subprocess.check_output(
    ["git", "-C", "/torchbench", "rev-parse", "HEAD"], text=True
).strip()
manifest = {
    "native_head": NATIVE_SHA,
    "native_packages": native_packages,
    "source_sha256": source_hashes,
    "torchbench_revision": torchbench_revision,
    "native_build_artifact_sha256": (
        WORKSPACE / "build/pin-a100-native-artifact.sha256"
    )
    .read_text()
    .split()[0],
}
(REPORT / "native-build-and-source-provenance.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)

results = []
for phase, pin in PINS.items():
    wheels = list((WORKSPACE / "build/pin-a100-packages" / phase).rglob("*.whl"))
    if len(wheels) != 1 or digest(wheels[0]) != pin["wheel_sha256"]:
        raise RuntimeError(f"Wrong complete {phase} Triton wheel: {wheels}")
    package = DEBUG / phase / "triton-site"
    with zipfile.ZipFile(wheels[0]) as archive:
        archive.extractall(package)
    metadata = list(package.glob("*.dist-info"))
    if len(metadata) != 1 or pin["short"] not in Distribution.at(metadata[0]).version:
        raise RuntimeError(f"Wrong package metadata: {metadata}")
    overlay = DEBUG / phase / "overlay"
    overlay.mkdir(parents=True, exist_ok=True)
    workers = REPORT / phase / "workers"
    workers.mkdir(parents=True, exist_ok=True)
    hook = """import hashlib, importlib.metadata, json, os, pathlib
import torch, triton
import triton.compiler, triton.compiler.compiler as compiler
from triton._C import libtriton
native_site = pathlib.Path(os.environ["PIN_A100_NATIVE_SITE"])
triton_site = pathlib.Path(os.environ["PIN_A100_TRITON_SITE"])
trace = pathlib.Path(os.environ["PIN_A100_WORKERS"])
def digest(path):
    with pathlib.Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()
if torch.version.git_version != os.environ["PIN_A100_NATIVE_HEAD"]:
    raise RuntimeError("Native Torch head mismatch")
if not pathlib.Path(torch.__file__).resolve().is_relative_to(native_site):
    raise RuntimeError("Torch package escaped complete native site")
if not pathlib.Path(triton.__file__).resolve().is_relative_to(triton_site):
    raise RuntimeError("Triton package escaped complete SDK site")
if digest(libtriton.__file__) != os.environ["PIN_A100_TRITON_NATIVE_SHA256"]:
    raise RuntimeError("Triton native compiler does not match its complete wheel")
provenance = dict(pid=os.getpid(), torch_sha=torch.version.git_version,
                  torch_package=torch.__file__, torch_native=torch._C.__file__,
                  torch_native_sha256=digest(torch._C.__file__),
                  triton_version=importlib.metadata.version("triton"),
                  triton_package=triton.__file__, triton_native=libtriton.__file__,
                  triton_native_sha256=digest(libtriton.__file__))
trace.joinpath(str(os.getpid()) + "-import.json").write_text(json.dumps(provenance, indent=2))
original_compile = compiler.compile
def traced_compile(*args, **kwargs):
    path = trace / (str(os.getpid()) + "-compile.json")
    if not path.exists():
        path.write_text(json.dumps(provenance, indent=2))
    return original_compile(*args, **kwargs)
compiler.compile = triton.compiler.compile = triton.compile = traced_compile
"""
    (overlay / "sitecustomize.py").write_text(hook)
    environment = dict(os.environ)
    for name in list(environment):
        if name.startswith("TRITON_") or name in (
            "PYTHONPATH",
            "TORCH_CUSTOM_PYTHONPATH",
        ):
            environment.pop(name)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                (str(package), str(overlay), str(NATIVE), "/torchbench")
            ),
            "TORCH_CUDA_ARCH_LIST": "8.0",
            "OMP_NUM_THREADS": "4",
            "TRITON_CACHE_DIR": str(DEBUG / phase / "triton-cache"),
            "TORCHINDUCTOR_CACHE_DIR": str(DEBUG / phase / "inductor-cache"),
            "TORCHINDUCTOR_WORKER_LOGPATH": str(REPORT / phase / "compile-worker.log"),
            "PIN_A100_NATIVE_SITE": str(NATIVE),
            "PIN_A100_NATIVE_HEAD": NATIVE_SHA,
            "PIN_A100_TRITON_SITE": str(package),
            "PIN_A100_WORKERS": str(workers),
            "PIN_A100_TRITON_NATIVE_SHA256": pin["native_sha256"],
        }
    )
    environment["TORCH_CUSTOM_PYTHONPATH"] = environment["PYTHONPATH"]
    preflight = """import json, os, pathlib, subprocess, torch, torchvision, triton
from triton._C import libtriton
from torch._inductor.utils import python_subprocess_env
worker_pythonpath = python_subprocess_env()["PYTHONPATH"]
if worker_pythonpath != os.environ["PYTHONPATH"]:
    raise RuntimeError("Compiler subprocess does not preserve the complete package paths")
gpu=torch.cuda.get_device_name()
if "A100" not in gpu or "sm_80" not in torch.cuda.get_arch_list():
    raise RuntimeError("Original A100/native architecture mismatch")
if torch.version.cuda != "13.2" or torch.backends.cudnn.version() != 92700:
    raise RuntimeError("Original CUDA/cuDNN stack mismatch")
print(json.dumps(dict(torch_sha=torch.version.git_version, cuda=torch.version.cuda,
 cudnn=torch.backends.cudnn.version(), gpu=gpu, arches=torch.cuda.get_arch_list(),
 torch_package=torch.__file__, torchvision_package=torchvision.__file__,
 torchvision_version=torchvision.__version__, triton_package=triton.__file__,
 triton_native=libtriton.__file__, compiler_subprocess_pythonpath=worker_pythonpath,
 config=torch.__config__.show(),
 nvidia_smi=subprocess.check_output(["nvidia-smi"], text=True)), indent=2))
"""
    preflight_log = REPORT / phase / "preflight.json"
    with preflight_log.open("w") as output:
        completed = subprocess.run(
            [sys.executable, "-c", preflight],
            cwd=DEBUG,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=180,
        )
    print(preflight_log.read_text(), flush=True)
    if completed.returncode:
        raise RuntimeError(
            f"Invalid {phase} native setup; no test result: {completed.returncode}"
        )
    output_csv = REPORT / phase / "yolov3.csv"
    command = [
        sys.executable,
        "benchmarks/dynamo/torchbench.py",
        "--device",
        "cuda",
        "--performance",
        "--backend",
        "inductor",
        "--amp",
        "--training",
        "--disable-cudagraphs",
        "--batch-size-file",
        str(WORKSPACE / "benchmarks/dynamo/torchbench_models_list.txt"),
        "--only",
        "yolov3",
        "--output",
        str(output_csv),
    ]
    log = REPORT / phase / "yolov3.log"
    with log.open("w") as output:
        completed = subprocess.run(
            command,
            cwd=WORKSPACE,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=1800,
        )
    print(log.read_text()[-20000:], flush=True)
    if completed.returncode or not output_csv.exists():
        raise RuntimeError(
            f"Original {phase} benchmark did not complete: {completed.returncode}"
        )
    with output_csv.open() as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 1 or rows[0]["name"] != "yolov3" or rows[0]["batch_size"] != "8":
        raise RuntimeError(f"Original benchmark input/result mismatch: {rows}")
    compile_traces = [
        json.loads(path.read_text()) for path in workers.glob("*-compile.json")
    ]
    if not compile_traces or any(
        item["torch_sha"] != NATIVE_SHA
        or item["triton_native_sha256"] != pin["native_sha256"]
        for item in compile_traces
    ):
        raise RuntimeError(
            f"Missing or incorrect actual compilation-worker provenance: {compile_traces}"
        )
    check_log = REPORT / phase / "memory-gate.log"
    with check_log.open("w") as output:
        checked = subprocess.run(
            [
                sys.executable,
                "benchmarks/dynamo/check_memory_compression_ratio.py",
                "--actual",
                str(output_csv),
                "--expected",
                "benchmarks/dynamo/expected_ci_perf_inductor_torchbench.csv",
            ],
            cwd=WORKSPACE,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
    print(check_log.read_text(), flush=True)
    if checked.returncode not in (0, 1):
        raise RuntimeError(
            f"Invalid original memory checker result: {checked.returncode}"
        )
    item = {
        "phase": phase,
        "native_head": NATIVE_SHA,
        "wheel": wheels[0].name,
        "wheel_sha256": digest(wheels[0]),
        "native_compiler_sha256": pin["native_sha256"],
        "benchmark_exit": completed.returncode,
        "original_memory_gate_exit": checked.returncode,
        "csv": rows[0],
        "actual_compilation_worker_count": len(compile_traces),
        "log_sha256": digest(log),
        "command": command,
    }
    results.append(item)
    (REPORT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("A100_MEMORY_CONTROL " + json.dumps(item), flush=True)
if len(results) != 2:
    raise RuntimeError("Both complete-package controls are required")
print(
    "Both unchanged exact-native A100 old/candidate memory controls captured; gate failures remain failures.",
    flush=True,
)
