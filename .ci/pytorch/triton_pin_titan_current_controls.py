"""Compare complete default pins in the refreshed original native SFT case."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TITAN = ROOT / "torchtitan"
REPORT = ROOT / "test/test-reports/triton-pin-titan-current-controls"
WORK = ROOT / "build/triton-pin-titan-current-controls"
NATIVE = WORK / "native-site"
NATIVE_HEAD = "9e8a5dd5bddcadf7b6016e8838e55acb2eafb300"
TITAN_HEAD = "d0ab95e46d601a7cce33b951bfc799b55d103788"
SOURCES = {
    "tests/integration_tests/features.py": "44499e6248b92346e1c1debc67e610fbdad333930813882476c207311b506c93",
    "tests/integration_tests/run_tests.py": "1697816c331dcd50767426324c8a881f0f3b3c51518c2d55ea4e88d6a330ac39",
    "scripts/loss_compare.py": "3773c8093311b691fe4c4aa6eaa425f5ea02b87fb180cd767a6c0acbbd25514c",
    "tests/assets/losses/real_pg/a10g/sft.txt": "0c679eb0937b96c1c67b67ac1a06c20e344bd5bef30b244d625302b9196fe678",
}
PINS = {
    "old": {
        "short": "b7ddcf86",
        "source_sha": "b7ddcf86c095de79527ba3098503e73158bb94bd",
        "wheel_sha256": "677e31599ea197fbc6aa73b76bb3c2562de7dd815c3ee3cf10a2a53b0d3b6c30",
        "compiler_sha256": "8e74638cb55a4ebcc95edf7c6707b9ad221ace6977b0e0cccfcb28cad58b9e3a",
        "source_run": 40454125718,
    },
    "candidate": {
        "short": "1b9ee7ae",
        "source_sha": "1b9ee7ae35e6e56e3f1e53650154910bdbeb4768",
        "wheel_sha256": "eb7ca1d7f99b8a3411a98936dfab8d25c585260d1c5b2a9721712b3b5dda448b",
        "compiler_sha256": "e36fd071c5add9bef81574b9f564d1c5b34fad2e2761b9a919d80a0aaf00c37f",
        "source_run": 42148622972,
    },
}


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def extract_wheel(wheel, destination):
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(destination)
        for member in archive.infolist():
            mode = (member.external_attr >> 16) & 0o777
            if mode:
                (destination / member.filename).chmod(mode)


def run(command, name, *, environment=None, check=True, timeout=1800):
    path = REPORT / f"{name}.log"
    with path.open("w") as output:
        result = subprocess.run(
            command,
            cwd=TITAN,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    print(name, "exit", result.returncode, flush=True)
    for line in path.read_text(errors="replace").splitlines()[-70:]:
        print(line[:2500], flush=True)
    if check and result.returncode:
        raise RuntimeError(f"{name} failed before a valid original control")
    return result.returncode


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=TITAN, text=True
    ).strip()
    if revision != TITAN_HEAD or any(
        digest(TITAN / p) != value for p, value in SOURCES.items()
    ):
        raise RuntimeError("Refreshed original TorchTitan source/gold differs")
    subprocess.run(["git", "diff", "--exit-code"], cwd=TITAN, check=True)
    wheels = list((ROOT / "dist").rglob("*.whl"))
    torch_wheels = [p for p in wheels if p.name.startswith("torch-")]
    if len(torch_wheels) != 1 or "git9e8a5dd" not in torch_wheels[0].name:
        raise RuntimeError("Complete original native Torch artifact differs")
    native_packages = []
    for wheel in wheels:
        extract_wheel(wheel, NATIVE)
        native_packages.append({"wheel": wheel.name, "sha256": digest(wheel)})
    torch_native = list((NATIVE / "torch").glob("_C*.so"))
    if len(torch_native) != 1:
        raise RuntimeError("Original native Torch extension is ambiguous")
    torch_native_sha256 = digest(torch_native[0])
    manifest = {
        "native_head": NATIVE_HEAD,
        "native_artifact_run": 42148713138,
        "native_artifact_sha256": (
            ROOT / "build/pin-titan-current-native-artifact.sha256"
        )
        .read_text()
        .split()[0],
        "native_packages": native_packages,
        "torchtitan_head": TITAN_HEAD,
        "unchanged_torchtitan_source_sha256": SOURCES,
        "pins": PINS,
        "controls": {},
    }
    # Match the original standalone reader's sibling imports.
    sys.path.insert(0, str(TITAN / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "current_loss_compare", TITAN / "scripts/loss_compare.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Original metric reader is unavailable")
    comparison = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = comparison
    spec.loader.exec_module(comparison)
    gold = {"loss": {}, "grad_norm": {}}
    for line in (
        (TITAN / "tests/assets/losses/real_pg/a10g/sft.txt").read_text().splitlines()
    ):
        if line and not line.startswith("#"):
            step, loss, norm = line.split()
            gold["loss"][int(step)] = float(loss)
            gold["grad_norm"][int(step)] = float(norm)
    for label, pin in PINS.items():
        phase = WORK / label
        package = phase / "triton-site"
        inputs = list(
            (ROOT / "build/pin-titan-current-packages" / label).rglob("*.whl")
        )
        if len(inputs) != 1 or digest(inputs[0]) != pin["wheel_sha256"]:
            raise RuntimeError(f"Complete official {label} package differs")
        extract_wheel(inputs[0], package)
        phase_report = REPORT / label
        workers = phase_report / "workers"
        phase_report.mkdir(parents=True, exist_ok=True)
        workers.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        for name in list(environment):
            if name.startswith("TRITON_") or name in (
                "PYTHONPATH",
                "TORCH_CUSTOM_PYTHONPATH",
            ):
                environment.pop(name)
        overlay = ROOT / ".ci/pytorch/triton_pin_titan_current_trace"
        environment.update(
            {
                "PYTHONPATH": os.pathsep.join(
                    (str(package), str(overlay), str(NATIVE), str(TITAN))
                ),
                "TRITON_CACHE_DIR": str(phase / "triton-cache"),
                "TORCHINDUCTOR_CACHE_DIR": str(phase / "inductor-cache"),
                "CUDA_CACHE_PATH": str(phase / "cuda-cache"),
                "TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1",
                "CUDA_VISIBLE_DEVICES": "0,1",
                "PIN_TITAN_CURRENT_NATIVE_SITE": str(NATIVE),
                "PIN_TITAN_CURRENT_NATIVE_HEAD": NATIVE_HEAD,
                "PIN_TITAN_CURRENT_TORCH_NATIVE_SHA256": torch_native_sha256,
                "PIN_TITAN_CURRENT_TRITON_SITE": str(package),
                "PIN_TITAN_CURRENT_TRITON_SHORT": pin["short"],
                "PIN_TITAN_CURRENT_COMPILER_SHA256": pin["compiler_sha256"],
                "PIN_TITAN_CURRENT_WORKERS": str(workers),
            }
        )
        environment["TORCH_CUSTOM_PYTHONPATH"] = environment["PYTHONPATH"]
        preflight = """import json, os, pathlib, subprocess, torch, triton
from triton.backends.nvidia.compiler import get_ptxas
from torch._inductor.utils import python_subprocess_env
if torch.cuda.device_count() != 2 or any("A10G" not in torch.cuda.get_device_name(i) for i in range(2)):
    raise RuntimeError("Original two-A10G execution differs")
if torch.version.cuda != "13.2" or torch.backends.cudnn.version() != 92700:
    raise RuntimeError("Original native CUDA/cuDNN stack differs")
worker_path = python_subprocess_env()["PYTHONPATH"]
if worker_path != os.environ["PYTHONPATH"]:
    raise RuntimeError("Compiler workers do not retain the complete package paths")
assembler = pathlib.Path(get_ptxas(86).path)
if not assembler.resolve().is_relative_to(pathlib.Path(triton.__file__).parent.resolve()):
    raise RuntimeError("Assembler escaped its own SDK")
version = subprocess.check_output([str(assembler), "--version"], text=True)
print(json.dumps(dict(torch_sha=torch.version.git_version, cuda=torch.version.cuda,
 cudnn=torch.backends.cudnn.version(), gpus=[torch.cuda.get_device_name(i) for i in range(2)],
 capability=torch.cuda.get_device_capability(), arches=torch.cuda.get_arch_list(),
 torch_package=torch.__file__, triton_package=triton.__file__, compiler_subprocess_pythonpath=worker_path,
 ptxas_path=str(assembler), ptxas_version=version, config=torch.__config__.show()), indent=2))
"""
        run(
            [sys.executable, "-c", preflight],
            label + "-preflight",
            environment=environment,
            timeout=180,
        )
        command = [
            sys.executable,
            "-m",
            "tests.integration_tests.run_tests",
            "--test_suite",
            "features",
            "--test_name",
            "sft",
            "--ngpu",
            "2",
            str(phase_report),
        ]
        result = run(
            command, label + "-original-sft", environment=environment, check=False
        )
        log = (REPORT / f"{label}-original-sft.log").read_text(errors="replace")
        if (
            result not in (0, 1)
            or "LOSS_COMPARE" not in log
            or "Training completed" not in log
        ):
            raise RuntimeError("Original SFT control did not fully execute")
        metrics = comparison.extract_metrics_from_tensorboard(
            str(phase_report / "sft"), "tb_baseline", ["loss", "grad_norm"]
        )
        if any(set(metrics[name]) != set(range(1, 11)) for name in gold):
            raise RuntimeError("Original twenty metric values are incomplete")
        differences = [
            {"metric": name, "step": step, "actual": metrics[name][step], "gold": value}
            for name, values in gold.items()
            for step, value in values.items()
            if metrics[name][step] != value
        ]
        if (result == 0) != (not differences):
            raise RuntimeError(
                "Original strict exit status and metric comparisons disagree"
            )
        records = [
            json.loads(line)
            for path in workers.glob("*.jsonl")
            for line in path.read_text().splitlines()
        ]
        compiled = [r for r in records if r["event"] == "compiled"]
        if not compiled or any(
            r["torch_sha"] != NATIVE_HEAD
            or r["triton_native_sha256"] != pin["compiler_sha256"]
            or pin["short"] not in r["triton_version"]
            for r in records
        ):
            raise RuntimeError(
                "Actual compilation worker identities are incomplete or differ"
            )
        manifest["controls"][label] = {
            "original_exit_code": result,
            "command": command,
            "metrics": metrics,
            "gold": gold,
            "differences": differences,
            "actual_compiling_pids": sorted({r["pid"] for r in compiled}),
            "compiler_records": records,
        }
        kernel_archive = REPORT / f"{label}-generated-kernels.zip"
        with zipfile.ZipFile(
            kernel_archive, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for cache in ("triton-cache", "inductor-cache"):
                cache_root = phase / cache
                if cache_root.exists():
                    for path in cache_root.rglob("*"):
                        if path.is_file() and path.suffix in (
                            ".py",
                            ".json",
                            ".ttir",
                            ".ttgir",
                            ".llir",
                            ".ptx",
                        ):
                            archive.write(
                                path, Path(cache) / path.relative_to(cache_root)
                            )
        (REPORT / "current-native-controls.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    subprocess.run(["git", "diff", "--exit-code"], cwd=TITAN, check=True)
    destination = ROOT / "test/debug/triton-pin-titan-current-controls"
    destination.mkdir(parents=True, exist_ok=True)
    for path in REPORT.iterdir():
        if path.is_file():
            shutil.copy2(path, destination / path.name)
    print(
        "CURRENT_NATIVE_TITAN_DEFAULT_CONTROLS",
        json.dumps(
            {
                label: {
                    "original_exit_code": value["original_exit_code"],
                    "differing_metrics": len(value["differences"]),
                }
                for label, value in manifest["controls"].items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
