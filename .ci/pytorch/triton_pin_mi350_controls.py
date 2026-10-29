"""Temporary native-pin controls for draft PR #202417; no expectation changes."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "test/test-reports/triton-pin-controls"
WORK = ROOT / "build/triton-pin-controls"
PINS = {
    "old": (
        "https://github.com/triton-lang/triton.git",
        "b7ddcf86c095de79527ba3098503e73158bb94bd",
    ),
    "candidate": (
        "https://github.com/warrendeng/triton.git",
        "1b9ee7ae35e6e56e3f1e53650154910bdbeb4768",
    ),
}
NODES = {
    "log1p": "test_batch_invariance_log1p_backend_inductor_default_cuda_float32",
    "tan": "test_unary_ufunc_numerical_tan_backend_inductor_default_cuda_float32",
}


def run(command, *, cwd=ROOT, env=None, name=None, check=True, timeout=7200):
    print(f"CONTROL {name or command[0]}: {' '.join(map(str, command))}", flush=True)
    start = time.monotonic()
    if name:
        with (REPORT / f"{name}.log").open("w") as log:
            result = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        output = (REPORT / f"{name}.log").read_text(errors="replace")
        print(output[-12000:], flush=True)
    else:
        result = subprocess.run(command, cwd=cwd, env=env, timeout=timeout)
    print(
        f"CONTROL exit={result.returncode} elapsed={time.monotonic() - start:.1f}s",
        flush=True,
    )
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    return result.returncode


def build_pin(label, repository, sha):
    phase = WORK / label
    phase.mkdir(parents=True)
    source = phase / "source"
    source.mkdir()
    run(["git", "init", str(source)])
    run(
        ["git", "-C", str(source), "fetch", "--depth=1", repository, sha],
        name=f"{label}-fetch",
    )
    run(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"])
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != sha:
        raise AssertionError((actual, sha))
    run(
        ["git", "submodule", "update", "--init", "--recursive"],
        cwd=source,
        name=f"{label}-submodules",
    )
    python = phase / "venv/bin/python"
    run([sys.executable, "-m", "venv", "--system-site-packages", str(phase / "venv")])
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("TRITON_") or key in {
            "LLVM_SYSPATH",
            "JSON_SYSPATH",
            "LLVM_INCLUDE_DIRS",
            "LLVM_LIBRARY_DIR",
            "LLVM_DIR",
            "MLIR_DIR",
            "LLD_DIR",
            "CMAKE_PREFIX_PATH",
        }:
            env.pop(key)
    env.update(
        {
            "PATH": str(python.parent) + os.pathsep + env["PATH"],
            "TRITON_HOME": str(phase / "dependencies"),
            "MAX_JOBS": "4",
            "TRITON_PARALLEL_LINK_JOBS": "1",
        }
    )
    requirements = tomllib.loads((source / "pyproject.toml").read_text())[
        "build-system"
    ]["requires"]
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "build",
            *requirements,
            "setuptools==78.1.1",
        ],
        env=env,
        name=f"{label}-build-dependencies",
    )
    run(
        [
            str(python),
            "-c",
            "import setuptools,setuptools.build_meta; print(setuptools.__version__,setuptools.__file__)",
        ],
        env=env,
        name=f"{label}-build-backend-preflight",
    )
    run(
        [str(python), "-m", "build", "--wheel", "--no-isolation"],
        cwd=source,
        env=env,
        name=f"{label}-native-build",
    )
    cmake_caches = list((source / "build").glob("**/CMakeCache.txt"))
    llvm_directories = [
        line.split("=", 1)[1]
        for cache in cmake_caches
        for line in cache.read_text().splitlines()
        if line.startswith("LLVM_DIR:")
    ]
    if not (
        llvm_directories
        and all(str(phase / "dependencies") in path for path in llvm_directories)
    ):
        raise AssertionError(f"LLVM package escaped this pin's SDK: {llvm_directories}")
    wheels = list((source / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise AssertionError(wheels)
    site = phase / "site"
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(site),
            str(wheels[0]),
        ],
        name=f"{label}-install",
    )
    manifests = {
        str(path.relative_to(source)): path.read_text()
        for path in (source / "cmake").glob("*llvm*info.json")
    }
    (REPORT / f"{label}-build-provenance.json").write_text(
        json.dumps(
            {
                "source_sha": actual,
                "wheel": wheels[0].name,
                "wheel_sha256": hashlib.file_digest(
                    wheels[0].open("rb"), "sha256"
                ).hexdigest(),
                "sdk_manifests": manifests,
                "cmake_llvm_directories": llvm_directories,
                "compiler": subprocess.check_output(["c++", "--version"], text=True),
            },
            indent=2,
        )
    )
    if label in PINS:
        debug = ROOT / "test/debug/triton-pin-native-wheels" / label
        debug.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(wheels[0], debug / wheels[0].name)
    return site


def configure_fusion(env, fusion):
    for key in (
        "TRITON_DEFAULT_FP_FUSION",
        "TRITON_ALLOW_FP_FUSION",
        "TRITON_FORCE_DISABLE_FP_FUSION",
    ):
        env.pop(key, None)
    if fusion is not None:
        env["TRITON_DEFAULT_FP_FUSION"] = str(fusion)
        env["TRITON_ALLOW_FP_FUSION"] = "1"


def fusion_provenance(fusion):
    code = (
        "language=triton.knobs.language; "
        "print('FUSION_CONTROL',language.fp_fusion_enabled(True),"
        "language.fp_fusion_enabled(None)); "
    )
    if fusion is not None:
        code += 'assert language.fp_fusion_enabled(True), "Implicit fusion control was not enabled"; '
    return code


def test_pin(label, site, sha, *, fusion=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(site) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    trace_site = ROOT / ".ci/pytorch/triton_pin_titan_trace_site"
    env["PYTHONPATH"] = str(trace_site) + os.pathsep + env["PYTHONPATH"]
    env["TORCH_CUSTOM_PYTHONPATH"] = env["PYTHONPATH"]
    trace = ROOT / "test/debug/triton-pin-native-property-trace" / label
    env["PIN_TRITON_TRACE_DIR"] = str(trace)
    configure_fusion(env, fusion)
    cache = WORK / f"cache-{label}"
    env["TRITON_CACHE_DIR"] = str(cache / "triton")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
    env["CUDA_CACHE_PATH"] = str(cache / "cuda")
    env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    probe = (
        "import json,torch,triton,importlib.metadata,pathlib; "
        "p=torch.cuda.get_device_properties(0); "
        "assert torch.version.hip and 'gfx950' in p.gcnArchName, str(p); "
        "v=importlib.metadata.version('triton'); "
        f"assert '{sha[:8]}' in v, v; "
        f"assert pathlib.Path(triton.__file__).resolve().is_relative_to({str(site)!r}), triton.__file__; "
        + fusion_provenance(fusion)
        + "print(json.dumps(dict(torch_version=torch.__version__,torch_sha=torch.version.git_version,"
        "hip=torch.version.hip,device=str(p),triton_version=triton.__version__,triton_distribution=v,triton_file=triton.__file__),indent=2))"
    )
    run(
        [sys.executable, "-c", probe],
        cwd=ROOT / "test",
        env=env,
        name=f"{label}-runtime-provenance",
    )
    results = {}
    for operation, node in NODES.items():
        name = f"{label}-{operation}"
        results[operation] = run(
            [
                sys.executable,
                str(ROOT / ".ci/pytorch/triton_pin_trace_property.py"),
                "-s",
                "--tb=short",
                "--reruns=0",
                f"--junitxml={REPORT / (name + '.xml')}",
                f"inductor/test_torchinductor_opinfo_properties.py::TestOpInfoPropertiesCUDA::{node}",
            ],
            cwd=ROOT / "test",
            env=env,
            name=name,
            check=False,
            timeout=1200,
        )
        log = (REPORT / f"{name}.log").read_text(errors="replace")
        if results[operation] not in (0, 1):
            raise AssertionError(f"Invalid test execution: {name}")
        if not ("1 passed" in log or "1 xfailed" in log or "XPASS:" in log):
            raise AssertionError(f"Original case did not complete: {name}")
    records = [
        json.loads(line)
        for path in trace.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    compiled = [record for record in records if record["event"] == "compiled"]
    (REPORT / f"{label}-compile-provenance.json").write_text(
        json.dumps(compiled, indent=2)
    )
    if not compiled or not all(
        Path(record["triton_file"]).resolve().is_relative_to(site)
        and Path(record["triton_core_file"]).resolve().is_relative_to(site)
        for record in records
    ):
        raise AssertionError(f"Compiler pin provenance changed for {label}")
    return results


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "-c",
            "import torch; p=torch.cuda.get_device_properties(0); "
            "assert torch.version.hip and 'gfx950' in p.gcnArchName, str(p); print(p)",
        ],
        cwd=ROOT / "test",
        name="mi350-preflight",
        timeout=180,
    )
    (REPORT / "pytorch-source-provenance.json").write_text(
        json.dumps(
            {
                "draft_head": "914c15e40b4d5818fb0e85fe7b94d77b5f00c332",
                "control_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "property_test_sha256": hashlib.sha256(
                    (
                        ROOT / "test/inductor/test_torchinductor_opinfo_properties.py"
                    ).read_bytes()
                ).hexdigest(),
                "docker_tree": subprocess.check_output(
                    ["git", "rev-parse", "HEAD:.ci/docker"], cwd=ROOT, text=True
                ).strip(),
            },
            indent=2,
        )
    )
    results = {}
    sites = {}
    selected_pins = (
        {"candidate": PINS["candidate"]} if "--fusion-only" in sys.argv else PINS
    )
    for label, (repository, sha) in selected_pins.items():
        sites[label] = build_pin(label, repository, sha)
        results[label] = test_pin(label, sites[label], sha)
        (REPORT / "results.json").write_text(json.dumps(results, indent=2))
    results["candidate-implicit-fusion-restored"] = test_pin(
        "candidate-implicit-fusion-restored",
        sites["candidate"],
        PINS["candidate"][1],
        fusion=1,
    )
    (REPORT / "results.json").write_text(json.dumps(results, indent=2))
    if "--verify-fix" in sys.argv:
        if set(results) != {"old", "candidate", "candidate-implicit-fusion-restored"}:
            raise AssertionError(results)
        if not all(code == 0 for phase in results.values() for code in phase.values()):
            raise AssertionError(results)
        for phase, outcome in [
            ("old", "1 xfailed"),
            ("candidate", "1 passed"),
            ("candidate-implicit-fusion-restored", "1 xfailed"),
        ]:
            for operation in NODES:
                log = (REPORT / f"{phase}-{operation}.log").read_text(errors="replace")
                if not (outcome in log and "XPASS:" not in log):
                    raise AssertionError((phase, operation, outcome))
        print("MI350_ORIGINAL_FIX_VERIFICATION", json.dumps(results), flush=True)
    print(
        "Completed native controls; original XPASS is recorded as a diagnostic outcome."
    )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
