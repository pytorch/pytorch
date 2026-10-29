"""Run the original strict two-A10G TorchTitan SFT comparison with full native pins."""

import json
import hashlib
import os
from pathlib import Path
import shutil
import sys
import zipfile
import urllib.request

import triton_pin_mi350_controls as controls


def reuse_candidate_wheel():
    phase = controls.WORK / "candidate"
    phase.mkdir(parents=True)
    suffix = "test-torchtitan_triton_pin_fusion_controls-1-1-lf-l-x86aavx2-45-167-a10g-4_123707907351.zip"
    url = (
        "https://gha-artifacts.s3.amazonaws.com/pytorch/pytorch/40947731381/1/artifact/debug-"
        + suffix
    )
    archive_path = phase / "native-wheel-artifact.zip"
    urllib.request.urlretrieve(url, archive_path)
    assert (
        hashlib.sha256(archive_path.read_bytes()).hexdigest()
        == "c6256c2e043818464e1fb0676bd39d24e72d17b0a7782dab813bbd48ea2e3d53"
    )
    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".whl")]
        assert len(members) == 1
        wheel = phase / Path(members[0]).name
        with archive.open(members[0]) as source, wheel.open("wb") as target:
            shutil.copyfileobj(source, target)
    assert (
        hashlib.sha256(wheel.read_bytes()).hexdigest()
        == "17583b66d96d0158a00458c22baefd5a563fa25f9878ed0aef0e1138806eb08b"
    )
    site = phase / "site"
    controls.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(site),
            str(wheel),
        ],
        name="reuse-full-native-candidate-wheel",
    )
    provenance = {
        "source_sha": "1b9ee7ae35e6e56e3f1e53650154910bdbeb4768",
        "whole_native_wheel_sha256": "17583b66d96d0158a00458c22baefd5a563fa25f9878ed0aef0e1138806eb08b",
        "native_source_run": 40947731381,
        "native_source_job": 123707907351,
        "artifact_url": url,
        "source_sdk_provenance_url": "https://gha-artifacts.s3.amazonaws.com/pytorch/pytorch/40947731381/1/artifact/test-jsons-"
        + suffix,
    }
    (controls.REPORT / "candidate-reused-native-provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )
    return site


def main():
    titan = controls.ROOT / "torchtitan"
    controls.REPORT = controls.ROOT / "test/test-reports/triton-pin-titan-controls"
    controls.WORK = controls.ROOT / "build/triton-pin-titan-controls"
    controls.REPORT.mkdir(parents=True, exist_ok=True)
    controls.WORK.mkdir(parents=True, exist_ok=True)
    controls.run(
        [
            sys.executable,
            "-c",
            "import torch; assert torch.cuda.device_count()>=2; "
            "assert 'A10G' in torch.cuda.get_device_name(0); assert torch.version.cuda=='13.2'; "
            "print(torch.version.git_version,torch.cuda.get_device_properties(0))",
        ],
        cwd=titan,
        name="titan-preflight",
        timeout=180,
    )
    results = {}
    metrics = {}
    selected_pins = (
        {"candidate": controls.PINS["candidate"]}
        if "--fusion-only" in sys.argv
        else controls.PINS
    )
    if "--boundary" in sys.argv:
        selected_pins = {
            "parent": (
                "https://github.com/triton-lang/triton.git",
                "7e013d6897e2037a73e8ddfe1aa2d8693d4bf349",
            ),
            "child": (
                "https://github.com/triton-lang/triton.git",
                "1d503a35275625d2c236605a42855bfd81267047",
            ),
        }
    for label, (repository, sha) in selected_pins.items():
        if "--reuse-candidate-wheel" in sys.argv:
            assert "--fusion-only" in sys.argv and label == "candidate"
            site = reuse_candidate_wheel()
        else:
            site = controls.build_pin(label, repository, sha)
        phases = [(label, None)] + (
            [(label + "-implicit-fusion-restored", "1")]
            if label in {"candidate", "child"}
            else []
        )
        if "--reuse-candidate-wheel" in sys.argv:
            phases.insert(1, (label + "-explicit-fusion-only", "0"))
        for phase, fma in phases:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(site) + (
                os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
            )
            env["TRITON_CACHE_DIR"] = str(controls.WORK / phase / "triton-cache")
            env["TORCHINDUCTOR_CACHE_DIR"] = str(
                controls.WORK / phase / "inductor-cache"
            )
            controls.configure_fusion(env, fma)
            if "--reuse-candidate-wheel" in sys.argv:
                trace_site = controls.ROOT / ".ci/pytorch/triton_pin_titan_trace_site"
                env["PYTHONPATH"] = str(trace_site) + os.pathsep + env["PYTHONPATH"]
                env["PIN_TRITON_TRACE_DIR"] = str(
                    controls.ROOT / "test/debug/triton-pin-titan-controls" / (phase + "-compile-trace")
                )
            env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
            env["CUDA_VISIBLE_DEVICES"] = "0,1"
            probe = (
                "import triton,importlib.metadata,pathlib,torch; v=importlib.metadata.version('triton'); "
                f"assert '{sha[:8]}' in v,v; assert pathlib.Path(triton.__file__).resolve().is_relative_to({str(site)!r}); "
                + controls.fusion_provenance(fma)
                + "print(v,triton.__file__,torch.version.git_version)"
            )
            controls.run(
                [sys.executable, "-c", probe],
                cwd=titan,
                env=env,
                name=phase + "-provenance",
                timeout=180,
            )
            results[phase] = controls.run(
                [
                    sys.executable,
                    "-m",
                    "tests.integration_tests.run_tests",
                    "--test_suite",
                    "features",
                    "--test_name",
                    "sft",
                    "--ngpu",
                    "2",
                    str(controls.REPORT / phase),
                ],
                cwd=titan,
                env=env,
                name=phase + "-original-sft",
                check=False,
                timeout=1800,
            )
            log = (controls.REPORT / (phase + "-original-sft.log")).read_text(
                errors="replace"
            )
            assert results[phase] in (0, 1) and (
                "LOSS_COMPARE" in log or "All tests passed" in log
            ), phase
            from tensorboard.backend.event_processing.event_accumulator import (
                EventAccumulator,
            )

            event_root = controls.REPORT / phase / "sft/tb_baseline"
            events = list(event_root.rglob("events.out.tfevents.*"))
            assert events, str(event_root)
            values = {}
            for path in events:
                accumulator = EventAccumulator(str(path))
                accumulator.Reload()
                for tag in ["loss_metrics/global_avg_loss", "grad_norm"]:
                    if tag in accumulator.Tags()["scalars"]:
                        for value in accumulator.Scalars(tag):
                            values.setdefault(tag, {})[value.step] = value.value
            assert all(
                len(values.get(tag, {})) == 10
                for tag in ["loss_metrics/global_avg_loss", "grad_norm"]
            ), values
            metrics[phase] = values
            (controls.REPORT / "metrics.json").write_text(json.dumps(metrics, indent=2))
            (controls.REPORT / "results.json").write_text(json.dumps(results, indent=2))
            if env.get("PIN_TRITON_TRACE_DIR"):
                records = [
                    json.loads(line)
                    for path in Path(env["PIN_TRITON_TRACE_DIR"]).glob("*.jsonl")
                    for line in path.read_text().splitlines()
                ]
                imports = [r for r in records if r["event"] == "import"]
                compiled = [r for r in records if r["event"] == "compiled"]
                (controls.REPORT / (phase + "-compile-trace-summary.json")).write_text(
                    json.dumps({"imports": imports, "compiled": compiled}, indent=2)
                )
                assert imports and compiled
                assert all(
                    sha[:8] in r["triton_version"]
                    and Path(r["triton_file"]).resolve().is_relative_to(site)
                    for r in imports
                ), imports
            kernel_archive = controls.REPORT / f"{phase}-generated-kernels.zip"
            with zipfile.ZipFile(
                kernel_archive, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                for cache_name in ["triton-cache", "inductor-cache"]:
                    cache_root = controls.WORK / phase / cache_name
                    if cache_root.exists():
                        for path in cache_root.rglob("*"):
                            if path.is_file() and path.suffix in {
                                ".py",
                                ".json",
                                ".ttir",
                                ".ttgir",
                                ".llir",
                                ".ptx",
                            }:
                                archive.write(
                                    path,
                                    Path(cache_name) / path.relative_to(cache_root),
                                )
            debug_root = controls.ROOT / "test/debug/triton-pin-titan-controls"
            debug_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(kernel_archive, debug_root / kernel_archive.name)
            if os.environ.get("OUTPUT_DIR"):
                artifact_root = (
                    Path(os.environ["OUTPUT_DIR"]) / "triton-pin-titan-controls"
                )
                artifact_root.mkdir(parents=True, exist_ok=True)
                for path in controls.REPORT.iterdir():
                    if path.is_file() and path.suffix in {".log", ".json", ".zip"}:
                        shutil.copy2(path, artifact_root / path.name)
    print("TITAN_NATIVE_CONTROLS", json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
