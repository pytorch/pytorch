"""Ramjet: build PyTorch on Modal CPUs, then run TCI tests on CPU and L4."""

from __future__ import annotations

import argparse
import getpass
import gzip
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.ramjet_build import build_wheel
from tools.ramjet_source import BOOTSTRAP, create_source_delta, snapshot_source


BASE_IMAGE = (
    "ghcr.io/pytorch/ci-image@sha256:"
    "6ea3168a198bed21b697ece2eff4be926329bd93d2f207fa792b4c313e736273"
)
T4_PLATFORM = "tci.platform:linux_x86_64_cuda_t4"
CPU_PLATFORM = "tci.platform:linux_x86_64"
L4_PLATFORM = "tci.platform:linux_x86_64_cuda_l4"
PLATFORMS = {
    "cpu": (CPU_PLATFORM, None),
    "l4": (L4_PLATFORM, "L4"),
    "t4": (T4_PLATFORM, "T4"),
}
CUDA_ARCHS = "7.5;8.9"
REMOTE_ROOT = "/opt/ramjet"


def prepare_runtime(root: str = REMOTE_ROOT) -> None:
    destination = Path(root)
    ready = destination / ".ramjet-ready"
    if ready.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open("/tmp/ramjet-tests.tar.gz") as archive:
        archive.extractall(destination, filter="data")
    for source in Path("/opt/pytorch").iterdir():
        target = destination / source.name
        if source.name not in {
            "torch",
            "torchgen",
            "functorch",
            "build",
            "__pycache__",
        }:
            if not target.exists() and not target.is_symlink():
                target.symlink_to(source)
    ready.touch()


def cached_build(volume, path: str) -> dict | None:
    try:
        return json.loads(b"".join(volume.read_file(path)))
    except FileNotFoundError:
        return None


def prepare_image(
    root: Path, snapshots: dict, base_image: str, volume, plan: dict, clean: bool
):
    import modal  # pyrefly: ignore[missing-import]  # Optional runner dependency.

    environment = {
        "PATH": "/opt/python-3.11-venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": REMOTE_ROOT,
        "CUDA_HOME": "/usr/local/cuda",
        "CUDA_PATH": "/usr/local/cuda",
        "CC": "/usr/bin/gcc-11",
        "CXX": "/usr/bin/g++-11",
        "USE_CUDA": "1",
        "USE_CCACHE": "1",
        "BUILD_TEST": "0",
        "MAX_JOBS": "32",
        "TORCH_CUDA_ARCH_LIST": CUDA_ARCHS,
        "TORCH_NATIVE_AOT": "0",
        "CCACHE_DIR": "/cache/ccache",
        "CCACHE_TEMPDIR": "/tmp/ccache",
        "CCACHE_BASEDIR": "/opt/pytorch",
        "CCACHE_MAXSIZE": "50G",
        "CCACHE_COMPILERCHECK": "content",
    }
    manifest, digest = snapshots["build"]
    bootstrap, bootstrap_digest = snapshots["bootstrap"]
    recipe = {
        "base_image": base_image,
        "environment": environment,
        "builder": hashlib.sha256(
            bootstrap.joinpath("tools/ramjet_build.py").read_bytes()
        ).hexdigest(),
    }
    recipe_key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    prefix = f"/ramjet/{recipe_key}"
    record = None if clean else cached_build(volume, f"{prefix}/{digest}.json")
    plan.update(build_recipe_sha256=recipe_key, build_cache_hit=record is not None)
    if record is not None:
        if record.get("build_sha256") != digest:
            raise RuntimeError(
                "Cached build metadata is inconsistent; use --clean-build"
            )
        plan.update(build_image_id=record["image_id"], source_delta_bytes=0)
        print(f"Reusing PyTorch image {record['image_id']}", flush=True)
        return modal.Image.from_id(record["image_id"])

    parent = None if clean else cached_build(volume, f"{prefix}/latest.json")
    previous = {}
    if parent is not None:
        content = b"".join(
            volume.read_file(f"/ramjet/manifests/{parent['build_sha256']}.json.gz")
        )
        content = gzip.decompress(content)
        if hashlib.sha256(content).hexdigest() != parent["build_sha256"]:
            raise RuntimeError(
                "Cached source manifest is inconsistent; use --clean-build"
            )
        previous = json.loads(content)
        image = modal.Image.from_id(parent["image_id"])
        plan["incremental_parent_image_id"] = parent["image_id"]
    else:
        image = (
            modal.Image.from_registry(base_image)
            .dockerfile_commands("USER root")
            .apt_install("ccache")
            .env(environment)
            .workdir(REMOTE_ROOT)
        )
    delta = create_source_delta(root, manifest, previous, manifest.parent)
    plan.update(source_delta=str(delta), source_delta_bytes=delta.stat().st_size)
    print(f"Sending {delta.stat().st_size / 2**20:.2f} MiB of build inputs", flush=True)
    context = manifest.parent / "build-inputs" / f"{delta.stem}-{bootstrap_digest}"
    for source, target in (
        (delta, "tmp/ramjet-source-delta.tar.gz"),
        (bootstrap / "tools/ramjet_build.py", "opt/ramjet/tools/ramjet_build.py"),
        (bootstrap / "tools/__init__.py", "opt/ramjet/tools/__init__.py"),
    ):
        destination = context / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            destination.hardlink_to(source)
        except FileExistsError:
            pass
    return image.add_local_dir(context, "/", copy=True).run_function(
        build_wheel,
        cpu=32,
        memory=131072,
        gpu=None,
        timeout=7200,
        volumes={"/cache/ccache": volume},
        include_source=False,
        force_build=clean,
        kwargs={
            "cache_volume": plan["cache_volume"],
            "build_sha256": digest,
            "delta_path": "/tmp/ramjet-source-delta.tar.gz",
        },
    )


def save_build(image, volume, manifest: Path, plan: dict) -> None:
    if plan["build_cache_hit"]:
        return
    digest = plan["build_sha256"]
    recipe_key = plan["build_recipe_sha256"]
    tag = hashlib.sha256(f"{recipe_key}:{digest}".encode()).hexdigest()
    image.publish(f"ramjet-build:{tag}")
    record = {"image_id": image.object_id, "build_sha256": digest}
    data = json.dumps(record).encode()
    with volume.batch_upload(force=True) as upload:
        upload.put_file(
            io.BytesIO(gzip.compress(manifest.read_bytes(), mtime=0)),
            f"/ramjet/manifests/{digest}.json.gz",
        )
        upload.put_file(io.BytesIO(data), f"/ramjet/{recipe_key}/{digest}.json")
        upload.put_file(io.BytesIO(data), f"/ramjet/{recipe_key}/latest.json")
    plan["build_image_id"] = image.object_id


def create_run_directory(root: Path) -> Path:
    base = Path(tempfile.gettempdir()) / f"ramjet-{getpass.getuser()}"
    base.mkdir(mode=0o700, exist_ok=True)
    if base.is_symlink() or (
        hasattr(os, "getuid") and base.stat().st_uid != os.getuid()
    ):
        raise RuntimeError(f"Ramjet temporary directory must be owned by you: {base}")
    base.chmod(0o700)
    root = root.resolve()
    digest = hashlib.sha256(os.fsencode(root)).hexdigest()[:12]
    project = base / f"{root.name}-{digest}"
    project.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="run-", dir=project))


def normalize_targets(
    targets: list[str], root: Path, cwd: Path | None = None
) -> list[str]:
    cwd = Path.cwd() if cwd is None else cwd
    normalized = []
    for target in targets or ["."]:
        file, separator, selector = target.partition("::")
        path = (cwd / file).resolve()
        if path == root:
            # Match pytest.ini's testpaths without collecting tools/test remotely.
            path = root / "test"
        if (
            not path.is_relative_to(root / "test")
            or not (path.is_dir() or (path.is_file() and path.suffix == ".py"))
            or (separator and (not path.is_file() or not selector))
        ):
            raise ValueError(
                f"Invalid target {target!r}: expected a directory or Python file "
                "under test/, with optional :: selectors on files"
            )
        normalized.append(f"{path.relative_to(root).as_posix()}{separator}{selector}")
    return list(dict.fromkeys(normalized))


def collection_targets(targets: list[str], root: str = REMOTE_ROOT) -> list[list[str]]:
    import sys

    if root == REMOTE_ROOT:
        prepare_runtime(root)
    by_file: dict[str, set[str]] = {}
    directories = []
    for target in targets:
        file = target.split("::", 1)[0]
        if Path(root, file).is_dir():
            directories.append(file)
        else:
            by_file.setdefault(file, set()).add(target)
    if directories:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", *directories, "--list-test-files"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(
                f"File discovery failed:\n{result.stdout}\n{result.stderr}"
            )
        for file in json.loads(result.stdout):
            by_file.setdefault(file, set()).add(file)
    if not by_file:
        raise ValueError("No test files found in the selected targets")
    return [sorted(by_file[file]) for file in sorted(by_file)]


def collect_markers(
    targets: list[str], root: str = REMOTE_ROOT
) -> dict[str, list[str]]:
    import sys

    if root == REMOTE_ROOT:
        prepare_runtime(root)

    def collect(command: list[str]) -> dict[str, list[str]]:
        result = subprocess.run(
            [*command, "--list-markers"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(
                f"Collection failed for {targets}:\n{result.stdout}\n{result.stderr}"
            )
        return json.loads(result.stdout)

    markers = collect([sys.executable, "-m", "pytest", *targets])
    by_file: dict[str, list[str]] = {}
    for node, names in markers.items():
        if any(name.startswith("tci.") for name in names):
            by_file.setdefault(node.split("::", 1)[0], []).append(node)
    # Recollect candidates with each script's __main__ setup before scheduling.
    for file, nodes in by_file.items():
        current = collect(
            [sys.executable, file, "--use-pytest", "--pytest-single-test", *nodes]
        )
        if set(current) != set(nodes):
            raise RuntimeError(f"Test candidates changed during collection of {file}")
        markers.update(current)
    return markers


def select_tests(
    markers: dict[str, list[str]], platforms: tuple[str, ...] = ("cpu", "l4")
) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {platform: [] for platform in platforms}
    allowed = {"tci.size:small", "tci.schedule:pull"}
    for node, names in sorted(markers.items()):
        constraints = {name for name in names if name.startswith("tci.")}
        matches = [
            platform for platform in selected if PLATFORMS[platform][0] in constraints
        ]
        if not matches:
            continue
        requirements = {
            name
            for name in constraints
            if not name.startswith(
                ("tci.platform:", "tci.device:", "tci.exclude_device:")
            )
        }
        if requirements - allowed:
            raise ValueError(f"Unsupported TCI requirements for {node}: {constraints}")
        for platform in matches:
            selected[platform].append(node)
    selected = {platform: nodes for platform, nodes in selected.items() if nodes}
    if not selected:
        raise ValueError(f"No tests declare the requested platforms: {platforms}")
    return selected


def shard_tests(nodes: list[str], jobs: int | None = None) -> list[list[str]]:
    if jobs is not None and jobs < 1:
        raise ValueError("jobs must be positive")
    nodes = sorted(set(nodes))
    if jobs is None:
        return [[node] for node in nodes]
    size = max(1, (len(nodes) + jobs - 1) // jobs)
    by_file: dict[str, list[str]] = {}
    for node in nodes:
        by_file.setdefault(node.split("::", 1)[0], []).append(node)
    shards = [
        selected[start : start + size]
        for selected in by_file.values()
        for start in range(0, len(selected), size)
    ]
    # Fill unused workers when rounding leaves too few shards.
    while len(shards) < min(jobs, len(nodes)):
        index = max(range(len(shards)), key=lambda i: len(shards[i]))
        middle = len(shards[index]) // 2
        shards[index : index + 1] = [shards[index][:middle], shards[index][middle:]]
    return shards


def allocate_workers(
    shards: dict[str, list[list[str]]], jobs: int | None, gpu_jobs: int | None = None
) -> list[dict[str, int | None]]:
    if jobs is not None and jobs < 1:
        raise ValueError("jobs must be positive")
    if gpu_jobs is not None and gpu_jobs < 0:
        raise ValueError("gpu_jobs must be nonnegative")
    remaining = {
        platform: len(batches) for platform, batches in shards.items() if batches
    }
    if gpu_jobs == 0 and any(PLATFORMS[platform][1] for platform in remaining):
        raise ValueError("GPU tests require a positive --gpu-jobs cap")
    waves = []
    while remaining:
        limits: dict[str, int | None] = {
            platform: (
                None
                if jobs is None and (gpu_jobs is None or not PLATFORMS[platform][1])
                else 0
            )
            for platform in remaining
        }
        free = (
            jobs
            if jobs is not None
            else sum(
                demand
                for platform, demand in remaining.items()
                if limits[platform] == 0
            )
        )
        free_gpus = gpu_jobs if gpu_jobs is not None else free
        while free:
            candidates = [
                (platform, limit)
                for platform, limit in limits.items()
                if limit is not None
                and limit < remaining[platform]
                and (not PLATFORMS[platform][1] or free_gpus)
            ]
            if not candidates:
                break
            for platform, limit in candidates:
                if not free or (PLATFORMS[platform][1] and not free_gpus):
                    continue
                limits[platform] = limit + 1
                free -= 1
                if PLATFORMS[platform][1]:
                    free_gpus -= 1
        wave = {platform: limit for platform, limit in limits.items() if limit != 0}
        waves.append(wave)
        remaining = {
            name: demand for name, demand in remaining.items() if name not in wave
        }
    return waves


def drain_workers(functions: dict, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    pending = dict(functions)
    while pending:
        states = {}
        for name, function in list(pending.items()):
            stats = function.get_current_stats()
            if not (
                stats.num_total_runners or stats.num_running_inputs or stats.backlog
            ):
                del pending[name]
            else:
                states[name] = (
                    f"{stats.num_total_runners} containers, "
                    f"{stats.num_running_inputs} running inputs, {stats.backlog} queued"
                )
        if not pending:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            details = "; ".join(f"{name}: {state}" for name, state in states.items())
            raise RuntimeError(
                f"Timed out waiting for Ramjet workers to stop ({details}). "
                "Inspect this app's containers in Modal before retrying."
            )
        time.sleep(min(2.0, remaining))


def run_shards(
    test,
    shards: dict[str, list[list[str]]],
    jobs: int | None,
    gpu_jobs: int | None = None,
) -> list:
    waves = allocate_workers(shards, jobs, gpu_jobs)
    results_by_platform = {}
    for index, wave in enumerate(waves):
        workers = {
            platform: test.with_options(
                gpu=PLATFORMS[platform][1],
                max_containers=limit,
                buffer_containers=0,
                scaledown_window=2,
            )
            for platform, limit in wave.items()
        }

        def run_platform(platform: str) -> list:
            batches = shards[platform]
            results = []
            try:
                results.extend(
                    workers[platform].map(
                        batches,
                        kwargs={"platform": PLATFORMS[platform][0]},
                        order_outputs=True,
                        return_exceptions=True,
                    )
                )
            except Exception as error:
                results.extend([error] * (len(batches) - len(results)))
            return results

        with ThreadPoolExecutor(max_workers=len(workers)) as executor:
            results_by_platform.update(
                zip(workers, executor.map(run_platform, workers))
            )
        if index + 1 < len(waves):
            try:
                drain_workers(workers)
            except Exception as error:
                for platform, batches in shards.items():
                    if platform not in results_by_platform:
                        results_by_platform[platform] = [error] * len(batches)
                break
    records = []
    for platform in shards:
        for result in results_by_platform.get(platform, []):
            if isinstance(result, Exception):
                result = {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": f"{type(result).__name__}: {result}",
                    "junit": "",
                }
            records.append({**result, "platform": PLATFORMS[platform][0]})
    return records


def merge_results(shards: list[list[str]], results: list[dict | Exception]) -> dict:
    import xml.etree.ElementTree as ET

    if not shards or any(not nodes for nodes in shards) or len(shards) != len(results):
        raise ValueError("Expected one result for every nonempty set of shards")
    suites = ET.Element("testsuites")
    records: list[dict] = []
    metadata = {}
    for nodes, result in zip(shards, results):
        if isinstance(result, Exception):
            record: dict = {
                "returncode": 1,
                "stdout": "",
                "stderr": f"{type(result).__name__}: {result}",
                "junit": "",
            }
        else:
            record = dict(result)
        record["nodes"] = nodes
        try:
            report = ET.fromstring(record.get("junit", ""))
            if "platform" in record:
                for suite in report.iter("testsuite"):
                    suite.set(
                        "name", f"{suite.get('name', 'pytest')} [{record['platform']}]"
                    )
            if report.tag == "testsuites":
                suites.extend(report)
            elif report.tag == "testsuite":
                suites.append(report)
            else:
                raise ValueError(f"Unexpected JUnit root: {report.tag}")
            cases = report.findall(".//testcase")
            if len(cases) != len(nodes) or any(
                case.find(status) is not None
                for case in cases
                for status in ("skipped", "error", "failure")
            ):
                raise ValueError("Expected every selected test to pass without skips")
        except (ET.ParseError, ValueError) as error:
            record["returncode"] = record.get("returncode", 0) or 1
            record["stderr"] = f"{record.get('stderr', '')}\nShard {nodes}: {error}"
        if not metadata:
            metadata = {
                key: record[key]
                for key in ("torch_path", "torch_version", "source", "build")
                if key in record
            }
        records.append(record)
    return {
        **metadata,
        "returncode": int(any(record.get("returncode", 0) for record in records)),
        "stdout": "\n".join(record.get("stdout", "") for record in records),
        "stderr": "\n".join(record.get("stderr", "") for record in records),
        "junit": ET.tostring(suites, encoding="unicode"),
        "shards": records,
    }


def execute_tests(
    nodes: list[str], root: str = REMOTE_ROOT, platform: str | None = None
) -> dict:
    import sys
    import xml.etree.ElementTree as ET

    by_file: dict[str, list[str]] = {}
    for node in nodes:
        by_file.setdefault(node.split("::", 1)[0], []).append(node)
    suites = ET.Element("testsuites")
    stdout = []
    stderr = []
    returncode = 0
    env = os.environ.copy()
    if platform is not None:
        env["PYTORCH_TEST_TCI_PLATFORM"] = platform.removeprefix("tci.platform:")
    with tempfile.TemporaryDirectory(prefix="ramjet-tests-") as directory:
        for index, (file, selected) in enumerate(by_file.items()):
            report = Path(directory) / f"{index}.xml"
            command = [sys.executable, file, "--use-pytest", "--pytest-single-test"]
            result = subprocess.run(
                [*command, *selected, "-v", f"--junitxml={report}"],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
            )
            stdout.append(result.stdout)
            stderr.append(result.stderr)
            returncode = returncode or result.returncode
            if report.exists():
                suites.extend(ET.parse(report).getroot())
            else:
                stderr.append(f"Missing test report for {file}")
                returncode = returncode or 1
    return {
        "returncode": returncode,
        "stdout": "\n".join(stdout),
        "stderr": "\n".join(stderr),
        "junit": ET.tostring(suites, encoding="unicode"),
    }


def run_tests(nodes: list[str], platform: str) -> dict:
    prepare_runtime()
    import time

    started_at = time.time()
    import modal  # pyrefly: ignore[missing-import]  # Optional runner dependency.

    import torch
    from torch.testing._internal import tci

    installed = Path(torch.__file__).resolve()
    if "site-packages" not in installed.parts:
        raise RuntimeError(f"Expected the installed wheel, imported {installed}")
    source = json.loads(Path(REMOTE_ROOT, "ramjet-source.json").read_text())
    build = json.loads(Path("/opt/ramjet-build.json").read_text())
    if source["build_sha256"] != build["build_sha256"]:
        raise RuntimeError("Test source manifest does not match the wheel build")
    if not torch.__version__.endswith(f"+ramjet.{build['build_sha256'][:12]}"):
        raise RuntimeError(f"Wheel version does not match build: {torch.__version__}")
    actual = tci.platform.current()
    if str(actual) != platform:
        raise RuntimeError(f"Expected {platform}, got {actual}")
    tests_started_at = time.time()
    result = execute_tests(nodes, platform=platform)
    return {
        **result,
        "torch_path": str(installed),
        "torch_version": str(torch.__version__),
        "platform": platform,
        "gpu": torch.cuda.get_device_name() if actual.gpu is not None else None,
        "source": source,
        "build": build,
        "input_id": modal.current_input_id(),
        "started_at": started_at,
        "tests_started_at": tests_started_at,
        "finished_at": time.time(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "targets",
        nargs="*",
        help="test files, directories, or pytest node IDs (default: current directory)",
    )
    parser.add_argument("--base-image", default=BASE_IMAGE)
    parser.add_argument("--cache-volume", default="ramjet-ccache")
    parser.add_argument(
        "--platform",
        choices=tuple(PLATFORMS),
        action="append",
        help="execution platform; repeat to select several (default: cpu and l4)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=os.getenv("RAMJET_JOBS"),
        help="total worker cap (default: RAMJET_JOBS or Modal limits)",
    )
    parser.add_argument(
        "--gpu-jobs",
        type=int,
        default=os.getenv("RAMJET_GPU_JOBS"),
        help="shared GPU worker cap (default: RAMJET_GPU_JOBS or Modal limits)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--clean-build",
        action="store_true",
        help="start a fresh native build, retaining ccache",
    )
    args = parser.parse_args()
    platforms = tuple(dict.fromkeys(args.platform or ("cpu", "l4")))
    if "@sha256:" not in args.base_image:
        parser.error("--base-image must use an immutable @sha256: digest")
    if args.jobs is not None and args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.gpu_jobs is not None and args.gpu_jobs < 0:
        parser.error("--gpu-jobs must be nonnegative")
    root = Path(__file__).resolve().parents[1]
    try:
        targets = normalize_targets(args.targets, root)
    except ValueError as error:
        parser.error(str(error))
    invocation_started = time.monotonic()
    output = create_run_directory(root)
    print(f"Ramjet output: {output}", flush=True)
    snapshots = snapshot_source(root, output.parent / "snapshots")
    snapshot, digest = snapshots["build"]
    test_snapshot, test_digest = snapshots["tests"]
    plan = {
        "output_dir": str(output),
        "build_sha256": digest,
        "test_sha256": test_digest,
        "build_manifest": str(snapshot),
        "source_preparation_seconds": time.monotonic() - invocation_started,
        "clean_build": args.clean_build,
        "test_snapshot": str(test_snapshot),
        "base_image": args.base_image,
        "builder": {"cpu": 32, "memory_mib": 131072, "gpu": None},
        "cuda_arch": CUDA_ARCHS,
        "cache_volume": args.cache_volume,
        "collect": targets,
        "platforms": {
            name: {"tci": PLATFORMS[name][0], "gpu": PLATFORMS[name][1]}
            for name in platforms
        },
        "jobs": args.jobs,
        "gpu_jobs": args.gpu_jobs,
    }
    print(json.dumps(plan, indent=2))
    output.joinpath("plan.json").write_text(json.dumps(plan, indent=2))
    if args.dry_run:
        return

    import modal  # pyrefly: ignore[missing-import]  # Optional runner dependency.

    inputs_started = time.monotonic()
    cache = modal.Volume.from_name(args.cache_volume, create_if_missing=True)
    builder_image = prepare_image(
        root, snapshots, args.base_image, cache, plan, args.clean_build
    )
    image = builder_image.add_local_file(test_snapshot, "/tmp/ramjet-tests.tar.gz")
    bootstrap_root, _ = snapshots["bootstrap"]
    for bootstrap in sorted(BOOTSTRAP):
        image = image.add_local_file(
            bootstrap_root / bootstrap, f"{REMOTE_ROOT}/{bootstrap}"
        )
    plan["build_input_preparation_seconds"] = time.monotonic() - inputs_started
    output.joinpath("plan.json").write_text(json.dumps(plan, indent=2))
    app = modal.App("ramjet")
    options = dict(
        image=image,
        cpu=2,
        memory=8192,
        timeout=600,
        retries=0,
        include_source=False,
        min_containers=0,
        buffer_containers=0,
        scaledown_window=2,
    )
    discover = app.function(**options, max_containers=1)(collection_targets)
    collect = app.function(**options, max_containers=args.jobs)(collect_markers)
    test = app.function(**options)(run_tests)
    result: dict = {"returncode": 1, "test_sha256": test_digest}
    app_started = time.monotonic()
    timings = {
        key: plan[key]
        for key in ("source_preparation_seconds", "build_input_preparation_seconds")
    }
    result["timings"] = timings
    try:
        with modal.enable_output(), app.run():
            save_build(builder_image, cache, snapshot, plan)
            timings["image_preparation_seconds"] = time.monotonic() - app_started
            collection_started = time.monotonic()
            plan["app_id"] = app.app_id
            output.joinpath("plan.json").write_text(json.dumps(plan, indent=2))
            collection_shards = discover.remote(targets)
            if args.jobs is not None:
                drain_workers({"discovery": discover})
            plan["collection_shards"] = collection_shards
            output.joinpath("plan.json").write_text(json.dumps(plan, indent=2))
            print(f"Collecting {len(collection_shards)} test files on CPU workers")
            markers = {}
            failures = []
            for collected in collect.map(
                collection_shards, order_outputs=False, return_exceptions=True
            ):
                if isinstance(collected, Exception):
                    failures.append(f"{type(collected).__name__}: {collected}")
                else:
                    markers.update(collected)
            output.joinpath("markers.json").write_text(json.dumps(markers, indent=2))
            if failures:
                errors = "\n\n".join(failures)
                output.joinpath("collection_errors.txt").write_text(errors)
                raise RuntimeError(f"Test collection failed:\n{errors}")
            if args.jobs is not None:
                drain_workers({"collection": collect})
            timings["collection_seconds"] = time.monotonic() - collection_started
            selected = select_tests(markers, platforms)
            by_platform = {
                platform: shard_tests(nodes, args.jobs)
                for platform, nodes in selected.items()
            }
            shards = [shard for batches in by_platform.values() for shard in batches]
            plan["shards"] = by_platform
            plan["worker_waves"] = allocate_workers(
                by_platform, args.jobs, args.gpu_jobs
            )
            output.joinpath("plan.json").write_text(json.dumps(plan, indent=2))
            limit = str(args.jobs) if args.jobs is not None else "Modal default"
            gpu_limit = (
                str(args.gpu_jobs) if args.gpu_jobs is not None else "Modal default"
            )
            count = sum(len(nodes) for nodes in selected.values())
            print(
                f"Running {count} test placements in {len(shards)} shards; "
                f"total worker cap: {limit}; shared GPU worker cap: {gpu_limit}"
            )
            tests_started = time.monotonic()
            results = run_shards(test, by_platform, args.jobs, args.gpu_jobs)
            timings["test_seconds"] = time.monotonic() - tests_started
            result.update(merge_results(shards, results))
        output.joinpath("junit.xml").write_text(result["junit"])
        print(result["stdout"])
        print(result["stderr"])
        if result["returncode"] or not result["junit"]:
            raise RuntimeError(f"Test run failed; see {output / 'result.json'}")
        print(f"Passed {count} test placements on {', '.join(selected)}")
    except BaseException as error:
        result["returncode"] = 1
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        timings["total_seconds"] = time.monotonic() - invocation_started
        result["build_cache_hit"] = plan["build_cache_hit"]
        result["build_image_id"] = plan.get("build_image_id")
        result["source_delta_bytes"] = plan["source_delta_bytes"]
        dashboard_url = f"https://modal.com/id/{app.app_id}" if app.app_id else None
        result.update(app_id=app.app_id, dashboard_url=dashboard_url)
        output.joinpath("result.json").write_text(json.dumps(result, indent=2))
        if dashboard_url:
            print(f"Modal run: {dashboard_url}", flush=True)


if __name__ == "__main__":
    main()
