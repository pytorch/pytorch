"""Collector engine.

In-process worker: apply a platform descriptor, import a test file (optionally with
test bodies gutted), and either enumerate concrete tests or observe run/skip status.
Parent helpers spawn the worker in a subprocess (descriptor globals are import-time
single-shot, so one process per (file, platform)).

Run directly as a worker (by path, so the wheel torch isn't shadowed by repo/torch):
    python tools/testing/introspection/collector.py <platform/config> <select|enumerate|status> [relpath]
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import ModuleType


# This file is also executed directly as a worker subprocess (see _run_worker). When
# run by path, the package root is not importable, so append it -- appending (not
# prepending) keeps the wheel-installed torch in site-packages ahead of any torch/
# source tree in the repo, which is essential for a wheel-installed torch in CI.
if not __package__:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from tools.testing.introspection.platforms import get_job, Job, Platform


REPO = Path(__file__).resolve().parents[3]
_COLLECTOR = str(Path(__file__).resolve())
_SENTINEL = "__INTROSPECT_JSON__"


def _root() -> Path:
    """Test-tree root to read test files / run_test from. Defaults to this checkout;
    B1 (diff) points it at a per-ref git worktree via TESTINTRO_ROOT. torch itself
    always comes from the installed (current) build, so a single build serves all
    refs whose native/codegen surface is unchanged."""
    return Path(os.environ.get("TESTINTRO_ROOT") or str(REPO))


_SM_ORLATER = {
    "SM53OrLater": (5, 3),
    "SM60OrLater": (6, 0),
    "SM70OrLater": (7, 0),
    "SM75OrLater": (7, 5),
    "SM80OrLater": (8, 0),
    "SM89OrLater": (8, 9),
    "SM90OrLater": (9, 0),
    "SM100OrLater": (10, 0),
    "SM120OrLater": (12, 0),
}
_SM_EXACT = {"IS_SM89": (8, 9), "IS_SM90": (9, 0), "IS_SM100": (10, 0)}


# --------------------------------------------------------------------------- #
# Descriptor application (the validated monkeypatch surface)
# --------------------------------------------------------------------------- #
def apply_descriptor(platform: Platform) -> None:
    import torch
    import torch.testing._internal.common_device_type as cdt
    import torch.testing._internal.common_utils as cu

    # Pre-import the inductor flag module while the subprocess still sees no
    # accelerator, so its import-time LazyVals don't touch a real driver; we override
    # its flags to the platform's declared values afterward.
    iu = _safe_import("torch.testing._internal.inductor_utils")

    # jit test files assert GRAPH_EXECUTOR is set (normally via parse_cmd_line_args).
    if hasattr(cu, "ProfilingMode") and getattr(cu, "GRAPH_EXECUTOR", None) is None:
        cu.GRAPH_EXECUTOR = cu.ProfilingMode.PROFILING

    # The gut-and-run (status) path drives TestCase.run; TEST_SAVE_XML defaults to ""
    # (not None), which pulls in the optional xmlrunner dep on the early-stop path.
    # We never emit XML, so disable it.
    if hasattr(cu, "TEST_SAVE_XML"):
        cu.TEST_SAVE_XML = None

    if platform.device_type == "cpu":
        torch.cuda.is_available = lambda: False
        cdt.device_type_test_bases = [cdt.CPUTestBase]
        return

    if platform.device_type == "cuda":
        _apply_cuda(platform, torch, cdt, cu)
        _override_inductor_utils(iu, platform, "cuda")
        _patch_has_triton()
        return

    if platform.device_type == "mps":
        if hasattr(torch.backends, "mps"):
            torch.backends.mps.is_available = lambda: True
            torch.backends.mps.is_built = lambda: True
        cdt.TEST_MPS = True
        cu.TEST_MPS = True
        _stub_macos_probes()
        cdt.device_type_test_bases = [cdt.CPUTestBase]
        return

    if platform.device_type == "xpu":
        torch.xpu.is_available = lambda: True
        torch.xpu.device_count = lambda: 1
        torch.xpu.current_device = lambda: 0
        cdt.TEST_XPU = True
        cu.TEST_XPU = True
        try:
            import torch.testing._internal.common_xpu as cx

            for n in dir(cx):
                if n.startswith("PLATFORM_SUPPORTS_"):
                    setattr(cx, n, True)
        except Exception:
            pass
        cdt.device_type_test_bases = [cdt.CPUTestBase]
        _override_inductor_utils(iu, platform, "xpu")
        _patch_has_triton()
        return

    raise ValueError(f"unsupported device_type {platform.device_type!r}")


def _stub_macos_probes() -> None:
    # test_mps.py reads total RAM at import via `sysctl -n hw.memsize` (macOS-only),
    # which errors on a non-Mac simulation host. Fake just that probe; pass the rest
    # through. The value only feeds test sizing heuristics, never a real allocation.
    import torch

    real = subprocess.check_output

    def fake(cmd, *args, **kwargs):
        # Only substitute when the real (macOS-only) probe is unavailable, so this is a
        # strict no-op on an actual Mac.
        if isinstance(cmd, (list, tuple)) and "hw.memsize" in cmd:
            try:
                return real(cmd, *args, **kwargs)
            except Exception:
                return b"17179869184"  # 16 GiB
        return real(cmd, *args, **kwargs)

    subprocess.check_output = fake

    # MPS C APIs are macOS-only and absent from a non-Mac build; fake the ones test
    # files call at import/decorator time (in method bodies they don't run here).
    if not hasattr(torch._C, "_mps_isCaptureEnabled"):
        torch._C._mps_isCaptureEnabled = lambda *a, **k: False
    if not hasattr(torch._C, "_mps_maxBufferLength"):
        torch._C._mps_maxBufferLength = lambda *a, **k: 1 << 34


def _patch_has_triton() -> None:
    # triton_utils.py defines add_kernel et al. only `if has_triton():`, which is
    # False when the GPU is hidden. Declare triton present so those helpers exist
    # (triton itself is installed; only device detection is what fails).
    t = _safe_import("torch.utils._triton")
    if t is not None:
        t.has_triton = lambda: True


def _safe_import(name: str):
    try:
        import importlib

        return importlib.import_module(name)
    except Exception:
        return None


def _override_inductor_utils(iu, platform: Platform, gpu_type: str) -> None:
    if iu is None:
        return
    cap = platform.cuda_capability or (8, 0)
    vals = {
        "HAS_CPU": True,
        "HAS_TRITON": True,
        "HAS_CUDA_AND_TRITON": gpu_type == "cuda",
        "HAS_XPU_AND_TRITON": gpu_type == "xpu",
        "HAS_MPS": gpu_type == "mps",
        "HAS_GPU": True,
        "HAS_GPU_AND_TRITON": True,
        "GPU_TYPE": gpu_type,
        "RUN_GPU": True,
        "RUN_CPU": True,
        "HAS_MULTIGPU": False,
        "IS_A100": gpu_type == "cuda" and cap == (8, 0),
        "IS_H100": gpu_type == "cuda" and cap == (9, 0),
        "IS_BIG_GPU": gpu_type == "cuda",
    }
    for k, v in vals.items():
        if hasattr(iu, k):
            setattr(iu, k, v)


def _apply_cuda(platform: Platform, torch, cdt, cu) -> None:
    import torch.testing._internal.common_cuda as cc

    cap = platform.cuda_capability or (8, 0)

    class _Props:
        major, minor = cap
        total_memory = 80 * 1024**3
        name = "Simulated"
        multi_processor_count = 108
        gcnArchName = "gfx942" if platform.rocm else ""

    torch.cuda.is_available = lambda: True
    torch.cuda.get_device_capability = lambda *a, **k: cap
    torch.cuda.get_device_properties = lambda *a, **k: _Props()
    torch.cuda.device_count = lambda: 1
    torch.cuda.current_device = lambda: 0
    torch.cuda.is_initialized = lambda: True

    cu.TEST_CUDA = True
    cc.TEST_CUDA = True
    cc.CUDA_DEVICE = torch.device("cuda:0")
    cc.TEST_CUDNN = True
    cc.TEST_CUDNN_VERSION = platform.cudnn_version
    for name, sm in _SM_ORLATER.items():
        if hasattr(cc, name):
            setattr(cc, name, cap >= sm)
    for name, sm in _SM_EXACT.items():
        if hasattr(cc, name):
            setattr(cc, name, cap == sm)
    for name in ["IS_SM12X"]:
        if hasattr(cc, name):
            setattr(cc, name, cap[0] == 12)
    for name in ["IS_THOR", "IS_JETSON"]:
        if hasattr(cc, name):
            setattr(cc, name, False)
    for name, val in platform.caps.items():
        if hasattr(cc, name):
            setattr(cc, name, val)

    if platform.rocm:
        cu.TEST_WITH_ROCM = True
        if hasattr(cc, "TEST_WITH_ROCM"):
            cc.TEST_WITH_ROCM = True

    # setUpClass runs at generation time and does real device work; its outputs feed
    # runtime skips, not the generated name set.
    def _fake_setupclass(klass):
        klass.no_magma = False
        klass.no_cudnn = False
        klass.cudnn_version = platform.cudnn_version
        klass.primary_device = "cuda:0"

    cdt.CUDATestBase.setUpClass = classmethod(_fake_setupclass)
    cdt.device_type_test_bases = [cdt.CPUTestBase, cdt.CUDATestBase]

    # The gut-and-run path would otherwise run the cuda memory-leak / stream checks
    # (real cuda APIs) around each test and can trip the early-stop suite logic.
    cdt.CUDATestBase._do_cuda_memory_leak_check = False
    cdt.CUDATestBase._do_cuda_non_default_stream = False
    cdt.CUDATestBase._should_stop_test_suite = lambda self: False

    # inductor's device interface binds get_device_properties/current_device as
    # staticmethods to the ORIGINAL torch.cuda.* at its import, bypassing the patches
    # above (e.g. is_big_gpu() -> DeviceProperties.create). Re-point them.
    di = _safe_import("torch._dynamo.device_interface")
    if di is not None and hasattr(di, "CudaInterface"):
        di.CudaInterface.get_device_properties = staticmethod(lambda *a, **k: _Props())
        di.CudaInterface.current_device = staticmethod(lambda *a, **k: 0)
        di.CudaInterface.device_count = staticmethod(lambda *a, **k: 1)


# --------------------------------------------------------------------------- #
# Import + harvest + gut-and-run
# --------------------------------------------------------------------------- #
_GUT_METHODS = {"setUp", "tearDown"}
_GUT_MODULE = {"setUpModule", "tearDownModule"}


class _Gut(ast.NodeTransformer):
    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if stmt.name.startswith("test") or stmt.name in _GUT_METHODS:
                    stmt.body = [ast.Pass()]
        self.generic_visit(node)
        return node


def _import_target(
    relpath: str, gut: bool, mod_name: str = "introspect_target_mod"
) -> ModuleType:
    path = str(_root() / relpath)
    # Mimic running the file directly: its own directory is on sys.path, so sibling
    # imports (e.g. functorch's `from attn_ft import ...`) resolve. Dedup so warm
    # batching doesn't grow sys.path unboundedly.
    parent = str((_root() / relpath).parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    with open(path) as f:
        src = f.read()
    if gut:
        tree = ast.parse(src, filename=path)
        _Gut().visit(tree)
        for stmt in tree.body:
            if (
                isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
                and stmt.name in _GUT_MODULE
            ):
                stmt.body = [ast.Pass()]
        ast.fix_missing_locations(tree)
        code = compile(tree, path, "exec")
        mod = ModuleType(mod_name)
        mod.__file__ = path
        sys.modules[mod_name] = mod
        exec(code, mod.__dict__)
        return mod
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _testcase_classes(mod: ModuleType) -> dict[str, type]:
    return {
        obj.__name__: obj
        for obj in vars(mod).values()
        if isinstance(obj, type) and issubclass(obj, unittest.TestCase)
    }


def _enumerate(mod: ModuleType) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, cls in _testcase_classes(mod).items():
        methods = unittest.defaultTestLoader.getTestCaseNames(cls)
        if methods:
            out[name] = sorted(methods)
    return out


def _locate(mod: ModuleType) -> dict[str, list]:
    """{"Class::method": [relfile, lineno]} for tests whose definition lives in the
    test tree (TESTINTRO_ROOT). Generated/parametrized variants resolve to their
    template function's def; tests defined outside the tree (e.g. torch-internal
    mixins) are omitted, so the renderer leaves them unlinked."""
    root = str(_root())
    out: dict[str, list] = {}
    for cname, cls in _testcase_classes(mod).items():
        for mname in unittest.defaultTestLoader.getTestCaseNames(cls):
            fn = getattr(cls, mname, None)
            if fn is None:
                continue
            try:
                fn = inspect.unwrap(fn)
                src = inspect.getsourcefile(fn)
                line = inspect.getsourcelines(fn)[1]
            except (TypeError, OSError):
                continue
            if not src:
                continue
            ap = os.path.abspath(src)
            if ap == root or ap.startswith(root + os.sep):
                out[f"{cname}::{mname}"] = [os.path.relpath(ap, root), line]
    return out


class _Recorder(unittest.TestResult):
    def __init__(self) -> None:
        super().__init__()
        self.ran: set[str] = set()
        self.skipped_reasons: dict[str, str] = {}

    @staticmethod
    def _key(test: unittest.TestCase) -> str:
        return f"{type(test).__name__}::{test._testMethodName}"

    def addSuccess(self, test):
        self.ran.add(self._key(test))

    def addError(self, test, err):
        self.ran.add(self._key(test))

    def addFailure(self, test, err):
        self.ran.add(self._key(test))

    def addExpectedFailure(self, test, err):
        self.ran.add(self._key(test))

    def addUnexpectedSuccess(self, test):
        self.ran.add(self._key(test))

    def addSkip(self, test, reason):
        self.skipped_reasons[self._key(test)] = reason


def _status(mod: ModuleType) -> dict:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in _testcase_classes(mod).values():
        tests = loader.loadTestsFromTestCase(cls)
        if tests.countTestCases():
            suite.addTests(tests)
    result = _Recorder()
    suite.run(result)
    return {
        "ran": sorted(result.ran),
        "skipped": sorted(result.skipped_reasons.items()),
    }


# --------------------------------------------------------------------------- #
# File selection (reuse run_test.get_selected_tests)
# --------------------------------------------------------------------------- #
def _select(job: Job) -> dict:
    """Compute the test files this job runs, reusing run_test.get_selected_tests.
    Returns repo-relative .py paths (existing only) plus any that map to no file."""
    import run_test  # on sys.path via REPO/test

    saved_argv = sys.argv
    sys.argv = ["run_test"]
    try:
        options = run_test.parse_args()
    finally:
        sys.argv = saved_argv
    for key, val in job.config.options.items():
        setattr(options, key, val)

    modules = run_test.get_selected_tests(options)
    files, missing = [], []
    for m in modules:
        rel = f"test/{m}.py"
        if (_root() / rel).exists():
            files.append(rel)
        else:
            missing.append(m)  # cpp tests, special handlers, etc.
    return {"files": files, "missing": missing}


def _collect_one(relpath: str, op: str, mod_name: str) -> dict:
    if op in ("enumerate", "enumloc"):
        mod = _import_target(relpath, gut=False, mod_name=mod_name)
        out = {"classes": _enumerate(mod)}
        if op == "enumloc":
            out["locations"] = _locate(mod)
        return out
    if op == "status":
        return _status(_import_target(relpath, gut=True, mod_name=mod_name))
    raise SystemExit(f"unknown op {op!r}")


# --------------------------------------------------------------------------- #
# Worker entrypoint
# --------------------------------------------------------------------------- #
def _emit(payload: dict) -> None:
    print(_SENTINEL + json.dumps(payload))
    sys.stdout.flush()


def _worker_main(argv: list[str]) -> int:
    job_name, op, *rest = argv
    job = get_job(job_name)
    sys.path.insert(0, str(_root() / "test"))
    apply_descriptor(job.platform)

    if op == "select":
        _emit({**_select(job), "job": job_name, "op": op})
        return 0

    if op == "batch":
        # rest = [inner_op, filelist_path]. Warm op_db once, then stream per file.
        inner_op, filelist = rest
        with open(filelist) as f:
            files = json.load(f)
        _safe_import("torch.testing._internal.common_methods_invocations")  # warm op_db
        test_root = str(_root() / "test")
        for i, relpath in enumerate(files):
            name = f"introspect_target_{i}"
            before_paths = list(sys.path)
            before_mods = set(sys.modules)
            try:
                payload = _collect_one(relpath, inner_op, name)
                _emit({"file": relpath, **payload})
            except (Exception, SystemExit) as e:
                _emit({"file": relpath, "error": repr(e).splitlines()[0][:300]})
            finally:
                # Per-file isolation: restore sys.path so a prior file's directory
                # can't shadow a later file's sibling import (e.g. test/export's
                # test_sparse shadowing the top-level test_sparse), and drop test-tree
                # modules so the next file re-imports its siblings fresh. torch.* and
                # stdlib stay warm.
                sys.path[:] = before_paths
                for m in set(sys.modules) - before_mods:
                    mod = sys.modules.get(m)
                    f = getattr(mod, "__file__", None) or ""
                    if m == name or f.startswith(test_root):
                        sys.modules.pop(m, None)
        return 0

    # Single-file ops (the --file fast path / isolated retry).
    _emit(
        {
            **_collect_one(rest[0], op, "introspect_target_mod"),
            "job": job_name,
            "op": op,
        }
    )
    return 0


# --------------------------------------------------------------------------- #
# Parent-side API
# --------------------------------------------------------------------------- #
# Bump when descriptor/collection logic changes in a way that invalidates cached
# results (the cache key already includes job name + file content hash).
CACHE_VERSION = "1"
CACHE_DIR = Path(
    os.environ.get("TESTINTRO_CACHE", str(Path.home() / ".cache" / "testintro"))
)


def _job_env(job: Job) -> dict[str, str]:
    env = dict(os.environ)
    env.update(job.subprocess_env())
    # Each worker imports torch, whose OpenMP/MKL pools default to all cores. With
    # many parallel workers that is N*cores threads -> catastrophic oversubscription.
    # Collection doesn't run test bodies, so pin every worker to a single thread.
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    return env


def _run_worker(
    job: Job, op: str, relpath: str | None = None, timeout: int = 1800
) -> dict:
    """Single-payload ops: select / enumerate / status (one file, isolated)."""
    # Run by path, not `-m`: `-m` puts cwd (REPO) on sys.path[0], so `import torch`
    # would load the repo's torch/ source instead of the installed wheel.
    cmd = [sys.executable, _COLLECTOR, job.name, op]
    if relpath is not None:
        cmd.append(relpath)
    proc = subprocess.run(
        cmd,
        cwd=str(REPO),
        env=_job_env(job),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    for line in proc.stdout.splitlines():
        if line.startswith(_SENTINEL):
            return json.loads(line[len(_SENTINEL) :])
    tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
    raise RuntimeError(
        f"collector worker failed (op={op}, target={relpath}, job={job.name}, rc={proc.returncode}).\n"
        f"Often means the platform descriptor is missing a probe.\nstderr tail:\n{tail}"
    )


def _select_signature() -> str:
    """Cheap, torch-free signature of what selection depends on: the set of test
    files (catches add/remove) plus the selection-logic sources (catches blocklist
    edits). Lets a fully-cached run avoid importing torch entirely."""
    import glob

    h = hashlib.sha256()
    root = _root()
    # Relative paths: absolute paths include the ephemeral worktree dir, which would
    # make the key differ every run and defeat the cache.
    names = sorted(
        os.path.relpath(p, root)
        for p in glob.glob(str(root / "test" / "**" / "*.py"), recursive=True)
    )
    h.update("\n".join(names).encode())
    for rel in ["test/run_test.py", "tools/testing/discover_tests.py"]:
        try:
            h.update((_root() / rel).read_bytes())
        except Exception:
            pass
    return h.hexdigest()[:16]


def select_files(job: Job, use_cache: bool = True) -> dict:
    if use_cache:
        sig = _select_signature()
        key = hashlib.sha256(
            f"select|{CACHE_VERSION}|{job.name}|{sig}".encode()
        ).hexdigest()
        path = CACHE_DIR / f"{key}.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
        res = _run_worker(job, "select")
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(res))
        except Exception:
            pass
        return res
    return _run_worker(job, "select")


# ---- content-hash cache -------------------------------------------------- #
def _cache_path(job: Job, op: str, relpath: str) -> Path:
    sha = hashlib.sha256((_root() / relpath).read_bytes()).hexdigest()
    key = hashlib.sha256(
        f"{CACHE_VERSION}|{job.name}|{op}|{relpath}|{sha}".encode()
    ).hexdigest()
    return CACHE_DIR / f"{key}.json"


def _cache_get(job: Job, op: str, relpath: str) -> dict | None:
    p = _cache_path(job, op, relpath)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _cache_put(job: Job, op: str, relpath: str, payload: dict) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(job, op, relpath).write_text(json.dumps(payload))
    except Exception:
        pass


# ---- warm batch workers -------------------------------------------------- #
def _run_batch_worker(
    job: Job, inner_op: str, files: list[str], timeout: int = 3600, on_done=None
) -> dict:
    """One warm worker (torch + op_db imported once) over `files` in order. Returns
    {relpath: payload} for every file that completed before the process exited -- a
    prefix, since the worker is sequential, so a hard crash leaves a clean boundary."""
    tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)  # noqa: SIM115
    json.dump(files, tf)
    tf.close()
    out: dict[str, dict] = {}
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                _COLLECTOR,  # by path, not -m (avoid REPO/torch shadowing the wheel)
                job.name,
                "batch",
                inner_op,
                tf.name,
            ],
            cwd=str(REPO),
            env=_job_env(job),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for line in proc.stdout:  # type: ignore[union-attr]
            if line.startswith(_SENTINEL):
                d = json.loads(line[len(_SENTINEL) :])
                out[d.pop("file")] = d
                if on_done:
                    on_done()
        proc.wait(timeout=timeout)
    finally:
        os.unlink(tf.name)
    return out


def _warm_chunk(job: Job, inner_op: str, files: list[str], on_done=None) -> dict:
    """Process a chunk in a warm worker; if the worker dies on a file (hard crash,
    not a caught exception), isolate that file in its own process and continue."""
    out: dict[str, dict] = {}
    remaining = list(files)
    while remaining:
        got = _run_batch_worker(job, inner_op, remaining, on_done=on_done)
        done = [f for f in remaining if f in got]
        for f in done:
            out[f] = got[f]
        if len(done) == len(remaining):
            break
        culprit = remaining[len(done)]  # first file with no emitted result
        try:
            out[culprit] = _run_worker(job, inner_op, culprit)
        except Exception as e:
            out[culprit] = {"error": str(e).splitlines()[0]}
        if on_done:
            on_done()  # count the isolated file
        remaining = remaining[len(done) + 1 :]
    return out


def _default_workers() -> int:
    """Concurrent collector subprocesses. Kept modest on purpose: the parent parses
    each worker's per-file JSON (large for op_db/inductor files) under the GIL, so
    too many parallel streams contend and anti-scale. Overridable via
    TESTINTRO_WORKERS for idle, many-core boxes."""
    override = os.environ.get("TESTINTRO_WORKERS")
    if override:
        return max(1, int(override))
    return max(1, min(16, (os.cpu_count() or 4) - 2))


def collect(
    job: Job,
    op: str,
    files: list[str],
    use_cache: bool = True,
    max_workers: int | None = None,
    on_progress=None,
) -> dict:
    """Collect `op` (enumerate|status) for many files: cache hits returned directly,
    misses processed across warm workers. Returns {relpath: payload-or-{'error':...}}.
    on_progress(done, total) is called as files complete (thread-safe)."""
    from concurrent.futures import ThreadPoolExecutor

    total = len(files)
    done = [0]
    lock = threading.Lock()

    def bump() -> None:
        if on_progress is None:
            return
        with lock:
            done[0] += 1
            on_progress(done[0], total)

    results: dict[str, dict] = {}
    todo: list[str] = []
    for f in files:
        cached = _cache_get(job, op, f) if use_cache else None
        if cached is not None:
            results[f] = cached
            bump()
        else:
            todo.append(f)

    if todo:
        n = max_workers or _default_workers()
        chunks = [todo[i::n] for i in range(n)]  # round-robin balances slow files
        chunks = [c for c in chunks if c]
        with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
            for part in ex.map(lambda c: _warm_chunk(job, op, c, on_done=bump), chunks):
                for f, r in part.items():
                    results[f] = r
                    if use_cache and "error" not in r:
                        _cache_put(job, op, f, r)
    return results


def enumerate_tests(
    relpath: str, job: Job, use_cache: bool = True
) -> dict[str, list[str]]:
    """{ClassName: [method, ...]} of concrete tests generated for the job."""
    r = collect(job, "enumerate", [relpath], use_cache)[relpath]
    if "error" in r:
        raise RuntimeError(r["error"])
    return r["classes"]


def status(relpath: str, job: Job, use_cache: bool = True) -> dict:
    """{'ran': [Class::method,...], 'skipped': [[Class::method, reason],...]}"""
    r = collect(job, "status", [relpath], use_cache)[relpath]
    if "error" in r:
        raise RuntimeError(r["error"])
    return r


def list_job(
    job: Job,
    with_status: bool = False,
    files_filter: str | None = None,
    use_cache: bool = True,
    max_workers: int | None = None,
) -> dict:
    """Behavior 3: aggregate concrete tests (and optionally run/skip status) across
    every file the job selects, using warm batched workers + the content-hash cache.
    Per-file failures are captured as {'error': ...} rather than aborting the run."""
    sel = select_files(job, use_cache=use_cache)
    files = sel["files"]
    if files_filter:
        files = [f for f in files if files_filter in f]

    op = "status" if with_status else "enumerate"
    raw = collect(job, op, files, use_cache, max_workers)
    results: dict[str, dict] = {}
    for f in files:
        r = raw.get(f, {"error": "no result"})
        if "error" in r:
            results[f] = {"error": r["error"]}
        elif with_status:
            results[f] = {"status": r}
        else:
            results[f] = {"classes": r["classes"]}
    return {
        "job": job.name,
        "files": files,
        "missing": sel["missing"],
        "results": results,
    }


if __name__ == "__main__":
    raise SystemExit(_worker_main(sys.argv[1:]))
