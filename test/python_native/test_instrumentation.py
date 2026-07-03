# Owner(s): ["module: dsl-native-ops"]

import ast
import json
import logging
import os
import shutil
import subprocess
import tempfile
import unittest
from collections import namedtuple

from torch._logging._internal import TorchLogsFormatter, trace_log
from torch._native.instrumentation import (
    CompileEvent,
    instrument_cutedsl_compile,
    instrument_triton_kernel,
)
from torch.testing._internal.common_utils import run_tests, TestCase


# No shared tlparse harness exists in torch (see test/dynamo/test_structured_trace.py,
# which defines these locally too), so mirror its minimal pattern here.
HAS_TLPARSE = shutil.which("tlparse") is not None
requires_tlparse = unittest.skipUnless(HAS_TLPARSE, "requires tlparse")


# Mirror of torch._vendor.quack.cache.CacheInfo. Defined locally so the test
# doesn't import quack (which pulls in cutlass, absent on CPU-only builds).
_CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])


class _FakeJITFunction:
    """Stand-in for a ``@triton.jit`` kernel. GPU- and Triton-free.

    ``instrument_triton_kernel`` watches ``device_caches[dev][0]`` (the dict
    of compiled variants) and launches via ``kernel[grid](*args)``. We model
    one fake device whose cache grows on each distinct ``variant`` kwarg (a
    fresh Triton compile / miss); a repeated variant leaves it unchanged (a
    cache hit).
    """

    def __init__(self):
        # defaultdict-like: one device "cuda:0", value is a (cache_dict, ...) tuple.
        self._cache: dict = {}
        self.device_caches = {"cuda:0": (self._cache, None)}
        self.raise_on_launch = False

    def __getitem__(self, grid):
        def launcher(*args, variant="v0", **kwargs):
            if self.raise_on_launch:
                raise RuntimeError("kaboom")
            if variant not in self._cache:
                self._cache[variant] = object()
            return "launched"

        return launcher


class _FakeJitCache:
    """Stand-in for a ``@jit_cache``-decorated compile function.

    Mimics the bits ``instrument_cutedsl_compile`` observes: a ``cache_info()``
    whose ``misses`` advances on a cold key, plus a controllable wall time
    via the compiled callable. Keeps the test GPU- and CuTeDSL-free.
    """

    def __init__(self):
        self._cache: dict = {}
        self.hits = 0
        self.misses = 0
        self.raise_on_call = False

    def __call__(self, *args, **kwargs):
        if self.raise_on_call:
            raise RuntimeError("boom")
        key = args + tuple(sorted(kwargs.items()))
        if key in self._cache:
            self.hits += 1
        else:
            self.misses += 1
            self._cache[key] = object()
        return self._cache[key]

    def cache_info(self):
        return _CacheInfo(
            hits=self.hits,
            misses=self.misses,
            maxsize=None,
            currsize=len(self._cache),
        )


class _CapturingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


class _LoggerCaptureTest(TestCase):
    """Captures the native_dsl logger so tests can assert on emitted lines."""

    def setUp(self):
        super().setUp()
        self.log = logging.getLogger("torch._native.instrumentation")
        self._orig_level = self.log.level
        self._orig_propagate = self.log.propagate
        self.log.setLevel(logging.INFO)
        self.log.propagate = False
        self.handler = _CapturingHandler()
        self.log.addHandler(self.handler)

    def tearDown(self):
        self.log.removeHandler(self.handler)
        self.log.setLevel(self._orig_level)
        self.log.propagate = self._orig_propagate
        super().tearDown()

    @property
    def messages(self):
        return [r.getMessage() for r in self.handler.records]


class TestInstrumentation(_LoggerCaptureTest):
    def test_first_call_reports_compiled(self):
        fake = _FakeJitCache()
        compile_fn = instrument_cutedsl_compile("aten::topk")(fake)

        compile_fn(256, 64)

        self.assertEqual(len(self.messages), 1)
        msg = self.messages[0]
        self.assertIn("aten::topk", msg)
        self.assertIn("[cutedsl]", msg)
        self.assertIn("compiled", msg)
        self.assertIn("misses=1", msg)

    def test_second_call_reports_cache_hit(self):
        fake = _FakeJitCache()
        compile_fn = instrument_cutedsl_compile("aten::topk")(fake)

        compile_fn(256, 64)
        compile_fn(256, 64)

        self.assertEqual(len(self.messages), 2)
        self.assertIn("compiled", self.messages[0])
        self.assertIn("cache_hit", self.messages[1])
        self.assertIn("hits=1", self.messages[1])

    def test_distinct_keys_each_compile(self):
        fake = _FakeJitCache()
        compile_fn = instrument_cutedsl_compile("aten::topk")(fake)

        compile_fn(256, 64)
        compile_fn(512, 128)

        self.assertEqual(fake.misses, 2)
        for msg in self.messages:
            self.assertIn("compiled", msg)

    def test_key_fn_used_in_log(self):
        fake = _FakeJitCache()
        compile_fn = instrument_cutedsl_compile(
            "aten::topk", key_fn=lambda N, K: f"radix N={N} K={K}"
        )(fake)

        compile_fn(256, 64)

        self.assertIn("radix N=256 K=64", self.messages[0])

    def test_error_is_reported_and_reraised(self):
        fake = _FakeJitCache()
        fake.raise_on_call = True
        compile_fn = instrument_cutedsl_compile("aten::topk")(fake)

        with self.assertRaises(RuntimeError):
            compile_fn(256, 64)

        self.assertEqual(len(self.messages), 1)
        self.assertIn("error", self.messages[0])

    def test_cache_attrs_forwarded(self):
        fake = _FakeJitCache()
        compile_fn = instrument_cutedsl_compile("aten::topk")(fake)

        # jit_cache exposes cache_info / cache_clear; the wrapper must keep
        # them reachable so it's a drop-in replacement.
        self.assertTrue(hasattr(compile_fn, "cache_info"))
        self.assertEqual(compile_fn.cache_info().misses, 0)

    def test_works_without_cache_info(self):
        # A plain callable (no cache_info) must still be timed and reported,
        # just without compiled/cache_hit ground truth (defaults to cache_hit).
        calls = []

        def plain(N, K):
            calls.append((N, K))
            return "ok"

        compile_fn = instrument_cutedsl_compile("aten::topk")(plain)
        self.assertEqual(compile_fn(256, 64), "ok")
        self.assertEqual(calls, [(256, 64)])
        self.assertEqual(len(self.messages), 1)

    def test_no_work_when_not_listening(self):
        # With no listener (logger above INFO, no trace handlers), the wrapper
        # must run the wrapped fn but skip all instrumentation: no log line,
        # and crucially no key_fn call (user code we shouldn't run for nothing).
        key_calls = []

        def key_fn(N, K):
            key_calls.append((N, K))
            return "k"

        fake = _FakeJitCache()
        compile_fn = instrument_cutedsl_compile("aten::topk", key_fn=key_fn)(fake)

        self.log.setLevel(logging.WARNING)
        saved = list(trace_log.handlers)
        for h in saved:
            trace_log.removeHandler(h)
        try:
            compile_fn(256, 64)
        finally:
            for h in saved:
                trace_log.addHandler(h)

        self.assertEqual(fake.misses, 1)  # wrapped fn still ran
        self.assertEqual(key_calls, [])  # ... but no instrumentation work
        self.assertEqual(self.messages, [])

    def test_compile_event_json_roundtrip(self):
        event = CompileEvent(
            op="aten::topk",
            dsl="cutedsl",
            outcome="compiled",
            compiled=True,
            wall_ms=12.5,
            key="radix N=256 K=64",
            hits=0,
            misses=1,
        )
        loaded = json.loads(json.dumps(event.as_dict(), sort_keys=True))
        self.assertEqual(loaded["op"], "aten::topk")
        self.assertEqual(loaded["compiled"], True)
        self.assertEqual(loaded["misses"], 1)


class TestTritonKernelInstrumentation(_LoggerCaptureTest):
    GRID = (1,)

    def _kernel(self, op="aten::bmm", key_fn=None):
        return instrument_triton_kernel(op, key_fn=key_fn)(_FakeJITFunction())

    def test_first_launch_reports_compiled(self):
        kernel = self._kernel()

        self.assertEqual(kernel[self.GRID](), "launched")

        self.assertEqual(len(self.messages), 1)
        msg = self.messages[0]
        self.assertIn("aten::bmm", msg)
        self.assertIn("[triton]", msg)
        self.assertIn("compiled", msg)

    def test_repeated_launch_reports_cache_hit(self):
        kernel = self._kernel()

        kernel[self.GRID](variant="v0")
        kernel[self.GRID](variant="v0")

        self.assertIn("compiled", self.messages[0])
        self.assertIn("cache_hit", self.messages[1])

    def test_new_variant_recompiles(self):
        kernel = self._kernel()

        kernel[self.GRID](variant="v0")
        kernel[self.GRID](variant="v1")

        for msg in self.messages:
            self.assertIn("compiled", msg)
        # This kernel's running variant count surfaces as misses=2.
        self.assertIn("misses=2", self.messages[1])

    def test_error_is_reported_and_reraised(self):
        fake = _FakeJITFunction()
        fake.raise_on_launch = True
        kernel = instrument_triton_kernel("aten::bmm")(fake)

        with self.assertRaises(RuntimeError):
            kernel[self.GRID]()

        self.assertEqual(len(self.messages), 1)
        self.assertIn("error", self.messages[0])

    def test_key_fn_used_in_log(self):
        kernel = self._kernel(key_fn=lambda variant="v0": f"variant={variant}")

        kernel[self.GRID](variant="abc")

        self.assertIn("variant=abc", self.messages[0])

    def test_two_kernels_in_one_module_dont_collide(self):
        # The core guarantee of per-kernel scoping: each kernel watches only
        # its own cache, so compiling kernel B must NOT make kernel A look
        # like it compiled. The old module-scan summed both and would here
        # falsely report A as "compiled" on its second call.
        kernel_a = self._kernel(op="aten::a", key_fn=lambda variant="v0": "A")
        kernel_b = self._kernel(op="aten::b", key_fn=lambda variant="v0": "B")

        kernel_a[self.GRID](variant="v0")  # A compiles
        kernel_b[self.GRID](variant="v0")  # B compiles (must not touch A)
        kernel_a[self.GRID](variant="v0")  # A hit -- NOT a recompile

        a_msgs = [m for m in self.messages if "aten::a" in m]
        b_msgs = [m for m in self.messages if "aten::b" in m]
        self.assertEqual(len(a_msgs), 2)
        self.assertEqual(len(b_msgs), 1)
        self.assertIn("compiled", a_msgs[0])
        self.assertIn("cache_hit", a_msgs[1])  # would be "compiled" if collided
        self.assertIn("compiled", b_msgs[0])
        # Each reports its own variant count, not the module-wide sum.
        self.assertIn("misses=1", a_msgs[1])

    def test_jit_kernel_exposes_raw_kernel(self):
        # wrap_triton needs the raw JITFunction; the proxy exposes it.
        fake = _FakeJITFunction()
        kernel = instrument_triton_kernel("aten::bmm")(fake)
        self.assertIs(kernel.jit_kernel, fake)

    def test_attribute_passthrough(self):
        # Non-launch attribute access is delegated to the wrapped kernel.
        fake = _FakeJITFunction()
        fake.cache_key = "abc123"
        kernel = instrument_triton_kernel("aten::bmm")(fake)
        self.assertEqual(kernel.cache_key, "abc123")


class TestTlparseOutput(TestCase):
    """The instrumentation's whole point on production jobs is tlparse-
    retrievable artifacts. Assert the structured-trace plumbing actually
    fires and that tlparse parses the emitted artifact in --strict mode.
    """

    def setUp(self):
        super().setUp()
        self.old_level = trace_log.level
        trace_log.setLevel(logging.DEBUG)

        # Raw trace file in the on-disk format tlparse consumes, written via
        # the same TorchLogsFormatter(trace=True) that TORCH_TRACE installs.
        # NB: this handler must be registered BEFORE the capture handler --
        # TorchLogsFormatter(trace=True) populates record.metadata as a side
        # effect of formatting, and the capture handler reads that field.
        # delete=False: the file is reopened by name (for tlparse and the
        # raw-content read), so don't let close() unlink it; tearDown removes it.
        self.raw_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="w", delete=False
        )
        self.raw_handler = logging.StreamHandler(self.raw_file)
        self.raw_handler.setFormatter(TorchLogsFormatter(trace=True))
        trace_log.addHandler(self.raw_handler)

        # Capture the records so we can assert on metadata/payload without
        # re-parsing the raw file.
        self.records: list[logging.LogRecord] = []
        self.capture = _CapturingHandler()
        self.capture.records = self.records
        trace_log.addHandler(self.capture)

    def tearDown(self):
        trace_log.removeHandler(self.capture)
        trace_log.removeHandler(self.raw_handler)
        self.raw_file.close()
        os.unlink(self.raw_file.name)
        trace_log.setLevel(self.old_level)
        super().tearDown()

    def _emit_one(self):
        fake = _FakeJitCache()
        compile_fn = instrument_cutedsl_compile(
            "aten::topk", key_fn=lambda N, K: f"radix N={N} K={K}"
        )(fake)
        compile_fn(256, 64)
        self.raw_file.flush()

    def _artifact_records(self):
        return [
            r
            for r in self.records
            if getattr(r, "metadata", {}).get("artifact", {}).get("name")
            == "native_dsl_compile"
        ]

    def test_emits_artifact_record(self):
        self._emit_one()

        recs = self._artifact_records()
        self.assertEqual(len(recs), 1)
        meta = recs[0].metadata["artifact"]
        self.assertEqual(meta["encoding"], "json")
        # Payload must be valid JSON carrying the CompileEvent fields.
        payload = json.loads(recs[0].payload)
        self.assertEqual(payload["op"], "aten::topk")
        self.assertEqual(payload["dsl"], "cutedsl")
        self.assertTrue(payload["compiled"])
        self.assertEqual(payload["key"], "radix N=256 K=64")

    def test_eager_record_has_no_compile_context(self):
        # In eager dispatch there's no live CompileContext, so the record
        # carries no frame id (and -- via expect_trace_id=False -- no
        # diagnostic stack either).
        self._emit_one()

        meta = self._artifact_records()[0].metadata
        self.assertNotIn("frame_id", meta)
        self.assertNotIn("stack", meta)

    def test_picks_up_live_compile_context(self):
        # When a native op compiles inside a torch.compile (CompileContext is
        # live), the artifact is auto-tagged with the ambient frame ids so it
        # nests under that compile in tlparse -- like a Dynamo artifact.
        from torch._guards import compile_context, CompileContext, CompileId

        cid = CompileId(frame_id=7, frame_compile_id=3)
        with compile_context(CompileContext(cid)):
            self._emit_one()

        meta = self._artifact_records()[0].metadata
        self.assertEqual(meta["frame_id"], 7)
        self.assertEqual(meta["frame_compile_id"], 3)
        self.assertEqual(meta["attempt"], 0)

    @requires_tlparse
    def test_tlparse_parses_artifact(self):
        self._emit_one()

        # Guard against a false pass: --strict over an empty file exits 0, so
        # assert the artifact was actually written before parsing it.
        with open(self.raw_file.name) as f:
            raw = f.read()
        self.assertIn("native_dsl_compile", raw)

        out = tempfile.mkdtemp()
        try:
            # --strict makes tlparse exit non-zero on any unparsable line, so
            # check_call alone is the assertion.
            subprocess.check_call(
                [
                    "tlparse",
                    "-o",
                    out,
                    "--overwrite",
                    "--no-browser",
                    "--strict",
                    self.raw_file.name,
                ]
            )
        finally:
            shutil.rmtree(out, ignore_errors=True)


def _dotted(node):
    """Dotted name of an AST expression, e.g. ``cute.compile`` / ``triton.jit``.

    Unwraps a Call to its callee, so ``@deco(...)`` and ``deco(...)`` both
    resolve to ``"deco"``. Returns "" for anything we don't model.
    """
    if isinstance(node, ast.Call):
        return _dotted(node.func)
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _decorator_names(fn_node):
    """Set of dotted decorator names on a FunctionDef node.

    e.g. ``@triton.jit`` -> "triton.jit", ``@instrument_triton_kernel(...)``
    -> "instrument_triton_kernel".
    """
    return {_dotted(d) for d in fn_node.decorator_list}


# One rule per DSL. ``jit_decorator`` is the DSL's raw cache/jit decorator;
# ``instrument_decorator`` is the explicit instrumentation wrapper; and
# ``combined_decorator`` is the one-decorator form that applies both. A
# compile site satisfies the guard if it carries the combined decorator, OR
# both the jit and instrument decorators explicitly. To onboard a new DSL,
# add a row -- every scan and existence check below picks it up. All three
# decorator names must be real entry points in torch._native.instrumentation
# (instrument names) / the DSL (jit name); test_required_decorators_exist
# enforces the instrumentation ones.
_DSL_INSTRUMENTATION_RULES = (
    # (dsl, jit_decorator, instrument_decorator, combined_decorator)
    ("triton", "triton.jit", "instrument_triton_kernel", "instrumented_triton_cache"),
    (
        "cutedsl",
        "jit_cache",
        "instrument_cutedsl_compile",
        "instrumented_cutedsl_cache",
    ),
)


def _is_instrumented(decos, jit_deco, instrument_deco, combined_deco):
    """True if a function's decorator set satisfies a DSL rule.

    Either the one-shot combined decorator, or the explicit jit + instrument
    stack.
    """
    return combined_deco in decos or (jit_deco in decos and instrument_deco in decos)


def _scan_for_missing_instrumentation(source, label):
    """Return (violations, n_compile_sites) for one Python source string.

    A compile site is any function carrying a DSL rule's ``jit_decorator`` or
    its ``combined_decorator``. A violation is such a site that isn't fully
    instrumented (see :func:`_is_instrumented`). ``n_compile_sites`` lets
    callers assert the scan saw something rather than passing vacuously.
    """
    violations = []
    n_compile_sites = 0
    tree = ast.parse(source, filename=label)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        decos = _decorator_names(node)
        for _, jit_deco, instrument_deco, combined_deco in _DSL_INSTRUMENTATION_RULES:
            if jit_deco in decos or combined_deco in decos:
                n_compile_sites += 1
                if not _is_instrumented(
                    decos, jit_deco, instrument_deco, combined_deco
                ):
                    violations.append(
                        f"{label}:{node.lineno} {node.name} has @{jit_deco} but "
                        f"is not instrumented (use @{combined_deco}, or add "
                        f"@{instrument_deco})"
                    )
    return violations, n_compile_sites


# cute.compile must be cached so it is also instrumented: the enclosing
# function must carry @jit_cache or the combined @instrumented_cutedsl_cache.
# A raw call elsewhere would compile uncached *and* invisibly to the
# instrumentation, so it is banned.
_CACHED_COMPILE_CALL = "cute.compile"
_CACHE_DECORATORS = ("jit_cache", "instrumented_cutedsl_cache")


def _scan_for_raw_cute_compile(source, label):
    """Return (violations, n_calls) for one Python source string.

    A violation is a ``cute.compile`` call whose nearest enclosing function
    carries neither ``@jit_cache`` nor ``@instrumented_cutedsl_cache`` (or
    that sits at module level). ``n_calls`` counts compile calls seen.
    """
    tree = ast.parse(source, filename=label)
    # Map each node to its parent so we can climb to the nearest FunctionDef.
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    def enclosing_function(node):
        cur = parent.get(node)
        while cur is not None:
            if isinstance(cur, ast.FunctionDef):
                return cur
            cur = parent.get(cur)
        return None

    violations = []
    n_calls = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _dotted(node) == _CACHED_COMPILE_CALL):
            continue
        n_calls += 1
        fn = enclosing_function(node)
        cached = fn is not None and any(
            d in _decorator_names(fn) for d in _CACHE_DECORATORS
        )
        if not cached:
            where = fn.name if fn is not None else "<module level>"
            allowed = " or ".join(f"@{d}" for d in _CACHE_DECORATORS)
            violations.append(
                f"{label}:{node.lineno} {_CACHED_COMPILE_CALL}() in {where} is "
                f"not wrapped by {allowed}"
            )
    return violations, n_calls


class TestInstrumentationCoverage(TestCase):
    """CI guard: every DSL compile site must carry its instrumentation.

    Static AST scan of ``torch/_native/ops`` -- adding a kernel for a known
    DSL (per :data:`_DSL_INSTRUMENTATION_RULES`) without the matching
    instrumentation decorator turns this test (and thus CI) red.

    Extending to a new DSL: add a row to ``_DSL_INSTRUMENTATION_RULES``.

    For CuTeDSL this also enforces the full chain: every ``cute.compile()``
    must be wrapped by ``@jit_cache`` (``test_no_raw_cute_compile_calls``),
    and every ``@jit_cache`` must be instrumented -- so no compile can escape
    caching or instrumentation.

    Out of scope by construction: ops that wrap *vendored* compile fns at the
    call site (e.g. norm's rmsnorm, which has no local ``@jit_cache`` or
    ``cute.compile``) aren't decorator/call sites here -- they have their own
    tests.
    """

    def _ops_files(self):
        import torch._native

        ops_dir = os.path.join(os.path.dirname(torch._native.__file__), "ops")
        for root, _, files in os.walk(ops_dir):
            for name in files:
                if name.endswith(".py"):
                    yield os.path.join(root, name)

    def test_required_decorators_exist(self):
        # Ties each rule to the real API: a typo'd or removed instrumentation
        # entry point fails here rather than letting the scan pass vacuously.
        import torch._native.instrumentation as instr

        for dsl, _, instrument_deco, combined_deco in _DSL_INSTRUMENTATION_RULES:
            for name in (instrument_deco, combined_deco):
                self.assertTrue(
                    hasattr(instr, name),
                    f"rule for DSL {dsl!r} names {name!r}, which is not in "
                    f"torch._native.instrumentation",
                )

    def test_every_compile_site_is_instrumented(self):
        missing = []
        checked = 0
        for path in self._ops_files():
            with open(path) as f:
                violations, n = _scan_for_missing_instrumentation(
                    f.read(), os.path.relpath(path)
                )
            missing += violations
            checked += n

        self.assertTrue(checked, "scan found no compile sites -- test is stale")
        self.assertEqual(
            missing,
            [],
            "DSL compile sites missing instrumentation:\n" + "\n".join(missing),
        )

    def test_scan_flags_uninstrumented_kernel_per_dsl(self):
        # Meta-test: prove the guard fires. For each DSL: a bare jit decorator
        # (no instrumentation) must be flagged, while BOTH accepted forms --
        # the explicit jit+instrument stack and the one-shot combined
        # decorator -- must pass.
        templates = {
            # dsl: (bad, explicit_ok, combined_ok)
            "triton": (
                "@triton.jit\ndef k(): ...\n",
                "@instrument_triton_kernel('aten::x')\n@triton.jit\ndef k(): ...\n",
                "@instrumented_triton_cache('aten::x')\ndef k(): ...\n",
            ),
            "cutedsl": (
                "@jit_cache\ndef c(): ...\n",
                "@instrument_cutedsl_compile('aten::x')\n@jit_cache\ndef c(): ...\n",
                "@instrumented_cutedsl_cache('aten::x')\ndef c(): ...\n",
            ),
        }
        for dsl, (bad, explicit_ok, combined_ok) in templates.items():
            bad_v, bad_n = _scan_for_missing_instrumentation(bad, f"<{dsl}-bad>")
            self.assertEqual(bad_n, 1, f"{dsl}: scan didn't see the compile site")
            self.assertEqual(
                len(bad_v), 1, f"{dsl}: uninstrumented kernel was NOT flagged"
            )

            for form, src in (("explicit", explicit_ok), ("combined", combined_ok)):
                v, n = _scan_for_missing_instrumentation(src, f"<{dsl}-{form}>")
                self.assertEqual(n, 1, f"{dsl}/{form}: scan missed the site")
                self.assertEqual(v, [], f"{dsl}/{form}: wrongly flagged")

    def test_no_raw_cute_compile_calls(self):
        # Caching is compulsory for cutedsl compiles: every cute.compile() must
        # sit inside a function decorated with @jit_cache or the combined
        # @instrumented_cutedsl_cache. A raw call would be uncached and
        # invisible to instrumentation.
        bad = []
        seen = 0
        for path in self._ops_files():
            with open(path) as f:
                violations, n = _scan_for_raw_cute_compile(
                    f.read(), os.path.relpath(path)
                )
            bad += violations
            seen += n

        self.assertTrue(seen, "scan found no cute.compile calls -- test is stale")
        self.assertEqual(
            bad,
            [],
            "uncached cute.compile() calls:\n" + "\n".join(bad),
        )

    def test_scan_flags_raw_cute_compile(self):
        # Meta-test: prove the caching requirement fires, and that both the raw
        # @jit_cache and the combined decorator satisfy it.
        raw = "def f():\n    return cute.compile(k)\n"
        raw_v, raw_n = _scan_for_raw_cute_compile(raw, "<raw>")
        self.assertEqual(raw_n, 1, "scan didn't see the cute.compile call")
        self.assertEqual(len(raw_v), 1, "raw cute.compile was NOT flagged")

        module_level = "x = cute.compile(k)\n"
        ml_v, ml_n = _scan_for_raw_cute_compile(module_level, "<module>")
        self.assertEqual(ml_n, 1)
        self.assertEqual(len(ml_v), 1, "module-level cute.compile was NOT flagged")

        for form, deco in (
            ("jit_cache", "@jit_cache"),
            ("combined", "@instrumented_cutedsl_cache('aten::x')"),
        ):
            src = f"{deco}\ndef f():\n    return cute.compile(k)\n"
            v, n = _scan_for_raw_cute_compile(src, f"<{form}>")
            self.assertEqual(n, 1)
            self.assertEqual(v, [], f"{form}-wrapped cute.compile was wrongly flagged")


if __name__ == "__main__":
    run_tests()
