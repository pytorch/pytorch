# Owner(s): ["module: dynamo"]

import contextlib
import copy
import dataclasses
import functools
import importlib
import inspect
import itertools
import math
import os
import pickle
import queue
import sys
import sysconfig
import textwrap
import threading
import types
from unittest import mock

import torch
import torch._dynamo.package as dynamo_package
import torch._dynamo.precompile_package as dynamo_package_lint
import torch._dynamo.testing
import torch._inductor.config
import torch._inductor.test_case
from torch._C._dynamo.eval_frame import _debug_get_precompile_entries
from torch._dynamo.exc import PackageError, RecompileError
from torch._dynamo.package import (
    _defining_module_name,
    CompilePackage,
    DynamoCache,
    SystemInfo,
)
from torch._dynamo.precompile_context import PrecompileContext
from torch._dynamo.precompile_package import (
    _dynamo_alias_module,
    _fact_order,
    _GuardFact,
    _SingleFileStore,
    precompile_capture,
    serving,
)
from torch._functorch import config as functorch_config
from torch._inductor.runtime.runtime_utils import cache_dir
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


def staged_with_graph_breaks(x):
    x = x * 2
    torch._dynamo.graph_break()
    x = x + 3
    torch._dynamo.graph_break()
    return x.sum()


def _graph_on(device):
    graph = torch.fx.Graph()
    graph.call_function(torch.ones, args=(torch.device(device),))
    return graph


@contextlib.contextmanager
def _counting_cpu_probe(toolchain=True):
    """Count pick_vec_isa calls; without a toolchain every probe raises."""
    import torch._inductor.cpu_vec_isa as cpu_vec_isa
    from torch._inductor.exc import InvalidCxxCompiler

    calls = []
    real_pick = cpu_vec_isa.pick_vec_isa

    def pick():
        calls.append(1)
        if not toolchain:
            raise InvalidCxxCompiler
        return real_pick()

    with mock.patch.object(cpu_vec_isa, "pick_vec_isa", pick):
        yield calls


_MIPS_TARGET = ("mips", "DEFAULT", None, "INVALID")

# recorded target, whether the host probe works, what the comparison raises
_CPU_TARGET_CASES = {
    "no_target_recorded": (None, True, None),
    "skewed_target": (_MIPS_TARGET, True, "built for machine 'mips'"),
    "host_probe_failed": (("x86_64", "AVX512", None, "avx512"), False, "no usable"),
}


# collections.abc, three lines of `from _collections_abc import *`, with the
# private file spoofing __name__ back to the public name so tracebacks read
# right. Every model that inlines through such a module hits both traps.
_SHIM_IMPL_SRC = """\
# Renamed out of the way for import-time reasons, exactly as _collections_abc.
__name__ = "shim_abc"


def helper(x):
    return x * 3.0
"""

_SHIM_SRC = """\
from _shim_impl import *
"""


def _rebind(scope):
    scope["g"] = "bystander"


# Values successive packages bind to one name, what a bystander does to the
# binding, and what each unload (last installed first) must leave bound. The
# binding stack is bookkeeping, not ownership of the namespace.
_BINDING_STACK_CASES = {
    # Rebinding by plain torch.compile, a CleanupHook or user code is theirs
    # to keep: restoring a surviving package's value hands them a compiled
    # value they never asked for, and the last unload would then delete it.
    "rebound": (("first", "second"), _rebind, ("bystander", "bystander")),
    # The "or gone" arm: a name deleted out from under a still-serving earlier
    # package is put back by the later package's unload.
    "deleted": (("first", "second"), lambda scope: scope.pop("g"), ("first", None)),
    # One value stacked twice with another between them: deregistering by
    # value alone finds the earlier frame and leaves an unloaded package's
    # value bound for good.
    "phantom": (("shared", "other", "shared"), lambda scope: None, ("other", "shared", None)),
}  # fmt: skip


class PrecompileBlock(torch.nn.Module):
    """With resume_work, the frame resumed after the break compiles too."""

    def __init__(self, i, resume_work):
        super().__init__()
        self.i = i
        self.resume_work = resume_work

    def forward(self, x):
        x = x * 2 + self.i
        torch._dynamo.graph_break()
        return x + 1.0 if self.resume_work else x


class PrecompileStack(torch.nn.Module):
    """All blocks share one forward code object, so variants pile onto it."""

    def __init__(self, n, resume_work=False):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [PrecompileBlock(i, resume_work) for i in range(n)]
        )

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x.sum()


PRECOMPILE_CONFIG = {"mode": "sum"}


def staged_with_global_dict_conditional(x):
    # The global is read on both sides of the break, so the entry frame and the
    # resume frame each carry a guard on it.
    if PRECOMPILE_CONFIG["mode"] == "sum":
        x = x * 2
    else:
        x = x * 3
    torch._dynamo.graph_break()
    if PRECOMPILE_CONFIG["mode"] == "sum":
        return x.sum()
    return x.mean() * 10.0


def _precompile_scale(t):
    return t * 2


class PrecompileNoDispatchSlot(torch.nn.Module):
    """Ordinary code with no swappable slot: a torch op, stdlib, a local def."""

    def forward(self, x):
        y = torch.relu(_precompile_scale(x)) * math.sqrt(2.0)
        torch._dynamo.graph_break()
        return (y + 1).sum()


PRECOMPILE_INV_CONFIG = {"mode": "sum"}


class PrecompileInvariantModel(torch.nn.Module):
    """Reads a global across a graph break, so the resume frame guards it."""

    def forward(self, x):
        y = x * 2
        torch._dynamo.graph_break()
        if PRECOMPILE_INV_CONFIG["mode"] == "sum":
            return y.sum()
        return y.mean()


_TWO_SHAPES = [(torch.ones(4, 8),), (torch.ones(5, 8),)]


class PrecompileSelfAct(torch.nn.Module):
    """self.act = <callable> -- how configurable activations are usually written."""

    def __init__(self, act):
        super().__init__()
        self.act = act

    def forward(self, x):
        y = self.act(x)
        torch._dynamo.graph_break()
        return (y + 1).sum()


class PrecompileEmptyGraph(torch.nn.Module):
    """Used to construct a legacy package with no guarded code in skip tests."""

    def forward(self, x):
        return x.sin()


class PrecompilePartialForward(torch.nn.Module):
    """self.forward = functools.partial(...) shadows the class method."""

    def __init__(self, scale):
        super().__init__()
        self.forward = functools.partial(self._impl, scale)

    def _impl(self, scale, x):
        return (x * scale).sum()


def _precompile_sin(t):
    return t.sin()


PRECOMPILE_ACTIVATION = _precompile_sin


def staged_with_global_function_ref(x):
    y = PRECOMPILE_ACTIVATION(x) + 1
    torch._dynamo.graph_break()
    return (y * 10).sum()


@instantiate_parametrized_tests
class TestPrecompilePackage(torch._inductor.test_case.TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        DynamoCache.clear()
        PrecompileContext.clear()

    def dir(self):
        path = os.path.join(cache_dir(), f"precompile_{self.id()}")
        os.makedirs(path, exist_ok=True)
        return path

    def path(self, name="artifact.pt"):
        """An artifact FILE inside this test's scratch dir; save() writes files."""
        return os.path.join(self.dir(), name)

    def _write_module(self, dirname, name, src):
        pkg_dir = os.path.join(self.dir(), dirname)
        os.makedirs(pkg_dir, exist_ok=True)
        with open(os.path.join(pkg_dir, f"{name}.py"), "w") as f:
            f.write(src)
        return pkg_dir

    def _forget_modules(self, *names):
        for name in names:
            sys.modules.pop(name, None)
            self.addCleanup(lambda n=name: sys.modules.pop(n, None))

    def _import_module(self, pkg_dir, name):
        sys.path.insert(0, pkg_dir)
        self.addCleanup(lambda: pkg_dir in sys.path and sys.path.remove(pkg_dir))
        self.addCleanup(lambda: sys.modules.pop(name, None))
        sys.modules.pop(name, None)
        importlib.invalidate_caches()
        return importlib.import_module(name)

    def _bare_package(self):
        """A package that installs no codes: these exercise the name stack only."""

        def fn(x):
            return x + 1

        return CompilePackage(fn)

    def _capture(self, fn, *args, no_grad=True):
        grad = torch.no_grad() if no_grad else contextlib.nullcontext()
        session = precompile_capture(fn, backend="eager", dynamic=False)
        with session as compiled, grad:
            compiled(*args)
        return session

    @parametrize("case", sorted(_CPU_TARGET_CASES))
    def test_cpu_codegen_target_is_compared_only_when_recorded(self, case):
        # None means "written before the field existed", not "no vector ISA".
        # Treating it as a mismatch would reject every artifact already on disk
        # over a target they never recorded. An artifact that DOES record one
        # must still be compared against this host, or an AVX-512 capture
        # silently miscomputes on AVX2 -- and when the host probe cannot run,
        # "current=None" alone reads like a precompile bug rather than a
        # missing compiler.
        recorded, toolchain, error = _CPU_TARGET_CASES[case]
        if toolchain and SystemInfo.current().cpu_codegen_target is None:
            self.skipTest("no C++ toolchain to determine a CPU codegen target")
        captured = dataclasses.replace(
            SystemInfo.current(), cpu_codegen_target=recorded
        )

        def expect():
            if error is None:
                return contextlib.nullcontext()
            return self.assertRaisesRegex(RuntimeError, error)

        with _counting_cpu_probe(toolchain):
            current = SystemInfo.current()
        self.assertEqual(current.cpu_codegen_target is None, not toolchain)
        with expect():
            captured.check_compatibility(current, "cpu")

        # check_versions is what a load runs. Computing the current target is
        # pure cost when the artifact recorded none -- and a hard failure with
        # no compiler -- so the probe must not run at all, not merely not crash.
        entry = dynamo_package._DynamoCacheEntry(
            codes=[],
            source_info=dynamo_package.SourceInfo(inlined_sources=set()),
            device_type="cpu",
            device_types=frozenset(("cpu",)),
            requires_native_backend_compatibility=True,
            system_info=captured,
        )
        with _counting_cpu_probe(toolchain) as calls, expect():
            entry.check_versions()
        self.assertEqual(len(calls), 0 if recorded is None else 1)

    @parametrize("backend", ("eager", "inductor"))
    def test_caching_precompile_probes_the_cpu_only_for_native_code(self, backend):
        # torch.compile(backend="eager") under caching_precompile builds its own
        # CompilePackage, and that package used to demand native backend
        # compatibility -- so an eager compile dry-compiled a C++ probe for a
        # vector width no eager artifact can ever bake. The relaxation must not
        # reach the backend that actually bakes one into the code it ships.
        seen = []
        real = dynamo_package.CompilePackage.cache_entry

        def spy(package):
            seen.append(real(package))
            return seen[-1]

        def f(x):
            return x.sin() + 1

        with (
            torch._dynamo.config.patch(caching_precompile=True),
            mock.patch.object(dynamo_package.CompilePackage, "cache_entry", spy),
            _counting_cpu_probe() as calls,
        ):
            torch.compile(f, backend=backend)(torch.randn(4))
        native = backend == "inductor"
        self.assertTrue(seen)
        self.assertEqual(
            {e.requires_native_backend_compatibility for e in seen}, {native}
        )
        if native:
            self.assertTrue(
                all(e.system_info.cpu_codegen_target is not None for e in seen)
            )
        else:
            self.assertEqual(calls, [])

    def test_accelerator_capture_defers_the_cpu_toolchain_probe(self):
        # A capture that has only produced accelerator graphs has no CPU vector
        # width to record, so it must not run the probe. The first cpu graph
        # backfills the field: the artifact still has to say what it baked.
        with _counting_cpu_probe() as calls:
            package = CompilePackage(staged_with_graph_breaks)
            package.update_device_type(_graph_on("cuda"))
            self.assertEqual(calls, [])
            self.assertIsNone(package.cache_entry().system_info.cpu_codegen_target)

            package.update_device_type(_graph_on("cpu"))
            self.assertEqual(len(calls), 1)
            self.assertEqual(
                package.cache_entry().system_info.cpu_codegen_target,
                SystemInfo.current().cpu_codegen_target,
            )

    def test_device_type_records_the_accelerator_not_the_last_graph(self):
        # _DynamoCacheEntry.device_type gates every GPU check in
        # SystemInfo.check_compatibility, and one package holds many graphs, so
        # a cpu epilogue after a graph break must not erase the accelerator.
        package = CompilePackage(staged_with_graph_breaks)
        package.update_device_type(_graph_on("cuda"))
        package.update_device_type(_graph_on("cpu"))
        self.assertEqual(package.cache_entry().device_type, "cuda")
        self.assertEqual(package.cache_entry().device_types, frozenset(("cpu", "cuda")))

        # A load, a cpu recompile and a re-save must not downgrade it either.
        # Loading a "cuda" entry runs check_versions and so needs a GPU; the
        # restore path does not care which accelerator it is, so the round trip
        # uses one SystemInfo does not gate and stays runnable on a cpu host.
        accel = CompilePackage(staged_with_graph_breaks)
        accel.update_device_type(_graph_on("mtia"))
        reloaded = CompilePackage(staged_with_graph_breaks, accel.cache_entry())
        reloaded.update_device_type(_graph_on("cpu"))
        self.assertEqual(reloaded.cache_entry().device_type, "mtia")

        cpu_only = CompilePackage(staged_with_graph_breaks)
        cpu_only.update_device_type(_graph_on("cpu"))
        self.assertEqual(cpu_only.cache_entry().device_type, "cpu")

    def test_scan_sys_modules_retries_after_a_later_import(self):
        # A miss usually means the module is not imported YET. Caching that
        # permanently drops the source checksum for every lazily imported file
        # for the rest of the process.
        name = "_precompile_late_import"
        pkg_dir = self._write_module("late", name, "def f(x):\n    return x + 1\n")
        self.addCleanup(dynamo_package._MODULE_KEY_BY_FILE.clear)
        path = os.path.join(pkg_dir, f"{name}.py")
        self.assertIsNone(dynamo_package._scan_sys_modules_for_file(path))
        late = self._import_module(pkg_dir, name)
        self.assertEqual(dynamo_package._scan_sys_modules_for_file(late.__file__), name)

    def test_inlined_code_is_named_by_the_file_that_holds_it(self):
        # inspect.getmodule maps a file to the name the module knows itself by,
        # and a private implementation file that spoofs __name__ answers with
        # the shim re-exporting it. The key is what load-time revalidation
        # re-imports, and __name__ imports to the shim, so it has to be the key.
        pkg = self._write_module("shim", "_shim_impl", _SHIM_IMPL_SRC)
        self._write_module("shim", "shim_abc", _SHIM_SRC)
        self._forget_modules("_shim_impl", "shim_abc")
        shim = self._import_module(pkg, "shim_abc")
        impl = sys.modules["_shim_impl"]
        self.assertIsNot(inspect.getmodule(shim.helper.__code__), impl)
        name = _defining_module_name(shim.helper.__code__)
        self.assertEqual(name, "_shim_impl")
        self.assertIs(importlib.import_module(name), impl)

    def test_a_failed_load_leaves_no_device_type_behind(self):
        # eval_frame's caching_precompile path retries a failed load on the SAME
        # package object, and update_device_type only widens cpu -> accelerator,
        # so a device_type left behind by the failed load can never be corrected
        # and gets re-saved as this capture's. One outside CHECK_GPUS turns off
        # every GPU check for whoever loads the artifact next.
        drifted = dynamo_package.SourceInfo(
            inlined_sources={
                dynamo_package.InlinedSource(
                    module="torch._dynamo.package",
                    firstlineno=1,
                    lastlineno=3,
                    checksum="not-the-checksum-on-disk",
                    content="",
                )
            }
        )
        stale = dynamo_package._DynamoCacheEntry(
            codes=[], source_info=drifted, device_type="mtia"
        )

        package = CompilePackage(None)
        with self.assertRaisesRegex(RuntimeError, "Source code changes detected"):
            package.initialize(staged_with_graph_breaks, stale)
        self.assertFalse(package.is_initialized())

        package.initialize(staged_with_graph_breaks, None)
        package.update_device_type(_graph_on("cuda"))
        entry = package.cache_entry()
        self.assertEqual(entry.device_type, "cuda")

        # "mtia" is not in CHECK_GPUS, so had it survived it would have waved
        # this artifact onto any host; what survives instead is gated.
        other_host = dataclasses.replace(entry.system_info, gpu_name="Some Other GPU")
        entry.system_info.check_compatibility(other_host, "mtia")
        self.assertIn(entry.device_type, SystemInfo.CHECK_GPUS)
        # The GPU-name check is what a surviving "mtia" would have skipped; it
        # sits behind the availability check, so a cpu host reaches it mocked.
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            self.assertRaisesRegex(RuntimeError, "different GPU"),
        ):
            entry.system_info.check_compatibility(other_host, entry.device_type)

    @parametrize("interference", sorted(_BINDING_STACK_CASES))
    def test_unload_rebinds_a_global_through_the_binding_stack(self, interference):
        values, interfere, expected = _BINDING_STACK_CASES[interference]
        module = types.ModuleType(f"test_precompile_{interference}")
        packages = [self._bare_package() for _ in values]
        for package, value in zip(packages, values):
            package._install_global(module, "g", value)
        interfere(module.__dict__)
        for package, want in zip(reversed(packages), expected):
            package.uninstall()
            self.assertEqual(module.__dict__.get("g"), want)
        self.assertEqual(dynamo_package._GLOBAL_BINDINGS.get(module, {}), {})

    def test_source_graph_module_copies_are_isolated(self):
        # _src and the exec'd forward are shared between copies; everything else
        # nn.Module keeps on an instance is state, hook dicts included, and a
        # copy sharing it lets an update on one copy silently edit the other.
        # __reduce__ used to pickle the SHARED _src, whose body aliases the
        # original's parameter/buffer containers -- so mutating the original
        # after a deepcopy round-tripped the mutated tensors into the copy's
        # pickle even though live calls were isolated. __reduce__ now
        # snapshots the instance's own state.
        from torch._dynamo.precompile_context import (
            _EagerGraphSource,
            _SourceGraphModule,
        )

        src = _EagerGraphSource(
            code="def forward(self, x):\n    return x + self.b\n",
            import_block="",
            body={"_buffers": {"b": torch.ones(3)}},
        )
        original = _SourceGraphModule(src)
        dup = copy.deepcopy(original)
        dup.register_forward_hook(lambda *args: None)
        dup._non_persistent_buffers_set.add("b")
        self.assertEqual(len(original._forward_hooks), 0)
        self.assertEqual(original._non_persistent_buffers_set, set())
        self.assertIs(dup._src, original._src)

        x = torch.randn(3)
        original._buffers["b"].mul_(100)
        self.assertEqual(original(x), x + 100)
        self.assertEqual(dup(x), x + 1)  # live isolation
        self.assertEqual(pickle.loads(pickle.dumps(dup))(x), x + 1)
        # And an instance pickles its CURRENT state, not its load-time state.
        self.assertEqual(pickle.loads(pickle.dumps(original))(x), x + 100)

    def test_library_module_requires_the_name_to_resolve_to_the_stdlib(self):
        # The risky-drop waiver keys on the OWNER's module name, and a name is
        # not an identity: graphlib, queue, code and distutils are all stdlib
        # names a third party ships, and purelib NESTS inside stdlib (conda) or
        # platstdlib (venv), so a __file__ prefix check waived every shadow.
        stdlib_root = sysconfig.get_paths()["stdlib"]
        shadowed = types.ModuleType("graphlib")
        shadowed.__file__ = os.path.join(
            stdlib_root, "site-packages", "graphlib", "__init__.py"
        )
        unlocated = types.ModuleType("graphlib")  # no __file__, no __spec__

        for module in (shadowed, unlocated):
            with mock.patch.dict(sys.modules, {"graphlib": module}):
                self.assertFalse(dynamo_package_lint._is_library_module("graphlib"))

        # Real stdlib and real torch, including the shapes with no file at all
        # (torch._C._nn owns F.gelu; pyexpat.errors is a stdlib submodule with
        # no location evidence of its own), must keep their waiver, so
        # descendants only have to not be located somewhere else.
        import xml.parsers.expat  # noqa: F401

        for name in (
            "torch",
            "torch._C",
            "torch._C._nn",
            "torch.ops",
            "os.path",
            "collections.abc",
            "sys",
            "zipimport",
            "pyexpat.errors",
            "xml.parsers.expat.model",
        ):
            self.assertTrue(
                dynamo_package_lint._is_library_module(name), f"{name} lost its waiver"
            )
        self.assertFalse(dynamo_package_lint._is_library_module("numpy"))
        self.assertFalse(dynamo_package_lint._is_library_module(None))

    def test_an_import_alias_decodes_to_the_module_it_names(self):
        self.assertIs(_dynamo_alias_module("__import_torch"), torch)
        self.assertIs(
            _dynamo_alias_module("__import_torch_dot_nn_dot_functional"),
            torch.nn.functional,
        )
        self.assertIsNone(_dynamo_alias_module("__import_not_a_module_at_all"))
        self.assertIsNone(_dynamo_alias_module("CFG"))

    def test_facts_differing_only_in_value_sort_apart(self):
        # Once the boilerplate code parts are filtered a TENSOR_MATCH renders no
        # code at all, so two shape specializations tie on every other component
        # of the sort key and their order falls to set iteration, which is hash
        # seeded: the file then differs between PROCESSES, which two captures in
        # one process cannot show.
        def fact(shape):
            return _GuardFact("TENSOR_MATCH", "x", (), f"shape={shape}", True)

        self.assertNotEqual(_fact_order(fact((4, 8))), _fact_order(fact((5, 8))))

    def test_code_fingerprint_recurses_into_container_and_nested_consts(self):
        # _code_fingerprint names a callable by its body so an ACT2FN-style table
        # can be told apart. Two lambdas can differ ONLY inside a constant the
        # outer co_code does not distinguish: a tuple, a frozenset, or a nested
        # code object. Filtering those out whole -- rather than recursing -- gives
        # both the same digest, _object_identity names them identically, and the
        # guard that split the two compilations is reported as an invariant of
        # each.
        from torch._dynamo.precompile_package import _code_fingerprint

        pairs = {
            "tuple const": (lambda x: x * (1, 2), lambda x: x * (1, 3)),
            "frozenset const": (lambda x: x in {1, 2}, lambda x: x in {1, 3}),
            # Not called: what matters is the nested code object in co_consts.
            "nested code": (lambda x: (lambda y: y + 1), lambda x: (lambda y: y + 2)),
        }
        for label, (left, right) in pairs.items():
            self.assertEqual(
                left.__code__.co_code,
                right.__code__.co_code,
                f"{label}: the pair must differ only in co_consts",
            )
            self.assertNotEqual(
                _code_fingerprint(left.__code__),
                _code_fingerprint(right.__code__),
                f"{label}: two different bodies share a fingerprint",
            )

    def test_guard_policy_classification_is_total(self):
        # survives() defaults to KEEP for a guard type in no set, so the
        # policy can only ever drop what _INVARIANT_DROPPABLE_GUARD_TYPES
        # names. This test is what makes the never-drop claim enforceable:
        # a guard type added to GuardBuilder fails here until someone triages
        # it into exactly one of the four sets.
        from torch._dynamo.guards import GuardBuilder
        from torch._dynamo.precompile_package import (
            _INVARIANT_DROPPABLE_GUARD_TYPES,
            _NOOP_GUARD_TYPES,
            _SHAPE_BEARING_GUARD_TYPES,
            _UNMODELLED_GUARD_TYPES,
        )

        guard_types = {
            name
            for name, value in vars(GuardBuilder).items()
            if name.isupper() and callable(value)
        }
        self.assertGreater(len(guard_types), 40)  # the enumeration itself works
        sets = {
            "_SHAPE_BEARING_GUARD_TYPES": _SHAPE_BEARING_GUARD_TYPES,
            "_UNMODELLED_GUARD_TYPES": _UNMODELLED_GUARD_TYPES,
            "_INVARIANT_DROPPABLE_GUARD_TYPES": _INVARIANT_DROPPABLE_GUARD_TYPES,
            "_NOOP_GUARD_TYPES": _NOOP_GUARD_TYPES,
        }
        classified: frozenset[str] = frozenset().union(*sets.values())
        self.assertEqual(
            sorted(guard_types - classified),
            [],
            "unclassified GuardBuilder guard type(s): add each to exactly one "
            "policy set in torch/_dynamo/precompile_package.py (KEPT until then)",
        )
        self.assertEqual(
            sorted(classified - guard_types),
            [],
            "phantom entries: no GuardBuilder method by these names",
        )
        for (a_name, a), (b_name, b) in itertools.combinations(sets.items(), 2):
            self.assertEqual(sorted(a & b), [], f"{a_name} overlaps {b_name}")

    def test_subgraph_renaming_leaves_lookalikes_alone(self):
        # Per-subgraph renaming is driven by AST positions. An attribute
        # (runner.call), a nested def (call inside class Runner) and a dotted
        # import path (torch._inductor.async_compile) each contain a renamed
        # name as text and must not be rewritten.
        from torch._dynamo.precompile_package import _namespace_module_names

        source = textwrap.dedent(
            """
            import torch._inductor.async_compile
            from torch._inductor import async_compile


            class Runner:
                def call(self, x):
                    return x


            runner = Runner()


            def call(x):
                return runner.call(x) + torch._inductor.async_compile.__name__
            """
        )
        (renamed,) = _namespace_module_names({"b0": source}).values()
        self.assertIn("class Runner_s0:", renamed)
        self.assertIn("runner_s0 = Runner_s0()", renamed)
        self.assertIn("def call_s0(x):", renamed)
        self.assertIn("    def call(self, x):", renamed)
        self.assertIn(
            "return runner_s0.call(x) + torch._inductor.async_compile.__name__", renamed
        )
        self.assertIn("import torch._inductor.async_compile\n", renamed)

    def test_summary_reports_dropped_guards(self):
        # Guard types the filter discards are recorded with their source name
        # rather than silently disappearing; dropping one widens what a graph
        # gets reused for, and only the name says whether that matters.
        session = precompile_capture(
            staged_with_global_dict_conditional, backend="eager", dynamic=False
        )
        with session as compiled:
            compiled(torch.randn(4, 8))
        summary = session.summary()
        # The filter cannot serialize identity guards, so referencing the torch
        # module at all produces at least one drop, reported as (type, source).
        self.assertTrue(summary.dropped_guards)
        for guard_type, source in summary.dropped_guards:
            self.assertIsInstance(guard_type, str)
            self.assertIsInstance(source, str)
        self.assertEqual(
            sum(summary.dropped_guard_types().values()), len(summary.dropped_guards)
        )
        self.assertIn("dropped guards", str(summary))
        # save() can be made to enforce that nothing was dropped.
        with self.assertRaisesRegex(PackageError, "dropped .* guard"):
            session.save(self.path(), require_no_dropped_guards=True)

    def test_strict_and_risky_drop_requirements_fail_closed(self):
        # Strict save rejects every drop. Relaxing that still keeps the risky
        # lint as a second gate unless the caller acknowledges it separately.
        clean = precompile_capture(
            PrecompileNoDispatchSlot(), backend="eager", dynamic=False
        )
        with clean as compiled, torch.no_grad():
            compiled(torch.randn(3, 4))
        clean.save(self.path())
        with self.assertRaisesRegex(PackageError, "not serialized"):
            clean.save(self.path(), require_no_dropped_guards=True)

        torch._dynamo.reset()
        session = precompile_capture(
            staged_with_global_function_ref, backend="eager", dynamic=False
        )
        with session as compiled:
            compiled(torch.randn(4, 8))
        with (
            self.assertNoLogs("torch._dynamo.precompile_package", "WARNING"),
            self.assertRaisesRegex(PackageError, "PRECOMPILE_ACTIVATION"),
        ):
            session.save(self.path(), require_no_dropped_guards=False)
        with self.assertLogs("torch._dynamo.precompile_package", "WARNING") as logs:
            session.save(
                self.path(),
                require_no_risky_drops=False,
                require_no_dropped_guards=False,
            )
        reported = [m for m in logs.output if "dropped guard" in m]
        self.assertEqual(len(reported), 1, logs.output)
        self.assertIn("PRECOMPILE_ACTIVATION", reported[0])
        self.assertIn("require_no_risky_drops=False", logs.output[0])

    def test_example_inputs_drive_the_capture(self):
        # example_inputs is just "run these for me": capture is by execution, so
        # the block body becomes optional.
        session = precompile_capture(
            PrecompileInvariantModel(),
            backend="eager",
            dynamic=False,
            example_inputs=_TWO_SHAPES,
        )
        with session:
            pass
        summary = session.summary()
        self.assertEqual(summary.frames, 2)
        self.assertEqual(summary.guarded_codes, 4)

    def test_a_failing_example_input_does_not_wedge_the_session(self):
        # __enter__ runs example_inputs, and a __enter__ that raises never gets
        # its __exit__, so the config patch the session holds would stay on for
        # the life of the process and the session would refuse to save what it
        # did capture. A bare tensor instead of a 1-tuple is the likely way in.
        before = functorch_config.bundled_autograd_cache
        session = precompile_capture(
            PrecompileInvariantModel(),
            backend="eager",
            dynamic=False,
            example_inputs=[torch.ones(4, 8)],
        )
        with self.assertRaisesRegex(TypeError, "tuples of positional args"):
            with session:
                pass
        self.assertEqual(functorch_config.bundled_autograd_cache, before)
        with self.assertRaisesRegex(PackageError, "capture raised"):
            session.save(self.path())

    def test_save_rejects_capture_that_ran_nothing(self):
        # Capture is by execution. A session whose callable was never run has
        # nothing to serve, and install() would just skip the frame, so
        # serving() could not report the gap either.
        session = precompile_capture(
            staged_with_graph_breaks, backend="eager", dynamic=False
        )
        with session:
            pass
        summary = session.summary()
        self.assertEqual(summary.guarded_codes, 0)
        self.assertFalse(summary.complete)
        with self.assertRaisesRegex(PackageError, "captured no compiled code"):
            session.save(self.path())

    def test_truncation_report_is_a_lower_bound(self):
        # Hitting recompile_limit sets FrameExecStrategy(RUN_ONLY, RUN_ONLY),
        # whose recursive half silences every frame called beneath the offender,
        # yet only the offender is recorded. Pin the gap the message must admit.
        inputs = [torch.randn(*s) for s in [(4, 8), (5, 8), (6, 8)]]

        def capture(limit):
            torch._dynamo.reset()
            session = precompile_capture(
                PrecompileStack(5, resume_work=True),
                backend="eager",
                recompile_limit=limit,
                dynamic=False,
            )
            with session as compiled:
                for x in inputs:
                    compiled(x)
            return session

        self.assertEqual(capture(64).summary().truncated, ())
        session = capture(8)
        summary = session.summary()
        # One frame named, but the resume frames beneath it also lost variants.
        self.assertEqual(len(summary.truncated), 1)
        self.assertIn(">=1 TRUNCATED", str(summary))
        with self.assertRaisesRegex(PackageError, "lower bound"):
            session.save(self.path())

    def test_capture_rejects_a_callable_with_no_code_object(self):
        # An nn.Module reaches the same dead end one level down: self.forward =
        # functools.partial(...) in __init__ shadows the class method, so the
        # entry function has no __code__ either. Saying so is the whole point of
        # the check; without it this died on a bare AttributeError from inside
        # CompilePackage.
        def scaled(scale, x):
            return x * scale

        class OnlyCall:
            def __call__(self, x):
                return x + 1.0

        for fn in (
            functools.partial(scaled, 2.0),
            OnlyCall(),
            PrecompilePartialForward(2.0),
        ):
            with self.assertRaisesRegex(TypeError, "no __code__"):
                precompile_capture(fn, backend="eager")

    @torch._dynamo.config.patch(caching_precompile=True)
    def test_truncated_frame_is_not_also_reported_uncovered(self):
        # Hitting the recompile limit happens INSIDE code_context, whose exit
        # saw an attempt that added no guarded code and called the frame
        # uncovered. But it has working variants: "uncovered" means no guarded
        # code at all, which is what install() skip_code()s and what save()'s
        # message describes.
        session = precompile_capture(
            PrecompileSelfAct(torch.relu),
            backend="eager",
            dynamic=False,
            recompile_limit=2,
        )
        with session as compiled, torch.no_grad():
            for n in (3, 4, 5, 6, 7, 8):
                compiled(torch.randn(n, 4))
        summary = session.summary()
        self.assertTrue(summary.truncated)
        self.assertGreater(summary.guarded_codes, 0)
        self.assertEqual(summary.uncovered_frames, ())

    def test_install_skips_backends_only_a_bypassed_entry_references(self):
        # Deserializing an inductor artifact is expensive and can fail on a
        # serving host, so install() must not touch one for an entry it will
        # skip anyway.
        class _Exploding:
            def after_deserialization(self):
                raise AssertionError("install deserialized an unusable backend")

        model = PrecompileSelfAct(torch.relu)
        session = precompile_capture(model, backend="eager", dynamic=False)
        with session as compiled, torch.no_grad():
            compiled(torch.randn(3, 4))
        package = session._package
        backend_ids = {
            backend_id
            for entry in package._codes.values()
            for backend_id in entry.backend_ids
        }
        self.assertTrue(backend_ids)
        for entry in package._codes.values():
            entry.bypassed = True
        package.install({backend_id: _Exploding() for backend_id in backend_ids})
        package.uninstall()

    def test_inductor_artifact_records_compile_time_cpu_target(self):
        model = PrecompileEmptyGraph()
        x = torch.randn(16)
        with torch._inductor.config.patch({"cpp.simdlen": 256}):
            session = precompile_capture(
                model,
                backend="inductor",
                dynamic=False,
                example_inputs=[(x,)],
            )
            with session:
                pass
            capture_target = session._package.cache_entry().system_info
        session.save(self.path())
        saved = _SingleFileStore().read(self.path()).dynamo.system_info

        self.assertEqual(capture_target.cpu_codegen_target[2], 256)
        self.assertEqual(saved.cpu_codegen_target, capture_target.cpu_codegen_target)

    @torch._dynamo.config.patch(caching_precompile=True)
    def test_failed_cache_install_cold_falls_back_on_same_package(self):
        from torch._dynamo.precompile_context import EagerCacheArtifact

        x = torch.randn(4, 8)
        session = precompile_capture(
            staged_with_graph_breaks, backend="eager", dynamic=False
        )
        with session as compiled:
            compiled(x)
        session.save(self.path())
        entry = _SingleFileStore().read(self.path())

        torch._dynamo.reset()
        with (
            mock.patch.object(DynamoCache, "load", return_value=entry),
            mock.patch.object(
                EagerCacheArtifact,
                "after_deserialization",
                side_effect=RuntimeError("cache install failed"),
            ),
            self.assertLogs("torch._dynamo.eval_frame", level="WARNING"),
        ):
            compiled = torch.compile(
                staged_with_graph_breaks, backend="eager", dynamic=False
            )
        self.assertEqual(compiled(x), staged_with_graph_breaks(x))

    @torch.compiler.set_stance("default")
    def test_overlapping_serving_contexts_keep_compilation_disabled(self):
        torch._dynamo.reset()

        @torch.compile(backend="eager", dynamic=False)
        def compiled(x):
            return x + 1

        compiled(torch.randn(2))
        both_serving = threading.Barrier(2, timeout=10)
        a_exited = threading.Event()
        errors = queue.SimpleQueue()
        rejected = queue.SimpleQueue()

        def serve_a():
            try:
                with serving():
                    both_serving.wait()
                a_exited.set()
            except Exception as e:
                errors.put(e)

        def serve_b():
            try:
                with serving():
                    both_serving.wait()
                    if not a_exited.wait(10):
                        raise RuntimeError("timed out waiting for serving thread")
                    try:
                        compiled(torch.randn(3))
                    except RuntimeError as e:
                        rejected.put("fail_on_recompile" in str(e))
                    else:
                        rejected.put(False)
            except Exception as e:
                errors.put(e)

        threads = [threading.Thread(target=fn) for fn in (serve_a, serve_b)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertTrue(errors.empty())
        self.assertTrue(rejected.get_nowait())
        self.assertTrue(rejected.empty())
        self.assertEqual(torch._dynamo.eval_frame._stance.stance, "default")

    def test_precompile_entries_are_region_scoped_in_both_directions(self):
        # Two rails that pull against each other. A precompile entry installed
        # for the DEFAULT region must not be served to an isolated one (the
        # identity guards that would tell two artifacts apart are the ones
        # precompile drops), and the caching_precompile loader must therefore
        # install into the region its own context looks up in.
        x = torch.randn(3, 4)
        model = PrecompileSelfAct(torch.relu)
        self._capture(model, x).save(self.path(), require_no_risky_drops=False)

        def installed(into_region):
            torch._dynamo.reset()
            cache_entry = _SingleFileStore().load_cache_entry(self.path())
            package = CompilePackage(model.forward, cache_entry.dynamo)
            ctx = torch._dynamo.optimize(
                "eager", dynamic=False, isolate_recompiles=True
            )
            region = {"isolate_recompiles_id": ctx._isolate_recompiles_id}
            package.install(cache_entry.backends, **(region if into_region else {}))
            return package, ctx(model)

        # Rail 1: default-region install is NOT visible from an isolated region.
        package, isolated = installed(into_region=False)
        try:
            with (
                torch.no_grad(),
                serving(),
                self.assertRaisesRegex(RecompileError, "Detected recompile"),
            ):
                isolated(x)
        finally:
            package.uninstall()

        # Rail 2: installing into the region the context uses does serve.
        package, isolated = installed(into_region=True)
        try:
            with torch.no_grad(), serving():
                self.assertEqual(isolated(x), model(x))
        finally:
            package.uninstall()

    @parametrize("error_type", (RuntimeError, KeyboardInterrupt))
    def test_a_failed_install_leaves_nothing_installed(self, error_type):
        self._capture(staged_with_graph_breaks, torch.randn(3, 4)).save(self.path())

        torch._dynamo.reset()
        cache_entry = _SingleFileStore().load_cache_entry(self.path())
        backends = cache_entry.backends

        class _Boom:
            def after_deserialization(self):
                raise error_type("artifact will not deserialize on this host")

        # The last frame's backend is the one that fails, so the earlier frames
        # are already installed when install() gives up. A backend rejected by
        # the serving host is the realistic way into this.
        backends[cache_entry.dynamo.codes[-1].backend_ids[-1]] = _Boom()

        scope = staged_with_graph_breaks.__globals__
        before = set(scope)
        package = CompilePackage(staged_with_graph_breaks, cache_entry.dynamo)
        torch._dynamo.optimize("eager", package=package, dynamic=False)(
            staged_with_graph_breaks
        )
        with self.assertRaisesRegex(error_type, "will not deserialize"):
            package.install(backends)

        # install() raising leaves the caller no handle to unload with, so a
        # partial install would be permanent: some frames served, some not.
        leftover = [n for n in set(scope) - before if not n.startswith("__import_")]
        self.assertEqual(sorted(leftover), [])
        self.assertEqual(
            _debug_get_precompile_entries(staged_with_graph_breaks.__code__), []
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
