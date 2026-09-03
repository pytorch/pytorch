# Owner(s): ["module: dynamo"]

import contextlib
import copy
import dataclasses
import importlib
import inspect
import itertools
import os
import pickle
import sys
import sysconfig
import textwrap
import types
from unittest import mock

import torch
import torch._dynamo.package as dynamo_package
import torch._dynamo.precompile_package as dynamo_package_lint
import torch._dynamo.testing
import torch._inductor.config
import torch._inductor.test_case
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
)
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


# One source LINE, so the two entries agree on file AND lineno: what separates
# them can only come from the code body. This is the ACT2FN shape.
# The stateless corpus shapes are plain functions; precompile_capture takes
# either, and only the guard sources matter here.
# The modules the corpus needs on disk, because a dispatch read off another
# module cannot be spelled inside this file. Written under one directory so
# they import each other by plain name, exactly as a user package would.
# Every shape below was once a silent wrong answer on a serving machine that
# somebody had to find by hand. _is_risky_drop regressed three review rounds
# running -- each fix closed the previous round's false negative and opened a
# new one -- so the shapes live in a table rather than in prose: adding one is
# a single entry, and nothing here may ever stop being flagged. Each row names
# the guard sources (by suffix) the risky-drop report must carry.
# The other half of the corpus, and the half that keeps the report worth
# reading: the lint only warns by default, so if ordinary code trips it the
# warning is noise nobody audits and nobody ever opts into enforcement. Each
# row names the identity guards (by suffix) that ARE dropped without being
# flagged: torch internals, stdlib modules and their attributes, a global bound
# to a def of its own name, reads off Dynamo's import aliases and its builtins
# dict.
# A value crossing a graph break or arriving as a non-tensor argument is
# guarded by equality, so the artifact only serves inputs reproducing it, and
# nothing else in the summary says so. Model config -- LayerNorm eps, Dropout p,
# a plain int attribute -- is CONSTANT_MATCH too but constant for the model the
# artifact is loaded onto; counting it would flag every model. Rows: builder,
# args, the wont_generalize report, and a (type, source substring) guard that
# must have been KEPT, so a row cannot pass by Dynamo not emitting the guard.
# Rows: builder, example_inputs, text the invariants file must contain, and a
# pattern it must not -- object ids, the per-process counter in Dynamo's
# builtins-dict name, and the address install_global_by_id bakes into an
# identifier ("<prefix>_<id(value)>_c<n>": `type(x) is torch.Tensor` installs
# one) are all normalized so the file is stable enough to commit and diff.
# Shapes where the guard that SPLIT two compilations must show up as varying
# and never as an invariant of both. Rows: builder, capture kwargs, calls,
# substrings some varying fact must carry, substrings some invariant fact must
# carry, substrings no invariant fact may carry.
# A pair that differs only after the break, so a resume function borrowed from
# the wrong artifact still runs and still returns a number.
# Content-addressing the resume code tells the first pair apart, but not two
# instances of ONE model class: the same script captured in two processes
# mints the same __resume_at_<offset>_<n> AND byte-identical resume code.
# Rows: fn, the captured variants as (PRECOMPILE_CONFIG mode, extra args), one
# uncaptured variant, guard types that must be emitted AND kept, and whether
# save() has to be told the drops it reports are acknowledged risky ones.
# functools.wraps copies __qualname__, so an instrumented wrapper around a
# different callable passes the qualname check while being a different code
# object.
# CompilePackage rebinds the stored guards onto whatever callable it is given,
# so load has to refuse anything but the captured callable itself; a __code__
# that is present and None gets past the capture-side hasattr check.
# How the saved artifact is damaged (None: the path is a directory, the
# pre-single-file layout) and the error precompile_load must name it with.
# A legacy package's empty frame installs a skip only in its region; whatever
# global strategy the frame had before the load, or gained while loaded, has
# to be what unload leaves. Rows: applied before the load, applied after it,
# the (cur, recursive) strategy expected after unload.
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

    # Every shape found to split a compilation while the report called it an
    # invariant. The fingerprint has failed open three times -- shapes, then
    # python type and conj/neg, then the dispatch key set -- each fix revealing
    # the next, so the shapes are asserted here rather than described. A new
    # one is one line. See _value_fingerprint.


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
