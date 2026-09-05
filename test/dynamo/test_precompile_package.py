# Owner(s): ["module: dynamo"]

import contextlib
import dataclasses
import gc
import importlib
import inspect
import os
import sys
import types
from unittest import mock

import torch
import torch._dynamo.package as dynamo_package
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


_MIPS_TARGET = ("mips", "DEFAULT", 128, ("INVALID",), None, "INVALID")

# recorded target, whether the host probe works, what the comparison raises
_CPU_TARGET_CASES = {
    "no_target_recorded": (None, True, None),
    "skewed_target": (_MIPS_TARGET, True, "built for machine 'mips'"),
    "host_probe_failed": (
        ("x86_64", "AVX512", 512, ("CPU_CAPABILITY_AVX512",), None, "avx512"),
        False,
        "no usable",
    ),
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

    def test_joining_a_dead_owners_binding_survives_its_cleanup(self):
        # A package collected without uninstall() parks its cleanup when the
        # registry lock is busy. A later install of the same artifact writes the
        # same builtins dict under the same name; the drain must find the new
        # owner on the binding and keep the name, not delete it as the dead
        # package's leftover.
        module = types.ModuleType("test_precompile_dead_owner")
        shared = {}
        dead = self._bare_package()
        dead._install_global(module, "g", shared)
        installed = dead._installed_globals
        del dead
        gc.collect()
        dynamo_package._DEAD_PACKAGES.append(
            dynamo_package._DeadPackageState(installed, [])
        )
        live = self._bare_package()
        live._install_global(module, "g", shared)
        self.assertIs(module.__dict__.get("g"), shared)
        live.uninstall()
        self.assertNotIn("g", module.__dict__)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
