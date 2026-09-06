# Owner(s): ["module: inductor"]
import sys
import threading
import types
from unittest import mock

import torch
import torch.fx
from torch._inductor.codegen import common
from torch._inductor.codegen.common import (
    get_scheduling_for_device,
    init_backend_registration,
    register_backend_for_device,
)
from torch._inductor.test_case import TestCase


class DeviceBackendInitTest(TestCase):
    """
    Tests the optional ``_inductor_backend_init`` hook: a no-arg callable on a
    privateuse1 device module (the one registered via
    ``torch._register_device_module``).  Inductor invokes it on each compile
    while the device is unregistered (on the ``compile_fx`` paths, before the
    decomposition table is selected); the hook must register the device via
    ``register_backend_for_device`` itself, which is what stops the
    re-invocation.
    """

    device = "fakedevice"

    def setUp(self) -> None:
        super().setUp()
        # Ensure the once-per-process built-in registration has run before
        # snapshotting, so tearDown only removes what each test added.
        common._init_builtin_backend_registration()
        self._orig_codegen = dict(common.device_codegens)
        self._orig_custom_passes = dict(common.custom_backend_passes)
        self._orig_custom_configs = dict(common.custom_backend_codegen_configs)
        common._privateuse1_backend_init_in_progress = False

    def tearDown(self) -> None:
        # Drop whatever the hooks registered, remove the fake device module
        # from torch, and clear the in-progress flag so later tests start clean.
        for registry, orig in [
            (common.device_codegens, self._orig_codegen),
            (common.custom_backend_passes, self._orig_custom_passes),
            (common.custom_backend_codegen_configs, self._orig_custom_configs),
        ]:
            for device in [d for d in registry if d not in orig]:
                del registry[device]
        if hasattr(torch, self.device):
            delattr(torch, self.device)
        sys.modules.pop(f"torch.{self.device}", None)
        common._privateuse1_backend_init_in_progress = False
        super().tearDown()

    def _install_device_module(self, **attrs) -> None:
        # rename_privateuse1_backend() is once-per-process, so mock the name
        # lookup instead of renaming for real.  Both references must be
        # patched: the probe in common.py reads the C-level attribute, while
        # backend_registration imported its own copy at module load.
        setattr(torch, self.device, types.SimpleNamespace(**attrs))
        for target in (
            "torch._C._get_privateuse1_backend_name",
            "torch.utils.backend_registration._get_privateuse1_backend_name",
        ):
            patcher = mock.patch(target, return_value=self.device)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _install_hook_registering_device(self) -> dict:
        calls = {"n": 0}

        def hook() -> None:
            calls["n"] += 1
            register_backend_for_device(self.device, object, object)

        self._install_device_module(_inductor_backend_init=hook)
        return calls

    # ------------------------------------------------------------------
    # Hook semantics
    # ------------------------------------------------------------------

    def test_backend_init_called_once(self) -> None:
        calls = self._install_hook_registering_device()
        init_backend_registration()
        self.assertIsNotNone(get_scheduling_for_device(self.device))
        # The device is registered now, so the hook is not re-fired.
        init_backend_registration()
        self.assertEqual(calls["n"], 1)

    def test_backend_init_takes_precedence_over_class_probe(self) -> None:
        # When both the hook and the legacy class attributes are present,
        # only the hook runs.
        calls = {"n": 0}

        class LegacyScheduling:
            pass

        class LegacyWrapper:
            pass

        class LegacyCppWrapper:
            pass

        class LegacyFxWrapper:
            pass

        def hook() -> None:
            calls["n"] += 1
            register_backend_for_device(self.device, object, object)

        self._install_device_module(
            _inductor_backend_init=hook,
            Scheduling=LegacyScheduling,
            PythonWrapperCodegen=LegacyWrapper,
            CppWrapperCodegen=LegacyCppWrapper,
            WrapperFxCodegen=LegacyFxWrapper,
        )
        init_backend_registration()
        self.assertEqual(calls["n"], 1)
        self.assertIs(get_scheduling_for_device(self.device), object)

    def test_class_probe_fallback_without_hook(self) -> None:
        # A device module without the hook still goes through the legacy
        # four-class probe, which registers only when Scheduling,
        # PythonWrapperCodegen and CppWrapperCodegen all resolve.
        class Scheduling:
            pass

        class Wrapper:
            pass

        class CppWrapper:
            pass

        class FxWrapper:
            pass

        self._install_device_module(
            Scheduling=Scheduling,
            PythonWrapperCodegen=Wrapper,
            CppWrapperCodegen=CppWrapper,
            WrapperFxCodegen=FxWrapper,
        )
        init_backend_registration()
        self.assertIs(get_scheduling_for_device(self.device), Scheduling)

    def test_class_probe_skips_partial_modules(self) -> None:
        # A module exposing only some of the four legacy classes registers
        # nothing.
        class Scheduling:
            pass

        self._install_device_module(Scheduling=Scheduling)
        init_backend_registration()
        self.assertIsNone(get_scheduling_for_device(self.device))

    def test_backend_init_failure_propagates_and_retries(self) -> None:
        calls = {"n": 0}

        def hook() -> None:
            calls["n"] += 1
            raise RuntimeError("boom")

        self._install_device_module(_inductor_backend_init=hook)
        # The vendor's error propagates instead of being swallowed.
        with self.assertRaisesRegex(RuntimeError, "boom"):
            init_backend_registration()
        # The hook registered nothing, so the next compile invokes it again.
        with self.assertRaisesRegex(RuntimeError, "boom"):
            init_backend_registration()
        self.assertEqual(calls["n"], 2)
        self.assertIsNone(get_scheduling_for_device(self.device))

    def test_reentrant_hook_failure_does_not_block_retry(self) -> None:
        # A hook that re-enters init_backend_registration and then raises
        # must not leave the in-progress flag set, or every later compile
        # would silently skip the hook.
        calls = {"n": 0}

        def hook() -> None:
            calls["n"] += 1
            common.get_backend_features("cpu")
            raise RuntimeError("boom")

        self._install_device_module(_inductor_backend_init=hook)
        with self.assertRaisesRegex(RuntimeError, "boom"):
            init_backend_registration()
        # The in-progress flag was reset, so the retry is not skipped.
        with self.assertRaisesRegex(RuntimeError, "boom"):
            init_backend_registration()
        self.assertEqual(calls["n"], 2)
        self.assertIsNone(get_scheduling_for_device(self.device))

    def test_noop_hook_refires_while_unregistered(self) -> None:
        # A hook that returns without registering leaves the device
        # unregistered, so each compile invokes it again; registration is the
        # signal that initialization is done.
        calls = {"n": 0}

        def hook() -> None:
            calls["n"] += 1

        self._install_device_module(_inductor_backend_init=hook)
        init_backend_registration()
        init_backend_registration()
        self.assertEqual(calls["n"], 2)
        self.assertIsNone(get_scheduling_for_device(self.device))

    def test_replaced_device_module_is_rediscovered(self) -> None:
        # No permanent "fired" state: swapping in a new device module (A/B
        # testing a new backend) is discovered once the old registration is
        # dropped.
        first = self._install_hook_registering_device()
        init_backend_registration()
        self.assertEqual(first["n"], 1)

        for registry in (
            common.device_codegens,
            common.custom_backend_passes,
            common.custom_backend_codegen_configs,
        ):
            del registry[self.device]

        second = self._install_hook_registering_device()
        init_backend_registration()
        self.assertEqual(second["n"], 1)
        self.assertIsNotNone(get_scheduling_for_device(self.device))

    def test_reentrant_hook_does_not_recurse(self) -> None:
        # A vendor import may reach code that queries backend features, which
        # re-enters init_backend_registration before the hook has registered
        # the device; the compile lock is re-entrant and the in-progress flag
        # prevents the hook from firing again.
        calls = {"n": 0}

        def hook() -> None:
            calls["n"] += 1
            common.get_backend_features("cpu")
            register_backend_for_device(self.device, object, object)

        self._install_device_module(_inductor_backend_init=hook)
        init_backend_registration()
        self.assertEqual(calls["n"], 1)
        self.assertIsNotNone(get_scheduling_for_device(self.device))

    def test_concurrent_caller_waits_for_hook(self) -> None:
        # A caller that reaches init_backend_registration while the hook is
        # in flight must block until registration has finished, not return
        # against a half-registered device. The hook re-enters first (the
        # re-entrant call returns early via the in-progress flag); the second
        # thread blocks on the compile lock until the first thread's hook
        # finishes, then sees the registration.
        hook_entered = threading.Event()
        release_hook = threading.Event()
        calls = {"n": 0}

        def hook() -> None:
            calls["n"] += 1
            common.get_backend_features("cpu")
            hook_entered.set()
            release_hook.wait(timeout=10)
            register_backend_for_device(self.device, object, object)

        self._install_device_module(_inductor_backend_init=hook)

        def caller() -> None:
            init_backend_registration()

        first = threading.Thread(target=caller)
        first.start()
        self.assertTrue(hook_entered.wait(timeout=10))
        second = threading.Thread(target=caller)
        second.start()
        second.join(timeout=0.5)
        self.assertTrue(second.is_alive())
        release_hook.set()
        first.join(timeout=10)
        second.join(timeout=10)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(calls["n"], 1)
        self.assertIsNotNone(get_scheduling_for_device(self.device))

    def test_backend_init_as_device_module_class_method(self) -> None:
        # The documented form: a static method on the device module class
        # registered via torch._register_device_module. The probe resolves it
        # with a plain getattr, so no special support is needed.
        calls = {"n": 0}

        class DeviceModule:
            @staticmethod
            def _inductor_backend_init() -> None:
                calls["n"] += 1
                register_backend_for_device(
                    DeviceBackendInitTest.device, object, object
                )

        # torch._register_device_module validates the device name, which needs
        # the once-per-process rename, so emulate its two side effects (the
        # attribute on torch and the sys.modules entry) directly.
        setattr(torch, self.device, DeviceModule)
        sys.modules[f"torch.{self.device}"] = DeviceModule
        for target in (
            "torch._C._get_privateuse1_backend_name",
            "torch.utils.backend_registration._get_privateuse1_backend_name",
        ):
            patcher = mock.patch(target, return_value=self.device)
            patcher.start()
            self.addCleanup(patcher.stop)
        init_backend_registration()
        self.assertEqual(calls["n"], 1)
        self.assertIsNotNone(get_scheduling_for_device(self.device))

    # ------------------------------------------------------------------
    # Placement on the compile path
    # ------------------------------------------------------------------

    def test_init_fires_before_decomp_table(self) -> None:
        # init_backend_registration must run at the top of _compile_fx_main,
        # before the decomposition table is snapshotted: a vendor hook may
        # register decompositions, and the first compile must see them.
        order = []
        compile_fx_mod = torch._inductor.compile_fx
        real_init = compile_fx_mod.init_backend_registration
        real_select_decomp = compile_fx_mod.select_decomp_table

        def record_init() -> None:
            order.append("init")
            real_init()

        def record_decomp():
            order.append("decomp")
            return real_select_decomp()

        gm = torch.fx.symbolic_trace(lambda x: (x + 1,))
        x = torch.randn(8)
        with (
            mock.patch.object(compile_fx_mod, "init_backend_registration", record_init),
            mock.patch.object(compile_fx_mod, "select_decomp_table", record_decomp),
        ):
            compile_fx_mod.compile_fx(gm, [x])

        self.assertEqual(order[0], "init")
        self.assertLess(order.index("init"), order.index("decomp"))

    def test_backend_init_fires_from_backend_features_query(self) -> None:
        # use_triton_template queries backend features per lowered GEMM; an
        # unregistered device is discovered from there too.
        calls = self._install_hook_registering_device()
        common.get_backend_features("cpu")
        self.assertEqual(calls["n"], 1)
        self.assertIsNotNone(get_scheduling_for_device(self.device))

    def test_backend_init_fires_on_torch_compile(self) -> None:
        calls = self._install_hook_registering_device()

        def fn(x):
            return x + 1

        x = torch.randn(8)
        self.assertEqual(torch.compile(fn)(x), fn(x))
        self.assertEqual(calls["n"], 1)
        self.assertIsNotNone(get_scheduling_for_device(self.device))

    def test_backend_init_fires_on_compile_fx(self) -> None:
        calls = self._install_hook_registering_device()
        gm = torch.fx.symbolic_trace(lambda x: (x + 1,))
        torch._inductor.compile_fx.compile_fx(gm, [torch.randn(8)])
        self.assertEqual(calls["n"], 1)
        self.assertIsNotNone(get_scheduling_for_device(self.device))

    def test_backend_init_fires_on_aot_compile(self) -> None:
        calls = self._install_hook_registering_device()

        class Model(torch.nn.Module):
            def forward(self, x):
                return (x + 1,)

        x = torch.randn(8)
        exported = torch.export.export(Model(), (x,))
        torch._inductor.aot_compile(exported.module(), (x,))
        self.assertEqual(calls["n"], 1)
        self.assertIsNotNone(get_scheduling_for_device(self.device))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
