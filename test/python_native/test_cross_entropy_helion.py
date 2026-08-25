# Owner(s): ["module: dsl-native-ops"]

import os
import subprocess
import sys
import textwrap
import threading
import unittest
import warnings
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch._native.ops.cross_entropy.helion_impl import (
    _B200_PRETUNED_SHAPES,
    _cross_entropy_cond,
)
from torch.testing._internal.common_cuda import IS_SM100, TEST_CUDA
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoHelionDSL,
    TestCase,
)
from torch.testing._internal.logging_utils import log_settings


_SHAPE_CONFIG_INDEX = {
    (2048, 32000): 2,
    (4096, 32000): 2,
    (8192, 32000): 2,
    (8192, 128000): 0,
    (16384, 128000): 0,
    (32768, 128000): 0,
    (2048, 128256): 1,
    (4096, 128256): 4,
    (8192, 128256): 4,
    (16384, 128256): 0,
    (2048, 129280): 1,
    (4096, 129280): 0,
    (8192, 129280): 1,
    (2048, 151936): 1,
    (4096, 151936): 1,
    (8192, 151936): 1,
    (2048, 152064): 3,
    (4096, 152064): 3,
    (8192, 152064): 0,
    (1024, 256000): 6,
    (2048, 256000): 5,
}

_CONFIG_SIGNATURES = (
    ([256], 4, "persistent_interleaved", [1]),
    ([8192], 8, "flat", [0]),
    ([None], 16, "flat", [0]),
    ([512], 8, "persistent_blocked", [2]),
    ([256], 4, "persistent_interleaved", [4]),
    ([1024], 16, "persistent_interleaved", [4]),
    ([2048], 32, "persistent_interleaved", [4]),
)

_AOT_RUNTIME_SHAPES = (
    (4096, 129280),
    (32768, 128000),
    (2048, 128256),
    (8192, 32000),
    (2048, 152064),
    (4096, 128256),
    (2048, 256000),
    (1024, 256000),
)


class TestHelionCrossEntropyAOTConfig(TestCase):
    def test_shapes_match_pretuned_set(self):
        self.assertEqual(_B200_PRETUNED_SHAPES, frozenset(_SHAPE_CONFIG_INDEX))

    @parametrize("shape, expected", tuple(_SHAPE_CONFIG_INDEX.items()))
    def test_aot_config(self, shape, expected):
        from torch._native.ops.cross_entropy._helion_aot_helion_kernel_cuda_sm100 import (
            autotune_cross_entropy,
        )

        logits = torch.empty(shape, device="meta", dtype=torch.bfloat16)
        config = autotune_cross_entropy(logits)
        signature = (
            config["reduction_loops"],
            config["num_warps"],
            config["pid_type"],
            config["range_unroll_factors"],
        )
        self.assertEqual(signature, _CONFIG_SIGNATURES[expected])


class TestHelionCrossEntropyRegistration(TestCase):
    def test_install_and_uninstall_are_atomic(self):
        import torch._native.ops.cross_entropy.helion_impl as impl

        install_entered = threading.Event()
        install_release = threading.Event()
        uninstall_started = threading.Event()
        uninstall_done = threading.Event()
        created = []
        errors = []

        class FakeLibrary:
            def __init__(self, *args):
                self.key = args[-1]
                self.destroyed = False
                created.append(self)

            def impl(self, *args, **kwargs):
                if self.key == "AutocastCUDA":
                    install_entered.set()
                    if not install_release.wait(timeout=5):
                        raise RuntimeError("install release timed out")

            def _destroy(self):
                self.destroyed = True

        def install():
            try:
                impl._install_autograd_fallthrough(object())
            except BaseException as exc:
                errors.append(exc)

        def uninstall():
            uninstall_started.set()
            try:
                impl._uninstall_autograd_fallthrough()
            except BaseException as exc:
                errors.append(exc)
            finally:
                uninstall_done.set()

        with (
            patch.object(impl, "_autocast_lib", None),
            patch.object(impl, "_autograd_lib", None),
            patch.object(impl.torch.library, "Library", FakeLibrary),
            patch.object(
                impl.torch._C,
                "_dispatch_has_kernel_for_dispatch_key",
                return_value=False,
            ),
        ):
            installer = threading.Thread(target=install)
            installer.start()
            self.assertTrue(install_entered.wait(timeout=5))

            remover = threading.Thread(target=uninstall)
            remover.start()
            self.assertTrue(uninstall_started.wait(timeout=5))
            self.assertFalse(uninstall_done.wait(timeout=0.25))

            install_release.set()
            installer.join(timeout=5)
            remover.join(timeout=5)

            self.assertFalse(installer.is_alive())
            self.assertFalse(remover.is_alive())
            self.assertEqual(errors, [])
            self.assertIsNone(impl._autocast_lib)
            self.assertIsNone(impl._autograd_lib)
            self.assertTrue(all(lib.destroyed for lib in created))

    @unittest.skipUnless(hasattr(os, "fork"), "requires os.fork")
    def test_partial_install_failure_can_be_cleaned_up_after_fork(self):
        import torch._native.ops.cross_entropy.helion_impl as impl

        parent_pid = os.getpid()
        cleanup_entered = threading.Event()
        cleanup_release = threading.Event()
        created = []
        errors = []

        class FakeLibrary:
            def __init__(self, *args):
                self.key = args[-1]
                self.active = False
                self.destroyed = False
                created.append(self)

            def impl(self, *args, **kwargs):
                if self.key == "AutocastCUDA":
                    self.active = True
                else:
                    raise KeyboardInterrupt("stop")

            def _destroy(self):
                if self.key == "AutocastCUDA" and os.getpid() == parent_pid:
                    cleanup_entered.set()
                    if not cleanup_release.wait(timeout=5):
                        raise RuntimeError("cleanup release timed out")
                self.active = False
                self.destroyed = True

        def install():
            try:
                impl._install_autograd_fallthrough(object())
            except BaseException as exc:
                errors.append(exc)

        with (
            patch.object(impl, "_autocast_lib", None),
            patch.object(impl, "_autograd_lib", None),
            patch.object(impl.torch.library, "Library", FakeLibrary),
            patch.object(
                impl.torch._C,
                "_dispatch_has_kernel_for_dispatch_key",
                return_value=False,
            ),
        ):
            worker = threading.Thread(target=install)
            worker.start()
            self.assertTrue(cleanup_entered.wait(timeout=5))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    child_pid = os.fork()
                if child_pid == 0:
                    try:
                        import signal

                        signal.signal(signal.SIGALRM, lambda *_: os._exit(42))
                        signal.alarm(2)
                        impl._uninstall_autograd_fallthrough()
                        clean = (
                            impl._autocast_lib is None
                            and impl._autograd_lib is None
                            and not any(lib.active for lib in created)
                        )
                        os._exit(0 if clean else 2)
                    except BaseException:
                        os._exit(3)
                _, status = os.waitpid(child_pid, 0)
            finally:
                cleanup_release.set()
                worker.join(timeout=5)

            self.assertFalse(worker.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], KeyboardInterrupt)
            self.assertIsNone(impl._autocast_lib)
            self.assertIsNone(impl._autograd_lib)
            self.assertEqual(len(created), 2)
            self.assertTrue(all(lib.destroyed for lib in created))
            self.assertTrue(os.WIFEXITED(status))
            self.assertEqual(os.WEXITSTATUS(status), 0)

    @unittest.skipUnless(hasattr(os, "fork"), "requires os.fork")
    def test_partial_install_can_be_cleaned_up_after_fork(self):
        import torch._native.ops.cross_entropy.helion_impl as impl

        entered = threading.Event()
        release = threading.Event()
        created = []
        errors = []

        class FakeLibrary:
            def __init__(self, *args):
                self.key = args[-1]
                self.destroyed = False
                created.append(self)

            def impl(self, *args, **kwargs):
                if self.key == "AutocastCUDA":
                    entered.set()
                    if not release.wait(timeout=5):
                        raise RuntimeError("install release timed out")

            def _destroy(self):
                self.destroyed = True

        def install():
            try:
                impl._install_autograd_fallthrough(object())
            except BaseException as exc:
                errors.append(exc)

        with (
            patch.object(impl, "_autocast_lib", None),
            patch.object(impl, "_autograd_lib", None),
            patch.object(impl.torch.library, "Library", FakeLibrary),
            patch.object(
                impl.torch._C,
                "_dispatch_has_kernel_for_dispatch_key",
                return_value=False,
            ),
        ):
            worker = threading.Thread(target=install)
            worker.start()
            self.assertTrue(entered.wait(timeout=5))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    child_pid = os.fork()
                if child_pid == 0:
                    try:
                        import signal

                        signal.signal(signal.SIGALRM, lambda *_: os._exit(42))
                        signal.alarm(2)
                        impl._uninstall_autograd_fallthrough()
                        clean = (
                            impl._autocast_lib is None
                            and impl._autograd_lib is None
                            and created[0].destroyed
                        )
                        os._exit(0 if clean else 2)
                    except BaseException:
                        os._exit(3)
                _, status = os.waitpid(child_pid, 0)
            finally:
                release.set()
                worker.join(timeout=5)

            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            self.assertTrue(os.WIFEXITED(status))
            self.assertEqual(os.WEXITSTATUS(status), 0)
            impl._uninstall_autograd_fallthrough()
            self.assertTrue(all(lib.destroyed for lib in created))


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@unittest.skipUnless(IS_SM100, "B200/GB200 sm100 required")
@skipIfNoHelionDSL
class TestHelionCrossEntropy(TestCase):
    def _inputs(self, device, requires_grad=False):
        logits = torch.randn(
            8192,
            32000,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=requires_grad,
        )
        labels = torch.randint(32000, (8192,), device=device, dtype=torch.int64)
        return logits, labels

    def test_condition_accepts_pretuned_contract(self, device):
        logits, labels = self._inputs(device)
        self.assertTrue(_cross_entropy_cond(logits, labels))

    @parametrize("shape", _AOT_RUNTIME_SHAPES)
    def test_correctness_optimized_path(self, device, shape):
        from torch._native.ops.cross_entropy._helion_aot_helion_kernel_cuda_sm100 import (
            autotune_cross_entropy,
        )
        from torch._native.ops.cross_entropy.helion_kernel import cross_entropy

        logits = torch.randn(shape, device=device, dtype=torch.bfloat16)
        labels = torch.randint(shape[1], (shape[0],), device=device)
        self.assertTrue(_cross_entropy_cond(logits, labels))
        actual = F.cross_entropy(logits, labels)
        with torch.backends.python_native.helion.disabled():
            expected = F.cross_entropy(logits, labels)
        self.assertEqual(actual, expected, rtol=1e-2, atol=1e-2)
        nonignored_count = torch.full((1,), shape[0], device=device, dtype=torch.int64)
        bound = cross_entropy.helion_kernel.bind((logits, labels, nonignored_count))
        self.assertEqual(bound._config.config, autotune_cross_entropy(logits))

    def test_target_gather_numerical_stability(self, device):
        n, v = 8192, 32000
        labels = torch.arange(n, device=device, dtype=torch.int64) % v
        logits = torch.full((n, v), 1000, device=device, dtype=torch.bfloat16)
        logits[torch.arange(n, device=device), labels] = 1016
        actual = F.cross_entropy(logits, labels)
        with torch.backends.python_native.helion.disabled():
            expected = F.cross_entropy(logits, labels)
        self.assertEqual(actual, expected, rtol=1e-2, atol=1e-4)

    def test_instrumentation_reports_compile_and_cache_hit(self, device):
        from torch._native.ops.cross_entropy.helion_kernel import cross_entropy

        logits, labels = self._inputs(device)
        nonignored_count = torch.full(
            (1,), logits.shape[0], device=device, dtype=torch.int64
        )
        kernel = cross_entropy.helion_kernel
        kernel.reset()
        try:
            with (
                log_settings("+native_dsl_compile"),
                self.assertLogs("torch._native.instrumentation", level="INFO") as logs,
            ):
                cross_entropy(logits, labels, nonignored_count)
                cross_entropy(logits, labels, nonignored_count)
        finally:
            kernel.reset()
        self.assertEqual(len(logs.output), 2)
        self.assertIn("compiled", logs.output[0])
        self.assertIn("misses=1", logs.output[0])
        self.assertIn("cache_hit", logs.output[1])

    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "requires at least 2 visible CUDA devices",
    )
    def test_non_current_device(self, device):
        if torch.cuda.get_device_capability(1) != (10, 0):
            self.skipTest("second device must be B200/GB200 sm100")
        old_device = torch.cuda.current_device()
        try:
            torch.cuda.set_device(0)
            logits = torch.randn(8192, 32000, device="cuda:1", dtype=torch.bfloat16)
            labels = torch.randint(32000, (8192,), device="cuda:1")
            with patch(
                "torch.cuda.is_current_stream_capturing",
                side_effect=lambda: torch.cuda.current_device() == 1,
            ):
                self.assertFalse(_cross_entropy_cond(logits, labels))
            self.assertEqual(torch.cuda.current_device(), 0)
            self.assertTrue(_cross_entropy_cond(logits, labels))
            actual = F.cross_entropy(logits, labels)
            with torch.backends.python_native.helion.disabled():
                expected = F.cross_entropy(logits, labels)
            self.assertEqual(torch.cuda.current_device(), 0)
            self.assertEqual(actual.device, torch.device("cuda:1"))
            self.assertEqual(actual, expected, rtol=1e-2, atol=1e-2)
        finally:
            torch.cuda.set_device(old_device)

    def test_cuda_graph_capture_falls_through(self, device):
        logits, labels = self._inputs(device)
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            F.cross_entropy(logits, labels)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = F.cross_entropy(logits, labels)
        graph.replay()
        with torch.backends.python_native.helion.disabled():
            expected = F.cross_entropy(logits, labels)
        self.assertEqual(actual, expected, rtol=1e-2, atol=1e-2)

    def test_optimized_path_does_not_synchronize(self, device):
        logits, labels = self._inputs(device)
        F.cross_entropy(logits, labels)
        torch.cuda.synchronize()

        marker = torch.cuda.Event()
        torch.cuda._sleep(200_000_000)
        marker.record()
        try:
            F.cross_entropy(logits, labels)
            self.assertFalse(marker.query())
        finally:
            torch.cuda.synchronize()

    def test_autocast_falls_through(self, device):
        logits, labels = self._inputs(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.assertFalse(_cross_entropy_cond(logits, labels))
            actual = F.cross_entropy(logits, labels)
            with torch.backends.python_native.helion.disabled():
                expected = F.cross_entropy(logits, labels)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual, expected)

    def test_fake_tensor_falls_through(self, device):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            logits = torch.empty(8192, 32000, device=device, dtype=torch.bfloat16)
            labels = torch.empty(8192, device=device, dtype=torch.int64)
            self.assertFalse(_cross_entropy_cond(logits, labels))

    def test_export_falls_through(self, device):
        from torch._native.registry import native_decomp_table

        class CrossEntropy(torch.nn.Module):
            def forward(self, logits, labels):
                return F.cross_entropy(logits, labels)

        logits, labels = self._inputs(device)
        exported = torch.export.export(CrossEntropy(), (logits, labels))
        exported = exported.run_decompositions(native_decomp_table())
        targets = [
            str(node.target)
            for node in exported.graph.nodes
            if node.op == "call_function"
        ]
        self.assertFalse(any(target.startswith("_native.") for target in targets))
        actual = exported.module()(logits, labels)
        with torch.backends.python_native.helion.disabled():
            expected = F.cross_entropy(logits, labels)
        self.assertEqual(actual, expected)

    def test_torch_compile_falls_through(self, device):
        logits, labels = self._inputs(device)
        compiled = torch.compile(
            lambda input, target: F.cross_entropy(input, target), fullgraph=True
        )
        actual = compiled(logits, labels)
        with torch.backends.python_native.helion.disabled():
            expected = F.cross_entropy(logits, labels)
        self.assertEqual(actual, expected)

    def test_torch_compile_autocast_falls_through(self, device):
        logits, labels = self._inputs(device)
        compiled = torch.compile(
            lambda input, target: F.cross_entropy(input, target), fullgraph=True
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = compiled(logits, labels)
            with torch.backends.python_native.helion.disabled():
                reference = torch.compile(
                    lambda input, target: F.cross_entropy(input, target),
                    fullgraph=True,
                )
                expected = reference(logits, labels)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual, expected)

    def test_autocast_probability_target_falls_through(self, device):
        logits = torch.randn(32, 128, device=device, dtype=torch.bfloat16)
        target = torch.softmax(torch.randn_like(logits), dim=-1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = F.cross_entropy(logits, target, label_smoothing=0.1)
            with torch.backends.python_native.helion.disabled():
                expected = F.cross_entropy(logits, target, label_smoothing=0.1)
        self.assertEqual(actual, expected)

    def test_preserves_existing_autograd_registration(self, device):
        script = textwrap.dedent(
            """\
            import torch
            import torch.nn.functional as F

            torch.backends.python_native.helion.disable()
            lib = torch.library.Library("aten", "IMPL", "AutogradCUDA")
            lib.impl(
                "cross_entropy_loss",
                lambda self, target, weight=None, reduction=1,
                       ignore_index=-100, label_smoothing=0.0:
                    torch.full((), 456.0, device=self.device),
                allow_override=True,
            )
            torch.backends.python_native.helion.enable()
            has_autocast = torch._C._dispatch_has_kernel_for_dispatch_key(
                "aten::cross_entropy_loss", "AutocastCUDA"
            )
            logits = torch.randn(2, 3, device="cuda")
            labels = torch.tensor([0, 1], device="cuda")
            value = F.cross_entropy(logits, labels).item()
            torch.backends.python_native.helion.disable()
            has_autograd_after = torch._C._dispatch_has_kernel_for_dispatch_key(
                "aten::cross_entropy_loss", "AutogradCUDA"
            )
            has_autocast_after = torch._C._dispatch_has_kernel_for_dispatch_key(
                "aten::cross_entropy_loss", "AutocastCUDA"
            )
            print(value, has_autocast, has_autograd_after, has_autocast_after)
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "456.0 True True False")

    def test_preserves_existing_autocast_registration(self, device):
        script = textwrap.dedent(
            """\
            import torch
            import torch.nn.functional as F

            torch.backends.python_native.helion.disable()
            lib = torch.library.Library("aten", "IMPL", "AutocastCUDA")
            lib.impl(
                "cross_entropy_loss",
                lambda self, target, weight=None, reduction=1,
                       ignore_index=-100, label_smoothing=0.0:
                    torch.full((), 456.0, device=self.device),
                allow_override=True,
            )
            torch.backends.python_native.helion.enable()
            has_autograd = torch._C._dispatch_has_kernel_for_dispatch_key(
                "aten::cross_entropy_loss", "AutogradCUDA"
            )
            logits = torch.randn(2, 3, device="cuda", requires_grad=True)
            labels = torch.tensor([0, 1], device="cuda")
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                autocast_result = F.cross_entropy(logits.detach(), labels)
            torch.backends.python_native.helion.disable()
            has_autograd_after = torch._C._dispatch_has_kernel_for_dispatch_key(
                "aten::cross_entropy_loss", "AutogradCUDA"
            )
            has_autocast_after = torch._C._dispatch_has_kernel_for_dispatch_key(
                "aten::cross_entropy_loss", "AutocastCUDA"
            )
            print(
                has_autograd,
                logits.grad is not None,
                autocast_result.item(),
                has_autograd_after,
                has_autocast_after,
            )
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "True True 456.0 False True")

    def test_autocast_uses_current_cuda_registration(self, device):
        script = textwrap.dedent(
            """\
            import torch
            import torch.nn.functional as F

            lib = torch.library.Library("aten", "IMPL", "CUDA")
            lib.impl(
                "cross_entropy_loss",
                lambda self, target, weight=None, reduction=1,
                       ignore_index=-100, label_smoothing=0.0:
                    torch.full((), 456.0, device=self.device),
                allow_override=True,
            )
            logits = torch.randn(2, 3, device="cuda")
            labels = torch.tensor([0, 1], device="cuda")
            plain = F.cross_entropy(logits, labels)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                autocast = F.cross_entropy(logits, labels)
            print(plain.item(), autocast.item())
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "456.0 456.0")

    def test_autocast_tensor_subclass_uses_current_cuda_registration(self, device):
        script = textwrap.dedent(
            """\
            import torch
            import torch.nn.functional as F
            from torch.utils._pytree import tree_map

            class Wrapper(torch.Tensor):
                elem: torch.Tensor
                __slots__ = ["elem"]

                @staticmethod
                def __new__(cls, elem):
                    out = torch.Tensor._make_wrapper_subclass(
                        cls,
                        elem.size(),
                        dtype=elem.dtype,
                        layout=elem.layout,
                        device=elem.device,
                        requires_grad=elem.requires_grad,
                        strides=elem.stride(),
                        storage_offset=elem.storage_offset(),
                    )
                    out.elem = elem
                    return out

                def __tensor_flatten__(self):
                    return ["elem"], None

                @staticmethod
                def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
                    return Wrapper(inner_tensors["elem"])

                @classmethod
                def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                    kwargs = kwargs or {}
                    unwrap = lambda value: value.elem if isinstance(value, Wrapper) else value
                    result = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
                    return tree_map(
                        lambda value: Wrapper(value)
                        if isinstance(value, torch.Tensor)
                        else value,
                        result,
                    )

            lib = torch.library.Library("aten", "IMPL", "CUDA")
            lib.impl(
                "cross_entropy_loss",
                lambda self, target, weight=None, reduction=1,
                       ignore_index=-100, label_smoothing=0.0:
                    torch.full((), 456.0, device=self.device),
                allow_override=True,
            )
            logits = Wrapper(torch.randn(2, 3, device="cuda"))
            labels = torch.tensor([0, 1], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = F.cross_entropy(logits, labels)
            print(result.elem.item())
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "456.0")

    def test_condition_rejects_unsupported_contracts(self, device):
        logits, labels = self._inputs(device)
        with self.subTest("weight"):
            self.assertFalse(
                _cross_entropy_cond(
                    logits,
                    labels,
                    torch.ones(32000, device=device, dtype=torch.bfloat16),
                )
            )
        with self.subTest("reduction"):
            self.assertFalse(_cross_entropy_cond(logits, labels, reduction=0))
        with self.subTest("ignore_index"):
            self.assertFalse(_cross_entropy_cond(logits, labels, ignore_index=-1))
        with self.subTest("label_smoothing"):
            self.assertFalse(_cross_entropy_cond(logits, labels, label_smoothing=0.1))
        with self.subTest("requires_grad"):
            self.assertFalse(_cross_entropy_cond(logits.requires_grad_(), labels))
        logits.requires_grad_(False)
        with self.subTest("target_dtype"):
            self.assertFalse(_cross_entropy_cond(logits, labels.to(torch.int32)))
        with self.subTest("target_layout"):
            labels_storage = torch.empty(
                labels.numel() * 2, device=device, dtype=torch.int64
            )
            self.assertFalse(_cross_entropy_cond(logits, labels_storage[::2]))
        with self.subTest("shape"):
            small_logits = torch.randn(32, 128, device=device, dtype=torch.bfloat16)
            small_labels = torch.randint(128, (32,), device=device)
            self.assertFalse(_cross_entropy_cond(small_logits, small_labels))
        with self.subTest("architecture"):
            with patch(
                "torch._native.ops.cross_entropy.helion_impl._is_sm100",
                return_value=False,
            ):
                self.assertFalse(_cross_entropy_cond(logits, labels))

    @parametrize("shape", ((2048, 32000), (4096, 32000)))
    def test_condition_rejects_performance_gated_shape(self, device, shape):
        logits = torch.randn(shape, device=device, dtype=torch.bfloat16)
        labels = torch.randint(shape[1], (shape[0],), device=device)
        self.assertFalse(_cross_entropy_cond(logits, labels))

    def test_misaligned_contiguous_inputs_fall_through(self, device):
        logits, labels = self._inputs(device)
        logits_storage = torch.empty(
            logits.numel() + 1, device=device, dtype=logits.dtype
        )
        misaligned_logits = logits_storage[1:].view_as(logits)
        self.assertTrue(misaligned_logits.is_contiguous())
        self.assertFalse(_cross_entropy_cond(misaligned_logits, labels))

        labels_storage = torch.empty(
            labels.numel() + 1, device=device, dtype=labels.dtype
        )
        misaligned_labels = labels_storage[1:]
        self.assertTrue(misaligned_labels.is_contiguous())
        self.assertFalse(_cross_entropy_cond(logits, misaligned_labels))

    def test_cow_inputs_do_not_materialize(self, device):
        from torch._native.ops.cross_entropy.helion_kernel import (
            cross_entropy,
            validate_labels_and_count,
        )

        logits, labels = self._inputs(device)
        cow_logits = logits._lazy_clone()
        cow_labels = labels._lazy_clone()
        self.assertTrue(_cross_entropy_cond(cow_logits, cow_labels))
        kernels = (validate_labels_and_count.helion_kernel, cross_entropy.helion_kernel)
        for kernel in kernels:
            kernel.reset()
        try:
            with (
                log_settings("+native_dsl_compile"),
                self.assertLogs("torch._native.instrumentation", level="INFO") as logs,
            ):
                actual = F.cross_entropy(cow_logits, cow_labels)
        finally:
            for kernel in kernels:
                kernel.reset()
        self.assertTrue(torch._C._is_cow_tensor(cow_logits))
        self.assertTrue(torch._C._is_cow_tensor(cow_labels))
        self.assertEqual(len(logs.output), 2)
        self.assertTrue(all("compiled" in message for message in logs.output))
        with torch.backends.python_native.helion.disabled():
            expected = F.cross_entropy(logits, labels)
        self.assertEqual(actual, expected, rtol=1e-2, atol=1e-2)

    @parametrize("case", ("weight", "reduction", "ignore_index", "label_smoothing"))
    def test_unsupported_arguments_fall_through(self, device, case):
        logits, labels = self._inputs(device)
        kwargs = {
            "weight": {"weight": torch.ones(32000, device=device, dtype=logits.dtype)},
            "reduction": {"reduction": "none"},
            "ignore_index": {"ignore_index": -1},
            "label_smoothing": {"label_smoothing": 0.1},
        }[case]
        reduction = {"none": 0, "mean": 1, "sum": 2}.get(kwargs.get("reduction"), 1)
        self.assertFalse(
            _cross_entropy_cond(
                logits,
                labels,
                kwargs.get("weight"),
                reduction,
                kwargs.get("ignore_index", -100),
                kwargs.get("label_smoothing", 0.0),
            )
        )
        actual = F.cross_entropy(logits, labels, **kwargs)
        with torch.backends.python_native.helion.disabled():
            expected = F.cross_entropy(logits, labels, **kwargs)
        self.assertEqual(actual, expected, rtol=1e-2, atol=1e-2)

    def test_correctness_with_ignored_nonfinite_rows(self, device):
        logits, labels = self._inputs(device)
        labels[::17] = -100
        logits[::51] = float("nan")
        logits[17::51] = float("inf")
        logits[34::51] = -float("inf")
        self.assertTrue(_cross_entropy_cond(logits, labels))
        actual = F.cross_entropy(logits, labels)
        with torch.backends.python_native.helion.disabled():
            expected = F.cross_entropy(logits, labels)
        self.assertEqual(actual, expected, rtol=1e-2, atol=1e-2)

    @parametrize("nonignored_count", (7, 31, 127, 2047))
    def test_sparse_nonignored_rows_match_eager(self, device, nonignored_count):
        torch.manual_seed(9173)
        logits, labels = self._inputs(device)
        labels[nonignored_count:] = -100
        actual = F.cross_entropy(logits, labels)
        with torch.backends.python_native.helion.disabled():
            expected = F.cross_entropy(logits, labels)
        self.assertEqual(actual, expected, rtol=0, atol=0)

    def test_all_labels_ignored(self, device):
        logits, labels = self._inputs(device)
        labels.fill_(-100)
        self.assertTrue(_cross_entropy_cond(logits, labels))
        actual = F.cross_entropy(logits, labels)
        self.assertTrue(torch.isnan(actual))

    @parametrize("invalid_label", (32000, -101))
    def test_invalid_target_raises(self, device, invalid_label):
        logits, labels = self._inputs(device)
        labels[0] = invalid_label
        self.assertTrue(_cross_entropy_cond(logits, labels))

        script = textwrap.dedent(
            """\
            import sys

            import torch
            import torch.nn.functional as F

            device = torch.device("cuda", int(sys.argv[1]))
            torch.cuda.set_device(device)
            logits = torch.randn(8192, 32000, device=device, dtype=torch.bfloat16)
            labels = torch.zeros(8192, device=device, dtype=torch.int64)
            labels[0] = int(sys.argv[2])
            F.cross_entropy(logits, labels)
            torch.cuda.synchronize()
            """
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(torch.cuda.current_device()),
                str(invalid_label),
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("device-side assert triggered", result.stderr)

    def test_requires_grad_falls_through(self, device):
        logits, labels = self._inputs(device, requires_grad=True)
        self.assertFalse(_cross_entropy_cond(logits, labels))
        loss = F.cross_entropy(logits, labels)
        self.assertIsNotNone(loss.grad_fn)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_forward_ad_falls_through(self, device):
        logits, labels = self._inputs(device)
        tangent = torch.randn_like(logits)
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(logits, tangent)
            self.assertFalse(_cross_entropy_cond(dual, labels))
            output = F.cross_entropy(dual, labels)
            _, output_tangent = torch.autograd.forward_ad.unpack_dual(output)
            self.assertIsNotNone(output_tangent)


instantiate_parametrized_tests(TestHelionCrossEntropyAOTConfig)
instantiate_device_type_tests(TestHelionCrossEntropy, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
