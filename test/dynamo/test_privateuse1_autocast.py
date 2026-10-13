# Owner(s): ["module: dynamo"]

import sys
import types
import typing
import unittest
from unittest import mock

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import EagerAndRecordGraphs


BACKEND = torch._C._get_privateuse1_backend_name()


class TestPrivateUse1Autocast(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Device-module registration is process-global, so keep these tests in
        # a dedicated file and process.
        if hasattr(torch, BACKEND):
            raise unittest.SkipTest(f"torch.{BACKEND} is already registered")
        device_module = types.ModuleType(f"torch.{BACKEND}")
        device_module.amp = types.SimpleNamespace()
        device_module.get_amp_supported_dtype = lambda: [
            torch.float16,
            torch.bfloat16,
        ]
        device_module._is_in_bad_fork = lambda: False
        device_module.manual_seed_all = lambda seed: None
        torch._register_device_module(BACKEND, device_module)
        cls.device_module = device_module
        cls._registered_device_module = True

    @classmethod
    def tearDownClass(cls):
        try:
            if cls._registered_device_module:
                # _register_device_module has no unregister API; these tests
                # own their temporary registration and clean it up directly.
                delattr(torch, BACKEND)
                sys.modules.pop(f"torch.{BACKEND}", None)
        finally:
            super().tearDownClass()

    def assert_enter_autocast_args(self, backend, expected):
        nodes = [
            node
            for node in backend.graphs[0].graph.nodes
            if node.target is torch.amp._enter_autocast
        ]
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].args, expected)

    def assert_no_enter_autocast(self, graph):
        self.assertFalse(
            any(node.target is torch.amp._enter_autocast for node in graph.graph.nodes)
        )

    def test_non_autocast_fast_path_does_not_query_backend(self):
        from torch._dynamo.variables import torch as torch_variables

        with mock.patch.object(
            torch_variables, "_get_privateuse1_autocast"
        ) as get_privateuse1_autocast:
            self.assertFalse(torch_variables._is_privateuse1_autocast(object))
            get_privateuse1_autocast.assert_not_called()

    def test_registered_wrapper_routes_in_graph(self):
        class BackendAutocast(torch.amp.autocast_mode.autocast):
            _dynamo_autocast_passthrough = True

            def __init__(self, enabled=True, dtype=torch.float16, cache_enabled=True):
                super().__init__(
                    BACKEND,
                    dtype=dtype,
                    enabled=enabled,
                    cache_enabled=cache_enabled,
                )

        device_module = self.device_module
        with mock.patch.object(
            device_module.amp, "autocast", BackendAutocast, create=True
        ):
            backend = EagerAndRecordGraphs()

            @torch.compile(backend=backend, fullgraph=True)
            def fn(x):
                with device_module.amp.autocast():
                    return x + 1

            x = torch.randn(4)
            self.assertEqual(fn(x), x + 1)

        self.assert_enter_autocast_args(backend, (BACKEND, torch.float16, True, True))

    def test_registered_module_entrypoint_routes_in_graph(self):
        class BackendAutocast(torch.amp.autocast_mode.autocast):
            _dynamo_autocast_passthrough = True

            def __init__(self, enabled=True, dtype=torch.float16, cache_enabled=True):
                super().__init__(
                    BACKEND,
                    dtype=dtype,
                    enabled=enabled,
                    cache_enabled=cache_enabled,
                )

        device_module = self.device_module
        with mock.patch.object(
            device_module.amp, "autocast", BackendAutocast, create=True
        ):
            backend = EagerAndRecordGraphs()

            @torch.compile(backend=backend, fullgraph=True)
            def fn(x):
                with getattr(torch, BACKEND).amp.autocast():
                    return x + 1

            x = torch.randn(4)
            self.assertEqual(fn(x), x + 1)

        self.assert_enter_autocast_args(backend, (BACKEND, torch.float16, True, True))

    def test_wrapper_argument_binding(self):
        class BackendAutocast(torch.amp.autocast_mode.autocast):
            _dynamo_autocast_passthrough = True

            def __init__(self, enabled=True, dtype=torch.float16, cache_enabled=True):
                super().__init__(
                    BACKEND, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled
                )

        calls = [
            ((False, torch.bfloat16, False), {}),
            ((), {"dtype": torch.bfloat16, "enabled": False, "cache_enabled": False}),
        ]
        with mock.patch.object(
            self.device_module.amp, "autocast", BackendAutocast, create=True
        ):
            for args, kwargs in calls:
                with self.subTest(args=args, kwargs=kwargs):
                    torch._dynamo.reset()
                    backend = EagerAndRecordGraphs()

                    def fn(x):
                        with BackendAutocast(*args, **kwargs):
                            return x + 1, torch.is_autocast_enabled(BACKEND)

                    x = torch.randn(4)
                    expected = fn(x)
                    actual = torch.compile(backend=backend, fullgraph=True)(fn)(x)
                    self.assertEqual(actual, expected)
                    self.assert_enter_autocast_args(
                        backend, (BACKEND, torch.bfloat16, False, False)
                    )

    def test_omitted_defaults_remain_deferred(self):
        class DeferredDefaultsAutocast(torch.amp.autocast_mode.autocast):
            _dynamo_autocast_passthrough = True

            def __init__(self, dtype=None, enabled=True, cache_enabled=None):
                super().__init__(
                    BACKEND,
                    dtype=dtype,
                    enabled=enabled,
                    cache_enabled=cache_enabled,
                )

        device_module = self.device_module
        with mock.patch.object(
            device_module.amp, "autocast", DeferredDefaultsAutocast, create=True
        ):
            backend = EagerAndRecordGraphs()

            @torch.compile(backend=backend, fullgraph=True)
            def fn(x):
                with device_module.amp.autocast():
                    return x + 1

            x = torch.randn(4)
            self.assertEqual(fn(x), x + 1)

        self.assert_enter_autocast_args(backend, (BACKEND, None, True, None))

    def test_wrapper_defaults_match_eager(self):
        class WrapperDefaultsAutocast(torch.amp.autocast_mode.autocast):
            _dynamo_autocast_passthrough = True

            def __init__(
                self,
                dtype=torch.bfloat16,
                enabled=False,
                cache_enabled=None,
            ):
                super().__init__(
                    BACKEND,
                    dtype=dtype,
                    enabled=enabled,
                    cache_enabled=cache_enabled,
                )

        device_module = self.device_module
        with mock.patch.object(
            device_module.amp, "autocast", WrapperDefaultsAutocast, create=True
        ):
            backend = EagerAndRecordGraphs()

            def fn(x):
                with device_module.amp.autocast():
                    return (
                        x + 1,
                        torch.is_autocast_enabled(BACKEND),
                        torch.get_autocast_dtype(BACKEND),
                    )

            x = torch.randn(4)
            expected = fn(x)
            actual = torch.compile(backend=backend, fullgraph=True)(fn)(x)
            self.assertEqual(actual, expected)

        self.assert_enter_autocast_args(backend, (BACKEND, torch.bfloat16, False, None))

    def test_registered_entrypoint_rebind_recompiles(self):
        class RegisteredAutocast(torch.amp.autocast_mode.autocast):
            _dynamo_autocast_passthrough = True

            def __init__(self, dtype=None, enabled=True, cache_enabled=None):
                super().__init__(
                    BACKEND,
                    dtype=dtype,
                    enabled=enabled,
                    cache_enabled=cache_enabled,
                )

        class ReplacementAutocast(torch.amp.autocast_mode.autocast):
            def __init__(self, dtype=None, enabled=True, cache_enabled=None):
                super().__init__(
                    BACKEND,
                    dtype=dtype,
                    enabled=enabled,
                    cache_enabled=cache_enabled,
                )

        autocast = RegisteredAutocast

        def fn(x):
            with autocast():
                return x + 1

        backend = EagerAndRecordGraphs()
        optimized_fn = torch.compile(fn, backend=backend)
        x = torch.randn(4)
        device_module = self.device_module
        with mock.patch.object(
            device_module.amp, "autocast", RegisteredAutocast, create=True
        ):
            self.assertEqual(optimized_fn(x), x + 1)
        with mock.patch.object(
            device_module.amp, "autocast", ReplacementAutocast, create=True
        ):
            self.assertEqual(optimized_fn(x), x + 1)
        self.assertEqual(len(backend.graphs), 2)
        self.assert_enter_autocast_args(backend, (BACKEND, None, True, None))
        self.assert_no_enter_autocast(backend.graphs[1])

    def test_sourceless_registered_entrypoint_rebind_recompiles(self):
        class RegisteredAutocast(torch.amp.autocast_mode.autocast):
            _dynamo_autocast_passthrough = True

            def __init__(self, dtype=None, enabled=True, cache_enabled=None):
                super().__init__(
                    BACKEND,
                    dtype=dtype,
                    enabled=enabled,
                    cache_enabled=cache_enabled,
                )

        class ReplacementAutocast(RegisteredAutocast):
            pass

        autocast = RegisteredAutocast

        def fn(x):
            autocast_cls = typing.Annotated[autocast, "metadata"].__origin__
            with autocast_cls():
                return x + 1

        backend = EagerAndRecordGraphs()
        optimized_fn = torch.compile(fn, backend=backend)
        x = torch.randn(4)
        device_module = self.device_module
        with mock.patch.object(
            device_module.amp, "autocast", RegisteredAutocast, create=True
        ):
            self.assertEqual(optimized_fn(x), x + 1)

        self.assert_enter_autocast_args(backend, (BACKEND, None, True, None))

        with mock.patch.object(
            device_module.amp, "autocast", ReplacementAutocast, create=True
        ):
            self.assertEqual(optimized_fn(x), x + 1)

        self.assertEqual(len(backend.graphs), 2)

    def test_unsupported_wrappers_keep_generic_route(self):
        class OptedInBaseAutocast(torch.amp.autocast_mode.autocast):
            _dynamo_autocast_passthrough = True

            def __init__(self, dtype=None, enabled=True, cache_enabled=None):
                super().__init__(
                    BACKEND,
                    dtype=dtype,
                    enabled=enabled,
                    cache_enabled=cache_enabled,
                )

        class InheritedOptInAutocast(OptedInBaseAutocast):
            def __init__(self, dtype=None, enabled=True, cache_enabled=None):
                super().__init__(
                    dtype=torch.bfloat16,
                    enabled=enabled,
                    cache_enabled=cache_enabled,
                )

        class TransformedDefaultsAutocast(torch.amp.autocast_mode.autocast):
            def __init__(self, dtype=None, enabled=True, cache_enabled=None):
                super().__init__(
                    BACKEND,
                    dtype=dtype or torch.bfloat16,
                    enabled=enabled,
                    cache_enabled=cache_enabled,
                )

        class VariadicArgumentsAutocast(torch.amp.autocast_mode.autocast):
            def __init__(self, *args, dtype=None, enabled=True, cache_enabled=None):
                dtype = args[0] if args else dtype or torch.bfloat16
                enabled = args[1] if len(args) > 1 else enabled
                cache_enabled = args[2] if len(args) > 2 else cache_enabled
                super().__init__(
                    BACKEND,
                    dtype=dtype,
                    enabled=enabled,
                    cache_enabled=cache_enabled,
                )

        class OptedInVariadicArgumentsAutocast(VariadicArgumentsAutocast):
            _dynamo_autocast_passthrough = True

        class OptedInVariadicPassthroughAutocast(torch.amp.autocast_mode.autocast):
            _dynamo_autocast_passthrough = True

            def __init__(self, *args, **kwargs):
                super().__init__(BACKEND, *args, **kwargs)

        class VariadicKeywordAutocast(torch.amp.autocast_mode.autocast):
            def __init__(self, **wrapper_kwargs):
                wrapper_kwargs["enabled"] = not wrapper_kwargs.get("enabled", True)
                super().__init__(BACKEND, **wrapper_kwargs)

        class OptedInKeywordPassthroughAutocast(torch.amp.autocast_mode.autocast):
            _dynamo_autocast_passthrough = True

            def __init__(self, **kwargs):
                super().__init__(BACKEND, **kwargs)

        class DeviceTypeAutocast(torch.amp.autocast_mode.autocast):
            _dynamo_autocast_passthrough = True

        class TransformedDeviceTypeAutocast(torch.amp.autocast_mode.autocast):
            _dynamo_autocast_passthrough = True

            def __init__(
                self, device_type=BACKEND, dtype=None, enabled=True, cache_enabled=None
            ):
                super().__init__(
                    "cpu", dtype=dtype, enabled=enabled, cache_enabled=cache_enabled
                )

        class ExtraParameterAutocast(torch.amp.autocast_mode.autocast):
            _dynamo_autocast_passthrough = True

            def __init__(
                self, dtype=None, enabled=True, cache_enabled=None, force_disabled=False
            ):
                super().__init__(
                    BACKEND,
                    dtype=dtype,
                    enabled=enabled and not force_disabled,
                    cache_enabled=cache_enabled,
                )

        class NarrowedAutocast(torch.amp.autocast_mode.autocast):
            _dynamo_autocast_passthrough = True

            def __init__(self, dtype=None):
                super().__init__(BACKEND, dtype=dtype, enabled=False)

        cases = [
            (InheritedOptInAutocast, (torch.float16,), {}),
            (TransformedDefaultsAutocast, (), {}),
            (
                VariadicArgumentsAutocast,
                (torch.bfloat16, False, None),
                {},
            ),
            (
                VariadicKeywordAutocast,
                (),
                {
                    "dtype": torch.bfloat16,
                    "enabled": True,
                    "cache_enabled": False,
                },
            ),
            (
                OptedInVariadicArgumentsAutocast,
                (torch.bfloat16, False, None),
                {},
            ),
            (
                OptedInVariadicPassthroughAutocast,
                (torch.bfloat16, False, None),
                {},
            ),
            (OptedInKeywordPassthroughAutocast, (), {}),
            (DeviceTypeAutocast, ("cpu",), {}),
            (TransformedDeviceTypeAutocast, (), {}),
            (ExtraParameterAutocast, (), {"force_disabled": True}),
            (NarrowedAutocast, (), {}),
        ]
        device_module = self.device_module
        x = torch.randn(4)
        for wrapper, args, kwargs in cases:
            with self.subTest(wrapper=wrapper.__name__, kwargs=kwargs):
                torch._dynamo.reset()
                with mock.patch.object(
                    device_module.amp, "autocast", wrapper, create=True
                ):

                    def fn(x):
                        with device_module.amp.autocast(*args, **kwargs):
                            return (
                                x + 1,
                                torch.is_autocast_enabled(BACKEND),
                                torch.get_autocast_dtype(BACKEND),
                                torch.is_autocast_enabled("cpu"),
                                torch.get_autocast_dtype("cpu"),
                            )

                    expected = fn(x)
                    backend = EagerAndRecordGraphs()
                    actual = torch.compile(backend=backend, fullgraph=True)(fn)(x)
                    self.assertEqual(actual, expected)
                    self.assert_no_enter_autocast(backend.graphs[0])


if __name__ == "__main__":
    run_tests()
