# Owner(s): ["module: dynamo"]

import sys
import types
from unittest import mock

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import EagerAndRecordGraphs
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


BACKEND = torch._C._get_privateuse1_backend_name()


class TestPrivateUse1Autocast(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Device-module registration is process-global, so keep these tests in
        # a dedicated file and process.
        if hasattr(torch, BACKEND):
            raise RuntimeError(f"torch.{BACKEND} is already registered")
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

    @classmethod
    def tearDownClass(cls):
        try:
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

    def test_registered_wrapper_routes_in_graph(self):
        class BackendAutocast(torch.amp.autocast_mode.autocast):
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

    def test_omitted_defaults_remain_deferred(self):
        class MinimalAutocast(torch.amp.autocast_mode.autocast):
            def __init__(self):
                super().__init__(BACKEND)

        device_module = self.device_module
        with mock.patch.object(
            device_module.amp, "autocast", MinimalAutocast, create=True
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

    @parametrize(
        "kwargs,expected",
        [
            (
                {
                    "dtype": torch.bfloat16,
                    "enabled": True,
                    "cache_enabled": False,
                },
                (BACKEND, torch.bfloat16, True, False),
            ),
            (
                {"dtype": torch.bfloat16, "enabled": False},
                (BACKEND, torch.bfloat16, False, None),
            ),
        ],
    )
    def test_variadic_keyword_arguments(self, kwargs, expected):
        class KwargsAutocast(torch.amp.autocast_mode.autocast):
            def __init__(self, **wrapper_kwargs):
                super().__init__(BACKEND, **wrapper_kwargs)

        device_module = self.device_module
        with mock.patch.object(
            device_module.amp, "autocast", KwargsAutocast, create=True
        ):
            backend = EagerAndRecordGraphs()

            @torch.compile(backend=backend, fullgraph=True)
            def fn(x):
                with device_module.amp.autocast(**kwargs):
                    return x + 1

            x = torch.randn(4)
            self.assertEqual(fn(x), x + 1)

        self.assert_enter_autocast_args(backend, expected)

    def test_explicit_device_type_is_preserved(self):
        class DeviceTypeAutocast(torch.amp.autocast_mode.autocast):
            pass

        device_module = self.device_module
        with mock.patch.object(
            device_module.amp, "autocast", DeviceTypeAutocast, create=True
        ):
            backend = EagerAndRecordGraphs()

            @torch.compile(backend=backend, fullgraph=True)
            def fn(x):
                with device_module.amp.autocast("cpu"):
                    return x + 1

            x = torch.randn(4)
            self.assertEqual(fn(x), x + 1)

        self.assert_enter_autocast_args(backend, ("cpu", None, True, None))

    def test_registered_entrypoint_rebind_recompiles(self):
        class RegisteredAutocast(torch.amp.autocast_mode.autocast):
            def __init__(self):
                super().__init__(BACKEND)

        class ReplacementAutocast(torch.amp.autocast_mode.autocast):
            def __init__(self):
                super().__init__(BACKEND)

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


instantiate_parametrized_tests(TestPrivateUse1Autocast)


if __name__ == "__main__":
    run_tests()
