# Owner(s): ["module: inductor"]

import subprocess
import sys
import textwrap
from unittest.mock import patch

from torch._inductor.heuristics.registry import (
    _HEURISTIC_CACHE,
    _HEURISTIC_REGISTRY,
    CodegenConfigHeuristics,
    get_codegen_heuristic,
    register_codegen_heuristic,
)
from torch._inductor.heuristics.triton_codegen.pointwise import (
    PointwiseHeuristic,
    ROCmPointwiseHeuristic,
    XPUPointwiseHeuristic,
)
from torch._inductor.runtime.hints import DeviceProperties, TileHint, TRITON_MAX_BLOCK
from torch._inductor.runtime.triton_heuristics import pointwise, triton_config
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def _recorded_config(size_hints, *block_sizes, **settings):
    return (size_hints, block_sizes, settings)


class _RecordingConfigFactory:
    def __init__(self):
        self.configs = []

    def __call__(self, size_hints, *block_sizes, **settings):
        config = _recorded_config(size_hints, *block_sizes, **settings)
        self.configs.append(config)
        return config


@instantiate_parametrized_tests
class TestCodegenHeuristicsRegistry(TestCase):
    def setUp(self):
        super().setUp()
        self.original_registry = _HEURISTIC_REGISTRY.copy()
        self.original_cache = _HEURISTIC_CACHE.copy()

    def tearDown(self):
        _HEURISTIC_REGISTRY.clear()
        _HEURISTIC_REGISTRY.update(self.original_registry)
        _HEURISTIC_CACHE.clear()
        _HEURISTIC_CACHE.update(self.original_cache)
        super().tearDown()

    def test_device_specific_registration_wins(self):
        name = "test_codegen_device_specific"

        @register_codegen_heuristic(name)
        class DefaultHeuristic(CodegenConfigHeuristics):
            pass

        @register_codegen_heuristic(name, "cuda")
        class DeviceHeuristic(CodegenConfigHeuristics):
            pass

        self.assertIsInstance(get_codegen_heuristic(name, "cuda"), DeviceHeuristic)

    def test_unmatched_device_falls_back_to_default(self):
        name = "test_codegen_default_fallback"

        @register_codegen_heuristic(name)
        class DefaultHeuristic(CodegenConfigHeuristics):
            pass

        self.assertIsInstance(get_codegen_heuristic(name, "xpu"), DefaultHeuristic)

    def test_lookup_reuses_cached_instance(self):
        name = "test_codegen_cache"

        @register_codegen_heuristic(name)
        class TestHeuristic(CodegenConfigHeuristics):
            pass

        first = get_codegen_heuristic(name, "cuda")
        second = get_codegen_heuristic(name, "cuda")
        self.assertIs(first, second)

    def test_register_false_does_not_add_entry(self):
        name = "test_codegen_disabled_registration"

        @register_codegen_heuristic(name, "cuda", register=False)
        class TestHeuristic(CodegenConfigHeuristics):
            pass

        self.assertNotIn((name, "cuda", None), _HEURISTIC_REGISTRY)

    def test_unknown_name_reports_name_and_device(self):
        name = "test_codegen_missing"
        with self.assertRaisesRegex(
            ValueError, rf"name={name}.*device_type=made_up_device"
        ):
            get_codegen_heuristic(name, "made_up_device")

    def test_lazy_registration(self):
        source = textwrap.dedent(
            """
            import importlib
            import sys

            registry = importlib.import_module("torch._inductor.heuristics.registry")
            package_name = "torch._inductor.heuristics.triton_codegen"
            if package_name in sys.modules:
                raise AssertionError(f"{package_name} was imported eagerly")

            heuristic = registry.get_codegen_heuristic("pointwise", "cuda")
            heuristic_type = type(heuristic)
            expected_module = "torch._inductor.heuristics.triton_codegen.pointwise"
            if heuristic_type.__name__ != "PointwiseHeuristic":
                raise AssertionError(
                    f"expected PointwiseHeuristic, got {heuristic_type.__name__}"
                )
            if heuristic_type.__module__ != expected_module:
                raise AssertionError(
                    f"expected module {expected_module}, got {heuristic_type.__module__}"
                )
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", source],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


class _RecordingHeuristic:
    def __init__(self, configs):
        self.configs = configs
        self.calls = []

    def get_configs(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.configs


@instantiate_parametrized_tests
class TestPointwiseDelegation(TestCase):
    def test_runtime_pointwise_delegates_to_registry(self):
        size_hints = {"x": 65536}
        device = DeviceProperties(
            type="synthetic",
            index=0,
            multi_processor_count=1,
            cc=0,
            warp_size=16,
        )
        triton_meta = {"device": device}
        inductor_meta = {"autotune_pointwise": True}
        hinted_configs = [object(), object()]
        returned_configs = [object(), object()]
        heuristic = _RecordingHeuristic(returned_configs)

        with (
            patch(
                "torch._inductor.runtime.triton_heuristics.autotune_hints_to_configs",
                return_value=hinted_configs,
            ) as hints_to_configs,
            patch(
                "torch._inductor.heuristics.registry.get_codegen_heuristic",
                return_value=heuristic,
            ) as get_heuristic,
        ):
            result = pointwise(
                size_hints,
                triton_meta=triton_meta,
                tile_hint=TileHint.DEFAULT,
                min_elem_per_thread=4,
                inductor_meta=inductor_meta,
                return_configs=True,
            )

        get_heuristic.assert_called_once_with("pointwise", "synthetic")
        hints_to_configs.assert_called_once()
        self.assertEqual(len(heuristic.calls), 1)
        args, kwargs = heuristic.calls[0]
        self.assertIs(args[0], size_hints)
        self.assertEqual(args[1], 512)
        self.assertIs(args[3], hinted_configs)
        self.assertIs(args[2].func, triton_config)
        self.assertEqual(args[2].keywords, {"min_elem_per_thread": 4, "warp_size": 16})
        self.assertEqual(kwargs["tile_hint"], TileHint.DEFAULT)
        self.assertIs(kwargs["inductor_meta"], inductor_meta)
        self.assertIs(result, returned_configs)


@instantiate_parametrized_tests
class TestPointwiseHeuristic(TestCase):
    def test_1d_autotune_disabled(self):
        size_hints = {"x": 8192}
        factory = _RecordingConfigFactory()
        hints = ["hint"]

        result = PointwiseHeuristic().get_configs(
            size_hints,
            1024,
            factory,
            hints,
            inductor_meta={"autotune_pointwise": False},
        )

        self.assertEqual(result, [_recorded_config(size_hints, 1024)])
        self.assertEqual(factory.configs, result)

    def test_1d_default_autotune_order(self):
        size_hints = {"x": 8192}
        factory = _RecordingConfigFactory()
        hints = ["hint-0", "hint-1"]

        result = PointwiseHeuristic().get_configs(size_hints, 1024, factory, hints)

        self.assertEqual(
            result,
            [
                _recorded_config(size_hints, 1024, num_elements_per_warp=256),
                _recorded_config(size_hints, 512, num_elements_per_warp=64),
                *hints,
            ],
        )
        self.assertEqual(factory.configs, result[:2])

    @parametrize("max_flag", ("max_autotune", "max_autotune_pointwise"))
    def test_1d_max_autotune_overrides_disabled(self, max_flag):
        size_hints = {"x": 8192}
        factory = _RecordingConfigFactory()
        metadata = {"autotune_pointwise": False, max_flag: True}

        result = PointwiseHeuristic().get_configs(
            size_hints, 1024, factory, [], inductor_meta=metadata
        )

        self.assertEqual(
            result,
            [
                _recorded_config(size_hints, 1024, num_elements_per_warp=256),
                _recorded_config(size_hints, 512, num_elements_per_warp=64),
            ],
        )

    @parametrize(
        "autotune_pointwise,tile_hint",
        ((False, None), (True, TileHint.SQUARE)),
    )
    def test_2d_shortcut(self, autotune_pointwise, tile_hint):
        size_hints = {"x": 1024, "y": 1024}
        factory = _RecordingConfigFactory()

        result = PointwiseHeuristic().get_configs(
            size_hints,
            1024,
            factory,
            ["hint"],
            tile_hint=tile_hint,
            inductor_meta={"autotune_pointwise": autotune_pointwise},
        )

        self.assertEqual(result, [_recorded_config(size_hints, 32, 32)])

    def test_2d_full_order(self):
        size_hints = {"x": 4096, "y": 4096}
        factory = _RecordingConfigFactory()
        hints = ["hint-0", "hint-1"]

        result = PointwiseHeuristic().get_configs(size_hints, 1024, factory, hints)

        self.assertEqual(
            result,
            [
                _recorded_config(size_hints, 32, 32),
                _recorded_config(size_hints, 64, 64),
                _recorded_config(size_hints, 256, 16),
                _recorded_config(size_hints, 16, 256),
                _recorded_config(size_hints, 1024, 1),
                _recorded_config(size_hints, 1, 1024),
                *hints,
            ],
        )
        self.assertEqual(factory.configs, result[:6])

    @parametrize("max_flag", ("max_autotune", "max_autotune_pointwise"))
    def test_2d_max_autotune_overrides_square_shortcut(self, max_flag):
        size_hints = {"x": 4096, "y": 4096}
        factory = _RecordingConfigFactory()
        metadata = {max_flag: True}

        result = PointwiseHeuristic().get_configs(
            size_hints,
            1024,
            factory,
            [],
            tile_hint=TileHint.SQUARE,
            inductor_meta=metadata,
        )

        self.assertEqual(len(result), 6)
        self.assertEqual(result[1], _recorded_config(size_hints, 64, 64))

    def test_3d_without_max_autotune(self):
        size_hints = {"x": 256, "y": 256, "z": 256}
        factory = _RecordingConfigFactory()

        result = PointwiseHeuristic().get_configs(
            size_hints, 1024, factory, ["hint"], inductor_meta={}
        )

        self.assertEqual(result, [_recorded_config(size_hints, 16, 16, 16)])

    @parametrize("max_flag", ("max_autotune", "max_autotune_pointwise"))
    def test_3d_max_autotune_order(self, max_flag):
        size_hints = {"x": 4096, "y": 4096, "z": 4096}
        factory = _RecordingConfigFactory()
        hints = ["hint-0", "hint-1"]

        result = PointwiseHeuristic().get_configs(
            size_hints,
            1024,
            factory,
            hints,
            inductor_meta={max_flag: True},
        )

        self.assertEqual(
            result,
            [
                _recorded_config(size_hints, 16, 16, 16),
                _recorded_config(size_hints, 64, 8, 8),
                _recorded_config(size_hints, 8, 64, 8),
                _recorded_config(size_hints, 8, 8, 64),
                _recorded_config(size_hints, 1024, 1, 1),
                _recorded_config(size_hints, 1, 1024, 1),
                _recorded_config(size_hints, 1, 1, 1024),
                *hints,
            ],
        )
        self.assertEqual(factory.configs, result[:7])

    @parametrize("size_hints", ({}, {"w": 1, "x": 2, "y": 3, "z": 4}))
    def test_unsupported_rank(self, size_hints):
        with self.assertRaisesRegex(NotImplementedError, str(size_hints)):
            PointwiseHeuristic().get_configs(
                size_hints, 1024, _RecordingConfigFactory(), []
            )


@instantiate_parametrized_tests
class TestBackendPointwiseHeuristics(TestCase):
    @parametrize("backend", ("rocm", "xpu"))
    def test_backend_1d_configs(self, backend):
        size_hints = {"x": 16384}
        factory = _RecordingConfigFactory()
        hints = ["hint"]
        heuristic = {
            "rocm": ROCmPointwiseHeuristic,
            "xpu": XPUPointwiseHeuristic,
        }[backend]()

        disabled = heuristic.get_configs(
            size_hints,
            1024,
            factory,
            hints,
            inductor_meta={"autotune_pointwise": False},
        )
        self.assertEqual(disabled, [_recorded_config(size_hints, 1024)])

        factory = _RecordingConfigFactory()
        enabled = heuristic.get_configs(size_hints, 1024, factory, hints)
        base = [
            _recorded_config(size_hints, 1024, num_elements_per_warp=256),
            _recorded_config(size_hints, 512, num_elements_per_warp=64),
            *hints,
        ]
        extras = {
            "rocm": [
                _recorded_config(size_hints, TRITON_MAX_BLOCK["X"], waves_per_eu=2),
                _recorded_config(size_hints, 4096),
                _recorded_config(
                    size_hints,
                    2048,
                    num_warps=8,
                    num_stages=2,
                    waves_per_eu=1,
                ),
            ],
            "xpu": [_recorded_config(size_hints, 32)],
        }[backend]
        self.assertEqual(enabled, [*base, *extras])

    @parametrize("atomic_add_found", (False, True))
    def test_rocm_disabled_1d_ignores_atomic_add(self, atomic_add_found):
        size_hints = {"x": 16384}
        metadata = {
            "autotune_pointwise": False,
            "atomic_add_found": atomic_add_found,
        }

        result = ROCmPointwiseHeuristic().get_configs(
            size_hints,
            1024,
            _RecordingConfigFactory(),
            ["hint"],
            inductor_meta=metadata,
        )

        self.assertEqual(result, [_recorded_config(size_hints, 1024)])

    def test_rocm_atomic_add_config_requires_full_autotuning(self):
        size_hints = {"x": 16384}
        factory = _RecordingConfigFactory()

        result = ROCmPointwiseHeuristic().get_configs(
            size_hints,
            1024,
            factory,
            [],
            inductor_meta={"atomic_add_found": True},
        )

        self.assertEqual(
            result[-1],
            _recorded_config(size_hints, 64, num_warps=1, num_stages=1),
        )
        self.assertEqual(len(result), 6)

    @parametrize("backend", ("rocm", "xpu"))
    def test_backend_2d_configs(self, backend):
        size_hints = {"x": 4096, "y": 4096}
        hints = ["hint"]
        heuristic = {
            "rocm": ROCmPointwiseHeuristic,
            "xpu": XPUPointwiseHeuristic,
        }[backend]()

        disabled = heuristic.get_configs(
            size_hints,
            1024,
            _RecordingConfigFactory(),
            hints,
            inductor_meta={"autotune_pointwise": False},
        )
        self.assertEqual(disabled, [_recorded_config(size_hints, 32, 32)])

        enabled = heuristic.get_configs(
            size_hints,
            1024,
            _RecordingConfigFactory(),
            hints,
            tile_hint=TileHint.SQUARE,
        )
        base = [
            _recorded_config(size_hints, 32, 32),
            _recorded_config(size_hints, 64, 64),
            _recorded_config(size_hints, 256, 16),
            _recorded_config(size_hints, 16, 256),
            _recorded_config(size_hints, 1024, 1),
            _recorded_config(size_hints, 1, 1024),
            *hints,
        ]
        extras = {
            "rocm": [
                _recorded_config(size_hints, 64, 32),
                _recorded_config(size_hints, 128, 16),
                _recorded_config(size_hints, 128, 32),
                _recorded_config(size_hints, 32, 512),
            ],
            "xpu": [
                _recorded_config(size_hints, 32, 32, num_warps=8),
                _recorded_config(size_hints, 4, 256),
            ],
        }[backend]
        self.assertEqual(enabled, [*base, *extras])

    @parametrize("max_flag", (None, "max_autotune", "max_autotune_pointwise"))
    def test_rocm_3d_matches_default(self, max_flag):
        size_hints = {"x": 4096, "y": 4096, "z": 4096}
        metadata = {} if max_flag is None else {max_flag: True}
        hints = ["hint"]

        rocm = ROCmPointwiseHeuristic().get_configs(
            size_hints,
            1024,
            _RecordingConfigFactory(),
            hints,
            inductor_meta=metadata,
        )
        default = PointwiseHeuristic().get_configs(
            size_hints,
            1024,
            _RecordingConfigFactory(),
            hints,
            inductor_meta=metadata,
        )

        self.assertEqual(rocm, default)

    def test_xpu_3d_always_uses_full_config_set(self):
        size_hints = {"x": 4096, "y": 4096, "z": 4096}
        hints = ["hint-0", "hint-1"]

        result = XPUPointwiseHeuristic().get_configs(
            size_hints,
            1024,
            _RecordingConfigFactory(),
            hints,
            inductor_meta={
                "autotune_pointwise": False,
                "max_autotune": False,
                "max_autotune_pointwise": False,
            },
        )

        self.assertEqual(
            result,
            [
                _recorded_config(size_hints, 16, 16, 16),
                _recorded_config(size_hints, 64, 8, 8),
                _recorded_config(size_hints, 8, 64, 8),
                _recorded_config(size_hints, 8, 8, 64),
                _recorded_config(size_hints, 1024, 1, 1),
                _recorded_config(size_hints, 1, 1024, 1),
                _recorded_config(size_hints, 1, 1, 1024),
                *hints,
            ],
        )


if __name__ == "__main__":
    run_tests()
