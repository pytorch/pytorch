# Owner(s): ["module: ci"]

import functools
import importlib
import os
import platform
import sys
import sysconfig
import unittest.mock

import torch
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    subtest,
    TestCase,
    TestEnvironment,
)


class TestReportHelpers(TestCase):
    @skipIfTorchDynamo("Dynamo calls the patched torch._C function while tracing")
    def test_capture_skips_torch_accelerator(self) -> None:
        environment = importlib.import_module(
            "torch.testing._internal.torchci.environment"
        )
        with unittest.mock.patch.object(
            torch._C, "_accelerator_getAccelerator"
        ) as probe:
            environment.capture()
        probe.assert_not_called()

    def test_rocm_version(self):
        environment = importlib.import_module(
            "torch.testing._internal.torchci.environment"
        )
        with unittest.mock.patch.multiple(
            torch.version, cuda=None, hip="7.16.26385", rocm="10.1.0"
        ):
            # The ROCm release, not HIP's version.
            self.assertEqual(environment._accelerator(), ("rocm", "10.1.0"))

    @parametrize(
        "config, expected",
        [
            # clang defines __GNUC__, so a clang build also prints a GCC line.
            subtest(
                (
                    "  - GCC 4.2\n  - C++ Version: 201703\n  - clang 21.1.0\n",
                    ("clang", "21"),
                ),
                name="clang",
            ),
            subtest(
                ("  - GCC 11.4\n  - C++ Version: 201703\n", ("gcc", "11")), name="gcc"
            ),
            # _MSC_FULL_VER of MSVC 19.41.34120.
            subtest(
                ("  - C++ Version: 201703\n  - MSVC 194134120\n", ("msvc", "19")),
                name="msvc",
            ),
            subtest(("  - C++ Version: 201703\n", ("", "")), name="unknown"),
        ],
    )
    def test_compiler(self, config, expected):
        environment = importlib.import_module(
            "torch.testing._internal.torchci.environment"
        )
        self.assertEqual(
            environment._compiler(f"PyTorch built with:\n{config}"), expected
        )

    def test_windows_os_version(self):
        environment = importlib.import_module(
            "torch.testing._internal.torchci.environment"
        )
        with (
            unittest.mock.patch.object(platform, "system", return_value="Windows"),
            unittest.mock.patch.object(platform, "version", return_value="10.0.17763"),
        ):
            # Server 2019's build, not the 10.0 shared by every Windows since 10.
            self.assertEqual(environment._os(), ("windows", "10.0.17763", "10.0.17763"))

    @skipIfTorchDynamo("environment capture does not need Dynamo coverage")
    def test_capture(self) -> None:
        environment = importlib.import_module(
            "torch.testing._internal.torchci.environment"
        )
        captured = environment.capture()
        env = captured.identity()
        self.assertEqual(
            list(env),
            [
                "os",
                "os_version",
                "cpu_architecture",
                "cpu_capability",
                "python_version",
                "cc_compiler",
                "cc_compiler_version",
                "accelerator",
                "accelerator_version",
                "device_count",
                "device_name",
            ],
        )
        for name, value in env.items():
            self.assertIsInstance(value, int if name == "device_count" else str)
        self.assertEqual(
            env["os"],
            {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}[
                platform.system()
            ],
        )
        free_threaded = "t" if sysconfig.get_config_var("Py_GIL_DISABLED") else ""
        self.assertEqual(
            env["python_version"],
            f"{sys.version_info.major}.{sys.version_info.minor}{free_threaded}",
        )
        self.assertIn(env["cc_compiler"], ("", "gcc", "clang", "msvc"))
        self.assertIn(env["accelerator"], ("cpu", "cuda", "rocm", "xpu", "mps"))
        if env["device_count"] == 0:
            self.assertEqual(env["device_name"], "")
        self.assertEqual(captured.flags, TestEnvironment.env_var_values)
        property_names = {
            "torch_version",
            "os_release",
            "device_memory_mib",
            "driver_version",
            "host_memory_mib",
            "build_environment",
            "test_config",
            "runner_name",
        }
        self.assertTrue(set(captured.properties) <= property_names)
        self.assertTrue(
            all(
                isinstance(value, str) and value
                for value in captured.properties.values()
            )
        )


instantiate_parametrized_tests(TestReportHelpers)


class TestEnvVarValues(TestCase):
    """TestEnvironment.env_var_values, which test run reports record as flags."""

    def setUp(self):
        super().setUp()
        self._defined: list[str] = []

    def tearDown(self):
        for name in self._defined:
            # def_flag and def_setting also bind the name in common_utils.
            if hasattr(common_utils, name):
                delattr(common_utils, name)
            TestEnvironment.env_var_values.pop(name, None)
            TestEnvironment.repro_env_vars.pop(name, None)
        super().tearDown()

    def _def_flag(self, name, **kwargs):
        self._defined.append(name)
        kwargs.setdefault("include_in_repro", False)
        return TestEnvironment.def_flag(name, **kwargs)

    def _def_setting(self, name, **kwargs):
        self._defined.append(name)
        return TestEnvironment.def_setting(name, **kwargs)

    def test_env_var_values(self):
        # What test run reports record: each env var's value as set, "" if unset,
        # an implied flag as "1", and include_in_repro=False ones left out.
        env = {k: v for k, v in os.environ.items() if not k.startswith("FOO_EV_")}
        with unittest.mock.patch.dict(
            os.environ,
            env | {"FOO_EV_SET": "1", "FOO_EV_ZERO": "0", "FOO_EV_STR": "triton"},
            clear=True,
        ):
            self._def_flag("FOO_EV_SET", env_var="FOO_EV_SET", include_in_repro=True)
            self._def_flag("FOO_EV_ZERO", env_var="FOO_EV_ZERO", include_in_repro=True)
            self._def_flag(
                "FOO_EV_IMPLIED",
                env_var="FOO_EV_IMPLIED",
                include_in_repro=True,
                implied_by_fn=lambda: True,
            )
            self._def_flag("FOO_EV_OFF", env_var="FOO_EV_OFF", include_in_repro=True)
            self._def_flag(
                "FOO_EV_EXCLUDED", env_var="FOO_EV_EXCLUDED", implied_by_fn=lambda: True
            )
            self._def_setting("FOO_EV_STR", env_var="FOO_EV_STR")
            self._def_setting("FOO_EV_UNSET", env_var="FOO_EV_UNSET")
        values = {
            k: v
            for k, v in TestEnvironment.env_var_values.items()
            if k.startswith("FOO_EV_")
        }
        self.assertEqual(
            values,
            {
                "FOO_EV_SET": "1",
                "FOO_EV_ZERO": "0",
                "FOO_EV_IMPLIED": "1",
                "FOO_EV_OFF": "",
                "FOO_EV_STR": "triton",
                "FOO_EV_UNSET": "",
            },
        )
        # The repro command only needs what was set explicitly.
        repro = {
            k: v
            for k, v in TestEnvironment.repro_env_vars.items()
            if k.startswith("FOO_EV_")
        }
        self.assertEqual(repro, {"FOO_EV_SET": "1", "FOO_EV_STR": "triton"})

    def test_env_var_values_are_unparsed(self):
        # Like EXPANDABLE_SEGMENTS, which reads PYTORCH_CUDA_ALLOC_CONF, and
        # OPINFO_RESTRICT_TO_DSL: the env var's string, not the flag's bool or the
        # setting's parsed value.
        conf = "garbage_collection_threshold:0.6,expandable_segments:True"
        with unittest.mock.patch.dict(
            os.environ,
            {"FOO_EV_ALLOC_CONF": conf, "FOO_EV_DSL": "triton", "FOO_EV_INT": "08"},
        ):
            enabled_fn = functools.partial(
                common_utils.allocator_option_enabled_fn, option="expandable_segments"
            )
            self.assertTrue(
                self._def_flag(
                    "FOO_EV_ALLOC_CONF",
                    env_var="FOO_EV_ALLOC_CONF",
                    include_in_repro=True,
                    enabled_fn=enabled_fn,
                )
            )
            self.assertEqual(
                self._def_setting(
                    "FOO_EV_DSL",
                    env_var="FOO_EV_DSL",
                    parse_fn=lambda val: None if val is None else str(val),
                ),
                "triton",
            )
            self.assertEqual(
                self._def_setting(
                    "FOO_EV_INT",
                    env_var="FOO_EV_INT",
                    parse_fn=lambda val: None if val is None else int(val),
                ),
                8,
            )
        values = {
            k: v
            for k, v in TestEnvironment.env_var_values.items()
            if k.startswith("FOO_EV_")
        }
        self.assertEqual(
            values,
            {"FOO_EV_ALLOC_CONF": conf, "FOO_EV_DSL": "triton", "FOO_EV_INT": "08"},
        )


if __name__ == "__main__":
    run_tests()
