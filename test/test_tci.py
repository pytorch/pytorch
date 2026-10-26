# Owner(s): ["module: tests"]

import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import pytest

import torch
from torch.testing._internal import common_device_type as device_tests, tci
from torch.testing._internal.common_utils import (
    _TestParametrizer,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TciTestLoader,
    TestCase,
)
from torch.testing._internal.tci import gpu, platform, schedule, Unsupported


@instantiate_parametrized_tests
class TestTci(TestCase):
    def setUp(self):
        super().setUp()
        for backend in ("cuda", "xpu", "backends.mps"):
            patcher = mock.patch(f"torch.{backend}.is_available", return_value=False)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_schedule_marker_strings(self):
        self.assertEqual(str(schedule.pull), "tci.schedule:pull")
        self.assertEqual(str(schedule.periodic), "tci.schedule:periodic")

    @parametrize(
        "name,expected_gpu,expected_platform",
        [
            ("Apple M1", gpu.mps_m1, platform.macos_arm64_mps_m1),
            ("Apple M2 Pro", gpu.mps_m2, platform.macos_arm64_mps_m2),
        ],
    )
    def test_mps_detection(self, name, expected_gpu, expected_platform):
        with (
            mock.patch("sys.platform", "darwin"),
            mock.patch("platform.machine", return_value="arm64"),
            mock.patch("torch.backends.mps.is_available", return_value=True),
            mock.patch("torch.backends.mps.get_name", return_value=name),
        ):
            self.assertIs(gpu.current(), expected_gpu)
            self.assertIs(platform.current(), expected_platform)

    def test_mps_unavailable(self):
        with mock.patch("torch.backends.mps.get_name") as get_name:
            self.assertIsNone(gpu.current())
            get_name.assert_not_called()

    def test_unsupported_mps(self):
        with (
            mock.patch("torch.backends.mps.is_available", return_value=True),
            mock.patch("torch.backends.mps.get_name", return_value="Apple M3"),
        ):
            with self.assertRaisesRegex(Unsupported, "unsupported gpu: Apple M3"):
                gpu.current()
            self.assertFalse(gpu.mps_m1.is_current())

    def test_gb300_detection(self):
        with (
            mock.patch("sys.platform", "linux"),
            mock.patch("platform.machine", return_value="aarch64"),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.version.hip", None),
            mock.patch("torch.cuda.get_device_name", return_value="NVIDIA GB300"),
            mock.patch("torch.cuda.get_device_capability", return_value=(10, 3)),
        ):
            self.assertIs(gpu.current(), gpu.cuda_gb300)
            self.assertIs(platform.current(), platform.linux_arm64_cuda_gb300)


@instantiate_parametrized_tests
class TestTciDecorator(TestCase):
    def test_shared_defaults_and_cuda_exclusion(self):
        class Example(TestCase):
            def test_example(self):
                pass

        self.assertEqual(
            tci.resolve(Example.test_example, Example).platforms,
            (platform.linux_x86_64, platform.linux_x86_64_cuda_l4),
        )
        self.assertEqual(
            tci.resolve(device_tests.CUDATestBase).platforms,
            (platform.linux_x86_64_cuda_l4,),
        )

    def test_metadata_collection_keeps_skipped_parametrizations(self):
        with (
            mock.patch.object(sys, "argv", ["test_tci.py", "--list-markers"]),
            mock.patch("torch.testing._internal.common_utils.OMIT_SKIPPED_TESTS", True),
        ):

            @instantiate_parametrized_tests
            class Example(TestCase):
                @unittest.skip("dependency unavailable during collection")
                @parametrize("value", [1, 2])
                def test_example(self, value):
                    pass

        names = unittest.defaultTestLoader.getTestCaseNames(Example)
        self.assertEqual(names, ["test_example_value_1", "test_example_value_2"])
        self.assertTrue(Example.test_example_value_1.__unittest_skip__)

    @parametrize(
        "platforms",
        [
            platform.linux_x86_64_cuda_t4,
            (platform.linux_x86_64_cuda_t4, platform.linux_x86_64_cuda_l4),
        ],
    )
    def test_markers_preserve_function(self, platforms):
        def target():
            return "called"

        decorated = tci.test(
            platform=platforms, size=tci.size.medium, schedule=schedule.periodic
        )(pytest.mark.serial(target))

        self.assertIs(decorated, target)
        self.assertEqual(decorated(), "called")
        self.assertEqual([mark.name for mark in decorated.pytestmark], ["serial"])
        expected_platforms = platforms if isinstance(platforms, tuple) else (platforms,)
        self.assertEqual(
            set(tci.resolve(decorated).markers),
            {
                "tci.size:medium",
                "tci.schedule:periodic",
                *(str(member) for member in expected_platforms),
            },
        )

    def test_unittest_runs_on_other_platform(self):
        calls = []

        class Example(TestCase):
            @tci.test(platform=platform.linux_x86_64_cuda_t4)
            def test_example(self):
                calls.append("called")

        with mock.patch.object(platform, "current", return_value=platform.linux_x86_64):
            result = unittest.TestResult()
            unittest.defaultTestLoader.loadTestsFromTestCase(Example).run(result)

        self.assertEqual(result.testsRun, 1)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.failures, [])
        self.assertEqual(result.skipped, [])
        self.assertEqual(calls, ["called"])
        self.assertEqual(
            set(tci.resolve(Example.test_example, Example).markers),
            {
                "tci.platform:linux_x86_64_cuda_t4",
                "tci.size:small",
                "tci.schedule:pull",
            },
        )

    @parametrize("value", [1, 2])
    def test_parametrized_markers(self, value):
        @instantiate_parametrized_tests
        @tci.test(
            platform=(platform.linux_x86_64, platform.linux_x86_64_cuda_l4),
            size=tci.size.large,
            schedule=schedule.periodic,
        )
        class Example(TestCase):
            @tci.test(exclude=platform.linux_x86_64)
            @parametrize("value", [1, 2])
            def test_example(self, value):
                pass

        generated = getattr(Example, f"test_example_value_{value}")
        self.assertEqual(
            set(tci.resolve(generated, Example).markers),
            {
                "tci.platform:linux_x86_64_cuda_l4",
                "tci.size:large",
                "tci.schedule:periodic",
            },
        )

    def test_inherited_defaults_and_method_overrides(self):
        cpu, l4, t4 = (
            platform.linux_x86_64,
            platform.linux_x86_64_cuda_l4,
            platform.linux_x86_64_cuda_t4,
        )

        @tci.test(platform=(cpu, l4), size=tci.size.medium)
        class Base(TestCase):
            def test_default(self):
                pass

        @tci.test(add=t4, exclude=cpu, schedule=schedule.periodic)
        class Example(Base):
            @tci.test(platform=cpu, size=tci.size.large)
            def test_replacement(self):
                pass

            @tci.test(add=cpu, exclude=l4)
            def test_delta(self):
                pass

        inherited = tci.resolve(Example.test_default, Example)
        self.assertEqual(inherited.platforms, (l4, t4))
        self.assertIs(inherited.size, tci.size.medium)
        self.assertIs(inherited.schedule, schedule.periodic)
        self.assertEqual(tci.resolve(Example().test_default), inherited)

        replacement = tci.resolve(Example.test_replacement, Example)
        self.assertEqual(replacement.platforms, (cpu,))
        self.assertIs(replacement.size, tci.size.large)
        self.assertIs(replacement.schedule, schedule.periodic)
        self.assertEqual(tci.resolve(Example.test_delta, Example).platforms, (t4, cpu))
        self.assertEqual(tci.resolve(Base).platforms, (cpu, l4))

    def test_multiple_inheritance_and_stacked_decorators(self):
        cpu, l4 = platform.linux_x86_64, platform.linux_x86_64_cuda_l4

        @tci.test(platform=(cpu, l4))
        class Base(TestCase):
            pass

        @tci.test(add=cpu, exclude=l4, size=tci.size.large)
        class Left(Base):
            pass

        @tci.test(exclude=cpu, size=tci.size.medium)
        class Right(Base):
            pass

        class Example(Left, Right):
            @tci.test(add=cpu)
            @tci.test(exclude=cpu)
            def test_example(self):
                pass

        metadata = tci.resolve(Example.test_example, Example)
        self.assertEqual(metadata.platforms, (cpu,))
        self.assertIs(metadata.size, tci.size.large)
        self.assertEqual(tci.resolve(Right).platforms, (l4,))
        self.assertEqual(tci.resolve(Base).platforms, (cpu, l4))

    def test_plain_function_stacked_decorators(self):
        cpu, l4 = platform.linux_x86_64, platform.linux_x86_64_cuda_l4

        @tci.test(add=(l4, cpu), exclude=cpu)
        @tci.test(platform=cpu, schedule=schedule.periodic)
        def target():
            return "called"

        metadata = tci.resolve(target)
        self.assertEqual(metadata.platforms, (l4,))
        self.assertIs(metadata.size, tci.size.small)
        self.assertIs(metadata.schedule, schedule.periodic)
        self.assertEqual(target(), "called")

    @parametrize("use_replacement", [False, True])
    def test_explicit_empty_differs_from_unmarked(self, use_replacement):
        cpu, l4 = platform.linux_x86_64, platform.linux_x86_64_cuda_l4

        class Unmarked:
            def test_example(self):
                pass

        @tci.test(platform=(cpu, l4))
        class Base(Unmarked):
            pass

        decorator = (
            tci.test(platform=()) if use_replacement else tci.test(exclude=(cpu, l4))
        )

        @decorator
        class Example(Base):
            pass

        self.assertIsNone(tci.resolve(Unmarked))
        self.assertIsNone(tci.resolve(len))
        self.assertIsNone(tci.resolve(Unmarked.test_example, Unmarked))
        self.assertIsNone(tci.resolve(Unmarked.test_example))
        metadata = tci.resolve(Example.test_example, Example)
        self.assertEqual(metadata.platforms, ())
        self.assertEqual(metadata.markers, ("tci.size:small", "tci.schedule:pull"))

    @parametrize(
        "field,value,error",
        [
            ("platform", "linux_x86_64_cuda_t4", TypeError),
            ("platform", (platform.linux_x86_64_cuda_t4, "cuda"), TypeError),
            ("add", "cuda", TypeError),
            ("exclude", (platform.linux_x86_64, "cuda"), TypeError),
            ("device", "cuda:0", TypeError),
            ("device", 1, TypeError),
            ("exclude_device", ("cpu", None), TypeError),
            ("size", "small", TypeError),
            ("schedule", "pull", TypeError),
        ],
    )
    def test_invalid_metadata(self, field, value, error):
        kwargs = {"platform": platform.linux_x86_64_cuda_t4, field: value}
        constraint = "platform" if field in ("add", "exclude") else field
        if field == "exclude_device":
            constraint = "device"
        with self.assertRaisesRegex(error, f"tci.{constraint}"):
            tci.test(**kwargs)


@instantiate_parametrized_tests
class TestTciDevicePolicy(TestCase):
    @parametrize("kind", ["property", "classmethod"])
    def test_dynamic_device_is_unknown_without_evaluation(self, kind):
        class Dynamic:
            @property
            def property_device(self):
                raise AssertionError("collection must not evaluate device properties")

            @classmethod
            def classmethod_device(cls):
                raise AssertionError("collection must not call device methods")

        @tci.test(platform=platform.linux_x86_64, device="cuda")
        def target():
            pass

        device = getattr(Dynamic, f"{kind}_device")
        self.assertTrue(
            tci.is_eligible(
                tci.resolve(target), platform=platform.linux_x86_64, device=device
            )
        )

    def test_bypass_keeps_device_exclusions_and_platform_policy(self):
        @tci.test(
            platform=platform.linux_x86_64_cuda_l4, device="cuda", exclude_device="meta"
        )
        def target():
            pass

        metadata = tci.resolve(target)
        self.assertFalse(metadata.supports_device("cpu"))
        self.assertTrue(metadata.supports_device("cpu", bypass=True))
        self.assertFalse(metadata.supports_device("meta", bypass=True))
        self.assertTrue(
            tci.is_eligible(
                metadata,
                platform=platform.linux_x86_64_cuda_l4,
                device="cpu",
                bypass=True,
            )
        )
        self.assertFalse(
            tci.is_eligible(
                metadata, platform=platform.linux_x86_64, device="cpu", bypass=True
            )
        )
        self.assertFalse(
            tci.is_eligible(
                metadata,
                platform=platform.linux_x86_64_cuda_l4,
                device="meta",
                bypass=True,
            )
        )

    def test_inherited_devices_and_replacement(self):
        @tci.test(device=("cpu", "cuda"))
        class Base(TestCase):
            pass

        @tci.test(exclude_device="cpu")
        class Example(Base):
            @tci.test(device="cpu")
            def test_cpu(self):
                pass

        metadata = tci.resolve(Example)
        self.assertFalse(metadata.supports_device("cpu"))
        self.assertTrue(metadata.supports_device("cuda"))
        self.assertFalse(metadata.supports_device("mps"))
        self.assertIn("tci.exclude_device:cpu", metadata.markers)
        method = tci.resolve(Example.test_cpu, Example)
        self.assertTrue(method.supports_device("cpu"))
        self.assertFalse(method.supports_device("cuda"))
        self.assertEqual(method.excluded_devices, ())
        self.assertTrue(tci.resolve(Base).supports_device("cpu"))

    def test_exclusion_only_and_empty_devices(self):
        @tci.test(exclude_device=("cpu", "meta"))
        def target():
            pass

        metadata = tci.resolve(target)
        self.assertIsNone(metadata.devices)
        self.assertFalse(metadata.supports_device("cpu"))
        self.assertFalse(metadata.supports_device("meta"))
        self.assertTrue(metadata.supports_device("cuda"))
        self.assertTrue(metadata.supports_device("xpu"))
        metadata = tci.resolve(tci.test(device=())(target))
        self.assertFalse(metadata.supports_device("cuda"))
        self.assertIn("tci.device:none", metadata.markers)

    @parametrize(
        "host,device,strict,expected",
        [
            (platform.linux_x86_64, "cuda", False, False),
            (platform.linux_x86_64_cuda_l4, "cuda", False, True),
            (platform.linux_x86_64_cuda_l4, "cpu", False, False),
            (platform.linux_x86_64_cuda_t4, "cuda", False, True),
            (platform.linux_x86_64_cuda_t4, "cpu", False, False),
            (None, "cuda", False, True),
            (None, "cpu", False, False),
            (platform.linux_x86_64_cuda_l4, None, False, True),
            (platform.linux_x86_64_cuda_t4, None, False, True),
            (platform.linux_x86_64_cuda_t4, "cuda", True, False),
            (platform.linux_x86_64_cuda_l4, "cuda", True, True),
            (None, None, True, False),
        ],
    )
    def test_eligibility(self, host, device, strict, expected):
        @tci.test(platform=platform.linux_x86_64_cuda_l4, device="cuda")
        def target():
            pass

        self.assertEqual(
            tci.is_eligible(
                tci.resolve(target), platform=host, device=device, strict=strict
            ),
            expected,
        )

    def test_unmarked_eligibility(self):
        self.assertTrue(tci.is_eligible(None, platform=platform.linux_x86_64))
        self.assertFalse(
            tci.is_eligible(None, platform=platform.linux_x86_64, strict=True)
        )


@instantiate_parametrized_tests
class TestTciPlatformSelection(TestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(mock.patch.dict(os.environ))
        os.environ.pop("PYTORCH_TEST_TCI_PLATFORM", None)
        self.enterContext(mock.patch("sys.platform", "linux"))
        self.enterContext(mock.patch("platform.machine", return_value="x86_64"))
        self.enterContext(mock.patch("torch.version.cuda", "13.2"))
        self.enterContext(mock.patch("torch.cuda._is_compiled", return_value=True))
        self.enterContext(mock.patch("torch.version.hip", None))
        self.enterContext(mock.patch("torch.version.xpu", None, create=True))
        for name in (
            "_lazy_init",
            "get_device_name",
            "get_device_capability",
            "is_available",
            "device_count",
        ):
            self.enterContext(
                mock.patch(
                    f"torch.cuda.{name}",
                    side_effect=AssertionError("selection must not initialize CUDA"),
                )
            )

    @parametrize(
        "count,name,expected",
        [
            (0, "", platform.linux_x86_64),
            (1, "NVIDIA L4", platform.linux_x86_64_cuda_l4),
            (1, "NVIDIA L40", None),
            (1, "NVIDIA T4", None),
            (-1, "", None),
        ],
    )
    def test_nvml_selection(self, count, name, expected):
        with (
            mock.patch("torch.cuda._device_count_nvml", return_value=count),
            mock.patch(
                "torch.testing._internal.torchci.environment._nvml_device",
                return_value=(name, "", ""),
            ),
        ):
            self.assertIs(tci.current_platform(), expected)

    def test_failed_probe_is_unknown(self):
        with mock.patch(
            "torch.cuda._device_count_nvml", side_effect=OSError("NVML unavailable")
        ):
            self.assertIsNone(tci.current_platform())

    def test_cpu_build_needs_no_probe(self):
        with (
            mock.patch("torch.cuda._is_compiled", return_value=False),
            mock.patch(
                "torch.cuda._device_count_nvml", side_effect=AssertionError("CPU build")
            ),
        ):
            self.assertIs(tci.current_platform(), platform.linux_x86_64)

    @parametrize(
        "system,machine,hip",
        [
            ("darwin", "x86_64", None),
            ("linux", "aarch64", None),
            ("linux", "x86_64", "7.2"),
        ],
    )
    def test_unmigrated_hosts_are_unknown(self, system, machine, hip):
        with (
            mock.patch("sys.platform", system),
            mock.patch("platform.machine", return_value=machine),
            mock.patch("torch.version.hip", hip),
            mock.patch(
                "torch.cuda._device_count_nvml",
                side_effect=AssertionError("unmigrated host"),
            ),
        ):
            self.assertIsNone(tci.current_platform())

    def test_scheduler_override_needs_no_probe(self):
        os.environ["PYTORCH_TEST_TCI_PLATFORM"] = platform.linux_x86_64_cuda_t4.name
        with mock.patch.object(
            tci.os, "current", side_effect=AssertionError("explicit platform")
        ):
            self.assertIs(tci.current_platform(), platform.linux_x86_64_cuda_t4)

    @parametrize("value", ["", "linux_typo"])
    def test_invalid_scheduler_override(self, value):
        os.environ["PYTORCH_TEST_TCI_PLATFORM"] = value
        with self.assertRaisesRegex(ValueError, "PYTORCH_TEST_TCI_PLATFORM"):
            tci.current_platform()


class TestTciFileDiscovery(TestCase):
    root: Path

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        temporary = tempfile.TemporaryDirectory(prefix="tci-files-")
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        for directory in ("test/nested", "test/package", "test/ignored"):
            cls.root.joinpath(directory).mkdir(parents=True)
        repo = Path(__file__).resolve().parents[1]
        for path in ("pytest.ini", "test/conftest.py", "test/pytest_shard_custom.py"):
            shutil.copyfile(repo / path, cls.root / path)
        for path in (
            "test/test_import_bomb.py",
            "test/nested/test_other.py",
            "test/nested/test_ignored.py",
            "test/nested/test_hook_ignored.py",
            "test/package/__init__.py",
            "test/package/test_package.py",
            "test/ignored/test_ignored.py",
            "test/check_custom.py",
            "test/helper.py",
        ):
            cls.root.joinpath(path).write_text(
                'raise RuntimeError("must not import")\n'
            )
        cls.root.joinpath("test/nested/conftest.py").write_text(
            'collect_ignore = ["test_ignored.py"]\n'
            "def pytest_ignore_collect(collection_path, config):\n"
            '    if collection_path.name == "test_hook_ignored.py":\n'
            "        return True\n"
        )
        cls.root.joinpath("test/test_markers.py").write_text(
            "import pytest\n"
            "@pytest.mark.serial\n"
            "def test_example():\n"
            '    raise RuntimeError("must not execute")\n'
        )

    def discover(self, *args):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--list-test-files", *args],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def test_file_discovery_respects_pytest_rules_without_imports(self):
        self.assertEqual(
            self.discover("test", "test/test_import_bomb.py", "--ignore=test/ignored"),
            [
                "test/nested/test_other.py",
                "test/package/test_package.py",
                "test/test_import_bomb.py",
                "test/test_markers.py",
            ],
        )

    def test_custom_pattern_and_explicit_file(self):
        self.assertEqual(
            self.discover("test", "-o", "python_files=check_*.py"),
            ["test/check_custom.py"],
        )
        self.assertEqual(self.discover("test/helper.py"), ["test/helper.py"])

    def test_missing_path_is_an_error(self):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--list-test-files", "test/missing.py"],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 4)
        self.assertIn("test/missing.py", result.stderr)

    def test_marker_collection_is_unchanged(self):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--list-markers", "test/test_markers.py"],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            json.loads(result.stdout),
            {"test/test_markers.py::test_example": ["multigpu", "serial"]},
        )


@instantiate_parametrized_tests
class TestTciDeviceCollection(TestCase):
    @parametrize(
        "generic_flag,base_flag,cpu_allowed",
        [(True, False, True), (None, True, True), (False, True, False)],
    )
    def test_generation_bypass_preserves_exclusions(
        self, generic_flag, base_flag, cpu_allowed
    ):
        class Example(TestCase):
            @tci.test(device="cuda")
            def test_positive(self, device):
                pass

            @tci.test(device="cuda", exclude_device="cpu")
            def test_excluded(self, device):
                pass

        if generic_flag is not None:
            Example.bypass_device_restrictions = generic_flag
        scope = {"Example": Example}
        with (
            mock.patch.object(sys, "argv", ["test_tci.py", "--list-markers"]),
            mock.patch.object(
                device_tests.DeviceTypeTestBase, "bypass_device_restrictions", base_flag
            ),
            mock.patch.object(
                device_tests,
                "device_type_test_bases",
                [device_tests.CPUTestBase, device_tests.CUDATestBase],
            ),
        ):
            device_tests.instantiate_device_type_tests(Example, scope)
        self.assertEqual(
            unittest.defaultTestLoader.getTestCaseNames(scope["ExampleCPU"]),
            ["test_positive_cpu"] if cpu_allowed else [],
        )
        self.assertEqual(
            unittest.defaultTestLoader.getTestCaseNames(scope["ExampleCUDA"]),
            ["test_excluded_cuda", "test_positive_cuda"],
        )

    def test_method_devices_override_class_defaults(self):
        parametrized = []

        class CudaParametrizer(_TestParametrizer):
            def _parametrize_test(self, test, generic_cls, device_cls):
                parametrized.append(device_cls.device_type)
                yield test, "", {}, lambda _: []

        @tci.test(device="cpu")
        class Example(TestCase):
            def test_cpu(self, device):
                pass

            @tci.test(device="cuda")
            @CudaParametrizer()
            def test_cuda(self, device):
                pass

        scope = {"Example": Example}
        with (
            mock.patch.object(sys, "argv", ["test_tci.py", "--list-markers"]),
            mock.patch.object(
                device_tests,
                "device_type_test_bases",
                [device_tests.CPUTestBase, device_tests.CUDATestBase],
            ),
        ):
            device_tests.instantiate_device_type_tests(Example, scope)
        self.assertEqual(parametrized, ["cpu", "cuda"])
        self.assertEqual(
            unittest.defaultTestLoader.getTestCaseNames(scope["ExampleCPU"]),
            ["test_cpu_cpu"],
        )
        self.assertEqual(
            unittest.defaultTestLoader.getTestCaseNames(scope["ExampleCUDA"]),
            ["test_cuda_cuda"],
        )

    def test_parameter_decorators_select_device_variants(self):
        @tci.test(device="cpu")
        class Example(TestCase):
            @parametrize(
                "value",
                [
                    subtest(1, name="cpu", decorators=[tci.test(device="cpu")]),
                    subtest(2, name="cuda", decorators=[tci.test(device="cuda")]),
                ],
            )
            def test_device(self, device, value):
                pass

        scope = {"Example": Example}
        with (
            mock.patch.object(sys, "argv", ["test_tci.py", "--list-markers"]),
            mock.patch.object(
                device_tests,
                "device_type_test_bases",
                [device_tests.CPUTestBase, device_tests.CUDATestBase],
            ),
        ):
            device_tests.instantiate_device_type_tests(Example, scope)
        self.assertEqual(
            unittest.defaultTestLoader.getTestCaseNames(scope["ExampleCPU"]),
            ["test_device_cpu_cpu"],
        )
        self.assertEqual(
            unittest.defaultTestLoader.getTestCaseNames(scope["ExampleCUDA"]),
            ["test_device_cuda_cuda"],
        )

    def test_collects_cuda_without_initializing_devices(self):
        captured, executed = [], []

        class CaptureDevice(_TestParametrizer):
            def _parametrize_test(self, test, generic_cls, device_cls):
                def decorators(kwargs):
                    captured.append((device_cls.device_type, kwargs.copy()))
                    return []

                yield test, "", {}, decorators

        @tci.test(
            platform=(platform.linux_x86_64, platform.linux_x86_64_cuda_l4),
            size=tci.size.medium,
        )
        class Example(TestCase):
            @CaptureDevice()
            @device_tests.dtypes(torch.float32, torch.float64)
            def test_dtype(self, device, dtype):
                executed.append((device, dtype))

            @CaptureDevice()
            @device_tests.deviceCountAtLeast(2)
            def test_multidevice(self, devices):
                executed.append(tuple(devices))

        scope = {"Example": Example}
        error = AssertionError("collection must not initialize or inspect CUDA")
        with (
            mock.patch.object(sys, "argv", ["test_tci.py", "--list-markers"]),
            mock.patch.object(device_tests, "IS_FBCODE", False),
            mock.patch.object(device_tests, "IS_SANDCASTLE", False),
            mock.patch.object(
                device_tests, "_is_privateuse1_backend_available", return_value=False
            ),
            mock.patch("torch.cuda.is_available", side_effect=error),
            mock.patch("torch.cuda.device_count", side_effect=error),
            mock.patch("torch.cuda._lazy_init", side_effect=error),
            mock.patch.object(
                device_tests.CUDATestBase, "setUpClass", side_effect=error
            ),
            mock.patch.object(
                device_tests.DeviceTypeTestBase,
                "_init_and_get_primary_device",
                side_effect=error,
            ),
        ):
            bases = device_tests.get_device_type_test_bases()
            self.assertEqual(
                bases, [device_tests.CPUTestBase, device_tests.CUDATestBase]
            )
            with mock.patch.object(device_tests, "device_type_test_bases", bases):
                device_tests.instantiate_device_type_tests(Example, scope)

        self.assertEqual(set(scope), {"ExampleCPU", "ExampleCUDA"})
        for device in ("cpu", "cuda"):
            cls = scope[f"Example{device.upper()}"]
            self.assertNotIn("_tci", vars(cls))
            self.assertIs(tci.resolve(cls).size, tci.size.medium)
            self.assertEqual(
                unittest.defaultTestLoader.getTestCaseNames(cls),
                [
                    f"test_dtype_{device}_float32",
                    f"test_dtype_{device}_float64",
                    f"test_multidevice_{device}",
                ],
            )

        cuda_kwargs = [kwargs for device, kwargs in captured if device == "cuda"]
        self.assertIn({"device": "cuda:0", "dtype": torch.float32}, cuda_kwargs)
        self.assertIn({"device": "cuda:0", "dtype": torch.float64}, cuda_kwargs)
        self.assertIn({"devices": ["cuda:0", "cuda:1"]}, cuda_kwargs)
        self.assertEqual(executed, [])

        cuda_cls = scope["ExampleCUDA"]
        with (
            mock.patch.object(
                cuda_cls, "_init_and_get_primary_device", return_value="cuda:7"
            ),
            mock.patch.object(
                cuda_cls, "get_all_devices", return_value=["cuda:7", "cuda:3"]
            ),
        ):
            cuda_cls().test_dtype_cuda_float32()
            cuda_cls().test_multidevice_cuda()
        self.assertEqual(executed, [("cuda:7", torch.float32), ("cuda:7", "cuda:3")])
        self.assertIn({"device": "cuda:0", "dtype": torch.float32}, cuda_kwargs)

    def test_normal_generation_binds_devices_eagerly(self):
        executed = []

        class Example(TestCase):
            def test_device(self, device):
                executed.append(device)

        scope = {"Example": Example}
        with (
            mock.patch.object(sys, "argv", ["test_tci.py"]),
            mock.patch.object(device_tests, "IS_FBCODE", False),
            mock.patch.object(device_tests, "IS_SANDCASTLE", False),
            mock.patch.object(
                device_tests, "_is_privateuse1_backend_available", return_value=False
            ),
            mock.patch("torch.cuda.is_available", return_value=False),
        ):
            self.assertEqual(
                device_tests.get_device_type_test_bases(), [device_tests.CPUTestBase]
            )
            with (
                mock.patch.object(
                    device_tests, "device_type_test_bases", [device_tests.CPUTestBase]
                ),
                mock.patch.object(
                    device_tests.CPUTestBase,
                    "_init_and_get_primary_device",
                    return_value="cpu:3",
                ) as primary,
            ):
                device_tests.instantiate_device_type_tests(Example, scope)
            primary.assert_called_once_with()

        with mock.patch.object(
            device_tests.CPUTestBase,
            "_init_and_get_primary_device",
            side_effect=AssertionError("normal generation already bound the device"),
        ):
            scope["ExampleCPU"]().test_device_cpu()
        self.assertEqual(executed, ["cpu:3"])


@instantiate_parametrized_tests
class TestTciUnittestLoader(TestCase):
    @parametrize("explicit", [False, True])
    def test_selection_precedes_class_fixtures(self, explicit):
        calls = []

        @tci.test(platform=tci.platform.linux_x86_64_cuda_l4)
        class OtherHost(TestCase):
            @classmethod
            def setUpClass(cls):
                raise AssertionError("an excluded class must not run fixtures")

            def test_wrong(self):
                raise AssertionError("an excluded test must not run")

        class CPUVariant(TestCase):
            device_type = "cpu"

            @tci.test(device="cpu")
            def test_allowed(self):
                calls.append("cpu")

            @tci.test(device="cuda")
            def test_wrong(self):
                raise AssertionError("an excluded device variant must not run")

        module = types.ModuleType("fixture")
        module.OtherHost = OtherHost
        module.CPUVariant = CPUVariant
        with mock.patch.dict(os.environ, PYTORCH_TEST_TCI_PLATFORM="linux_x86_64"):
            loader = TciTestLoader()
        suite = (
            loader.loadTestsFromNames(
                [
                    "OtherHost.test_wrong",
                    "CPUVariant.test_allowed",
                    "CPUVariant.test_wrong",
                ],
                module,
            )
            if explicit
            else loader.loadTestsFromModule(module)
        )
        result = unittest.TestResult()
        suite.run(result)
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(calls, ["cpu"])

    def test_load_tests_hook_and_invalid_selectors(self):
        @tci.test(platform=())
        class Excluded(TestCase):
            def test_wrong(self):
                raise AssertionError("load_tests must not bypass TCI")

        module = types.ModuleType("fixture")
        module.load_tests = lambda loader, tests, pattern: unittest.TestSuite(
            [unittest.defaultTestLoader.loadTestsFromTestCase(Excluded)]
        )
        with mock.patch.dict(os.environ, PYTORCH_TEST_TCI_PLATFORM="linux_x86_64"):
            loader = TciTestLoader()
        self.assertEqual(loader.loadTestsFromModule(module).countTestCases(), 0)
        result = unittest.TestResult()
        loader.loadTestsFromName("Missing.test_typo", module).run(result)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("Missing", result.errors[0][1])

    def test_unmigrated_host_keeps_platform_tests(self):
        class Example(TestCase):
            def test_example(self):
                pass

        with (
            mock.patch.dict(os.environ),
            mock.patch.object(tci, "current_platform", return_value=None),
        ):
            os.environ.pop("PYTORCH_TEST_TCI_PLATFORM", None)
            loader = TciTestLoader()
        self.assertEqual(loader.loadTestsFromTestCase(Example).countTestCases(), 1)


@instantiate_parametrized_tests
class TestTciRunnerIntegration(TestCase):
    root: Path

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        temporary = tempfile.TemporaryDirectory(prefix="tci-runners-")
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        cls.root.joinpath("test").mkdir()
        repo = Path(__file__).resolve().parents[1]
        for path in ("pytest.ini", "test/conftest.py", "test/pytest_shard_custom.py"):
            shutil.copyfile(repo / path, cls.root / path)
        cls.root.joinpath("test/test_placement.py").write_text(
            """import os
from torch.testing._internal import tci
from torch.testing._internal.common_utils import TestCase, run_tests

class TestDefault(TestCase):
    def test_default(self):
        pass

    @tci.test(exclude=tci.platform.linux_x86_64_cuda_l4)
    def test_cpu(self):
        self.assertEqual(os.environ["PYTORCH_TEST_TCI_PLATFORM"], "linux_x86_64")

    @tci.test(exclude=tci.platform.linux_x86_64)
    def test_l4(self):
        self.assertEqual(os.environ["PYTORCH_TEST_TCI_PLATFORM"], "linux_x86_64_cuda_l4")

class TestCPUVariant(TestCase):
    device_type = "cpu"

    @tci.test(device="cpu")
    def test_allowed(self):
        pass

    @tci.test(device="cuda")
    def test_wrong(self):
        raise AssertionError("wrong CPU variant")

class TestCUDAVariant(TestCase):
    device_type = "cuda"

    @tci.test(exclude_device=("cpu", "meta"))
    def test_allowed(self):
        pass

    @tci.test(device="cpu")
    def test_wrong(self):
        raise AssertionError("wrong CUDA variant")

def test_unmarked():
    raise AssertionError("strict placement must not run unmarked tests")

if __name__ == "__main__":
    run_tests()
"""
        )

    def invoke(self, *args, platform="linux_x86_64"):
        env = dict(os.environ, PYTORCH_TEST_TCI_PLATFORM=platform)
        return subprocess.run(
            [sys.executable, *args],
            cwd=self.root,
            env=env,
            text=True,
            capture_output=True,
            timeout=90,
        )

    @parametrize("platform", ["linux_x86_64", "linux_x86_64_cuda_l4"])
    @parametrize("pytest", [False, True])
    def test_default_selection(self, platform, pytest):
        args = ["-m", "pytest", "-q"] if pytest else []
        result = self.invoke(*args, "test/test_placement.py", platform=platform)
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("4 passed" if pytest else "Ran 4 tests", output)

    def test_script_pytest_and_repeat_modes(self):
        result = self.invoke("test/test_placement.py", "--use-pytest", "-q")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("4 passed", result.stdout)
        result = self.invoke("test/test_placement.py", "--repeat", "2")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stderr.count("Ran 4 tests"), 2)

    @parametrize("mode", ["--subprocess", "--run-parallel"])
    def test_process_modes(self, mode):
        args = [mode, "6"] if mode == "--run-parallel" else [mode]
        result = self.invoke("test/test_placement.py", *args)
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertEqual(output.count("Ran 1 test"), 4)

    def test_discovery_and_xml(self):
        result = self.invoke("test/test_placement.py", "--discover-tests")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            set(result.stdout.splitlines()),
            {
                "TestDefault.test_default",
                "TestDefault.test_cpu",
                "TestCPUVariant.test_allowed",
                "TestCUDAVariant.test_allowed",
            },
        )
        with tempfile.TemporaryDirectory(prefix="tci-xml-") as reports:
            result = self.invoke("test/test_placement.py", "--save-xml", reports)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Ran 4 tests", result.stderr)

    def test_marker_collection_is_platform_independent(self):
        result = self.invoke("-m", "pytest", "--list-markers", "test/test_placement.py")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        markers = json.loads(result.stdout)
        self.assertEqual(len(markers), 6)
        self.assertFalse(any(node.endswith("test_wrong") for node in markers))
        l4 = markers["test/test_placement.py::TestDefault::test_l4"]
        self.assertIn("tci.platform:linux_x86_64_cuda_l4", l4)
        self.assertNotIn("tci.platform:linux_x86_64", l4)

    def test_explicit_typo_stays_an_error(self):
        result = self.invoke("test/test_placement.py", "TestDefault.test_typo")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("test_typo", result.stdout + result.stderr)


if __name__ == "__main__":
    run_tests()
