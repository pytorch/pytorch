# Owner(s): ["oncall: export"]
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
from torch._dynamo.eval_frame import is_dynamo_supported
from torch.export import Dim
from torch.export.experimental import _ExportPackage
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def cmake_compile(base_dir):
    custom_env = os.environ.copy()
    custom_env["CMAKE_PREFIX_PATH"] = str(Path(torch.__file__).parent)
    build_path = Path(base_dir) / "build"
    build_path.mkdir()
    subprocess.run(
        ["cmake", ".."],
        cwd=build_path,
        env=custom_env,
        check=True,
    )
    subprocess.run(["make"], cwd=build_path, check=True)
    result = subprocess.run(
        ["./build/main"],
        cwd=base_dir,
        check=True,
        capture_output=True,
        text=True,
    )

    return result


@unittest.skipIf(not is_dynamo_supported(), "dynamo isn't supported")
class TestPackage(TestCase):
    def test_basic(self):
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        x = torch.randn(3, 2)
        package = _ExportPackage()
        self.assertEqual(
            package._exporter("fn", fn)(x),
            fn(x),
        )
        self.assertEqual(len(package.methods), 1)
        self.assertEqual(len(package.methods["fn"].fallbacks), 1)
        self.assertEqual(len(package.methods["fn"].overloads), 0)

    def test_more_than_once(self):
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        x = torch.randn(3, 2)
        package = _ExportPackage()
        exporter = package._exporter("fn", fn)
        exporter(x)
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot export .* more than once",
        ):
            exporter(x)

    def test_error(self):
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        x = torch.randn(3, 2)
        package = _ExportPackage()
        exporter = package._exporter("fn", fn, fallback="error")
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot export fallback .* when fallback policy is set to 'error'",
        ):
            exporter(x)

    def test_overloads(self):
        class Module(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if x.shape[0] == 4:
                    return x + 1
                elif x.shape[0] == 3:
                    return x - 1
                else:
                    return x + 2

        fn = Module()
        x = torch.randn(3, 2)
        x2 = torch.randn(4, 2)
        x3 = torch.randn(5, 2)

        def spec(self, x):
            assert x.shape[0] == 3

        def spec2(self, x):
            assert x.shape[0] == 4

        def spec3(self, x):
            assert x.shape[0] >= 5
            return {"x": (Dim("batch", min=5), Dim.STATIC)}

        package = _ExportPackage()
        exporter = (
            package._exporter("fn", fn)
            ._define_overload("spec", spec)
            ._define_overload("spec2", spec2)
            ._define_overload("spec3", spec3)
        )
        self.assertEqual(exporter(x), x - 1)
        self.assertEqual(exporter(x2), x2 + 1)
        self.assertEqual(exporter(x3), x3 + 2)
        self.assertEqual(len(package.methods), 1)
        self.assertEqual(len(package.methods["fn"].overloads), 3)

    @parametrize("package_example_inputs", [True, False])
    def test_package_static_linkage(self, package_example_inputs):
        class Model1(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        class Model2(torch.nn.Module):
            def forward(self, x, y):
                return x - y

        def default(*args, **kwargs):
            return None

        example_inputs = (
            torch.ones(3, 3),
            torch.ones(3, 3),
        )

        package = _ExportPackage()
        m1 = Model1()
        m2 = Model2()
        exporter1 = package._exporter("Plus", m1)._define_overload("default", default)
        exporter2 = package._exporter("Minus", m2)._define_overload("default", default)
        exporter1(*example_inputs)
        exporter2(*example_inputs)
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
        ):
            package._compiled_and_package(
                tmp_dir + "/package.pt2", True, package_example_inputs
            )

            # Test compiling generated files
            result = cmake_compile(tmp_dir)
            if package_example_inputs:
                self.assertEqual(
                    result.stdout,
                    "output_tensor1 2  2  2\n 2  2  2\n 2  2  2\n[ CPUFloatType{3,3} ]\noutput_tensor2 0  0  0\n"
                    " 0  0  0\n 0  0  0\n[ CPUFloatType{3,3} ]\n",
                )


instantiate_parametrized_tests(TestPackage)

if __name__ == "__main__":
    run_tests()
