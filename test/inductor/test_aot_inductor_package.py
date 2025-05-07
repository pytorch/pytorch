# Owner(s): ["module: inductor"]
import copy
import functools
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Callable

from parameterized import parameterized_class

import torch
from torch._inductor.package import AOTICompiledModel, load_package, package_aoti
from torch._inductor.test_case import TestCase
from torch._inductor.utils import fresh_inductor_cache
from torch.export import Dim
from torch.testing._internal.common_utils import IS_FBCODE, skipIfXpu, TEST_CUDA
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


def skipif(predicate: Callable[[str, bool], bool], reason: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if predicate(self.device, self.package_cpp_only):
                self.skipTest(reason)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def compile(
    model,
    args,
    kwargs=None,
    *,
    dynamic_shapes=None,
    package_path=None,
    inductor_configs=None,
) -> AOTICompiledModel:
    ep = torch.export.export(
        model,
        args,
        kwargs,
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    package_path = torch._inductor.aoti_compile_and_package(
        ep, package_path=package_path, inductor_configs=inductor_configs
    )  # type: ignore[arg-type]
    loaded = load_package(package_path)
    return loaded


@unittest.skipIf(sys.platform == "darwin", "No CUDA on MacOS")
@parameterized_class(
    [
        {"device": "cpu", "package_cpp_only": False},
    ]
    + (
        [
            # FIXME: AssertionError: AOTInductor compiled library does not exist at
            {"device": "cpu", "package_cpp_only": True}
        ]
        if not IS_FBCODE
        else []
    )
    + (
        [
            {"device": GPU_TYPE, "package_cpp_only": False},
            {"device": GPU_TYPE, "package_cpp_only": True},
        ]
        if sys.platform != "darwin"
        else []
    ),
    class_name_func=lambda cls, _, params: f"{cls.__name__}{'Cpp' if params['package_cpp_only'] else ''}_{params['device']}",
)
class TestAOTInductorPackage(TestCase):
    def check_model(
        self: TestCase,
        model,
        example_inputs,
        inductor_configs=None,
        dynamic_shapes=None,
        atol=None,
        rtol=None,
    ) -> AOTICompiledModel:
        with torch.no_grad():
            torch.manual_seed(0)
            model = model.to(self.device)
            ref_model = copy.deepcopy(model)
            ref_inputs = copy.deepcopy(example_inputs)
            expected = ref_model(*ref_inputs)

            inductor_configs = inductor_configs or {}
            inductor_configs["aot_inductor.package_cpp_only"] = self.package_cpp_only

            torch.manual_seed(0)
            with tempfile.NamedTemporaryFile(suffix=".pt2") as f:
                compiled_model = compile(
                    model,
                    example_inputs,
                    dynamic_shapes=dynamic_shapes,
                    inductor_configs=inductor_configs,
                    package_path=f.name,
                )

            actual = compiled_model(*example_inputs)

        self.assertEqual(actual, expected, atol=atol, rtol=rtol)
        return compiled_model

    def test_add(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    def test_remove_intermediate_files(self):
        # For CUDA, generated cpp files contain absolute path to the generated cubin files.
        # With the package artifact, that cubin path should be overriden at the run time,
        # so removing those intermeidate files in this test to verify that.
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        model = Model()
        with torch.no_grad():
            torch.manual_seed(0)
            model = model.to(self.device)
            ref_model = copy.deepcopy(model)
            ref_inputs = copy.deepcopy(example_inputs)
            expected = ref_model(*ref_inputs)

            torch.manual_seed(0)
            with tempfile.NamedTemporaryFile(suffix=".pt2") as f:
                ep = torch.export.export(model, example_inputs, strict=True)
                with fresh_inductor_cache():
                    # cubin files are removed when exiting this context
                    package_path = torch._inductor.aoti_compile_and_package(
                        ep,
                        package_path=f.name,
                    )  # type: ignore[arg-type]
                loaded = torch._inductor.aoti_load_package(package_path)
                actual = loaded(*example_inputs)

            self.assertEqual(actual, expected)

    def test_linear(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                return x + self.linear(y)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    @unittest.skipIf(IS_FBCODE, "cmake won't work in fbcode")
    @skipIfXpu  # build system may be different
    def test_compile_after_package(self):
        if not self.package_cpp_only:
            raise unittest.SkipTest("Only meant to test cpp package")
        if shutil.which("cmake") is None:
            raise unittest.SkipTest("cmake is not available")
        if shutil.which("make") is None:
            raise unittest.SkipTest("make is not available")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                return x + self.linear(y)

        with torch.no_grad():
            example_inputs = (
                torch.randn(10, 10, device=self.device),
                torch.randn(10, 10, device=self.device),
            )
            model = Model().to(device=self.device)
            expected = model(*example_inputs)

            options = {
                "aot_inductor.package_cpp_only": self.package_cpp_only,
            }
            ep = torch.export.export(model, example_inputs, strict=True)
            package_path = torch._inductor.aoti_compile_and_package(
                ep, inductor_configs=options
            )
            with tempfile.TemporaryDirectory() as tmp_dir, zipfile.ZipFile(
                package_path, "r"
            ) as zip_ref:
                zip_ref.extractall(tmp_dir)
                tmp_path = Path(tmp_dir) / "data" / "aotinductor" / "model"
                self.assertTrue(tmp_path.exists())
                build_path = tmp_path / "build"
                self.assertTrue(not build_path.exists())

                # Create a build directory to run cmake
                build_path.mkdir()
                custom_env = os.environ.copy()
                custom_env["CMAKE_PREFIX_PATH"] = str(Path(torch.__file__).parent)
                subprocess.run(
                    ["cmake", ".."],
                    cwd=build_path,
                    env=custom_env,
                )
                subprocess.run(["make"], cwd=build_path)

                # Check if the .so file was build successfully
                so_path = build_path / "libaoti_model.so"
                self.assertTrue(so_path.exists())
                optimized = torch._export.aot_load(str(so_path), self.device)
                actual = optimized(*example_inputs)
                self.assertTrue(torch.allclose(actual, expected))

    def test_metadata(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                return x + self.linear(y)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        metadata = {"dummy": "moo"}
        compiled_model = self.check_model(
            Model(),
            example_inputs,
            inductor_configs={"aot_inductor.metadata": metadata},
        )

        loaded_metadata = compiled_model.get_metadata()  # type: ignore[attr-defined]

        self.assertEqual(loaded_metadata.get("dummy"), "moo")

    def test_bool_input(self):
        # Specialize on whichever branch the example input for b is
        class Model(torch.nn.Module):
            def forward(self, x, b):
                if b:
                    return x * x
                else:
                    return x + x

        example_inputs = (torch.randn(3, 3, device=self.device), True)
        self.check_model(Model(), example_inputs)

    def test_multiple_methods(self):
        options = {
            "aot_inductor.package": True,
            "aot_inductor.package_cpp_only": self.package_cpp_only,
        }

        class Model1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                return torch.cat([a, b], dim=0)

        dim0_a = Dim("dim0_a", min=1, max=10)
        dim0_b = Dim("dim0_b", min=1, max=20)
        dynamic_shapes = {"a": {0: dim0_a}, "b": {0: dim0_b}}
        example_inputs1 = (
            torch.randn(2, 4, device=self.device),
            torch.randn(3, 4, device=self.device),
        )
        ep1 = torch.export.export(
            Model1(), example_inputs1, dynamic_shapes=dynamic_shapes, strict=True
        )
        aoti_files1 = torch._inductor.aot_compile(
            ep1.module(), example_inputs1, options=options
        )

        class Model2(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.device = device

            def forward(self, x):
                t = torch.tensor(x.size(-1), device=self.device, dtype=torch.float)
                t = torch.sqrt(t * 3)
                return x * t

        example_inputs2 = (torch.randn(5, 5, device=self.device),)
        ep2 = torch.export.export(Model2(self.device), example_inputs2, strict=True)
        aoti_files2 = torch._inductor.aot_compile(
            ep2.module(), example_inputs2, options=options
        )

        with tempfile.NamedTemporaryFile(suffix=".pt2") as f:
            package_path = package_aoti(
                f.name, {"model1": aoti_files1, "model2": aoti_files2}
            )
            loaded1 = load_package(package_path, "model1")
            loaded2 = load_package(package_path, "model2")

        self.assertEqual(loaded1(*example_inputs1), ep1.module()(*example_inputs1))
        self.assertEqual(loaded2(*example_inputs2), ep2.module()(*example_inputs2))

    @unittest.skipIf(not TEST_CUDA, "requires cuda")
    def test_duplicate_calls(self):
        options = {
            "aot_inductor.package": True,
        }

        device = "cuda"

        class Model1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                return torch.cat([a, b], dim=0)

        dim0_a = Dim("dim0_a", min=1, max=10)
        dim0_b = Dim("dim0_b", min=1, max=20)
        dynamic_shapes = {"a": {0: dim0_a}, "b": {0: dim0_b}}
        example_inputs1 = (
            torch.randn(2, 4, device=device),
            torch.randn(3, 4, device=device),
        )
        self.check_model(Model1(), example_inputs1)
        ep1 = torch.export.export(
            Model1(), example_inputs1, dynamic_shapes=dynamic_shapes, strict=True
        )
        aoti_files1 = torch._inductor.aot_compile(
            ep1.module(), example_inputs1, options=options
        )

        device = "cpu"
        example_inputs2 = (
            torch.randn(2, 4, device=device),
            torch.randn(3, 4, device=device),
        )
        ep2 = torch.export.export(
            Model1(), example_inputs2, dynamic_shapes=dynamic_shapes, strict=True
        )
        aoti_files2 = torch._inductor.aot_compile(
            ep2.module(), example_inputs2, options=options
        )

        with tempfile.NamedTemporaryFile(suffix=".pt2") as f:
            package_path = package_aoti(
                f.name, {"model1": aoti_files1, "model2": aoti_files2}
            )
            loaded1 = load_package(package_path, "model1")
            loaded2 = load_package(package_path, "model2")

        self.assertTrue(
            torch.allclose(loaded1(*example_inputs1), ep1.module()(*example_inputs1))
        )
        self.assertTrue(
            torch.allclose(loaded2(*example_inputs2), ep2.module()(*example_inputs2))
        )

    def test_specified_output_dir(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                return torch.cat([a, b], dim=0)

        example_inputs = (
            torch.randn(2, 4, device=self.device),
            torch.randn(3, 4, device=self.device),
        )
        ep = torch.export.export(Model(), example_inputs, strict=True)
        aoti_files = torch._inductor.aot_compile(
            ep.module(),
            example_inputs,
            options={
                "aot_inductor.output_path": "tmp_output_",
                "aot_inductor.package": True,
                "aot_inductor.package_cpp_only": self.package_cpp_only,
            },
        )
        with tempfile.NamedTemporaryFile(suffix=".pt2") as f:
            package_path = package_aoti(f.name, {"model1": aoti_files})
            loaded = load_package(package_path, "model1")
        self.assertTrue(
            torch.allclose(loaded(*example_inputs), ep.module()(*example_inputs))
        )

    def test_save_buffer(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                return torch.cat([a, b], dim=0)

        example_inputs = (
            torch.randn(2, 4, device=self.device),
            torch.randn(3, 4, device=self.device),
        )
        ep = torch.export.export(Model(), example_inputs, strict=True)

        buffer = io.BytesIO()
        buffer = torch._inductor.aoti_compile_and_package(ep, package_path=buffer)  # type: ignore[arg-type]
        for _ in range(2):
            loaded = load_package(buffer)
            self.assertTrue(
                torch.allclose(loaded(*example_inputs), ep.module()(*example_inputs))
            )

    @skipif(
        lambda device, package_cpp_only: device == "cpu" or package_cpp_only,
        "No support for cpp only and cpu",
    )
    def test_package_without_weight(self):
        class Model(torch.nn.Module):
            def __init__(self, n, k, device):
                super().__init__()
                self.linear = torch.nn.Linear(k, n, device=device)

            def forward(self, a):
                return self.linear(a)

        M, N, K = 128, 2048, 4096
        model = Model(N, K, self.device)
        example_inputs = (torch.randn(M, K, device=self.device),)

        inductor_configs = {
            "always_keep_tensor_constants": True,
            "aot_inductor.package_constants_in_so": False,
        }
        compiled = compile(model, example_inputs, inductor_configs=inductor_configs)

        self.assertEqual(
            set(compiled.get_constant_fqns()), set(model.state_dict().keys())
        )

        compiled.load_constants(model.state_dict(), check_full_update=True)

        test_inputs = torch.randn(M, K, device=self.device)
        expected = model(test_inputs)
        output = compiled(test_inputs)
        self.assertEqual(expected, output)

    @skipif(
        lambda device, package_cpp_only: device == "cpu" or package_cpp_only,
        "No support for cpp only and cpu",
    )
    def test_package_user_managed_weight(self):
        class Model(torch.nn.Module):
            def __init__(self, n, k, device):
                super().__init__()
                self.linear = torch.nn.Linear(k, n, device=device)

            def forward(self, a):
                return self.linear(a)

        M, N, K = 128, 4096, 4096
        model = Model(N, K, self.device)
        example_inputs = (torch.randn(M, K, device=self.device),)

        inductor_configs = {
            "always_keep_tensor_constants": True,
            "aot_inductor.package_constants_in_so": False,
        }
        compiled = compile(model, example_inputs, inductor_configs=inductor_configs)

        self.assertEqual(
            set(compiled.get_constant_fqns()), set(model.state_dict().keys())
        )

        compiled.load_constants(
            model.state_dict(), check_full_update=True, user_managed=False
        )

        test_inputs = torch.randn(M, K, device=self.device)
        expected = model(test_inputs)
        output = compiled(test_inputs)
        self.assertEqual(expected, output)

        # Let's try to modify the weight in-place, result shouldn't change.
        model.linear.weight.data *= 3.7
        new_output = compiled(test_inputs)
        self.assertEqual(new_output, output)

        # Recreate a new model that we will test against user_managed=True
        new_compiled = compile(model, example_inputs, inductor_configs=inductor_configs)
        new_compiled.load_constants(
            model.state_dict(), check_full_update=True, user_managed=True
        )

        expected = model(test_inputs)
        new_output = new_compiled(test_inputs)
        self.assertEqual(expected, new_output)

        # Try to modify the weight in-place, result should change.
        model.linear.weight.data *= 3.7
        expected = model(test_inputs)
        new_output = new_compiled(test_inputs)
        self.assertEqual(new_output, expected)

    def test_deepcopy_compiled_model(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )

        model = Model()

        compiled = compile(model, example_inputs)

        copmiled_copy = copy.deepcopy(compiled)

        expected = model(*example_inputs)
        output = compiled(*example_inputs)
        output_copy = copmiled_copy(*example_inputs)
        self.assertEqual(expected, output)
        self.assertEqual(expected, output_copy)

    @skipif(
        lambda device, package_cpp_only: device == "cpu" or package_cpp_only,
        "No support for cpp only and cpu",
    )
    def test_update_weights(self):
        class Model(torch.nn.Module):
            def __init__(self, n, k, device):
                super().__init__()
                self.linear = torch.nn.Linear(k, n, device=device)

            def forward(self, a):
                return self.linear(a)

        M, N, K = 128, 2048, 4096
        model = Model(N, K, self.device)
        example_inputs = (torch.randn(M, K, device=self.device),)

        compiled = self.check_model(model, example_inputs)

        new_state_dict = {
            "linear.weight": torch.randn(N, K, device=self.device),
            "linear.bias": torch.randn(N, device=self.device),
        }
        model.load_state_dict(new_state_dict)

        compiled.load_constants(model.state_dict(), check_full_update=True)

        test_inputs = torch.randn(M, K, device=self.device)
        expected = model(test_inputs)
        output = compiled(test_inputs)
        self.assertEqual(expected, output)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU or sys.platform == "darwin":
        run_tests(needs="filelock")
