# Owner(s): ["module: inductor"]
import copy
import sys
import tempfile
import unittest

from parameterized import parameterized_class

import torch
from torch._inductor.package import AOTICompiledModel, load_package, package_aoti
from torch._inductor.test_case import TestCase
from torch._inductor.utils import fresh_inductor_cache
from torch.export import Dim
from torch.testing._internal.common_utils import IS_FBCODE, TEST_CUDA
from torch.testing._internal.triton_utils import HAS_CUDA


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
            {"device": "cuda", "package_cpp_only": False},
            {"device": "cuda", "package_cpp_only": True},
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
        disable_constraint_solver=False,
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
                ep = torch.export.export(
                    model,
                    example_inputs,
                )
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

        b = torch.randn(3, 4, device=self.device)
        dim0_a = Dim("dim0_a", min=1, max=10)
        dim0_b = Dim("dim0_b", min=1, max=20)
        dynamic_shapes = {"a": {0: dim0_a}, "b": {0: dim0_b}}
        example_inputs1 = (
            torch.randn(2, 4, device=self.device),
            torch.randn(3, 4, device=self.device),
        )
        ep1 = torch.export.export(
            Model1(), example_inputs1, dynamic_shapes=dynamic_shapes
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
        ep2 = torch.export.export(Model2(self.device), example_inputs2)
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
            Model1(), example_inputs1, dynamic_shapes=dynamic_shapes
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
            Model1(), example_inputs2, dynamic_shapes=dynamic_shapes
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

        assert torch.allclose(loaded1(*example_inputs1), ep1.module()(*example_inputs1))
        assert torch.allclose(loaded2(*example_inputs2), ep2.module()(*example_inputs2))

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
        ep = torch.export.export(Model(), example_inputs)
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
        assert torch.allclose(loaded(*example_inputs), ep.module()(*example_inputs))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    # cpp_extension N/A in fbcode
    if HAS_CUDA or sys.platform == "darwin":
        run_tests(needs="filelock")
