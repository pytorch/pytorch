# Owner(s): ["module: inductor"]
import copy
import sys
import tempfile
import unittest

import torch
from torch._inductor import config
from torch._inductor.package import AOTICompiledModel, load_package, package_aoti
from torch._inductor.test_case import TestCase
from torch.export import Dim
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import IS_FBCODE
from torch.testing._internal.triton_utils import HAS_CUDA


try:
    try:
        from .test_torchinductor import copy_tests
    except ImportError:
        from test_torchinductor import copy_tests
except (unittest.SkipTest, ImportError) as e:
    if __name__ == "__main__":
        sys.exit(0)
    raise


def compile(
    model, example_inputs, dynamic_shapes, options, device
) -> AOTICompiledModel:
    ep = torch.export.export(
        model,
        example_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    gm = ep.module()
    aoti_files = torch._inductor.aot_compile(gm, example_inputs, options=options)  # type: ignore[arg-type]
    with tempfile.NamedTemporaryFile(suffix=".pt2") as f:
        package_path = package_aoti(f.name, aoti_files)
        loaded = load_package(package_path)
    return loaded


def check_model(
    self: TestCase,
    model,
    example_inputs,
    options=None,
    dynamic_shapes=None,
    disable_constraint_solver=False,
    atol=None,
    rtol=None,
) -> AOTICompiledModel:
    with torch.no_grad(), config.patch(
        {
            "aot_inductor.package": True,
            # TODO: "aot_inductor.force_mmap_weights": True,
        }
    ):
        torch.manual_seed(0)
        model = model.to(self.device)
        ref_model = copy.deepcopy(model)
        ref_inputs = copy.deepcopy(example_inputs)
        expected = ref_model(*ref_inputs)

        torch.manual_seed(0)
        compiled_model = compile(
            model,
            example_inputs,
            dynamic_shapes,
            options,
            self.device,
        )

        actual = compiled_model(*example_inputs)

    self.assertEqual(actual, expected, atol=atol, rtol=rtol)
    return compiled_model


class AOTInductorTestsTemplate:
    def test_add(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        self.check_model(Model(), example_inputs)

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
            Model(), example_inputs, options={"aot_inductor.metadata": metadata}
        )

        loaded_metadata = compiled_model.get_metadata()  # type: ignore[attr-defined]

        self.assertEqual(loaded_metadata.get("dummy"), "moo")

    def test_multiple_methods(self):
        options = {"aot_inductor.package": True}

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


common_utils.instantiate_parametrized_tests(AOTInductorTestsTemplate)


@unittest.skipIf(sys.platform == "darwin" or IS_FBCODE, "No CUDA on MacOS")
class AOTInductorTestPackagedABICompatibleCuda(TestCase):
    device = "cuda"
    check_model = check_model


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestPackagedABICompatibleCuda,
    "packaged_abi_compatible_cuda",
)


@unittest.skipIf(IS_FBCODE, "This is for OSS only")
class AOTInductorTestPackagedABICompatibleCpu(TestCase):
    device = "cpu"
    check_model = check_model


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestPackagedABICompatibleCpu,
    "packaged_abi_compatible_cpu",
)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    # cpp_extension N/A in fbcode
    if HAS_CUDA or sys.platform == "darwin":
        run_tests(needs="filelock")
