# Owner(s): ["module: inductor"]
import copy
import os
import sys
import unittest
from dataclasses import dataclass
from typing import Callable

import torch
from torch._inductor import config
from torch._inductor.package import load_package
from torch._inductor.test_case import TestCase
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


@dataclass
class CompiledResult:
    package_path: str
    compiled_model: Callable


def compile(model, example_inputs, dynamic_shapes, options, device) -> CompiledResult:
    ep = torch.export.export(
        model,
        example_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    gm = ep.module()
    package_path = torch._inductor.aot_compile(gm, example_inputs, options=options)  # type: ignore[arg-type]
    compiled_model = load_package(package_path, device)
    return CompiledResult(package_path, compiled_model)


def check_model(
    self: TestCase,
    model,
    example_inputs,
    options=None,
    dynamic_shapes=None,
    disable_constraint_solver=False,
    atol=None,
    rtol=None,
) -> CompiledResult:
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
        compiled_result = compile(
            model,
            example_inputs,
            dynamic_shapes,
            options,
            self.device,
        )

        actual = compiled_result.compiled_model(*example_inputs)

    self.assertEqual(actual, expected, atol=atol, rtol=rtol)
    return compiled_result


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
        metadata = {"device": str(self.device)}
        compiled_result = self.check_model(
            Model(), example_inputs, options={"aot_inductor.metadata": metadata}
        )

        so_path = next(
            os.path.join(root, file)
            for root, dirs, files in os.walk(compiled_result.package_path)
            for file in files
            if file.endswith("so")
        )

        if metadata["device"] == "cpu":
            runner = torch._C._aoti.AOTIModelContainerRunnerCpu(so_path, 1)  # type: ignore[call-arg]
        elif metadata["device"] == "cuda" or metadata["device"].startswith("cuda:"):
            runner = torch._C._aoti.AOTIModelContainerRunnerCuda(so_path, 1, metadata["device"])  # type: ignore[assignment, call-arg]

        loaded_metadata = runner.get_metadata()
        self.assertEqual(len(loaded_metadata), 1)
        self.assertEqual(loaded_metadata.get("device"), str(self.device))


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
