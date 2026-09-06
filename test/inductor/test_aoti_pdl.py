# Owner(s): ["module: inductor"]

import unittest

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache, run_and_get_cpp_code
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM90OrLater
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    TEST_WITH_ROCM,
)


@unittest.skipIf(
    not SM90OrLater or TEST_WITH_ROCM,
    "AOTInductor PDL requires NVIDIA sm90+",
)
class AOTInductorPDLTest(TestCase):
    @parametrize("enable_pdl", [False, True])
    def test_package_uses_host_pdl_launch(self, enable_pdl):
        class Model(torch.nn.Module):
            def forward(self, x):
                intermediate = torch.sin(x.sum(dim=1))
                return intermediate.sum(dim=0)

        model = Model().cuda()
        inputs = (torch.randn(1024, 1024, device="cuda"),)
        expected = model(*inputs)

        with config.patch({"triton.enable_pdl": enable_pdl}), fresh_cache():
            exported = torch.export.export(model, inputs, strict=True)

            def compile_package():
                return torch._inductor.aoti_compile_and_package(exported)

            package_path, code = run_and_get_cpp_code(compile_package)
            loaded = torch._inductor.aoti_load_package(package_path)
            actual = loaded(*inputs)

        self.assertEqual(actual, expected, atol=1e-4, rtol=1e-4)
        FileCheck().check_regex(
            rf"launchKernel\([^;]+stream_, {str(enable_pdl).lower()}\);"
        ).run(code)
        if enable_pdl:
            (
                FileCheck()
                .check("'launch_pdl': True")
                .check("gdc_wait")
                .check("gdc_launch_dependents")
                .run(code)
            )
        else:
            FileCheck().check("'launch_pdl': False").run(code)
            self.assertNotIn("gdc_wait", code)
            self.assertNotIn("gdc_launch_dependents", code)


instantiate_parametrized_tests(AOTInductorPDLTest)


if __name__ == "__main__":
    run_tests()
