# Owner(s): ["module: inductor"]

import sys

import unittest

from torch.testing._internal.common_utils import IS_CI, IS_WINDOWS, skipIfRocm
from torch.testing._internal.inductor_utils import HAS_CUDA

if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_memory_planning yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")  # noqa: F821

import torch
from torch._C import FileCheck
from torch._dynamo.utils import same
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_cpp_code
from torch.export import Dim
from torch.utils._triton import has_triton


@unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
@config.patch(memory_planning=True)
class TestMemoryPlanning(TestCase):
    def _generate(self, *, device):
        """
        Generate a simple test case that has multiple simultaneously-live intermediate tensors.
        """

        class Foo(torch.nn.Module):
            def forward(self, x, y, z):
                t0 = x.matmul(y)
                t1 = x.matmul(z)
                t0 = x.transpose(0, 1).matmul(t1)
                t1 = x.matmul(t0)
                return t0.sum() + t1.sum()

        x = torch.randn((3, 2), device=device)
        y = torch.randn((2, 4), device=device)
        z = torch.randn((2, 3), device=device)
        return (Foo(), (x, y, z))

    def test_python_wrapper(self):
        f, args = self._generate(device="cuda")
        compiled = torch.compile(f, dynamic=True)
        result, code = run_and_get_cpp_code(compiled, *args)

        FileCheck().check(
            "pool1 = empty_strided_cuda(((4*s0*s1) + (align(4*(s0*s0))), ), (1, )"
        ).check_next(
            "buf0 = alloc_from_pool(pool1, 0, torch.float32, (s0, s0), (s0, 1))"
        ).check(
            "buf1 = alloc_from_pool(pool1, align(4*(s0*s0)),"
        ).run(
            code
        )
        self.assertTrue(same(f(*args), result))

    def test_cpp_wrapper(self):
        f, args = self._generate(device="cuda")
        compiled = torch.compile(f, dynamic=True)
        with config.patch("cpp_wrapper", True):
            result, code = run_and_get_cpp_code(compiled, *args)

        FileCheck().check(
            "pool1 = at::detail::empty_strided_cuda({(4L*s0*s1) + (align(4L*(static_cast<long>(s0*s0)))), }, {1L, }"
        ).check_next(
            "auto buf0 = alloc_from_pool(pool1, 0, at::kFloat, {s0, s0}, {s0, 1L});"
        ).check(
            "auto buf1 = alloc_from_pool(pool1, align(4L*(static_cast<long>(s0*s0))),"
        ).run(
            code
        )
        self.assertTrue(same(f(*args), result))

    @skipIfRocm(msg="test_aot_inductor doesn't work on ROCm")
    def test_abi_compatible(self):
        from test_aot_inductor import AOTIRunnerUtil

        f, args = self._generate(device="cuda")
        dim0_x = Dim("dim0_x", min=1, max=2048)
        dynamic_shapes = ({0: dim0_x}, None, None)
        with config.patch("abi_compatible", True):
            result, code = run_and_get_cpp_code(
                lambda: AOTIRunnerUtil.run(
                    "cuda", f, args, dynamic_shapes=dynamic_shapes
                )
            )

        FileCheck().check(
            "int64_t int_array_2[] = {24L + (align(12L*s0)), };"
        ).check_next("int64_t int_array_3[] = {1L, };").check_next(
            "AtenTensorHandle pool1_handle;"
        ).check_next(
            "aoti_torch_empty_strided(1, int_array_2, int_array_3,"
        ).check_next(
            "RAIIAtenTensorHandle pool1(pool1_handle);"
        ).check_next(
            "int64_t int_array_4[] = {s0, 3L};"
        ).check_next(
            "int64_t int_array_5[] = {3L, 1L};"
        ).check_next(
            "AtenTensorHandle tmp_tensor_handle_1;"
        ).check_next(
            "aoti_torch__alloc_from_pool(pool1, 0"
        ).run(
            code
        )
        self.assertTrue(same(f(*args), result))


if __name__ == "__main__":
    if HAS_CUDA:
        run_tests()
