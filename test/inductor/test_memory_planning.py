# Owner(s): ["module: inductor"]

import sys
import unittest

from torch.testing._internal.common_utils import IS_CI, IS_WINDOWS, skipIfXpu
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, requires_gpu


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


try:
    from .test_aot_inductor import AOTIRunnerUtil
except ImportError:
    from test_aot_inductor import (  # @manual=fbcode//caffe2/test/inductor:test_aot_inductor-library
        AOTIRunnerUtil,
    )


@requires_gpu()
@config.patch(memory_planning=True)
class TestMemoryPlanning(TestCase):
    device = GPU_TYPE

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
        f, args = self._generate(device=GPU_TYPE)
        compiled = torch.compile(f, dynamic=True)
        result, code = run_and_get_cpp_code(compiled, *args)

        FileCheck().check(
            "pool1 = empty_strided_"
            + GPU_TYPE
            + "((4*s27*s77 + align(4*s77*s77), ), (1, )"
        ).check_next(
            "buf0 = alloc_from_pool(pool1, 0, torch.float32, (s77, s77), (s77, 1))"
        ).check("buf1 = alloc_from_pool(pool1, align(4*s77*s77),").run(code)
        self.assertTrue(same(f(*args), result))

    def test_cpp_wrapper(self):
        f, args = self._generate(device=GPU_TYPE)
        compiled = torch.compile(f, dynamic=True)
        with config.patch({"cpp_wrapper": True}):
            result, code = run_and_get_cpp_code(compiled, *args)

        FileCheck().check(
            "aoti_torch__alloc_from_pool(pool1, 0, cached_torch_dtype_float32, 2, int_array_2, int_array_3, &tmp_tensor_handle_0)"
        ).check_next("auto buf0 = RAIIAtenTensorHandle(tmp_tensor_handle_0);").check(
            "auto buf1 = RAIIAtenTensorHandle(tmp_tensor_handle_1);"
        ).run(code)
        self.assertTrue(same(f(*args), result))

    @skipIfXpu(msg="aoti doesn't work on XPU")
    def test_aoti(self):
        f, args = self._generate(device=GPU_TYPE)
        dim0_x = Dim("dim0_x", min=1, max=2048)
        dynamic_shapes = ({0: dim0_x}, None, None)
        result, code = run_and_get_cpp_code(
            lambda: AOTIRunnerUtil.run(f, args, dynamic_shapes=dynamic_shapes)
        )

        FileCheck().check(
            "int64_t int_array_0[] = {24L + align(12L*s77), };"
        ).check_next("int64_t int_array_1[] = {1L, };").check_next(
            "AtenTensorHandle pool1_handle;"
        ).check_next(
            "aoti_torch_empty_strided(1, int_array_0, int_array_1,"
        ).check_next("RAIIAtenTensorHandle pool1(pool1_handle);").check_next(
            "int64_t int_array_2[] = {s77, 3L};"
        ).check_next("int64_t int_array_3[] = {3L, 1L};").check_next(
            "AtenTensorHandle tmp_tensor_handle_0;"
        ).check_next("aoti_torch__alloc_from_pool(pool1, 0").run(code)
        self.assertTrue(same(f(*args), result))

    @config.patch({"triton.autotune_at_compile_time": False})
    def test_unbacked_symint(self):
        # when allocation's size has unbacked symints
        # the unbacked symints are only available after computed
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Repro(torch.nn.Module):
            def forward(self, x, y):
                x = x + 1
                u0 = x.item()
                torch._check(u0 >= 1)
                s0 = y.size(0)
                expr = u0 * s0
                sevens = torch.empty_strided(
                    size=(10, expr, 32), stride=(expr * 32, 32, 1), device=x.device
                ).fill_(7)
                return sevens * 3

        example_inputs = (
            torch.scalar_tensor(2, dtype=torch.int, device=self.device),
            torch.ones(8, device=self.device),
        )
        model = Repro().to(self.device)
        result, code = run_and_get_cpp_code(
            lambda: AOTIRunnerUtil.run(model, example_inputs)
        )
        self.assertTrue(same(model(*example_inputs), result))

        # check allocation is done after the unbacked symint is computed
        FileCheck().check("auto u0 = u0_raw;").check(
            "const int64_t int_array_2[] = {10L, 8L*u0, 32L};"
        ).check("AtenTensorHandle pool0_handle;").check(
            "aoti_torch_empty_strided(3, int_array_2, int_array_3"
        ).run(code)

        # all AtenTensorHandle allocated using aoti_torch__alloc_from_pool are wrapped with RAIIAtenTensorHandle
        # otherwise we'll have memory leak
        FileCheck().check_count(
            "aoti_torch__alloc_from_pool(pool1", 1, exactly=True
        ).check_count("aoti_torch__alloc_from_pool(pool0", 1, exactly=True).run(code)

        FileCheck().check(
            "AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch__alloc_from_pool(pool1, 0, cached_torch_dtype_int32, 0, int_array_1, int_array_1, &tmp_tensor_handle_0));"  # noqa: B950
        ).check("RAIIAtenTensorHandle(tmp_tensor_handle_0);").check(
            "AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch__alloc_from_pool(pool0, 0, cached_torch_dtype_float32, 3, int_array_4, int_array_5, &tmp_tensor_handle_1));"  # noqa: B950
        ).check("RAIIAtenTensorHandle(tmp_tensor_handle_1);").run(code)


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
