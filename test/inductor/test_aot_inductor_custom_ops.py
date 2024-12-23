# Owner(s): ["module: inductor"]
# This test requires libaoti_custom_ops.so to be built, which happnes when BUILD_TEST = 1
import logging
import sys
import unittest

import torch
import torch._export
import torch._inductor
import torch._inductor.config
from torch._inductor import config
from torch._inductor.test_case import TestCase
from torch.export import Dim, export
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import (
    find_library_location,
    IS_CI,
    IS_FBCODE,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
)
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test
from torch.testing._internal.triton_utils import HAS_CUDA


if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

try:
    try:
        from .test_aot_inductor_utils import (
            check_model,
            check_model_with_multiple_inputs,
            code_check_count,
        )
        from .test_torchinductor import copy_tests, TestFailure
    except ImportError:
        from test_aot_inductor_utils import (  # @manual=fbcode//caffe2/test/inductor:aot_inductor_utils-library
            check_model,
            check_model_with_multiple_inputs,
            code_check_count,
        )
        from test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
            copy_tests,
            TestFailure,
        )
except (unittest.SkipTest, ImportError):
    if __name__ == "__main__":
        sys.exit(0)
    raise


@torch.library.custom_op(
    "aoti_custom_ops::fn_with_incorrect_optional_tensor", mutates_args=()
)
def fn_with_incorrect_optional_tensor(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    if z is None:
        return x + y
    else:
        return x + y + z


@fn_with_incorrect_optional_tensor.register_fake
def fn_with_incorrect_optional_tensor_fake(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    if z is None:
        return x + y
    else:
        return x + y + z


class AOTInductorTestsTemplate:
    def test_custom_op_add(self) -> None:
        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.ops.aoti_custom_ops.custom_add(x, y)

        m = M().to(device=self.device)
        args = (
            torch.randn(3, 3, device=self.device),
            torch.randn(3, 3, device=self.device),
        )
        self.check_model(m, args)

    def test_custom_op_add_output_path(self) -> None:
        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.ops.aoti_custom_ops.custom_add(x, y)

        m = M().to(device=self.device)
        args = (
            torch.randn(3, 3, device=self.device),
            torch.randn(3, 3, device=self.device),
        )
        with config.patch("aot_inductor.output_path", "model.so"):
            with self.assertRaises(Exception):
                self.check_model(m, args)

    def test_custom_op_all_inputs(self) -> None:
        class MyModel(torch.nn.Module):
            # pyre-fixme[3]: Return type must be annotated.
            def __init__(self):
                super().__init__()

            # pyre-fixme[3]: Return type must be annotated.
            # pyre-fixme[2]: Parameter must be annotated.
            def forward(self, x, y):
                with torch.no_grad():
                    x_dim0 = x.shape[0]
                    x_dim1 = x.shape[1]
                    y_dim0 = y.shape[0]
                    y_dim1 = y.shape[1]
                    symint_0 = x_dim0 + x_dim1
                    symint_1 = y_dim0 * y_dim1

                    z = torch.concat((x, x))

                    _2547 = torch.ops.aoti_custom_ops.fn_with_all_inputs(
                        tensor=x,
                        tensors=[x, y],
                        optional_tensors=[None, z],
                        b8=False,
                        b8s=[True, False],
                        i64=42,
                        i64s=[16, 17],
                        symint=symint_0,
                        symints=[symint_0, symint_1],
                        f64=3.14,
                        f64s=[2.2, 3.3],
                        scalar=1.23,
                        scalars=[45, 67],
                        string="hello",
                        strings=["ab", "cde"],
                        # dtype=torch.float16,
                        # memory_format=torch.contiguous_format,
                        # layout=torch.strided,
                        device=torch.device("cpu"),
                        # optional
                        o_tensor=None,
                        o_tensors=[x, y],
                        o_b8=False,
                        o_b8s=[True, False],
                        o_i64=None,
                        o_i64s=[16, 17],
                        o_symint=symint_1,
                        o_symints=[symint_1, symint_0],
                        o_f64=3.14,
                        o_f64s=None,
                        o_scalar=None,
                        o_scalars=[89, 910],
                        o_string="hello",
                        o_strings=["ab", "cde"],
                        # o_dtype=None,
                        # o_memory_format=torch.contiguous_format,
                        # o_layout=torch.strided,
                        o_device=None,
                    )

                return _2547

        m = MyModel().to(device=self.device)
        x = torch.zeros(4, 8, device=self.device)
        y = torch.ones(3, 9, device=self.device)
        args = (x, y)
        m(*args)

        self.check_model(m, args)

    def test_custom_op_with_multiple_outputs(self) -> None:
        class Model(torch.nn.Module):
            def forward(self, x, y):
                out = x + y
                # tuple of Tensor output
                out3, out4 = torch.ops.aoti_custom_ops.fn_with_tuple_output(out, 1)
                # TensorList output
                out5, out6 = torch.ops.aoti_custom_ops.fn_with_list_output(
                    [out3, out4], 1
                )
                # tuple of Tensor and TensorList
                out7, [out8, out9] = torch.ops.aoti_custom_ops.fn_with_mix_outputs(
                    out5, [out6, out4]
                )
                return out3, out4, out5, out6, out7, out8, out9

        m = Model().to(device=self.device)
        args = (
            torch.randn(4, 4, device=self.device),
            torch.randn(4, 4, device=self.device),
        )
        m(*args)

        self.check_model(m, args)

    def test_custom_op_out_variant_without_return(self) -> None:
        class Model(torch.nn.Module):
            def forward(self, x, y):
                torch.ops.aoti_custom_ops.fn_out_variant_without_return(x, y)
                return y

        m = Model().to(device=self.device)
        args = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        m(*args)

        self.check_model(m, args)

    def test_custom_op_with_reinterpret_view_inputs(self) -> None:
        class Model(torch.nn.Module):
            def forward(self, x):
                out = x.permute([1, 0])
                return torch.ops.aoti_custom_ops.fn_with_default_input(out, 1)

        m = Model().to(device=self.device)
        args = (torch.randn(2, 3, device=self.device),)

        self.check_model(m, args)

    def test_custom_op_with_concat_inputs(self) -> None:
        class Model(torch.nn.Module):
            def forward(self, x, y):
                out = torch.concat([x, y], dim=0)
                return torch.ops.aoti_custom_ops.fn_with_default_input(out, 1)

        m = Model().to(device=self.device)
        args = (
            torch.randn(2, 3, device=self.device),
            torch.randn(2, 3, device=self.device),
        )

        self.check_model(m, args)

    def test_custom_op_missing_arg_with_default_value(self) -> None:
        class Model(torch.nn.Module):
            def forward(self, x):
                # missing second arg
                return torch.ops.aoti_custom_ops.fn_with_default_input(x)

        m = Model().to(device=self.device)
        args = (torch.randn(2, 3, device=self.device),)

        self.check_model(m, args)

    @unittest.skipIf(IS_FBCODE, "FbProxyExecutor doesn't have these error msgs")
    def test_incorrect_custom_op_schema(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.ops.aoti_custom_ops.fn_with_incorrect_optional_tensor(
                    x, y, None
                )

        m = M().to(device=self.device)
        args = (
            torch.randn(2, 3, device=self.device),
            torch.randn(2, 3, device=self.device),
        )

        with self.assertRaisesRegex(RuntimeError, "Expected extern kernel"):
            self.check_model(m, args)


class AOTInductorLoggingTest(LoggingTestCase):
    @make_logging_test(dynamic=logging.DEBUG)
    def test_shape_env_reuse(self, records):
        # make sure ShapeEnv is only created once and reused afterwards
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x + 2

        inputs = (torch.randn(4, 4),)
        dynamic_shapes = {
            "x": {0: Dim.AUTO, 1: Dim.AUTO},
        }
        ep = export(Foo(), inputs, dynamic_shapes=dynamic_shapes, strict=False)
        with torch.no_grad():
            torch._inductor.aot_compile(ep.module(), inputs)
        self.assertEqual([r.msg == "create_env" for r in records].count(True), 1)


common_utils.instantiate_parametrized_tests(AOTInductorTestsTemplate)


class AOTICustomOpTestCase(TestCase):
    def setUp(self):
        if IS_SANDCASTLE or IS_FBCODE:
            torch.ops.load_library("//caffe2/test/inductor:custom_ops")
        elif IS_MACOS:
            raise unittest.SkipTest("non-portable load_library call used in test")
        else:
            lib_file_path = find_library_location("libaoti_custom_ops.so")
            if IS_WINDOWS:
                lib_file_path = find_library_location("aoti_custom_ops.dll")
            torch.ops.load_library(str(lib_file_path))
        super().setUp()


def fail_cpu(is_skip=False):
    return TestFailure(
        ("cpu",),
        is_skip=is_skip,
    )


def fail_cuda(is_skip=False):
    return TestFailure(
        ("cuda"),
        is_skip=is_skip,
    )


# test_failures, xfail by default, set is_skip=True to skip
CPU_TEST_FAILURES = {
    # TODO: failed internally
    "test_multiple_output_alias": fail_cpu(is_skip=True),
}

# test_failures, xfail by default, set is_skip=True to skip
CUDA_TEST_FAILURES = {
    # quantized unsupported for GPU
    "test_quantized_linear": fail_cuda(),
    "test_quanatized_int8_linear": fail_cuda(),
}


class AOTInductorTestABICompatibleCpu(AOTICustomOpTestCase):
    device = "cpu"
    device_type = "cpu"
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = False
    use_minimal_arrayref_interface = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleCpu,
    "cpu",
    CPU_TEST_FAILURES,
)


@unittest.skipIf(sys.platform == "darwin", "No CUDA on MacOS")
class AOTInductorTestABICompatibleCuda(AOTICustomOpTestCase):
    device = "cuda"
    device_type = "cuda"
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = False
    use_minimal_arrayref_interface = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleCuda,
    "cuda",
    CUDA_TEST_FAILURES,
)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    # cpp_extension N/A in fbcode
    if HAS_CUDA or sys.platform == "darwin":
        run_tests(needs="filelock")
