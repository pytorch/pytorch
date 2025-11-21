# Owner(s): ["module: inductor"]
import importlib
import os
import re
import sys

import torch
from torch._dynamo.utils import disable_cache_limit
from torch._inductor import config
from torch._inductor.codegen.triton import OpDtypeSupport
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code, run_and_get_triton_code, triton_type
from torch.fx.operator_schemas import get_signature_for_torch_op
from torch.testing import FileCheck
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import parametrize
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_gpu


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)


importlib.import_module("functorch")
importlib.import_module("filelock")


from torch._inductor.lowering import lowerings
from torch.testing._internal.common_device_type import ops
from torch.testing._internal.inductor_utils import HAS_GPU


unique_pointwise_op_names = set()

for op in lowerings:
    if not isinstance(op, torch._ops.OpOverload):
        continue

    if torch.Tag.pointwise not in op.tags:
        continue

    if op._schema.is_mutable:
        continue

    op_name = (op.name().split("::")[-1]).split(".")[0]
    unique_pointwise_op_names.add(op_name)

pointwise_ops = [
    op
    for op in op_db
    if op.name in unique_pointwise_op_names and "reduction" not in op.variant_test_name
]


class TestCase(InductorTestCase):
    @ops(
        pointwise_ops,
        allowed_dtypes=(
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
            torch.bool,
        ),
    )
    # @config.patch("triton.codegen_upcast_to_fp32", False) # TODO enable
    @config.patch("test_configs.runtime_triton_dtype_assert", True)
    @config.patch("test_configs.runtime_triton_shape_assert", True)
    @config.patch("test_configs.static_cpp_dtype_assert", True)
    @disable_cache_limit()
    def test_op_dtype_propagation(self, op, dtype):
        def run(op, args, kwargs):
            return op(*args, **kwargs)

        sample_inputs_itr = op.sample_inputs(GPU_TYPE, dtype, requires_grad=False)
        for sample_input in sample_inputs_itr:
            args = (sample_input.input,) + sample_input.args
            kwargs = sample_input.kwargs
            out = run(op.get_op(), args, kwargs)

            # test_configs.runtime_triton_dtype_assert does not work well with dynamic shape so far.
            # Consider the following cases for torch.add:
            #   both lhs/rhs are int32 tensor, there is also a integer alpha argument.
            #   In dynamic shape case, alpha is passed in as an ks0 argument. To be safe,
            #   we use tl.int64 for ks0's dtype.
            #   But the dtype for alpha is also decided as tl.int32 during lowering when
            #   we promote alpha to a ir.Constant.
            #   Ideally to resolve this problem, we should track assignment like
            #     alpha = ks0
            #   so that we know alpha is actually tl.int64 rather than tl.int32.
            out_c = torch.compile(run, dynamic=False)(op.get_op(), args, kwargs)
            self.assertEqual(out, out_c)

    @requires_gpu()
    @parametrize("upcast_to_fp32", [False, True])
    @config.patch("triton.use_block_ptr", True)
    def test_codegen_upcast_to_fp32(self, upcast_to_fp32):
        @torch.compile
        def func(a, b, c, d):
            return a * b * c * d

        inps = (torch.rand((32, 32), device=GPU_TYPE, dtype=torch.float16),) * 4
        with config.patch("triton.codegen_upcast_to_fp32", upcast_to_fp32):
            func_opt = torch.compile(func, backend="inductor")
            code = run_and_get_triton_code(func_opt, *inps)
            fp32_cast_in_code = "to(tl.float32)" in code
            self.assertEqual(fp32_cast_in_code, upcast_to_fp32)

        @requires_gpu()
        @parametrize("input_shape", [(32, 32), (32, 128), (256, 32)])
        @parametrize(
            "reduction_func",
            [
                torch.prod,
                torch.sum,
                torch.argmax,
                torch.argmin,
                torch.min,
                torch.max,
            ],
        )
        @parametrize("input_dtype", [torch.float16, torch.bfloat16])
        @config.patch("triton.use_block_ptr", True)
        def test_low_precision_reduction(
            self, input_shape, reduction_func, input_dtype
        ):
            @torch.compile
            def func(a, b, c, d):
                return reduction_func(a * b * c * d)

            inps = (torch.rand(input_shape, device=GPU_TYPE, dtype=input_dtype),) * 4
            with config.patch("triton.codegen_upcast_to_fp32", False):
                func_opt = torch._dynamo.optimize("inductor")(func)
                code = run_and_get_triton_code(func_opt, *inps)
                self.assertTrue(".to(tl.float32)" in code)
                self.assertEqual(func(*inps), func_opt(*inps))

    def test_op_dtype_support(self):
        """
        Triton codegen upcasts values to float32 for certain ops.
        Check that those ops have accurate dtype information.
        """

        for op_name in [
            "rsqrt",
            "sqrt",
            "isnan",
            "floor",
            "ceil",
            "tan",
            "atan",
            "atanh",
            "sigmoid",
            "log2",
            "log10",
            "cosh",
            "sinh",
            "acosh",
            "asinh",
            "asin",
            "acos",
            "asinh",
            "erf",
            "lgamma",
            "sin",
            "cos",
            "exp",
            "expm1",
            "exp2",
            "abs",
            "hypot",
            "nextafter",
        ]:
            # These ops do not support float16 and bfloat16.
            supported_dtypes = OpDtypeSupport.supported_dtypes[op_name]
            self.assertNotIn(torch.float16, supported_dtypes)
            self.assertNotIn(torch.bfloat16, supported_dtypes)

            # These ops should support float32 and float64.
            self.assertIn(torch.float32, supported_dtypes)
            self.assertIn(torch.float64, supported_dtypes)

    @requires_gpu()
    @parametrize("op_name", OpDtypeSupport.supported_dtypes)
    @parametrize("load_upcast_to_fp32", [False, True])
    @parametrize("input_dtype", [torch.float16, torch.bfloat16])
    @config.patch("triton.use_block_ptr", True)
    def test_dtype_aware_codegen(self, op_name: str, load_upcast_to_fp32, input_dtype):
        """
        Test dtype aware codegen for some tl.math/libdevice calls.
        Operands should be upcast to float32, and the output should be downcast to float16.
        """

        # Check if the op's output should be upcasted/downcasted.
        supported_dtypes = OpDtypeSupport.supported_dtypes[op_name]
        convert_output = OpDtypeSupport.convert_outputs[op_name]
        self.assertNotIn(input_dtype, supported_dtypes)

        # Retrieve the corresponding torch op.
        torch_op_name = op_name.removeprefix("libdevice_")
        op = getattr(torch, torch_op_name)

        # Edge case: torch.round maps to libdevice.nearbyint.
        triton_op_name_overrides = {
            "round": "nearbyint",
            # torch.sqrt lowers to tl.sqrt_rn after switching away from libdevice.sqrt
            "sqrt": "sqrt_rn",
        }
        override = triton_op_name_overrides.get(op_name)
        triton_op_name = override if override is not None else torch_op_name

        # Get the number of args for the op.
        # Take the minimum over all signatures to isolate required args.
        signatures = get_signature_for_torch_op(op)
        num_args = min(len(signature.parameters) for signature in signatures)

        # Test codegen and check for casts.
        inps = (torch.rand((32, 32), device=GPU_TYPE, dtype=input_dtype),) * num_args
        tl_dtype_str = str(input_dtype).replace("torch", "tl")
        with config.patch("triton.codegen_upcast_to_fp32", load_upcast_to_fp32):
            compiled = torch.compile(op, backend="inductor")
            code = run_and_get_triton_code(compiled, *inps)

            # Search the code with a regex.
            # Example code: libdevice.floor(tmp3.to(tl.float32)).to(tl.float16)
            output_cast = rf"\.to\({tl_dtype_str}\)" if convert_output else ""
            pattern = rf"{triton_op_name}\(.*\.to\(tl\.float32\)\){output_cast}"
            cast_in_code = re.search(pattern, code, re.MULTILINE) is not None
            self.assertNotEqual(cast_in_code, load_upcast_to_fp32)

    @config.patch("triton.codegen_upcast_to_fp32", False)
    def test_binary_math_mixed_precision(self):
        """
        Test a binary math operator where only one input needs to be upcast.
        """
        # Create inputs of different dtypes.
        inputs = [
            torch.randn(8, device=GPU_TYPE, dtype=dtype)
            for dtype in (torch.float16, torch.float32)
        ]

        func = torch.hypot
        compiled = torch.compile(backend="inductor")(func)
        result, (code,) = run_and_get_code(compiled, *inputs)

        # Check accuracy.
        ref = func(*inputs)
        self.assertTrue(torch.allclose(ref, result))

        # Check for exactly one upcast.
        num_upcasts = code.count(".to(tl.float32)")
        self.assertEqual(num_upcasts, 1)

        # There should be no downcast, since the input is promoted to float32.
        self.assertNotIn(".to(tl.float16)", code)

    @config.patch("test_configs.static_cpp_dtype_assert", True)
    @config.patch("test_configs.runtime_triton_dtype_assert", True)
    @config.patch("test_configs.runtime_triton_shape_assert", True)
    @config.patch("triton.codegen_upcast_to_fp32", False)
    def test_downcast_div_mod(self):
        def fn(x, y):
            return x % y, x / y

        x, y = (torch.rand([8], dtype=torch.float16, device=GPU_TYPE) for _ in range(2))

        out, code = run_and_get_code(torch.compile(fn), x, y)

        FileCheck().check("static_assert").check_same(".dtype").run(code[0])
        self.assertEqual(fn(x, y), out)

    @config.patch("test_configs.static_cpp_dtype_assert", True)
    @config.patch("test_configs.runtime_triton_dtype_assert", True)
    @config.patch("test_configs.runtime_triton_shape_assert", True)
    def test_constant(self):
        def fn():
            return (torch.full((2, 3), 3.1416, device=GPU_TYPE, dtype=torch.float16),)

        out, code = run_and_get_code(torch.compile(fn))
        FileCheck().check("static_assert").check_same(".dtype").run(code[0])
        self.assertEqual(fn(), out)

    @config.patch("test_configs.runtime_triton_dtype_assert", True)
    @config.patch("test_configs.runtime_triton_shape_assert", True)
    @config.patch("test_configs.static_cpp_dtype_assert", True)
    @config.patch("triton.persistent_reductions", False)
    def test_any(self):
        def fn(x):
            return torch.any(x)

        x = torch.rand([40], device=GPU_TYPE).to(torch.bool)
        out, code = run_and_get_code(torch.compile(fn), x)
        self.assertEqual(fn(x), out)

    @config.patch("test_configs.runtime_triton_dtype_assert", True)
    @config.patch("test_configs.runtime_triton_shape_assert", True)
    @config.patch("test_configs.static_cpp_dtype_assert", True)
    def test_assoc_scan(self):
        from torch._higher_order_ops.associative_scan import associative_scan

        x = torch.randn(10, device=GPU_TYPE)
        # dtype check correctly
        associative_scan(
            lambda acc, curr: acc + torch.abs(curr), x, dim=-1, combine_mode="pointwise"
        )

    @parametrize("upcast_to_fp32", (False, True))
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_upcast_rank_0_cpu(self, dtype: torch.dtype, upcast_to_fp32: bool):
        """
        Test whether we implicitly upcast CPU tensors of rank 0 to float32.
        """

        # Test broadcasting a rank-0 CPU tensor to rank 1.
        x = torch.randn(1, dtype=dtype, device="cpu")[0]
        y = torch.randn(8, dtype=dtype, device=GPU_TYPE)
        self.assertEqual(len(x.shape), 0)
        self.assertEqual(len(y.shape), 1)
        inps = (x, y)
        func = torch.add

        with config.patch("triton.codegen_upcast_to_fp32", upcast_to_fp32):
            compiled = torch.compile(func)
            result, (code,) = run_and_get_code(compiled, *inps)

        # Check numerics.
        ref = func(*inps)
        self.assertTrue(torch.allclose(result, ref))

        # Inductor upcasts CPU arguments of rank 0 to float32. Check for a downcast to
        # the original dtype.
        num_downcasts = code.count(f".to({triton_type(dtype)})")
        self.assertEqual(num_downcasts, 0 if upcast_to_fp32 else 1)


instantiate_device_type_tests(
    TestCase, globals(), only_for=("cuda", "xpu"), allow_xpu=True
)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests(needs="filelock")
