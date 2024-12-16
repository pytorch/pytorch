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
from torch._inductor.utils import run_and_get_code, run_and_get_triton_code
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
    @disable_cache_limit()
    def test_op_dtype_propagation(self, op, dtype):
        def run(op, args, kwargs):
            return op(*args, **kwargs)

        sample_inputs_itr = op.sample_inputs("cuda", dtype, requires_grad=False)
        for sample_input in sample_inputs_itr:
            args = (sample_input.input,) + sample_input.args
            kwargs = sample_input.kwargs
            out = run(op.get_op(), args, kwargs)
            out_c = torch.compile(run)(op.get_op(), args, kwargs)
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
            func_opt = torch._dynamo.optimize("inductor")(func)
            code = run_and_get_triton_code(func_opt, *inps)
            fp32_cast_in_code = "to(tl.float32)" in code
            self.assertEqual(fp32_cast_in_code, upcast_to_fp32)

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
            compiled = torch._dynamo.optimize("inductor")(op)
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

    @config.patch("test_configs.runtime_triton_dtype_assert", True)
    def test_constant(self):
        def fn():
            return (torch.full((2, 3), 3.1416, device="cuda", dtype=torch.float16),)

        out, code = run_and_get_code(torch.compile(fn))
        FileCheck().check("static_assert").check_same(".dtype").run(code[0])
        self.assertEqual(fn(), out)

    @config.patch("test_configs.runtime_triton_dtype_assert", True)
    @config.patch("triton.persistent_reductions", False)
    def test_any(self):
        def fn(x):
            return torch.any(x)

        x = torch.rand([40], device="cuda").to(torch.bool)
        out, code = run_and_get_code(torch.compile(fn), x)
        self.assertEqual(fn(x), out)

    @config.patch("test_configs.runtime_triton_dtype_assert", True)
    def test_assoc_scan(self):
        from torch._higher_order_ops.associative_scan import associative_scan

        x = torch.randn(10, device="cuda")
        # dtype check correctly
        associative_scan(
            lambda acc, curr: acc + torch.abs(curr), x, dim=-1, combine_mode="pointwise"
        )


instantiate_device_type_tests(TestCase, globals(), only_for=("cuda",))

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests(needs="filelock")
