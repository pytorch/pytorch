# Owner(s): ["module: inductor"]
import importlib
import os
import re
import sys
from typing import Tuple

import torch
import torch._inductor.codegen.triton as triton_codegen
from torch._dynamo.utils import disable_cache_limit
from torch._inductor import config
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_triton_code
from torch.fx.operator_schemas import get_signature_for_torch_op
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


# Collect pointwise ops
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

    def test_upcast_ops_list(self):
        """
        Triton codegen registers various ops to be tested in this function.
        Check that some expected ops are present.
        """

        all_ops = {op for op, convert_output in triton_codegen.upcast_ops}

        for expected_op in [
            "rsqrt"
            "sqrt"
            "isnan"
            "floor"
            "ceil"
            "tan"
            "atan"
            "atanh"
            "sigmoid"
            "log2"
            "log10"
            "cosh"
            "sinh"
            "acosh"
            "asinh"
            "asin"
            "acos"
            "asinh"
            "erf"
            "lgamma"
            "sin"
            "cos"
            "exp"
            "expm1"
            "exp2"
            "abs"
        ]:
            self.assertIn(expected_op, all_ops)

    @requires_gpu()
    @parametrize("func_with_args", triton_codegen.upcast_funcs)
    @parametrize("load_upcast_to_fp32", [False, True])
    @parametrize("input_dtype", [torch.float16, torch.bfloat16])
    @config.patch("triton.use_block_ptr", True)
    def test_dtype_aware_codegen(
        self, func_with_args: Tuple[str, bool], load_upcast_to_fp32, input_dtype
    ):
        """
        Test dtype aware codegen for some tl.math/libdevice calls.
        Operands should be upcast to float32, and the output should be downcast to float16.
        """

        # Retrieve the corresponding torch op.
        override_func, convert_output = func_with_args
        op_str = override_func.__name__.removeprefix("libdevice_")
        op = getattr(torch, op_str)

        # Edge case: torch.round maps to libdevice.nearbyint
        op_str_overrides = {
            "round": "nearbyint",
        }
        override = op_str_overrides.get(op_str)
        if override is not None:
            op_str = override

        # Get the number of args for the op.
        signature = get_signature_for_torch_op(op)
        num_args = len(signature[0].parameters)

        # Test codegen and check for casts.
        inps = (torch.rand((32, 32), device=GPU_TYPE, dtype=input_dtype),) * num_args
        tl_dtype_str = str(input_dtype).replace("torch", "tl")
        with config.patch("triton.codegen_upcast_to_fp32", load_upcast_to_fp32):
            compiled = torch._dynamo.optimize("inductor")(op)
            code = run_and_get_triton_code(compiled, *inps)

            # Search the code with a regex.
            # Example code: libdevice.floor(tmp3.to(tl.float32)).to(tl.float16)
            output_cast = rf"\.to\({tl_dtype_str}\)" if convert_output else ""
            pattern = rf"{op_str}\(.*\.to\(tl\.float32\)\){output_cast}"
            cast_in_code = re.search(pattern, code, re.MULTILINE) is not None
            self.assertNotEqual(cast_in_code, load_upcast_to_fp32)


instantiate_device_type_tests(TestCase, globals(), only_for=("cuda",))

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests(needs="filelock")
