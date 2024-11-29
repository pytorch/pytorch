# Owner(s): ["module: inductor"]
import importlib
import os
import sys

import torch
from torch._dynamo.utils import disable_cache_limit
from torch._inductor import config
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_methods_invocations import op_db


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


instantiate_device_type_tests(TestCase, globals(), only_for=("cuda",))

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests(needs="filelock")
