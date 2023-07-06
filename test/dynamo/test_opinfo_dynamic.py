# Owner(s): ["module: dynamo"]
import atexit
import torch
import functools
from functools import partial
import unittest
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyNativeDeviceTypes,
    onlyCPU,
    onlyCUDA,
    OpDTypes,
    ops,
    skipCPUIf,
    skipCUDAIf,
)
from torch.testing._internal.common_methods_invocations import op_db, skipOps
from torch.testing._internal.common_utils import (
    dtype_abbrs,
    IS_MACOS,
    IS_X86,
    skipCUDAMemoryLeakCheckIf,
    skipIfCrossRef,
    skipIfTorchDynamo,
    suppress_warnings,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA
from unittest.mock import patch
from torch._dynamo.test_case import run_tests




# START = 0
# END= 2000 


records = []
specialized_records = []
def print_records():
    from tabulate import tabulate
    import csv
    global records, specialized_records
    records = set(records)
    specialized_records = set(specialized_records)
    print(tabulate(records))
    print(tabulate(specialized_records))
    with open("status.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(("op", "status", "exception"))
        writer.writerows(records)
    with open("specialized.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(("op", "specialized_input"))
        writer.writerows(specialized_records)



atexit.register(print_records)
def clone_preserve_strides(x, device=None):
    if not isinstance(x, torch.Tensor):
        return x
    buffer = torch.as_strided(
        x, (x.untyped_storage().size() // x.element_size(),), (1,), 0
    )
    if not device:
        buffer = buffer.clone()
    else:
        buffer = buffer.to(device, copy=True)
    out = torch.as_strided(buffer, x.size(), x.stride(), x.storage_offset())
    return out


def check_model(
    self,
    op,
    example_inputs,
    kwargs=None,
    *,
    atol=1e-5,
    rtol=1e-5,
):

    func = op.get_op()

    def fn(*args, **kwargs):
        return func(*args, **kwargs)

    model = fn

    kwargs = kwargs or {}
    torch._dynamo.reset()

    ref_inputs = [clone_preserve_strides(x) for x in example_inputs]
    ref_kwargs = kwargs
    torch.manual_seed(0)

    correct = model(*ref_inputs, **ref_kwargs)
    called = False

    def detect_specialization(gm, *args):
        for node in gm.graph.nodes:
            example_value = node.meta.get("example_value", None)
            if node.op == "placeholder" and isinstance(example_value, torch._subclasses.FakeTensor):
                size = example_value.size()
                for sz in size:
                    if isinstance(sz, int) and sz not in (0, 1):
                        # Found a specialization
                        specialized_records.append((op.name, node.meta["grapharg"].source.name()))

                        break
            elif node.op == "placeholder" and isinstance(example_value, torch.SymInt):
                si = node.meta["example_value"]
                if len(torch.fx.experimental.symbolic_shapes.free_symbols(si)) == 0:
                        # Found a specialization
                        specialized_records.append((op.name, node.meta["grapharg"].source.name()))
                        break


    def detect_specialization_backend(gm, *args):
        detect_specialization(gm, *args)
        nonlocal called
        called = True
        return gm.forward
        
    def run(*ex, **kwargs):
        return model(*ex, **kwargs)

    run = torch._dynamo.optimize(detect_specialization_backend, dynamic=True)(run)
    torch.manual_seed(0)
    actual = run(*example_inputs, **kwargs)
    assert called, "Ran graph without calling compile_fx"
    assert type(actual) == type(correct)

    self.assertEqual(
        actual,
        correct,
        atol=atol,
        rtol=rtol,
        equal_nan=True,
        exact_dtype=True,
    )
 
_ops = partial(
    ops, dtypes=OpDTypes.supported, allowed_dtypes=[torch.float32, ],
)


class TestDynamoDynamicOpInfo(TestCase):
    check_model = check_model

    @onlyNativeDeviceTypes
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @skipCUDAIf(not HAS_CUDA, "Skipped! Triton not found")
    @skipCPUIf(not HAS_CPU, "Skipped! Supported CPU compiler not found")
    @skipIfTorchDynamo("Test uses dynamo already")
    @skipIfCrossRef
    @_ops(op_db)
    # START:END])
    @patch("torch._dynamo.config.raise_on_unsafe_aot_autograd", True)
    @onlyCUDA
    def test_comprehensive(self, device, dtype, op):
        torch._dynamo.reset()
        with torch.no_grad():
            torch.cuda.empty_cache()
        op_name = op.name
        if op.variant_test_name:
            op_name += f".{op.variant_test_name}"

        device_type = torch.device(device).type

        assert device_type in ("cuda", "cpu")
        requires_grad = (
            op.supports_autograd
            and dtype in op.supported_backward_dtypes(device_type)
            # TODO: OpInfo really ought to error out for this case, but it's
            # not exercised in test_ops_gradients atm.  The problem is not
            # complex32 per-se (which is supported by data movement only ops)
            # but that when we do backwards we expect other ops like add to work
            and not dtype == torch.complex32
        )
 
        samples = op.sample_inputs(device, dtype, requires_grad=requires_grad)

        record = (op.name, "PASS", "N/A")
        try:
            for sample_input in samples:
                args = [sample_input.input] + list(sample_input.args)
                kwargs = sample_input.kwargs
                self.check_model(
                    op,
                    args,
                    kwargs,
                )
        except Exception as e:
            record = (op.name, "FAIL", str(type(e)))
            # raise e
        records.append(record)
        

instantiate_device_type_tests(TestDynamoDynamicOpInfo, globals())

if __name__ == "__main__":
    run_tests()

