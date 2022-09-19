# Owner(s): ["module: primTorch"]

from functools import partial

import torch
from torch._prims.executor import make_traced

from torch._prims.nvfuser_prims import nvprim_names
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
    OpDTypes,
    ops,
    skipCUDAIfRocm,
)
from torch.testing._internal.common_methods_invocations import python_ref_db
from torch.testing._internal.common_utils import (
    gradcheck,
    is_iterable_of_tensors,
    TestCase,
)
from torch.utils._pytree import tree_flatten, tree_unflatten


ref_nvprims_ops = tuple(
    filter(
        lambda op: op.torch_opinfo_name in nvprim_names,
        python_ref_db,
    )
)


class TestNvPrims(TestCase):
    # Adapted from test_ops_gradients.py
    def _check_helper(self, device, dtype, op, variant):
        if not op.supports_dtype(dtype, torch.device(device).type):
            self.skipTest(f"Skipped! {op.name} does not support dtype {str(dtype)}")

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        for sample in samples:
            # Check if result is boolean
            if op.torch_opinfo_name in ["eq", "ne", "gt", "ge", "lt", "le", "isfinite"]:
                self.skipTest("Skipped! Boolean result is not differentiable")

            all_args, _ = tree_flatten((sample.input, *sample.args, sample.kwargs))
            gradcheck_args = tuple(
                x for x in all_args if (isinstance(x, torch.Tensor) and x.requires_grad)
            )

            def _input_recomposition_helper(inputs, inp, input_idx):
                if is_iterable_of_tensors(inp):
                    tensor_list = []
                    for x in inp:
                        if isinstance(x, torch.Tensor) and x.requires_grad:
                            tensor_list.append(inputs[input_idx])
                            input_idx = input_idx + 1
                        else:
                            tensor_list.append(x)
                    return tensor_list, input_idx
                elif isinstance(inp, torch.Tensor) and inp.requires_grad:
                    return inputs[input_idx], input_idx + 1
                else:
                    return inp, input_idx

            def fn(*inputs):
                # Puts inputs back into sample properly
                positional_args = []
                input_idx = 0
                inp, input_idx = _input_recomposition_helper(
                    inputs, sample.input, input_idx
                )
                positional_args.append(inp)

                for x in sample.args:
                    inp, input_idx = _input_recomposition_helper(inputs, x, input_idx)
                    positional_args.append(inp)

                # Recreates kwargs
                kwargs = {}
                for k, v in sample.kwargs.items():
                    inp, input_idx = _input_recomposition_helper(inputs, v, input_idx)
                    kwargs[k] = inp

                output = op.gradcheck_wrapper(variant, *positional_args, **kwargs)
                if sample.output_process_fn_grad is not None:
                    return sample.output_process_fn_grad(output)
                return output

            self.assertTrue(
                gradcheck(
                    fn,
                    gradcheck_args,
                    check_grad_dtypes=True,
                    nondet_tol=op.gradcheck_nondet_tol,
                    fast_mode=op.gradcheck_fast_mode,
                    check_undefined_grad=True,
                    check_batched_grad=False,
                )
            )

    @onlyCUDA
    @skipCUDAIfRocm
    @ops(ref_nvprims_ops, dtypes=OpDTypes.supported, allowed_dtypes=(torch.float64,))
    def test_nvprims_grad(self, device, dtype, op):
        from copy import copy

        op = copy(op)
        # Here we are testing aten executor because we are verifying that the
        # backward pass is correct. gradcheck doesn't work with nvfuser executor
        # because nvfuser's result is a detached tensor.
        op.op = partial(make_traced(op.op), executor="aten")

        self._check_helper(device, dtype, op, op.get_op())


instantiate_device_type_tests(TestNvPrims, globals())


if __name__ == "__main__":
    run_tests()
