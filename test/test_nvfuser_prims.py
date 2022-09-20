# Owner(s): ["module: primTorch"]

from functools import partial

import torch
from torch._prims.context import TorchRefsNvfuserCapabilityMode
from torch._prims.executor import make_traced

from torch._prims.nvfuser_prims import nvprim_names
from torch.fx.experimental.proxy_tensor import make_fx, wrapper_and_args_for_make_fx
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
        lambda op: op.torch_opinfo_name in nvprim_names
        or op.name == "ops.nvprims.var_mean",
        python_ref_db,
    )
)


class TestNvPrims(TestCase):
    # TODO: some tests from test_nvprims_grad_trace_backward are failing
    # test_nvprims_grad_trace_backward__refs_amax_cuda_float64
    # test_nvprims_grad_trace_backward__refs_amin_cuda_float64
    # test_nvprims_grad_trace_backward__refs_atan2_cuda_float64
    # test_nvprims_grad_trace_backward__refs_div_floor_rounding_cuda_float64
    # test_nvprims_grad_trace_backward__refs_div_no_rounding_mode_cuda_float64
    # test_nvprims_grad_trace_backward__refs_div_trunc_rounding_cuda_float64
    # test_nvprims_grad_trace_backward__refs_fmod_cuda_float64
    # test_nvprims_grad_trace_backward__refs_mul_cuda_float64
    # test_nvprims_grad_trace_backward__refs_var_cuda_float64
    _do_cuda_memory_leak_check = False

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

    @onlyCUDA
    @skipCUDAIfRocm
    @ops(ref_nvprims_ops, dtypes=OpDTypes.supported, allowed_dtypes=(torch.float64,))
    def test_nvprims_grad_trace_backward(self, device, dtype, op):
        # Check if result is boolean
        if op.torch_opinfo_name in ["eq", "ne", "gt", "ge", "lt", "le", "isfinite"]:
            self.skipTest("Skipped! Boolean result is not differentiable")

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        for sample in samples:
            with TorchRefsNvfuserCapabilityMode():
                input, args, kwargs = sample.input, sample.args, sample.kwargs
                outputs = op.op(input, *args, **kwargs)

                outputs = (outputs,) if not isinstance(outputs, tuple) else outputs
                grad_outputs = [torch.randn_like(out) for out in outputs]

                def func(outputs, inputs, grad_outputs):
                    return torch.autograd.grad(outputs, inputs, grad_outputs)

                inputs = [
                    x
                    for x in (input, *args)
                    if isinstance(x, torch.Tensor) and x.requires_grad
                ]
                wrapped, all_args = wrapper_and_args_for_make_fx(
                    func, [outputs, inputs, grad_outputs], {}
                )

                gm = make_fx(wrapped)(all_args)

                nodes = list(gm.graph.nodes)
                aten_ops = set(
                    [
                        str(n.target)
                        for n in nodes
                        if n.op == "call_function"
                        and "aten" == str(n.target).split(".")[0]
                    ]
                )
                # Allow aten call from inside the Autograd engine
                if "aten.is_same_size.default" in aten_ops:
                    aten_ops.remove("aten.is_same_size.default")
                if "aten.detach.default" in aten_ops:
                    aten_ops.remove("aten.detach.default")
                if "aten.view.default" in aten_ops:
                    aten_ops.remove("aten.view.default")
                if "aten.sum.dim_IntList" in aten_ops:
                    aten_ops.remove("aten.sum.dim_IntList")
                if "aten.add.Tensor" in aten_ops:
                    aten_ops.remove("aten.add.Tensor")
                self.assertEqual(len(aten_ops), 0, f"aten ops found: {aten_ops}")
                prims_ops = set(
                    [
                        str(n.target)
                        for n in nodes
                        if n.op == "call_function"
                        and "prims" == str(n.target).split(".")[0]
                    ]
                )
                self.assertEqual(len(prims_ops), 0, f"prims ops found: {prims_ops}")
                nvprims_ops = set(
                    [
                        str(n.target)
                        for n in nodes
                        if n.op == "call_function"
                        and "nvprims" == str(n.target).split(".")[0]
                    ]
                )
                self.assertTrue(len(nvprims_ops) > 0, f"nvprims ops not found")


instantiate_device_type_tests(TestNvPrims, globals())


if __name__ == "__main__":
    run_tests()
