# Owner(s): ["oncall: pt2"]

import unittest

from common_utils import decorateForModules

import torch
import torch.utils._pytree as pytree
from functorch.compile import aot_function, nop
from torch.nn.utils.rnn import PackedSequence
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_modules import module_db, modules
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)
from torch.testing._internal.optests import (
    _test_aot_autograd_forwards_backwards_helper,
)


def _test_aot_autograd_module_helper(
    self, device, dtype, training, module_info, *, dynamic=False
):
    module_cls = module_info.module_cls
    module_inputs = module_info.module_inputs_func(
        module_info, device=device, dtype=dtype, requires_grad=True, training=training
    )
    for module_input in module_inputs:
        if module_input.forward_input is None:
            continue

        args, kwargs = (
            module_input.constructor_input.args,
            module_input.constructor_input.kwargs,
        )
        m = module_cls(*args, **kwargs)
        m.to(device).to(dtype)
        m.train(training)

        # Lazy modules need to see an input first to initialize params.
        args, kwargs = (
            module_input.forward_input.args,
            module_input.forward_input.kwargs,
        )
        flat_args, args_spec = pytree.tree_flatten((args, kwargs))

        # PackedSequence is only used for RNNs. It might be possible to fake-ify if they're pytrees but
        # torchdynamo already doesn't support RNNs
        if any(tuple(isinstance(flat_arg, PackedSequence) for flat_arg in flat_args)):
            continue

        if issubclass(module_info.module_cls, torch.nn.modules.lazy.LazyModuleMixin):
            with torch.no_grad():
                m(*args, **kwargs)

        sentinel_val = -42
        is_tensor_spec = [
            sentinel_val if isinstance(arg, torch.Tensor) else arg for arg in flat_args
        ]
        args = [arg for arg in flat_args if isinstance(arg, torch.Tensor)]

        def f(params_buffers_args):
            named_params, named_buffers, args = params_buffers_args
            cur_flat_args = list(is_tensor_spec)
            args = iter(args)
            for idx, v in enumerate(cur_flat_args):
                if v == sentinel_val:
                    cur_flat_args[idx] = next(args)
            c_args, c_kwargs = pytree.tree_unflatten(cur_flat_args, args_spec)
            params_and_buffers = {**named_params, **named_buffers}
            return torch.func.functional_call(m, params_and_buffers, c_args, c_kwargs)

        named_params = dict(m.named_parameters(remove_duplicate=False))
        named_buffers = dict(m.named_buffers(remove_duplicate=False))
        num_params_buffers = len(named_params) + len(named_buffers)
        compiled_f = aot_function(
            f, nop, num_params_buffers=num_params_buffers, dynamic=dynamic
        )
        params_buffers_args = [named_params, named_buffers, args]
        _test_aot_autograd_forwards_backwards_helper(
            f,
            compiled_f,
            params_buffers_args,
            self.assertRaisesRegex,
            self.assertEqual,
            True,
        )


aot_autograd_module_failures = set(
    {
        torch.nn.CTCLoss,  # torch._subclasses.fake_tensor.DynamicOutputShapeException: aten._ctc_loss.default
        torch.nn.GaussianNLLLoss,  # RuntimeError: It appears that you're trying to get value out
        # of a tracing tensor with aten._local_scalar_dense.default -
        # erroring out! It's likely that this is caused by data-dependent
        # control flow or similar.
        torch.nn.MultiLabelMarginLoss,  # AssertionError: The values for attribute 'shape' do not match:
        # torch.Size([1]) != torch.Size([]). Outputs of the operator are different in
        # eager-mode PyTorch vs AOTAutograd. This means the operator will have incorrect
        # output underneath torch.compile. This could be because the operator's
        # implementation not traceable or that there is a bug in AOTAutograd.
        torch.nn.TransformerEncoder,  # DataDependentOutputException: aten.eq compares a mask input
        # to a causal mask tensor, to see if Boolean is_causal should be set
        # for TransformerEncoder layers, MHA and sdp custom kernels
        torch.nn.Transformer,  # DataDependentOutputException: aten.equal compares a mask input
        # to a causal mask tensor, to see if Boolean is_causal should be set
        # for TransformerEncoder layers, MHA and sdp custom kernels
        # (this bubbles up to Transformer)
    }
)

symbolic_aot_autograd_module_failures = {
    torch.nn.Transformer,  # DataDependentOutputException: aten.equal compares a mask input to a mask producing a bool
    torch.nn.TransformerEncoder,  # DataDependentOutputException: aten.equal compares a mask input to a mask producing a bool
    torch.nn.GaussianNLLLoss,  # NotImplementedError: local_scalar_dense/item NYI for torch.bool
    torch.nn.FractionalMaxPool3d,  # int() argument must be a string, a bytes-like object or a number, not 'SymFloat'
}


class TestEagerFusionModuleInfo(TestCase):
    hw_classification = HardwareClassification.CPU

    @modules(module_db, allowed_dtypes=(torch.float,))
    @decorateForModules(unittest.expectedFailure, aot_autograd_module_failures)
    def test_aot_autograd_module_exhaustive(self, device, dtype, training, module_info):
        _test_aot_autograd_module_helper(self, device, dtype, training, module_info)

    @modules(module_db, allowed_dtypes=(torch.float,))
    @decorateForModules(
        unittest.expectedFailure,
        aot_autograd_module_failures | symbolic_aot_autograd_module_failures,
    )
    def test_aot_autograd_symbolic_module_exhaustive(
        self, device, dtype, training, module_info
    ):
        _test_aot_autograd_module_helper(
            self, device, dtype, training, module_info, dynamic=True
        )


instantiate_device_type_tests(TestEagerFusionModuleInfo, globals(), only_for="cpu")

if __name__ == "__main__":
    run_tests()
