
from functools import partial
import torch
import torch.nn as nn
from torch.testing._internal.common_device_type import OpDTypes, instantiate_device_type_tests, ops
from torch.testing._internal.common_utils import TestCase, freeze_rng_state, make_tensor, run_tests
from torch.testing._internal.common_methods_invocations import SampleInput, op_db
from torch._expanded_weights import ExpandedWeight
from torch.nn.utils._stateless import per_sample_call
from torch.testing._internal.common_nn import TestBase, module_tests, new_module_tests

from torch.overrides import handle_torch_function, has_torch_function_variadic

class TestExpandedWeightFunctional(TestCase):
    @ops(filter(lambda op: op.supports_expanded_weight, op_db), dtypes=OpDTypes.supported, allowed_dtypes=(torch.double,))
    def test_expanded_weight_per_sample_grad(self, device, dtype, op):
        sample_inputs = op.sample_inputs(device, dtype, requires_grad=True)
        for sample_input in supported_inputs(op, sample_inputs):
            if op.name == "nn.functional.embedding":  # embedding flips its argument order for autograd tests
                sample_input = SampleInput(sample_input.args[0], args=(sample_input.input,), kwargs=sample_input.kwargs)
            input = sample_input.input
            args = sample_input.args
            kwargs = sample_input.kwargs
            batch_size = input.shape[0] if len(input.shape) > 1 else 1

            # get per sample grads with ExpandedWeights objects
            (ew_input, ew_args, ew_kwargs) = make_expanded_weight(sample_input, batch_size)
            result = run_op(op, ew_input, *ew_args, **ew_kwargs)
            diff_input_list = (ew_input,) + tuple(ew_args) + tuple(ew_kwargs.values())
            diff_input_list = [i for i in diff_input_list if is_diff_tensor(i)]
            diff_input_list = [i.orig_weight if isinstance(i, ExpandedWeight) else i for i in diff_input_list]
            if not diff_input_list:
                continue
            result.sum().backward()  # grad doesn't work with ExpandedWeight because it calls __torch_function__
            expanded_weight_grad = tuple(i.grad_sample if hasattr(i, "grad_sample") else i.grad for i in diff_input_list)

            # get per sample grads with for loop
            func = partial(run_op, op)
            per_sample_grad = for_loop_per_sample_grad(batch_size, input, func, *args, **kwargs)

            # check equality
            self.assertEqual(len(per_sample_grad), len(expanded_weight_grad))
            for (result_grad, expected_grad) in zip(expanded_weight_grad, per_sample_grad):
                if result_grad is None:
                    result_grad = torch.zeros_like(expected_grad)
                assert torch.allclose(result_grad, expected_grad), f"Got {result_grad}, expected {expected_grad}"

    @ops(filter(lambda op: op.supports_expanded_weight, op_db), dtypes=OpDTypes.supported, allowed_dtypes=(torch.double,))
    def test_unsupported_expand_weights(self, device, dtype, op):
        sample_inputs = op.sample_inputs(device, dtype, requires_grad=True)
        unsupported_inputs = supported_inputs(op, sample_inputs, supported_inputs=False)
        for sample_input in unsupported_inputs:
            with self.assertRaisesRegex(RuntimeError, r"Expanded Weights"):
                if op.name == "nn.functional.embedding":  # embedding flips its argument order for autograd tests
                    sample_input = SampleInput(sample_input.args[0], args=(sample_input.input,), kwargs=sample_input.kwargs)
                input = sample_input.input

                batch_size = input.shape[0] if len(input.shape) > 1 else 1

                # get per sample grads with ExpandedWeights objects
                (ew_input, ew_args, ew_kwargs) = make_expanded_weight(sample_input, batch_size)
                result = run_op(op, ew_input, *ew_args, **ew_kwargs)
                diff_input_list = (ew_input,) + tuple(ew_args) + tuple(ew_kwargs.values())
                diff_input_list = [i for i in diff_input_list if is_diff_tensor(i)]
                diff_input_list = [i.orig_weight if isinstance(i, ExpandedWeight) else i for i in diff_input_list]
                result.sum().backward()  # grad doesn't work with ExpandedWeight because it calls __torch_function__

    @ops(filter(lambda op: op.supports_expanded_weight, op_db), dtypes=OpDTypes.supported)
    def test_expanded_weight_forward(self, device, dtype, op):
        sample_inputs = op.sample_inputs(device, dtype)
        for sample_input in supported_inputs(op, sample_inputs):
            batch_size = sample_input.input.shape[0] if len(sample_input.input.shape) > 1 else 1
            (ew_input, ew_args, ew_kwargs) = make_expanded_weight(sample_input, batch_size)
            expanded_weight_result = op(ew_input, *ew_args, **ew_kwargs)
            normal_result = op(sample_input.input, *sample_input.args, **sample_input.kwargs)
            self.assertEqual(expanded_weight_result, normal_result)

    def test_expanded_weight_fallback(self, device):
        def linear_fallback(input, weight, bias):
            if has_torch_function_variadic(input, weight, bias):
                return handle_torch_function(linear_fallback, (input, weight, bias,), input, weight, bias,)
            return torch.nn.functional.linear(input, weight, bias)

        batch_size = 3
        sample_linear_input = make_tensor((batch_size, 4), device, torch.float32, requires_grad=True)
        sample_weight = make_tensor((5, 4), device, torch.float32, requires_grad=True)
        sample_bias = make_tensor((5), device, torch.float32, requires_grad=True)

        # use fallback
        fallback_weight = torch.clone(sample_weight)
        fallback_bias = torch.clone(sample_bias)
        ew_constructor = partial(ExpandedWeight, batch_size=batch_size)
        res = linear_fallback(sample_linear_input, ew_constructor(fallback_weight), ew_constructor(fallback_bias)).sum()
        res.backward()
        fallback_grad = (sample_linear_input.grad, fallback_weight.grad_sample, fallback_bias.grad_sample)

        sample_linear_input.grad = None  # reset input to be used again

        # use for loop
        per_sample_grad = for_loop_per_sample_grad(batch_size, sample_linear_input, linear_fallback, sample_weight, sample_bias)

        # check equality
        self.assertEqual(len(per_sample_grad), len(fallback_grad))
        for (result_grad, expected_grad) in zip(fallback_grad, per_sample_grad):
            if result_grad is None:
                result_grad = torch.zeros_like(expected_grad)
            assert torch.allclose(result_grad, expected_grad)


class TestExpandedWeightModule(TestCase):
    def _do_test(self, module, input):
        if sum(1 for _ in module.parameters()) == 0:  # for norms with affine=False
            return
        batch_size = input.shape[0]
        with freeze_rng_state():
            # get per sample grads with ExpandedWeights context manager
            actual_res = per_sample_call(module, batch_size, input).sum()
            actual_res.backward()
            actual_grads = []
            for param in module.parameters():
                actual_grads.append(param.grad_sample)
                del param.grad_sample

            # get per sample grads with a for loop
            expected_res = torch.tensor(0.)
            expected_grads = []
            for i in range(batch_size):
                res = module(input[i].unsqueeze(0)).sum()
                expected_grads.append(torch.autograd.grad(res, module.parameters(), torch.ones_like(res)))
                expected_res += res
            expected_grads = tuple(torch.stack(grad) for grad in zip(*expected_grads))
        self.assertEqual(actual_res, expected_res)
        assert [torch.allclose(actual, expected) for (actual, expected) in zip(actual_grads, expected_grads)]

class ContextManagerTests(TestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def constructor_args(self):
        return self._get_arg('constructor_args', False)

    def test_context_manager(self, test_case):
        module = self.constructor(*self.constructor_args)
        input = self._get_input()
        if len(input.shape) == 0 or input.shape[0] == 0:
            return
        if self.constructor == torch.nn.Linear and len(input.shape) == 1:
            return
        test_case._do_test(module, input)

# TODO: Once all of these use ModuleInfo, replace with ModuleInfo tests
supported_modules = ['Linear', 'Conv2d', 'GroupNorm', 'LayerNorm', 'InstanceNorm', 'Embedding']
supported_tests = [t for t in module_tests + new_module_tests if 'module_name' in t and t['module_name'] in supported_modules]
for test_param in supported_tests:
    if 'constructor' not in test_param:
        name = test_param.pop('module_name')
        test_param['constructor'] = getattr(nn, name)
    decorator = test_param.pop('decorator', None)
    test = ContextManagerTests(**test_param)
    test_name = test.get_name()
    if hasattr(TestExpandedWeightModule, test_name):
        raise RuntimeError('Found two tests with the same name: ' + test_name)
    if decorator is not None:
        fn = decorator(fn)
    setattr(TestExpandedWeightModule, test_name, lambda self, test=test: test.test_context_manager(self))

# ------------- HELPER FUNCTIONS -----------------

def run_op(op, input, *args, **kwargs):
    r"""
    OpInfo for Embedding switches the input and weight so autograd tests will only check the derivative
    of the weight, not the input, which can't be differentiable since its dtype is int. Calls op,
    using the special ordering that Embedding's OpInfo expects for that case.
    """
    if op.name == "nn.functional.embedding":
        return op(args[0], input, **kwargs)
    else:
        return op(input, *args, **kwargs)

def make_expanded_weight(sample_input, batch_size):
    def expanded_weight_or_clone(arg):
        return ExpandedWeight(torch.clone(arg), batch_size) if is_diff_tensor(arg) else clone_if_tensor(arg)

    ew_input = clone_if_tensor(sample_input.input)
    ew_args = tuple(expanded_weight_or_clone(arg) for arg in sample_input.args)
    ew_kwargs = {name: expanded_weight_or_clone(arg) for (name, arg) in sample_input.kwargs.items()}
    return ew_input, ew_args, ew_kwargs

def supported_inputs(op, sample_inputs, supported_inputs=True):
    r"""
    ExpandedWeights currently does not support some use cases when there's no batch dimension or
    operations that would cause inter-batch operations. Removes all of the cases it cannot deal with
    """
    def filter_fn(input):
        if op.name == "nn.functional.linear":
            is_supported_input = len(input.input.shape) > 1  # input of rank 1 means no batch dim
        elif op.name == "nn.functional.layer_norm":
            normalized_shape = input.args[0]
            is_supported_input = input.input.shape != normalized_shape  # would cause inter-batch operations
        elif op.name == "nn.functional.conv2d":
            # currently can't deal with padding computation on Python level
            is_supported_input = 'padding' not in input.kwargs or not isinstance(input.kwargs['padding'], str)
        elif op.name == "nn.functional.embedding":
            idx = input.args[0]
            is_supported_input = len(idx.shape) > 1  # there's no batch size
        else:
            is_supported_input = True
        is_supported_input = is_supported_input and input.input.shape[0] > 0  # 0 is not a valid batch size
        return is_supported_input if supported_inputs else not is_supported_input
    return [input for input in sample_inputs if filter_fn(input)]

def for_loop_per_sample_grad(batch_size, input, func, *args, **kwargs):
    # get per sample grads by getting derivative for each input in a for loop
    per_sample_grad = []
    for i in range(batch_size):
        per_sample_input = input[i]
        result = func(per_sample_input.unsqueeze(0), *args, **kwargs)
        diff_input_list = (per_sample_input,) + tuple(args) + tuple(kwargs.values())
        diff_input_list = [i for i in diff_input_list if isinstance(i, torch.Tensor) and i.requires_grad]
        per_sample_grad.append(torch.autograd.grad(result, diff_input_list, torch.ones_like(result), allow_unused=True))
    if len(per_sample_grad) == batch_size:
        per_sample_grad = tuple(torch.stack(grad) for grad in zip(*per_sample_grad))
    return per_sample_grad

def is_diff_tensor(t):
    return isinstance(t, torch.Tensor) and t.requires_grad

def clone_if_tensor(t):
    if isinstance(t, torch.Tensor):
        res = torch.clone(t).detach()
        res.requires_grad_(t.requires_grad)
        return res
    else:
        return t

instantiate_device_type_tests(TestExpandedWeightFunctional, globals())
if __name__ == '__main__':
    run_tests()
