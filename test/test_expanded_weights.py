from functools import partial
from itertools import product

import torch
import torch.nn as nn
from torch.nn.utils._per_sample_grad import call_for_per_sample_grads
from torch.testing._internal.common_device_type import OpDTypes, instantiate_device_type_tests, ops
from torch.testing._internal.common_nn import TestBase, module_tests, new_module_tests
from torch.testing._internal.common_utils import TestCase, freeze_rng_state, make_tensor, run_tests
from torch.testing._internal.common_methods_invocations import SampleInput, op_db
from torch.nn.utils._expanded_weights import ExpandedWeight
from torch.nn.utils._expanded_weights.expanded_weights_utils import forward_helper, grad_if_exists, \
    grad_if_exists_for_input, unpack_expanded_weight_or_tensor, sum_over_all_but_batch_and_last_n

class TestContext:
    pass

class TestExpandedWeightHelperFunction(TestCase):
    def test_forward_helper(self, device):
        input = torch.randn(3, 4, device=device)
        weight = torch.randn(5, 4, device=device)
        bias = torch.randn(5, device=device)
        for (weight_batched, bias_batched) in product([True, False], [True, False]):
            maybe_batched_weight = ExpandedWeight(weight.clone().requires_grad_(), 3) if weight_batched else weight
            maybe_batched_bias = ExpandedWeight(bias.clone().requires_grad_(), 3) if bias_batched else bias
            args = (input, maybe_batched_weight, maybe_batched_bias)
            (res, ctx_args, aux_outputs) = forward_helper(nn.functional.linear, args, 1)
            expected = nn.functional.linear(input, weight, bias)
            self.assertEqual(res, expected)

            self.assertEqual(len(ctx_args), len(args))
            for (helper_arg, orig_arg) in zip(ctx_args, args):
                assert(helper_arg is orig_arg)  # avoids the many property checks in assertEquals
            self.assertEqual(aux_outputs, None)

    def test_forward_helper_failure_args(self, device):
        weight = torch.randn(5, 4, device=device)
        bias = torch.randn(5, device=device)
        with self.assertRaisesRegex(RuntimeError, r"do not support inputs that are also ExpandedWeights."):
            input = ExpandedWeight(torch.randn(3, 4, requires_grad=True), 3)
            forward_helper(nn.functional.linear, (input, weight, bias), 1)
        with self.assertRaisesRegex(RuntimeError, r"requires a Tensor as the first input"):
            forward_helper(nn.functional.linear, (3, weight, bias), 1)
        with self.assertRaisesRegex(RuntimeError, r"requires a batch dimension but got an input of size 0"):
            input = torch.tensor(3)
            forward_helper(nn.functional.linear, (input, weight, bias), 1)
        with self.assertRaisesRegex(RuntimeError, r"0 is not a valid batch size for Expanded Weights"):
            input = torch.randn(0, 1, 2)
            forward_helper(nn.functional.linear, (input, weight, bias), 1)
        input = torch.randn(3, 4)
        for (weight_batched, bias_batched) in product([True, False], [True, False]):
            if not weight_batched and not bias_batched:
                continue
            maybe_batched_weight = ExpandedWeight(weight.clone().requires_grad_(), 4) if weight_batched else weight
            maybe_batched_bias = ExpandedWeight(bias.clone().requires_grad_(), 4) if bias_batched else bias
            with self.assertRaisesRegex(RuntimeError, r"Expected ExpandedWeights to have batch size matching input"):
                forward_helper(nn.functional.linear, (input, maybe_batched_weight, maybe_batched_bias), 1)

    def test_forward_helper_failure_outputs(self, device):
        input = torch.randn(3, 4, device=device)
        weight = torch.randn(5, 4, device=device)
        bias = torch.randn(5, device=device)
        with self.assertRaisesRegex(RuntimeError, r"Got single output but expected at least 4"):
            forward_helper(nn.functional.linear, (input, weight, bias), 4)

    def test_grad_if_exists(self, device):
        def test_fn(_):
            return True

        orig_weight = torch.randn(4, device=device, requires_grad=True)
        expanded_weight = ExpandedWeight(orig_weight, 3)
        grad_if_exists(expanded_weight, test_fn)
        self.assertTrue(hasattr(orig_weight, 'grad_sample'))
        self.assertTrue(orig_weight.grad_sample)

        basic_tensor = torch.randn(4, device=device)
        grad_if_exists(basic_tensor, test_fn)
        self.assertFalse(hasattr(basic_tensor, 'grad_sample'))

        non_tensor = 3
        grad_if_exists(non_tensor, test_fn)
        self.assertFalse(hasattr(non_tensor, 'grad_sample'))

    def test_grad_if_exists_failure(self, device):
        def test_fn(_):
            return True

        grad_tensor = torch.randn(4, requires_grad=True, device=device)
        with self.assertRaisesRegex(RuntimeError, r"does not support a mixture of ExpandedWeight parameters and normal Parameters"):
            grad_if_exists(grad_tensor, test_fn)

    def test_grad_if_exists_for_input(self, device):
        def test_fn():
            return True

        input = torch.randn(4, requires_grad=True, device=device)
        self.assertTrue(grad_if_exists_for_input(input, test_fn))

        input.requires_grad_(False)
        self.assertTrue(grad_if_exists_for_input(input, test_fn) is None)
        self.assertTrue(grad_if_exists_for_input(4, test_fn) is None)

    def test_unpack_expanded_weight_or_tensor(self, device):
        input = torch.randn(3, requires_grad=True, device=device)
        self.assertEqual(input, unpack_expanded_weight_or_tensor(ExpandedWeight(input, 3)))

        input.requires_grad_(False)
        self.assertEqual(input, unpack_expanded_weight_or_tensor(input))
        self.assertTrue(unpack_expanded_weight_or_tensor(4) is None)

    def test_unpack_expanded_weight_or_tensor_with_custom_function(self, device):
        input = torch.randn(3, requires_grad=True, device=device)
        self.assertTrue(unpack_expanded_weight_or_tensor(ExpandedWeight(input, 3), lambda x: x is input))

        input.requires_grad_(False)
        self.assertTrue(unpack_expanded_weight_or_tensor(input, lambda x: x is input))
        self.assertTrue(unpack_expanded_weight_or_tensor(4, lambda x: x is input) is None)

    def test_unpack_expanded_weight_or_tensor_failure(self, device):
        input = torch.randn(3, requires_grad=True, device=device)
        with self.assertRaisesRegex(RuntimeError, r"does not support a mixture of ExpandedWeight parameters and normal Parameters"):
            unpack_expanded_weight_or_tensor(input)

        with self.assertRaisesRegex(RuntimeError, r"does not support a mixture of ExpandedWeight parameters and normal Parameters"):
            unpack_expanded_weight_or_tensor(input, lambda x: x is input)

    def test_sum_over_all_but_batch_and_last_n(self, device):
        input = torch.randn(1, 2, 3, 4, 5, device=device)
        res = sum_over_all_but_batch_and_last_n(input, 2)
        expected = input.sum((1, 2))
        self.assertEqual(res, expected)

        res = sum_over_all_but_batch_and_last_n(input, 0)
        expected = input.sum((1, 2, 3, 4))
        self.assertEqual(res, expected)

        res = sum_over_all_but_batch_and_last_n(input, 4)
        self.assertEqual(res, input)

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
            diff_input_list = (ew_input,) + tuple(ew_args) + tuple(ew_kwargs.values())
            diff_input_list = [i for i in diff_input_list if is_diff_tensor(i)]
            diff_input_list = [i.orig_weight if isinstance(i, ExpandedWeight) else i for i in diff_input_list]
            if not diff_input_list:
                continue
            result = run_op(op, ew_input, *ew_args, **ew_kwargs)
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
    def test_expanded_weight_per_sample_grad_multiple_calls(self, device, dtype, op):
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
            diff_input_list = (ew_input,) + tuple(ew_args) + tuple(ew_kwargs.values())
            diff_input_list = [i for i in diff_input_list if is_diff_tensor(i)]
            diff_input_list = [i.orig_weight if isinstance(i, ExpandedWeight) else i for i in diff_input_list]
            if not diff_input_list:
                continue

            result = run_op(op, ew_input, *ew_args, **ew_kwargs)
            result.sum().backward()  # grad doesn't work with ExpandedWeight because it calls __torch_function__
            result = run_op(op, ew_input, *ew_args, **ew_kwargs)  # running it again makes a list for multiple batches
            result.sum().backward()
            expanded_weight_grad = tuple(i.grad_sample if hasattr(i, "grad_sample") else i.grad for i in diff_input_list)
            expanded_weight_grad = tuple(torch.stack(i).flatten(end_dim=1) if isinstance(i, list) else i
                                         for i in expanded_weight_grad)

            # get per sample grads with for loop
            func = partial(run_op, op)
            repeated_input = input.unsqueeze(0).expand(2, *input.shape).reshape(2 * input.shape[0], *input.shape[1:])
            per_sample_grad = for_loop_per_sample_grad(2 * batch_size, repeated_input, func, *args, **kwargs)
            if op.name != "nn.functional.embedding":  # embedding's input is not differentiable
                per_sample_input = per_sample_grad[0].reshape(2, *input.shape).sum(0)
                per_sample_grad = (per_sample_input, *per_sample_grad[1:])

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

    def test_expanded_weight_error(self, device):
        batch_size = 3
        sample_input = make_tensor((batch_size, 4), device, torch.float32, requires_grad=True)
        sample_weight = make_tensor((4), device, torch.float32, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, r"Expanded Weights encountered but cannot handle function"):
            torch.add(sample_input, ExpandedWeight(sample_weight, batch_size))


class TestExpandedWeightModule(TestCase):
    def _do_test(self, module, input):
        if sum(1 for _ in module.parameters()) == 0:  # for norms with affine=False
            return
        batch_size = input.shape[0]
        with freeze_rng_state():
            # get per sample grads with ExpandedWeights context manager
            actual_res = call_for_per_sample_grads(module, batch_size, input).sum()
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

    def _do_test_multi_input(self, module, input):
        if sum(1 for _ in module.parameters()) == 0:  # for norms with affine=False
            return
        batch_size = input.shape[0]
        with freeze_rng_state():
            # get per sample grads with ExpandedWeights context manager, calling .backward() twice
            actual_res = call_for_per_sample_grads(module, batch_size, input).sum()
            actual_res.backward()
            actual_res = call_for_per_sample_grads(module, batch_size, input).sum()
            actual_res.backward()
            actual_grads = []
            for param in module.parameters():
                assert isinstance(param.grad_sample, list)
                actual_grads.append(torch.stack(param.grad_sample).flatten(end_dim=1))
                del param.grad_sample

            # get per sample grads with a for loop, running over the input twice
            expected_grads = []
            for i in range(2 * batch_size):
                res = module(input[i % batch_size].unsqueeze(0)).sum()
                expected_grads.append(torch.autograd.grad(res, module.parameters(), torch.ones_like(res)))
            expected_grads = tuple(torch.stack(grad) for grad in zip(*expected_grads))
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

    def test_context_manager_multiple_inputs(self, test_case):
        module = self.constructor(*self.constructor_args)
        input = self._get_input()
        if len(input.shape) == 0 or input.shape[0] == 0:
            return
        if self.constructor == torch.nn.Linear and len(input.shape) == 1:
            return
        test_case._do_test_multi_input(module, input)

# TODO: Once all of these use ModuleInfo, replace with ModuleInfo tests
supported_modules = ['Linear', 'Conv1d', 'Conv2d', 'Conv3d', 'GroupNorm', 'LayerNorm', 'InstanceNorm', 'Embedding']
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
    test_name_multi_input = test.get_name() + "_multiple_inputs"
    if hasattr(TestExpandedWeightModule, test_name_multi_input):
        raise RuntimeError('Found two tests with the same name: ' + test_name)
    if decorator is not None:
        fn = decorator(fn)
    setattr(TestExpandedWeightModule, test_name, lambda self, test=test: test.test_context_manager(self))
    setattr(TestExpandedWeightModule, test_name_multi_input, 
            lambda self, test=test: test.test_context_manager_multiple_inputs(self))

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
        convolutions = ["nn.functional.conv1d", "nn.functional.conv2d", "nn.functional.conv3d"]
        if op.name == "nn.functional.linear":
            is_supported_input = len(input.input.shape) > 1  # input of rank 1 means no batch dim
        elif op.name == "nn.functional.layer_norm":
            normalized_shape = input.args[0]
            is_supported_input = input.input.shape != normalized_shape  # would cause inter-batch operations
        elif op.name in convolutions:
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
    return isinstance(t, ExpandedWeight) or (isinstance(t, torch.Tensor) and t.requires_grad)

def clone_if_tensor(t):
    if isinstance(t, torch.Tensor):
        res = torch.clone(t).detach()
        res.requires_grad_(t.requires_grad)
        return res
    else:
        return t

instantiate_device_type_tests(TestExpandedWeightHelperFunction, globals())
instantiate_device_type_tests(TestExpandedWeightFunctional, globals())
if __name__ == '__main__':
    run_tests()
