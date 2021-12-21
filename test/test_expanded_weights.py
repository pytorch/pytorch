
from functools import partial
import torch
import torch.nn as nn
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase, freeze_rng_state, make_tensor, run_tests
from torch._expanded_weights import ExpandedWeight
from torch.nn.utils._stateless import per_sample_call
from torch.testing._internal.common_nn import TestBase, module_tests, new_module_tests

from torch.overrides import handle_torch_function, has_torch_function_variadic

class TestExpandedWeightFunctional(TestCase):
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
supported_modules = ['Linear']
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

instantiate_device_type_tests(TestExpandedWeightFunctional, globals())
if __name__ == '__main__':
    run_tests()
