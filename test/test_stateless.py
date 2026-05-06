# Owner(s): ["module: nn"]

import contextlib
import os
import re
import subprocess
import sys
import unittest
from copy import deepcopy

import torch
import torch.nn.utils.stateless as stateless
from torch.fx.experimental.proxy_tensor import make_fx
from torch.optim._stateless import _reparametrize_optimizer
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_utils import run_tests, TestCase, parametrize, instantiate_parametrized_tests, \
    subtest


class MockModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(1, 1)
        self.buffer = torch.nn.Buffer(torch.ones(1))
        self.foo = 0.0

    def forward(self, x):
        return self.l1(x) + self.buffer


class MockTiedModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(1, 1)
        self.tied_bias = self.l1.bias
        self.buffer = torch.nn.Buffer(torch.ones(1))
        self.tied_buffer = self.buffer

    def forward(self, x):
        return self.l1(x) + self.tied_bias + self.buffer + self.tied_buffer


class ChainedLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(2, 2)
        self.l2 = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.l2(self.l1(x))


class TestStatelessFunctionalAPI(TestCase):
    def _run_call_with_mock_module(self, module, functional_call, device='cpu', prefix=''):

        x = torch.rand((1, 1)).to(device)
        weight = torch.tensor([[1.0]], device=device)
        bias = torch.tensor([0.0], device=device)
        buffer = torch.tensor([0.0], device=device)
        if prefix != '':
            parameters = {f'{prefix}.l1.weight': weight,
                          f'{prefix}.l1.bias': bias,
                          f'{prefix}.buffer': buffer}
        else:
            parameters = {'l1.weight': weight,
                          'l1.bias': bias,
                          'buffer': buffer}
        to_check = module
        if prefix != '':
            to_check = getattr(module, prefix)
        prev_weight = to_check.l1.weight.clone()
        prev_buffer = to_check.buffer.clone()
        # the parameters represent an identity function contrary to the
        # existing params in module. So here we expect the result to be the
        # same as the input if the weight swapping went well.
        res = functional_call(module, parameters, x)
        self.assertEqual(x, res)
        # check that the weight remain unmodified
        cur_weight = to_check.l1.weight
        cur_buffer = to_check.buffer
        self.assertEqual(cur_weight, prev_weight)
        self.assertEqual(cur_buffer, prev_buffer)

    @contextlib.contextmanager
    def _ensure_module_unchanged(self, module, message):
        orig_parameters, orig_buffers = tuple(module.parameters()), tuple(module.buffers())
        orig_tensors = orig_parameters + orig_buffers
        orig_tensors_values = tuple(t.clone() for t in orig_tensors)
        try:
            yield module
        finally:
            parameters, buffers = tuple(module.parameters()), tuple(module.buffers())
            self.assertTrue(
                len(parameters) == len(orig_parameters)
                and len(buffers) == len(orig_buffers)
                and all(
                    t1 is t2 and torch.allclose(t1, t3)
                    for t1, t2, t3 in zip(
                        orig_tensors,
                        parameters + buffers,
                        orig_tensors_values,
                    )
                ),
                message,
            )

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_functional_call(self, functional_call):
        module = MockModule()
        self._run_call_with_mock_module(module, functional_call)

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_functional_call_with_jit(self, functional_call):
        module = MockModule()
        jit_module = torch.jit.script(module)
        with self.assertRaisesRegex(
            RuntimeError,
            r'used with Jitted modules'
        ):
            self._run_call_with_mock_module(jit_module, functional_call)
        x = torch.rand((1, 1))
        traced_module = torch.jit.trace(module, x)
        with self.assertRaisesRegex(
            RuntimeError,
            r'used with Jitted modules'
        ):
            self._run_call_with_mock_module(traced_module, functional_call)

    @unittest.skipIf(not TEST_MULTIGPU, 'multi-GPU not supported')
    @unittest.skip("This doesn't work right now")
    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_functional_call_with_data_parallel(self, functional_call):
        module = MockModule()
        module.cuda()
        dp_module = torch.nn.DataParallel(module, [0, 1])
        self._run_call_with_mock_module(dp_module, functional_call, device='cuda', prefix='module')

    @unittest.skipIf(not TEST_MULTIGPU, 'multi-GPU not supported')
    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_functional_call_with_data_parallel_error(self, functional_call):
        module = MockModule()
        module.cuda()
        dp_module = torch.nn.DataParallel(module, [0, 1])
        with self.assertRaisesRegex(RuntimeError, r'used with nn.DataParallel module'):
            functional_call(
                dp_module,
                {'module.weight': torch.zeros(5, device='cuda')},
                (torch.ones(2, 5, device='cuda'),))

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_functional_call_with_gradient(self, functional_call):
        module = MockModule()
        x = torch.rand((1, 1))
        weight = torch.tensor([[1.0]], requires_grad=True)
        bias = torch.tensor([0.0], requires_grad=True)
        buffer = torch.tensor([0.0])
        parameters = {'l1.weight': weight,
                      'l1.bias': bias,
                      'buffer': buffer}
        res = functional_call(module, parameters, x)
        # Check that a backward step calculates the gradient of the supplied parameters
        res.backward()
        self.assertIsNotNone(weight.grad)
        self.assertIsNotNone(bias.grad)
        self.assertIsNone(buffer.grad)
        # Gradient was not calculated for the module stated and buffers
        self.assertIsNone(module.l1.weight.grad)
        self.assertIsNone(module.l1.bias.grad)
        self.assertIsNone(module.buffer.grad)

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_functional_batch_norm(self, functional_call):
        module = torch.nn.BatchNorm1d(10)
        module.train()  # Allow stats update
        # lets replace the running_mean buffer and check if its correctly updated
        x = torch.full((20, 10), 128.0)
        rm = torch.zeros(10)
        parameters = {'running_mean': rm}
        prev_rm = module.running_mean.clone()
        functional_call(module, parameters, x)
        cur_rm = module.running_mean
        self.assertEqual(cur_rm, prev_rm)
        self.assertEqual(rm, torch.full((10,), 12.8))
        # Now run functional without reparameterization and check that the module has
        # been updated
        functional_call(module, {}, x)
        self.assertEqual(module.running_mean, torch.full((10,), 12.8))

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_circular_references(self, functional_call):
        module = MockModule()
        # Add a circular reference
        module.l1.m = module
        x = torch.rand((1, 1))
        weight = torch.tensor([[1.0]])
        bias = torch.tensor([0.0])
        buffer = torch.tensor([0.0])
        parameters = {'l1.m.l1.weight': weight,
                      'l1.bias': bias,
                      'l1.m.buffer': buffer}
        prev_weight = module.l1.weight.clone()
        prev_buffer = module.buffer.clone()
        res = functional_call(module, parameters, x, tie_weights=False)
        self.assertEqual(x, res)
        # check that the weights remain unmodified and were correctly accessed
        cur_weight = module.l1.weight
        cur_buffer = module.buffer
        self.assertEqual(cur_weight, prev_weight)
        self.assertEqual(cur_buffer, prev_buffer)

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_reparametrized_module_change_parametrization_original(self, functional_call):
        module = MockModule()
        torch.nn.utils.parametrizations.spectral_norm(module.l1)
        self.assertTrue('l1.parametrizations.weight.original' in dict(module.named_parameters()))
        orig_sn_weight = module.l1.weight.clone()
        x = torch.rand((1, 1))
        # We substitute the parameter inside the parametrization
        # the parametrization itself is not overwritten so it will be applied with a different
        # value for the original tensor
        parameters = {'l1.parametrizations.weight.original': torch.nn.Parameter(torch.tensor([[1.0]])),
                      'l1.bias': torch.tensor([0.0]),
                      'buffer': torch.tensor([0.0])}
        res = functional_call(module, parameters, x)
        self.assertEqual(x, res)
        # verify that the spectral normalization is still applied
        self.assertTrue('l1.parametrizations.weight.original' in dict(module.named_parameters()))
        self.assertEqual(orig_sn_weight, module.l1.weight)

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_reparametrize_module_fail_reset_to_original(self, functional_call):
        module = MockModule()
        torch.nn.utils.parametrizations.spectral_norm(module.l1)
        self.assertTrue('l1.parametrizations.weight.original' in dict(module.named_parameters()))
        orig_sn_weight = module.l1.weight.clone()
        # We substitute the parameter inside the parametrization
        # the parametrization itself is not overwritten so it will be applied with a different
        # value for the original tensor
        parameters = {'l1.parametrizations.weight.original': torch.nn.Parameter(torch.tensor([[1.0]])),
                      'l1.bias': torch.tensor([0.0]),
                      'buffer': torch.tensor([0.0])}

        with self.assertRaisesRegex(RuntimeError, "shapes cannot be multiplied"):
            @torch._dynamo.disable
            def _error_case():
                x = torch.rand((4, 5))  # to work, it should be of size (1, 1)
                functional_call(module, parameters, x)  # this call will fail because x is the wrong size
            _error_case()

        # verify that the spectral normalization is still applied
        self.assertTrue('l1.parametrizations.weight.original' in dict(module.named_parameters()))
        self.assertEqual(orig_sn_weight, module.l1.weight)

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_reparametrize_some_weights(self, functional_call):
        module = MockModule()
        weight = torch.tensor([[2.0]])
        extra = torch.tensor([1.0])

        parameters = {'l1.weight': weight}
        x = torch.randn(1, 1)
        out = functional_call(module, parameters, x)
        self.assertEqual(out, x * weight + module.l1.bias + module.buffer)

        parameters = {'l1.weight': weight,
                      'extra': extra}
        x = torch.randn(1, 1)
        out = functional_call(module, parameters, x)
        self.assertEqual(out, x * weight + module.l1.bias + module.buffer)

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_reparametrize_strict(self, functional_call):
        module = MockModule()
        weight = torch.tensor([[2.0]])
        bias = torch.tensor([5.0])
        buffer = torch.tensor([3.0])
        extra = torch.tensor([1.0])

        # All weights no error
        parameters = {'l1.weight': weight,
                      'l1.bias': bias,
                      'buffer': buffer}
        x = torch.randn(1, 1)
        with self._ensure_module_unchanged(
            module,
            'the module should not have been modified by a successful call',
        ):
            out = functional_call(module, parameters, x, strict=True)
            self.assertEqual(out, x * weight + bias + buffer)

        # Some weights
        parameters = {'l1.weight': weight}
        x = torch.randn(1, 1)
        with self._ensure_module_unchanged(
            module,
            'the module should not have been modified by a failed call',
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                re.escape("Missing key(s): 'buffer', 'l1.bias'."),
            ):
                out = functional_call(module, parameters, x, strict=True)

        # Extra keys
        parameters = {'l1.weight': weight,
                      'l1.bias': bias,
                      'buffer': buffer,
                      'extra': extra}
        x = torch.randn(1, 1)
        with self._ensure_module_unchanged(
            module,
            'the module should not have been modified by a failed call',
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                re.escape("Unexpected key(s): 'extra'."),
            ):
                out = functional_call(module, parameters, x, strict=True)

        # Some weights with extra keys
        parameters = {'l1.weight': weight,
                      'extra': extra}
        x = torch.randn(1, 1)
        with self._ensure_module_unchanged(
            module,
            'the module should not have been modified by a failed call',
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                re.escape("Unexpected key(s): 'extra'.") + r'\s+' + re.escape("Missing key(s): 'buffer', 'l1.bias'."),
            ):
                out = functional_call(module, parameters, x, strict=True)

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_reparametrize_special(self, functional_call):
        class NonTensor:
            def __repr__(self):
                return f'<{self.__class__.__name__}>'

        module = MockModule()
        weight = torch.tensor([[2.0]])
        bias = torch.tensor([5.0])
        buffer = torch.tensor([3.0])
        non_tensor = NonTensor()

        # Set to None
        parameters = {'l1.weight': weight,
                      'l1.bias': None,
                      'buffer': buffer}
        x = torch.randn(1, 1)
        with self._ensure_module_unchanged(
            module,
            'the module should not have been modified by a successful call',
        ):
            out = functional_call(module, parameters, x)
            self.assertEqual(out, x * weight + buffer)

        # Set non-tensor
        parameters = {'l1.weight': non_tensor}
        x = torch.randn(1, 1)
        with self._ensure_module_unchanged(
            module,
            'the module should not have been modified by a failed call',
        ):
            with self.assertRaisesRegex(
                TypeError,
                re.escape("<NonTensor> is not an instance of torch.Tensor"),
            ):
                out = functional_call(module, parameters, x)

        # Set non-tensor attribute
        parameters = {'l1.weight': weight, 'foo': torch.tensor([1.0])}
        x = torch.randn(1, 1)
        with self._ensure_module_unchanged(
            module,
            'the module should not have been modified by a failed call',
        ):
            with self.assertRaisesRegex(
                TypeError,
                re.escape("attribute `foo`: 0.0 is not an instance of torch.Tensor"),
            ):
                out = functional_call(module, parameters, x)

        # Set non-exist submodule
        parameters = {'l1.weight': weight,
                      'l2.bias': bias}
        x = torch.randn(1, 1)
        with self._ensure_module_unchanged(
            module,
            'the module should not have been modified by a failed call',
        ):
            with self.assertRaisesRegex(
                AttributeError,
                re.escape("MockModule has no attribute `l2`"),
            ):
                out = functional_call(module, parameters, x)

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_tied_weights_warns(self, functional_call):
        module = MockModule()
        module.tied_bias = module.l1.bias
        module.tied_buffer = torch.nn.Buffer(module.buffer)

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_reparametrize_tie_weights(self, functional_call):
        module = MockTiedModule()
        weight = torch.tensor([[2.0]])
        bias = torch.tensor([5.0])
        buffer = torch.tensor([3.0])
        extra = torch.tensor([1.0])

        parameters = {'l1.weight': weight,
                      'l1.bias': bias,
                      'buffer': buffer}
        x = torch.randn(1, 1)
        out = functional_call(module, parameters, x, tie_weights=True)
        self.assertEqual(out, x * weight + bias + bias + buffer + buffer)

        parameters = {'l1.weight': weight,
                      'l1.bias': bias,
                      'buffer': buffer,
                      'extra': extra}
        x = torch.randn(1, 1)
        out = functional_call(module, parameters, x, tie_weights=True)
        self.assertEqual(out, x * weight + bias + bias + buffer + buffer)

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_reparametrize_tie_some_weights(self, functional_call):
        module = MockTiedModule()
        weight = torch.tensor([[2.0]])
        buffer = torch.tensor([3.0])

        parameters = {'l1.weight': weight,
                      'buffer': buffer}
        x = torch.randn(1, 1)
        out = stateless.functional_call(module, parameters, x, tie_weights=True)
        self.assertEqual(out, x * 2. + module.l1.bias + module.tied_bias + buffer + buffer)

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless._functional_call, "stateless")
    ])
    def test_tied_weights_errors(self, functional_call):
        module = MockTiedModule()
        weight = torch.tensor([[1.0]])
        bias = torch.tensor([0.0])
        buffer = torch.tensor([0.0])

        parameters = {'l1.weight': weight,
                      'l1.bias': bias,
                      'buffer': buffer}
        x = torch.randn(1, 1)
        self.assertNotWarn(lambda: functional_call(module, parameters, x, tie_weights=True))

        # if tied values are the same tensors, shouldn't warn
        parameters['tied_bias'] = bias
        parameters['tied_buffer'] = buffer
        self.assertNotWarn(lambda: functional_call(module, parameters, x, tie_weights=True))
        del parameters['tied_bias']
        del parameters['tied_buffer']

        with self.assertRaisesRegex(
            ValueError,
            re.escape("functional_call got multiple values for keys ['l1.bias', 'tied_bias']"),
        ):
            parameters['tied_bias'] = torch.tensor([5.0])
            functional_call(module, parameters, x, tie_weights=True)
        del parameters['tied_bias']

        with self.assertRaisesRegex(
            ValueError,
            re.escape("functional_call got multiple values for keys ['buffer', 'tied_buffer']"),
        ):
            parameters['tied_buffer'] = torch.tensor([5.0])
            functional_call(module, parameters, x, tie_weights=True)

    def test_tied_weights_no_error_without_flag(self):
        module = MockTiedModule()
        weight = torch.tensor([[1.0]])
        bias = torch.tensor([0.0])
        buffer = torch.tensor([0.0])

        parameters = {'l1.weight': weight,
                      'l1.bias': bias,
                      'buffer': buffer}
        x = torch.randn(1, 1)
        self.assertNotWarn(lambda: stateless._functional_call(module, parameters, x, tie_weights=False))
        parameters['tied_bias'] = torch.tensor([5.0])
        self.assertNotWarn(lambda: stateless._functional_call(module, parameters, x, tie_weights=False))
        del parameters['tied_bias']
        parameters['tied_buffer'] = torch.tensor([5.0])
        self.assertNotWarn(lambda: stateless._functional_call(module, parameters, x, tie_weights=False))

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_reparametrize_tie_weights_strict(self, functional_call):
        module = MockTiedModule()
        weight = torch.tensor([[2.0]])
        bias = torch.tensor([5.0])
        buffer = torch.tensor([3.0])
        extra = torch.tensor([1.0])

        # Tie weights no error
        parameters = {'l1.weight': weight,
                      'l1.bias': bias,
                      'buffer': buffer}
        x = torch.randn(1, 1)
        with self._ensure_module_unchanged(
            module,
            'the module should not have been modified by a successful call',
        ):
            out = functional_call(module, parameters, x, tie_weights=True, strict=True)
            self.assertEqual(out, x * weight + bias + bias + buffer + buffer)

        # Tie weights without flag
        parameters = {'l1.weight': weight,
                      'l1.bias': bias,
                      'buffer': buffer}
        x = torch.randn(1, 1)
        with self._ensure_module_unchanged(
            module,
            'the module should not have been modified by a failed call',
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                re.escape("Missing key(s): 'tied_bias', 'tied_buffer'."),
            ):
                out = functional_call(module, parameters, x, tie_weights=False, strict=True)

        # Tie some weights
        parameters = {'l1.weight': weight,
                      'buffer': buffer}
        x = torch.randn(1, 1)
        with self._ensure_module_unchanged(
            module,
            'the module should not have been modified by a failed call',
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                re.escape("Missing key(s): 'l1.bias', 'tied_bias'."),
            ):
                out = stateless.functional_call(module, parameters, x, tie_weights=True, strict=True)

        # Tie weights with extra keys
        parameters = {'l1.weight': weight,
                      'l1.bias': bias,
                      'buffer': buffer,
                      'extra': extra}
        x = torch.randn(1, 1)
        with self._ensure_module_unchanged(
            module,
            'the module should not have been modified by a failed call',
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                re.escape("Unexpected key(s): 'extra'."),
            ):
                out = stateless.functional_call(module, parameters, x, tie_weights=True, strict=True)

        # Tie weights with extra keys and without flag
        parameters = {'l1.weight': weight,
                      'l1.bias': bias,
                      'buffer': buffer,
                      'extra': extra}
        x = torch.randn(1, 1)
        with self._ensure_module_unchanged(
            module,
            'the module should not have been modified by a failed call',
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                re.escape("Unexpected key(s): 'extra'.") + r'\s+' + re.escape("Missing key(s): 'tied_bias', 'tied_buffer'."),
            ):
                out = stateless.functional_call(module, parameters, x, tie_weights=False, strict=True)

        # Tie some weights with extra keys
        parameters = {'l1.weight': weight,
                      'buffer': buffer,
                      'extra': extra}
        x = torch.randn(1, 1)
        with self._ensure_module_unchanged(
            module,
            'the module should not have been modified by a failed call',
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                re.escape("Unexpected key(s): 'extra'.") + r'\s+' + re.escape("Missing key(s): 'l1.bias', 'tied_bias'."),
            ):
                out = stateless.functional_call(module, parameters, x, tie_weights=True, strict=True)

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_setattr(self, functional_call):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = torch.nn.Buffer(torch.tensor([0.0]))

            def forward(self, x):
                self.foo = self.foo + 1
                return x + self.foo

        foo = torch.tensor([2.0])
        x = torch.randn(1)
        a = {'foo': foo}
        mod = Foo()
        functional_call(mod, a, x)
        self.assertEqual(mod.foo, torch.tensor([0.0]))
        self.assertEqual(a['foo'], torch.tensor([3.0]))
        self.assertEqual(foo, torch.tensor([2.0]))
        self.assertTrue(a['foo'] is not foo)

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_in_place_operator(self, functional_call):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = torch.nn.Buffer(torch.tensor([0.0]))

            def forward(self, x):
                self.foo.add_(1)
                return x + self.foo

        foo = torch.tensor([2.0])
        x = torch.randn(1)
        a = {'foo': foo}
        mod = Foo()
        functional_call(mod, a, x)
        self.assertEqual(mod.foo, torch.tensor([0.0]))
        self.assertEqual(a['foo'], torch.tensor([3.0]))
        self.assertEqual(foo, torch.tensor([3.0]))
        self.assertTrue(a['foo'] is foo)

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_setattr_strict(self, functional_call):
        class Bar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if hasattr(self, 'extra'):
                    raise AssertionError("self should not have 'extra' attribute")

            def forward(self, x):
                return x + self.extra

        a = {'extra': torch.zeros(())}
        mod = Bar()
        self.assertTrue(not hasattr(mod, 'extra'))
        out = functional_call(mod, a, torch.ones(()))
        self.assertEqual(out, torch.ones(()))
        self.assertTrue(not hasattr(mod, 'extra'))

        a = {'extra': torch.zeros(())}
        with self.assertRaisesRegex(
            RuntimeError,
            re.escape("Unexpected key(s): 'extra'."),
        ):
            out = functional_call(mod, a, torch.ones(()), strict=True)
        self.assertTrue(not hasattr(mod, 'extra'))

        a = {}
        with self.assertRaisesRegex(
            AttributeError,
            re.escape("'Bar' object has no attribute 'extra'"),
        ):
            out = functional_call(mod, a, torch.ones(()))
        self.assertTrue(not hasattr(mod, 'extra'))

        a = {}
        with self.assertRaisesRegex(
            AttributeError,
            re.escape("'Bar' object has no attribute 'extra'"),
        ):
            out = functional_call(mod, a, torch.ones(()), strict=True)
        self.assertTrue(not hasattr(mod, 'extra'))


    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_functional_call_with_kwargs(self, functional_call):
        class Foo(torch.nn.Module):
            def __init__(self, x):
                super().__init__()
                self.x = x

            def forward(self, inp, *, other_inp):
                return inp * self.x + other_inp

        a = {'x': torch.zeros(2, 3)}
        mod = Foo(torch.randn(2, 3))
        inp, other_inp = torch.randn(2, 3), torch.randn(2, 3)
        with self.assertRaisesRegex(TypeError, "missing 1 required keyword-only argument: 'other_inp'"):
            functional_call(mod, a, inp)
        res = functional_call(mod, a, inp, {'other_inp': other_inp})
        self.assertEqual(res, other_inp)
        res_1 = functional_call(mod, a, (), {'inp': inp, 'other_inp': other_inp})
        self.assertEqual(res, res_1)
        res_2 = functional_call(mod, a, kwargs={'inp': inp, 'other_inp': other_inp})
        self.assertEqual(res, res_2)

    def test_functional_call_tuple_dicts(self):
        mod = MockModule()
        x = torch.rand((1, 1))
        parameters = {k: torch.ones_like(v) for k, v in mod.named_parameters()}
        buffers = {k: torch.zeros_like(v) for k, v in mod.named_buffers()}

        # two dictionaries
        res = torch.func.functional_call(mod, (parameters, buffers), x)
        self.assertEqual(res, x + 1)

        # no dictionaries
        res = torch.func.functional_call(mod, (), x)
        self.assertEqual(res, mod(x))

        # three dictionaries
        a = ({'l1.weight': torch.ones(1, 1)}, {'l1.bias': torch.ones(1)}, {'buffer': torch.zeros(1)})
        res = torch.func.functional_call(mod, a, x)
        self.assertEqual(res, x + 1)

    def test_functional_call_multiple_dicts_error(self):
        mod = MockModule()
        x = torch.rand((1, 1))
        parameters = {'l1.weight': torch.zeros((1, 1)), 'l1.bias': torch.zeros((1, 1))}
        repeated_parameters = {'l1.weight': torch.ones((1, 1))}
        with self.assertRaisesRegex(
            ValueError,
            re.escape("['l1.weight'] appeared in multiple dictionaries"),
        ):
            torch.func.functional_call(mod, (parameters, repeated_parameters), x)

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_functional_call_member_reference(self, functional_call):
        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l1 = torch.nn.Linear(1, 1)
                self.buffer = torch.nn.Buffer(torch.ones(1))

            def forward(self, x):
                parameters = tuple(self.parameters())
                buffers = tuple(self.buffers())
                return self.l1(x) + self.buffer, parameters, buffers

        module = Module()
        weight = torch.tensor([[2.0]])
        bias = torch.tensor([5.0])
        buffer = torch.tensor([3.0])
        extra = torch.tensor([1.0])
        extra_p = torch.nn.Parameter(extra)

        # All weights
        parameters = {'l1.weight': weight,
                      'l1.bias': bias,
                      'buffer': buffer}
        x = torch.randn(1, 1)
        out, parameters, buffers = functional_call(module, parameters, x)
        self.assertEqual(out, x * weight + bias + buffer)
        self.assertEqual(parameters, (weight, bias))
        self.assertEqual(buffers, (buffer,))
        self.assertTrue(all(t1 is t2 for t1, t2 in zip(parameters, (weight, bias))))
        self.assertTrue(all(t1 is t2 for t1, t2 in zip(buffers, (buffer,))))

        # Some weights
        parameters = {'l1.weight': weight}
        x = torch.randn(1, 1)
        out, parameters, buffers = functional_call(module, parameters, x)
        self.assertEqual(out, x * weight + module.l1.bias + module.buffer)
        self.assertEqual(parameters, (weight, module.l1.bias))
        self.assertEqual(buffers, (module.buffer,))
        self.assertTrue(all(t1 is t2 for t1, t2 in zip(parameters, (weight, module.l1.bias))))
        self.assertTrue(all(t1 is t2 for t1, t2 in zip(buffers, (module.buffer,))))

        # All weights with extra keys
        parameters = {'l1.weight': weight,
                      'l1.bias': bias,
                      'buffer': buffer,
                      'l1.extra': extra}
        x = torch.randn(1, 1)
        out, parameters, buffers = functional_call(module, parameters, x)
        self.assertEqual(out, x * weight + bias + buffer)
        self.assertEqual(parameters, (weight, bias))
        self.assertEqual(buffers, (buffer,))
        self.assertTrue(all(t1 is t2 for t1, t2 in zip(parameters, (weight, bias))))
        self.assertTrue(all(t1 is t2 for t1, t2 in zip(buffers, (buffer,))))

        # All weights with extra keys with parameters
        parameters = {'l1.weight': weight,
                      'l1.bias': bias,
                      'buffer': buffer,
                      'l1.extra': extra_p}
        x = torch.randn(1, 1)
        out, parameters, buffers = functional_call(module, parameters, x)
        self.assertEqual(out, x * weight + bias + buffer)
        self.assertEqual(parameters, (weight, bias, extra_p))
        self.assertEqual(buffers, (buffer,))
        self.assertTrue(all(t1 is t2 for t1, t2 in zip(parameters, (weight, bias, extra_p))))
        self.assertTrue(all(t1 is t2 for t1, t2 in zip(buffers, (buffer,))))

        # Some weights with extra keys
        parameters = {'l1.weight': weight,
                      'l1.extra': extra}
        x = torch.randn(1, 1)
        out, parameters, buffers = functional_call(module, parameters, x)
        self.assertEqual(out, x * weight + module.l1.bias + module.buffer)
        self.assertEqual(parameters, (weight, module.l1.bias))
        self.assertEqual(buffers, (module.buffer))
        self.assertTrue(all(t1 is t2 for t1, t2 in zip(parameters, (weight, module.l1.bias))))
        self.assertTrue(all(t1 is t2 for t1, t2 in zip(buffers, (module.buffer,))))

        # Some weights with extra keys with parameters
        parameters = {'l1.weight': weight,
                      'l1.extra': extra_p}
        x = torch.randn(1, 1)
        out, parameters, buffers = functional_call(module, parameters, x)
        self.assertEqual(out, x * weight + module.l1.bias + module.buffer)
        self.assertEqual(parameters, (weight, module.l1.bias, extra_p))
        self.assertEqual(buffers, (module.buffer))
        self.assertTrue(all(t1 is t2 for t1, t2 in zip(parameters, (weight, module.l1.bias, extra_p))))
        self.assertTrue(all(t1 is t2 for t1, t2 in zip(buffers, (module.buffer,))))

        # Set None
        parameters = {'l1.weight': weight,
                      'l1.bias': None}
        x = torch.randn(1, 1)
        out, parameters, buffers = functional_call(module, parameters, x)
        self.assertEqual(out, x * weight + module.buffer)
        self.assertEqual(parameters, (weight,))
        self.assertEqual(buffers, (module.buffer))
        self.assertTrue(all(t1 is t2 for t1, t2 in zip(parameters, (weight,))))
        self.assertTrue(all(t1 is t2 for t1, t2 in zip(buffers, (module.buffer,))))


class TestStatelessDeprecation(TestCase):
    def test_private_stateless_warns(self):
        script = """
import torch
import warnings

with warnings.catch_warnings(record=True) as w:
    from torch.nn.utils import _stateless

exit(len(w))
"""
        try:
            subprocess.check_output(
                [sys.executable, '-W', 'always', '-c', script],
                stderr=subprocess.STDOUT,
                # On Windows, opening the subprocess with the default CWD makes `import torch`
                # fail, so just set CWD to this script's directory
                cwd=os.path.dirname(os.path.realpath(__file__)),)
        except subprocess.CalledProcessError as e:
            self.assertEqual(e.returncode, 1)
        else:
            self.assertTrue(False, "No warning was raised.")

    def test_stateless_functional_call_warns(self):
        m = torch.nn.Linear(1, 1)
        params = dict(m.named_parameters())
        x = torch.randn(3, 1)
        with self.assertWarnsRegex(FutureWarning, "Please use `torch.func.functional_call`"):
            stateless.functional_call(m, params, x)

class TestPythonOptimizeMode(TestCase):
    def test_runs_with_optimize_flag(self):
        script = "import torch; import torch._functorch.deprecated"
        try:
            subprocess.check_output(
                [sys.executable, "-OO", "-c", script],
                stderr=subprocess.STDOUT,
                # On Windows, opening the subprocess with the default CWD makes `import torch`
                # fail, so just set CWD to this script's directory
                cwd=os.path.dirname(os.path.realpath(__file__)),)
        except subprocess.CalledProcessError as e:
            self.assertFalse(e.returncode, "Import failed while running python in optimized mode")

class TestStatelessOptimizerReparam(TestCase):
    class _TestException(Exception):
        pass

    def _init_optimizer(self, freeze_l1: bool = False):
        # ChainedLinear + AdamW stepped once. With freeze_l1=True, l1's
        # params have requires_grad=False so no state is lazy-init'd for them
        # (their packed ids appear in param_groups but are absent from state).
        module = ChainedLinear()
        if freeze_l1:
            for p in module.l1.parameters():
                p.requires_grad = False
            optimizer = torch.optim.AdamW(module.parameters(), lr=0.01)
        else:
            optimizer = torch.optim.AdamW(
                [
                    {"params": [module.l1.weight, module.l1.bias],
                     "lr": 0.1, "weight_decay": 0.01},
                    {"params": [module.l2.weight, module.l2.bias],
                     "lr": 0.01, "weight_decay": 0.0},
                ]
            )
        loss = module(torch.randn(4, 2)).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return module, optimizer

    def _make_reparam_inputs(self):
        module, optimizer = self._init_optimizer()
        parameters = {
            name: torch.randn_like(param, requires_grad=param.requires_grad)
            for name, param in module.named_parameters()
        }
        optimizer_state_dict = deepcopy(optimizer.state_dict())
        return module, optimizer, parameters, optimizer_state_dict

    def _assert_optimizer_restored(
        self, optimizer, state_before, state_dict_before, params_before
    ):
        self.assertIs(optimizer.state, state_before)
        self.assertEqual(optimizer.state_dict(), state_dict_before)
        for group, params in zip(optimizer.param_groups, params_before, strict=True):
            self.assertIs(group["params"], params)

    def test_reparametrize_optimizer_rejects_invalid_inputs(self):
        _, optimizer, parameters, base_osd = self._make_reparam_inputs()

        def expect(opt, params, bad_input, pattern):
            with self.assertRaisesRegex(RuntimeError, re.escape(pattern)):
                with _reparametrize_optimizer(opt, params, bad_input):
                    pass

        with self.subTest("uninitialized"):
            m = ChainedLinear()
            uninit_opt = torch.optim.AdamW(m.parameters(), lr=0.1)
            uninit_params = {
                n: torch.randn_like(p, requires_grad=p.requires_grad)
                for n, p in m.named_parameters()
            }
            expect(
                uninit_opt, uninit_params, deepcopy(uninit_opt.state_dict()),
                "_reparametrize_optimizer requires initialized optimizer state.",
            )

        with self.subTest("non_raw_state"):
            osd = deepcopy(base_osd)
            osd["state"] = {"l1.weight": {}}
            osd["param_groups"][0]["params"] = ["l1.weight", "l1.bias"]
            expect(
                optimizer, parameters, osd,
                "param_groups[*]['params'] entries keyed by packed parameter ids",
            )

        with self.subTest("bad_state_field"):
            expect(
                optimizer, parameters, {"state": [], "param_groups": []},
                "optimizer_state_dict['state'] to be a dict",
            )

        with self.subTest("bad_param_groups_field"):
            expect(
                optimizer, parameters, {"state": {}, "param_groups": {}},
                "optimizer_state_dict['param_groups'] to be a list",
            )

        with self.subTest("group_count_mismatch"):
            osd = deepcopy(base_osd)
            osd["param_groups"].pop()
            expect(optimizer, parameters, osd, "different number of parameter groups")

        with self.subTest("group_size_mismatch"):
            osd = deepcopy(base_osd)
            osd["param_groups"][0]["params"].pop()
            expect(
                optimizer, parameters, osd,
                "does not match the size of live optimizer param group 0",
            )

        with self.subTest("non_dict_param_state"):
            osd = deepcopy(base_osd)
            osd["state"][osd["param_groups"][0]["params"][0]] = torch.tensor(1.0)
            expect(
                optimizer, parameters, osd,
                "per-parameter optimizer state entries to be dictionaries",
            )

        with self.subTest("non_param_state"):
            osd = deepcopy(base_osd)
            osd["state"]["global_step"] = 42
            expect(
                optimizer, parameters, osd, "got extra keys ['global_step']",
            )

    def test_reparametrize_optimizer_handles_missing_param_state(self):
        # When some params have ``requires_grad=False`` their grads stay
        # ``None`` after ``loss.backward(); step()`` and the optimizer never
        # lazy-inits state for them. The resulting state_dict has packed ids
        # in param_groups that are absent from ``state`` —
        # _reparametrize_optimizer must accept this shape, and mutations the
        # trace makes to such params must stay local (not pollute the input
        # ``optimizer_state_dict``).
        module, optimizer = self._init_optimizer(freeze_l1=True)
        optimizer_state_dict = deepcopy(optimizer.state_dict())
        # Sanity: state has fewer entries than param_groups (2 trainable l2
        # params have state; 2 frozen l1 params do not).
        self.assertEqual(len(optimizer_state_dict["param_groups"][0]["params"]), 4)
        self.assertEqual(len(optimizer_state_dict["state"]), 2)
        frozen_param_id = optimizer_state_dict["param_groups"][0]["params"][0]
        self.assertNotIn(frozen_param_id, optimizer_state_dict["state"])

        parameters = {
            name: torch.randn_like(param, requires_grad=param.requires_grad)
            for name, param in module.named_parameters()
        }

        with _reparametrize_optimizer(
            optimizer, parameters, optimizer_state_dict
        ):
            # First param is frozen (l1.weight); mutations to its rebound
            # state must not leak back into the input ``optimizer_state_dict``.
            frozen_param = optimizer.param_groups[0]["params"][0]
            optimizer.state[frozen_param]["step"] = torch.tensor(7.0)

        self.assertNotIn(frozen_param_id, optimizer_state_dict["state"])

    def test_reparametrize_optimizer_supports_named_parameters_init(self):
        # Optimizer initialized with named_parameters() adds a ``param_names``
        # key to each param group; ``state`` is still keyed by packed ints.
        # Rebind must round-trip param_names along with the other group fields.
        module = ChainedLinear()
        optimizer = torch.optim.AdamW(module.named_parameters(), lr=0.01)
        x = torch.randn(4, 2)
        loss = module(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        param_names_before = [g["param_names"] for g in optimizer.param_groups]

        parameters = {
            name: torch.randn_like(p, requires_grad=p.requires_grad)
            for name, p in module.named_parameters()
        }
        optimizer_state_dict = deepcopy(optimizer.state_dict())

        with _reparametrize_optimizer(
            optimizer, parameters, optimizer_state_dict
        ):
            # param_names is re-applied to the rebound group
            self.assertEqual(
                [g["param_names"] for g in optimizer.param_groups],
                param_names_before,
            )

        # And restored on exit
        self.assertEqual(
            [g["param_names"] for g in optimizer.param_groups],
            param_names_before,
        )

    def test_reparametrize_optimizer_restores_after_exception(self):
        _, optimizer, parameters, optimizer_state_dict = self._make_reparam_inputs()
        state_before = optimizer.state
        state_dict_before = deepcopy(optimizer.state_dict())
        params_before = [group["params"] for group in optimizer.param_groups]
        try:
            with _reparametrize_optimizer(
                optimizer, parameters, optimizer_state_dict
            ):
                self.assertTrue(optimizer.state is not state_before)
                self.assertTrue(
                    optimizer.param_groups[0]["params"] is not params_before[0]
                )
                first_param = optimizer.param_groups[0]["params"][0]
                optimizer.state[first_param]["step"] = torch.tensor(123.0)
                optimizer.state[first_param]["exp_avg"].add_(1)
                optimizer.param_groups[0]["lr"] = 5.0
                raise self._TestException
        except self._TestException:
            self._assert_optimizer_restored(
                optimizer, state_before, state_dict_before, params_before
            )

    def test_reparametrize_optimizer_reflects_state_mutations(self):
        _, optimizer, parameters, optimizer_state_dict = self._make_reparam_inputs()
        state_before = optimizer.state
        state_dict_before = deepcopy(optimizer.state_dict())
        params_before = [group["params"] for group in optimizer.param_groups]

        first_param_id = optimizer_state_dict["param_groups"][0]["params"][0]
        exp_avg_before = optimizer_state_dict["state"][first_param_id]["exp_avg"].clone()

        with _reparametrize_optimizer(
            optimizer, parameters, optimizer_state_dict
        ):
            first_param = optimizer.param_groups[0]["params"][0]
            # In-place tensor op: tensor is shared with optimizer_state_dict
            # (no clone), so the mutation is visible after exit.
            optimizer.state[first_param]["exp_avg"].add_(1)
            # Dict rebind: per-param state dict is shallow-copied, so
            # structural changes do not leak back.
            optimizer.state[first_param]["step"] = torch.tensor(123.0)
            optimizer.param_groups[0]["lr"] = 5.0

        self._assert_optimizer_restored(
            optimizer, state_before, state_dict_before, params_before
        )
        self.assertEqual(
            optimizer_state_dict["state"][first_param_id]["exp_avg"],
            exp_avg_before + 1,
        )
        self.assertNotEqual(
            optimizer_state_dict["state"][first_param_id]["step"],
            torch.tensor(123.0),
        )

    def test_make_fx_reparametrize_optimizer_tensor_reassignment_stays_local(self):
        module = ChainedLinear()
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1, momentum=0.9)
        x = torch.randn(4, 2)
        loss = module(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        parameters = {
            name: torch.randn_like(param, requires_grad=param.requires_grad)
            for name, param in module.named_parameters()
        }
        optimizer_state_dict = deepcopy(optimizer.state_dict())
        first_param_id = optimizer_state_dict["param_groups"][0]["params"][0]
        original_momentum = optimizer_state_dict["state"][first_param_id][
            "momentum_buffer"
        ]

        def f(state, x):
            params, optimizer_state = state
            with stateless._reparametrize_module(module, params):
                with _reparametrize_optimizer(
                    optimizer, params, optimizer_state
                ):
                    p = optimizer.param_groups[0]["params"][0]
                    # Intentional tensor reassignment inside make_fx. The new
                    # tensor is a graph-internal value (proxy for ``zeros_like``)
                    # and the assignment lives in the shallow-copied per-param
                    # state dict, so it must not propagate to ``optimizer_state``.
                    optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p)
                    for param in module.parameters():
                        param.grad = torch.ones_like(param)
                    optimizer.step()
                    return module(x)

        gm = make_fx(f)((parameters, optimizer_state_dict), x)
        # Reassigned slot in the input dict still points at the original tensor.
        self.assertIs(
            optimizer_state_dict["state"][first_param_id]["momentum_buffer"],
            original_momentum,
        )
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, state, x):
    state_1, state_2, state_3, state_4, state_5, state_6, state_7, state_8, state_9, state_10, state_11, state_12, state_13, state_14, state_15, state_16, state_17, state_18, state_19, state_20, state_21, x_1, = fx_pytree.tree_flatten_spec([state, x], self._in_spec)
    zeros_like = torch.ops.aten.zeros_like.default(state_1, pin_memory = False)
    ones_like = torch.ops.aten.ones_like.default(state_1, pin_memory = False)
    ones_like_1 = torch.ops.aten.ones_like.default(state_2, pin_memory = False)
    ones_like_2 = torch.ops.aten.ones_like.default(state_3, pin_memory = False)
    ones_like_3 = torch.ops.aten.ones_like.default(state_4, pin_memory = False)
    _record_function_enter_new = torch.ops.profiler._record_function_enter_new.default('Optimizer.step#SGD.step')
    mul_ = torch.ops.aten.mul_.Tensor(zeros_like, 0.9);  zeros_like = None
    add_ = torch.ops.aten.add_.Tensor(mul_, ones_like);  mul_ = ones_like = None
    add__1 = torch.ops.aten.add_.Tensor(state_1, add_, alpha = -0.1);  state_1 = add_ = None
    mul__1 = torch.ops.aten.mul_.Tensor(state_6, 0.9);  state_6 = None
    add__2 = torch.ops.aten.add_.Tensor(mul__1, ones_like_1);  mul__1 = ones_like_1 = None
    add__3 = torch.ops.aten.add_.Tensor(state_2, add__2, alpha = -0.1);  state_2 = add__2 = None
    mul__2 = torch.ops.aten.mul_.Tensor(state_7, 0.9);  state_7 = None
    add__4 = torch.ops.aten.add_.Tensor(mul__2, ones_like_2);  mul__2 = ones_like_2 = None
    add__5 = torch.ops.aten.add_.Tensor(state_3, add__4, alpha = -0.1);  state_3 = add__4 = None
    mul__3 = torch.ops.aten.mul_.Tensor(state_8, 0.9);  state_8 = None
    add__6 = torch.ops.aten.add_.Tensor(mul__3, ones_like_3);  mul__3 = ones_like_3 = None
    add__7 = torch.ops.aten.add_.Tensor(state_4, add__6, alpha = -0.1);  state_4 = add__6 = None
    _record_function_exit = torch.ops.profiler._record_function_exit._RecordFunction(_record_function_enter_new);  _record_function_enter_new = _record_function_exit = None
    t = torch.ops.aten.t.default(add__1);  add__1 = None
    addmm = torch.ops.aten.addmm.default(add__3, x_1, t);  add__3 = x_1 = t = None
    t_1 = torch.ops.aten.t.default(add__5);  add__5 = None
    addmm_1 = torch.ops.aten.addmm.default(add__7, addmm, t_1);  add__7 = addmm = t_1 = None
    return pytree.tree_unflatten([addmm_1], self._out_spec)""",
        )

    def test_make_fx_reparametrize_module_and_optimizer_records_aten_ops(self):
        module = ChainedLinear()
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1, momentum=0.9)
        x = torch.randn(4, 2)
        loss = module(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        parameters = {
            name: torch.randn_like(param, requires_grad=param.requires_grad)
            for name, param in module.named_parameters()
        }
        optimizer_state_dict = deepcopy(optimizer.state_dict())

        def f(state, x):
            params, optimizer_state = state
            with stateless._reparametrize_module(module, params):
                with _reparametrize_optimizer(
                    optimizer, params, optimizer_state
                ):
                    y = module(x)
                    for param in module.parameters():
                        param.grad = torch.ones_like(param)
                    optimizer.step()
                    return y

        gm = make_fx(f)((parameters, optimizer_state_dict), x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, state, x):
    state_1, state_2, state_3, state_4, state_5, state_6, state_7, state_8, state_9, state_10, state_11, state_12, state_13, state_14, state_15, state_16, state_17, state_18, state_19, state_20, state_21, x_1, = fx_pytree.tree_flatten_spec([state, x], self._in_spec)
    t = torch.ops.aten.t.default(state_1)
    addmm = torch.ops.aten.addmm.default(state_2, x_1, t);  x_1 = t = None
    t_1 = torch.ops.aten.t.default(state_3)
    addmm_1 = torch.ops.aten.addmm.default(state_4, addmm, t_1);  addmm = t_1 = None
    ones_like = torch.ops.aten.ones_like.default(state_1, pin_memory = False)
    ones_like_1 = torch.ops.aten.ones_like.default(state_2, pin_memory = False)
    ones_like_2 = torch.ops.aten.ones_like.default(state_3, pin_memory = False)
    ones_like_3 = torch.ops.aten.ones_like.default(state_4, pin_memory = False)
    _record_function_enter_new = torch.ops.profiler._record_function_enter_new.default('Optimizer.step#SGD.step')
    mul_ = torch.ops.aten.mul_.Tensor(state_5, 0.9);  state_5 = None
    add_ = torch.ops.aten.add_.Tensor(mul_, ones_like);  mul_ = ones_like = None
    add__1 = torch.ops.aten.add_.Tensor(state_1, add_, alpha = -0.1);  state_1 = add_ = add__1 = None
    mul__1 = torch.ops.aten.mul_.Tensor(state_6, 0.9);  state_6 = None
    add__2 = torch.ops.aten.add_.Tensor(mul__1, ones_like_1);  mul__1 = ones_like_1 = None
    add__3 = torch.ops.aten.add_.Tensor(state_2, add__2, alpha = -0.1);  state_2 = add__2 = add__3 = None
    mul__2 = torch.ops.aten.mul_.Tensor(state_7, 0.9);  state_7 = None
    add__4 = torch.ops.aten.add_.Tensor(mul__2, ones_like_2);  mul__2 = ones_like_2 = None
    add__5 = torch.ops.aten.add_.Tensor(state_3, add__4, alpha = -0.1);  state_3 = add__4 = add__5 = None
    mul__3 = torch.ops.aten.mul_.Tensor(state_8, 0.9);  state_8 = None
    add__6 = torch.ops.aten.add_.Tensor(mul__3, ones_like_3);  mul__3 = ones_like_3 = None
    add__7 = torch.ops.aten.add_.Tensor(state_4, add__6, alpha = -0.1);  state_4 = add__6 = add__7 = None
    _record_function_exit = torch.ops.profiler._record_function_exit._RecordFunction(_record_function_enter_new);  _record_function_enter_new = _record_function_exit = None
    return pytree.tree_unflatten([addmm_1], self._out_spec)""",
        )


instantiate_parametrized_tests(
    TestStatelessFunctionalAPI,
)
instantiate_parametrized_tests(
    TestStatelessOptimizerReparam,
)

if __name__ == '__main__':
    run_tests()
