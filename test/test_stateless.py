# Owner(s): ["module: nn"]

import unittest
import sys
import os
import subprocess

import torch

import torch.nn.utils.stateless as stateless
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_utils import run_tests, TestCase, parametrize, instantiate_parametrized_tests, \
    subtest


class MockModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(1, 1)
        self.register_buffer('buffer', torch.ones(1))

    def forward(self, x):
        return self.l1(x) + self.buffer

class MockTiedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(1, 1)
        self.tied_bias = self.l1.bias
        self.register_buffer('buffer', torch.ones(1))
        self.register_buffer('tied_buffer', self.buffer)

    def forward(self, x):
        return self.l1(x) + self.tied_bias + self.buffer + self.tied_buffer


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
        res = functional_call(module, parameters, x)
        cur_rm = module.running_mean
        self.assertEqual(cur_rm, prev_rm)
        self.assertEqual(rm, torch.full((10,), 12.8))
        # Now run functional without reparametrization and check that the module has
        # been updated
        res = functional_call(module, {}, x)
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
        # check that the weights remain unmodified and were correctly accesed
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
    def test_reparamertize_module_fail_reset_to_original(self, functional_call):
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
            x = torch.rand((4, 5))  # to work, it should be of size (1, 1)
            functional_call(module, parameters, x)  # this call will fail because x is the wrong size

        # verify that the spectral normalization is still applied
        self.assertTrue('l1.parametrizations.weight.original' in dict(module.named_parameters()))
        self.assertEqual(orig_sn_weight, module.l1.weight)

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_tied_weights_warns(self, functional_call):
        module = MockModule()
        module.tied_bias = module.l1.bias
        module.register_buffer("tied_buffer", module.buffer)

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_reparamertize_tie_weights(self, functional_call):
        module = MockTiedModule()
        weight = torch.tensor([[2.0]],)
        bias = torch.tensor([5.0])
        buffer = torch.tensor([3.0])

        parameters = {'l1.weight': weight,
                      'l1.bias': bias,
                      'buffer': buffer}
        x = torch.randn(1, 1)
        out = functional_call(module, parameters, x, tie_weights=True)
        self.assertEqual(out, x * weight + bias + bias + buffer + buffer)


    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_reparamertize_tie_some_weights(self, functional_call):
        module = MockTiedModule()
        weight = torch.tensor([[2.0]],)
        buffer = torch.tensor([3.0])

        parameters = {'l1.weight': weight,
                      'buffer': buffer}
        x = torch.randn(1, 1)
        out = stateless.functional_call(module, parameters, x, tie_weights=True)
        self.assertEqual(out, x * 2. + module.l1.bias + module.tied_bias + buffer + buffer)

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_tied_weights_errors(self, functional_call):
        module = MockTiedModule()
        weight = torch.tensor([[1.0]],)
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

        with self.assertRaisesRegex(ValueError, "functional_call got values for both (l1.bias|tied_bias)"):
            parameters['tied_bias'] = torch.tensor([5.0])
            functional_call(module, parameters, x, tie_weights=True)
        del parameters['tied_bias']

        with self.assertRaisesRegex(ValueError, "functional_call got values for both (buffer|tied_buffer)"):
            parameters['tied_buffer'] = torch.tensor([5.0])
            functional_call(module, parameters, x, tie_weights=True)


    def test_tied_weights_no_error_without_flag(self):
        module = MockTiedModule()
        weight = torch.tensor([[1.0]],)
        bias = torch.tensor([0.0])
        buffer = torch.tensor([0.0])

        parameters = {'l1.weight': weight,
                      'l1.bias': bias,
                      'buffer': buffer}
        x = torch.randn(1, 1)
        self.assertNotWarn(lambda: stateless.functional_call(module, parameters, x, tie_weights=False))
        parameters['tied_bias'] = torch.tensor([5.0])
        self.assertNotWarn(lambda: stateless.functional_call(module, parameters, x, tie_weights=False))
        del parameters['tied_bias']
        parameters['tied_buffer'] = torch.tensor([5.0])
        self.assertNotWarn(lambda: stateless.functional_call(module, parameters, x, tie_weights=False))

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_setattr(self, functional_call):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('foo', torch.zeros(()))

            def forward(self, x):
                self.foo = self.foo + 1
                return x + self.foo

        a = {'foo': torch.zeros(())}
        mod = Foo()
        functional_call(mod, a, torch.ones(()))
        self.assertEqual(mod.foo, torch.zeros(()))
        self.assertEqual(a['foo'], torch.ones(()))

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

        # three dictonaries
        a = ({'l1.weight': torch.ones(1, 1)}, {'l1.bias': torch.ones(1)}, {'buffer': torch.zeros(1)})
        res = torch.func.functional_call(mod, a, x)
        self.assertEqual(res, x + 1)


    def test_functional_call_multiple_dicts_error(self):
        mod = MockModule()
        x = torch.rand((1, 1))
        parameters = {'l1.weight': torch.zeros((1, 1)), 'l1.bias': torch.zeros((1, 1))}
        repeated_parameters = {'l1.weight': torch.ones((1, 1))}
        with self.assertRaisesRegex(ValueError, "l1.weight appeared in multiple dictionaries"):
            torch.func.functional_call(mod, (parameters, repeated_parameters), x)


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
                [sys.executable, '-W', 'all', '-c', script],
                stderr=subprocess.STDOUT,
                # On Windows, opening the subprocess with the default CWD makes `import torch`
                # fail, so just set CWD to this script's directory
                cwd=os.path.dirname(os.path.realpath(__file__)),)
        except subprocess.CalledProcessError as e:
            self.assertEqual(e.returncode, 1)
        else:
            self.assertTrue(False, "No warning was raised.")

class TestPythonOptimizeMode(TestCase):
    def test_runs_with_optimize_flag(self):
        script = """
import torch
"""
        try:
            subprocess.check_output(
                [sys.executable, '-OO', '-c', script],
                stderr=subprocess.STDOUT,
                # On Windows, opening the subprocess with the default CWD makes `import torch`
                # fail, so just set CWD to this script's directory
                cwd=os.path.dirname(os.path.realpath(__file__)),)
        except subprocess.CalledProcessError as e:
            self.assertFalse(e.returncode, "Import failed while running python in optimized mode")


instantiate_parametrized_tests(
    TestStatelessFunctionalAPI,
)

if __name__ == '__main__':
    run_tests()
