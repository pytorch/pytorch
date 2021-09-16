import unittest

import torch

import torch.nn.utils._stateless as _stateless
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_utils import run_tests, TestCase


class MockModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(1, 1)
        self.register_buffer('buffer', torch.ones(1))

    def forward(self, x):
        return self.l1(x) + self.buffer


class TestStatelessFunctionalAPI(TestCase):
    def _run_call_with_mock_module(self, module, device='cpu', prefix=''):
        x = torch.rand((1, 1)).to(device)
        weight = torch.tensor([[1.0]], device=device)
        bias = torch.tensor([0.0], device=device)
        buffer = torch.tensor([0.0], device=device)
        if prefix != '':
            parameters = {f'{prefix}.l1.weight': weight,
                          f'{prefix}.l1.bias': bias,
                          f'{prefix}.buffer': buffer}
        else:
            parameters = {f'l1.weight': weight,
                          f'l1.bias': bias,
                          f'buffer': buffer}
        to_check = module
        if prefix != '':
            to_check = getattr(module, prefix)
        prev_weight = getattr(to_check.l1, 'weight').clone()
        prev_buffer = getattr(to_check, 'buffer').clone()
        # the parameters represent an identity function contrary to the
        # existing params in module. So here we expect the result to be the
        # same as the input if the weight swapping went well.
        res = _stateless.functional_call(module, parameters, x)
        self.assertEqual(x, res)
        # check that the weight remain unmodified
        cur_weight = getattr(to_check.l1, 'weight')
        cur_buffer = getattr(to_check, 'buffer')
        self.assertEqual(cur_weight, prev_weight)
        self.assertEqual(cur_buffer, prev_buffer)

    def test_functional_call(self):
        module = MockModule()
        self._run_call_with_mock_module(module)

    def test_functional_call_with_jit(self):
        module = MockModule()
        jit_module = torch.jit.script(module)
        with self.assertRaisesRegex(
            RuntimeError,
            r'delete methods or parameters'
        ):
            self._run_call_with_mock_module(jit_module)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_functional_call_with_data_parallel(self):
        module = MockModule()
        module.cuda()
        dp_module = torch.nn.DataParallel(module, [0, 1])
        self._run_call_with_mock_module(dp_module, device='cuda', prefix='module')

    def test_functional_call_with_gradient(self):
        module = MockModule()
        x = torch.rand((1, 1))
        weight = torch.tensor([[1.0]], requires_grad=True)
        bias = torch.tensor([0.0], requires_grad=True)
        buffer = torch.tensor([0.0])
        parameters = {'l1.weight': weight,
                      'l1.bias': bias,
                      'buffer': buffer}
        res = _stateless.functional_call(module, parameters, x)
        # Check that a backward step calculates the gradient of the supplied parameters
        res.backward()
        self.assertIsNotNone(weight.grad)
        self.assertIsNotNone(bias.grad)
        self.assertIsNone(buffer.grad)
        # Gradient was not calculated for the module stated and buffers
        self.assertIsNone(module.l1.weight.grad)
        self.assertIsNone(module.l1.bias.grad)
        self.assertIsNone(module.buffer.grad)

    def test_functional_batch_norm(self):
        module = torch.nn.BatchNorm1d(10)
        module.train()  # Allow stats update
        # lets replace the running_mean buffer and check if its correctly updated
        x = torch.full((20, 10), 128.0)
        rm = torch.zeros(10)
        parameters = {'running_mean': rm}
        prev_rm = module.running_mean.clone()
        res = _stateless.functional_call(module, parameters, x)
        cur_rm = module.running_mean
        self.assertEqual(cur_rm, prev_rm)
        self.assertEqual(rm, torch.full((10,), 12.8))
        # Now run functional without reparametrization and check that the module has
        # been updated
        res = _stateless.functional_call(module, {}, x)
        self.assertEqual(module.running_mean, torch.full((10,), 12.8))

    def test_circular_references(self):
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
        prev_weight = getattr(module.l1, 'weight').clone()
        prev_buffer = getattr(module, 'buffer').clone()
        res = _stateless.functional_call(module, parameters, x)
        self.assertEqual(x, res)
        # check that the weights remain unmodified and were correctly accesed
        cur_weight = getattr(module.l1, 'weight')
        cur_buffer = getattr(module, 'buffer')
        self.assertEqual(cur_weight, prev_weight)
        self.assertEqual(cur_buffer, prev_buffer)


if __name__ == '__main__':
    run_tests()
