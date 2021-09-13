import pytest
import torch

import torch.nn.utils._stateless as _stateless
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_utils import run_tests, TestCase


class MockModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.l1(x)


class TestStatelessFunctionalAPI(TestCase):
    def test_functional_call(self):
        module = MockModule()
        self._run_call_with_module(module)

    def _run_call_with_module(self, module, device='cpu', prefix=''):
        x = torch.rand((1, 1)).to(device)
        weight = torch.nn.Parameter(torch.tensor([[1.0]])).to(device)
        bias = torch.nn.Parameter(torch.tensor([0.0])).to(device)
        parameters = {f'{prefix}.l1.weight': weight,
                      f'{prefix}.l1.bias': bias}
        prev_weight = getattr(module, f'{prefix}.l1.weight', module)
        res = _stateless.functional_call(module, parameters, x)
        self.assertEqual(x, res.reshape((1, 1)))
        # check that the weight remain unmodified
        cur_weight = getattr(module, f'{prefix}.l1.weight', module)
        self.assertEqual(cur_weight, prev_weight)

    @pytest.mark.xfail
    def test_functional_call_with_jit(self):
        module = MockModule()
        jit_module = torch.jit.script(module)
        # RuntimeError: cannot delete methods or parameters of a script module
        self._run_call_with_module(jit_module)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_functional_call_with_data_parallel(self):
        module = MockModule()
        module.cuda()
        dp_module = torch.nn.DataParallel(module, [0, 1])
        self._run_call_with_module(dp_module, device='cuda', prefix='module')


if __name__ == '__main__':
    run_tests()
