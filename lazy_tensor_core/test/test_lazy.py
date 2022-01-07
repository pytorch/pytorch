import torch

from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.jit_utils import JitTestCase

import lazy_tensor_core

lazy_tensor_core._LAZYC._ltc_init_ts_backend()
torch.manual_seed(42)


class TestLazyTensor(JitTestCase):
    def testConvolutionBackward(self):
        def clone_move(t):
            dev = 'lazy'
            copy_t = t.detach().clone().requires_grad_(True).to(device=dev)
            return copy_t

        inp = torch.rand(1, 3, 128, 128, device='cuda', requires_grad=True)
        inp_copy = clone_move(inp)
        grad = torch.rand(1, 32, 121, 121, device='cuda')  # no requires_grad
        grad_copy = clone_move(grad)
        weight = torch.rand(32, 3, 8, 8, device='cuda', requires_grad=True)
        weight_copy = clone_move(weight)
        bias = torch.rand(32, device='cuda', requires_grad=True)
        bias_copy = clone_move(bias)

        # run eager
        conv_out = torch.nn.functional.conv2d(inp, weight, bias)
        (inp_grad, weight_grad, bias_grad) = torch.autograd.grad([conv_out], [inp, weight, bias], [grad])

        # run lazy
        conv_copy_out = torch.nn.functional.conv2d(inp_copy, weight_copy, bias_copy)
        (inp_copy_grad, weight_copy_grad, bias_copy_grad) = torch.autograd.grad(
            [conv_copy_out], [inp_copy, weight_copy, bias_copy], [grad_copy])

        jit_graph = lazy_tensor_core._LAZYC._get_ltc_tensors_backend([bias_copy_grad])

        # check numerics
        torch.testing.assert_allclose(bias_copy_grad.cpu(), bias_grad.cpu())
        torch.testing.assert_allclose(weight_copy_grad.cpu(), weight_grad.cpu())
        torch.testing.assert_allclose(inp_copy_grad.cpu(), inp_grad.cpu())


if __name__ == '__main__':
    run_tests()
