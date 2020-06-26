from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch.quantization.qconfig import default_qat_qconfig, _get_default_qat_qconfig_backward
from torch.testing._internal.common_utils import TestCase
from hypothesis import given
from hypothesis import strategies as st
import torch.testing._internal.hypothesis_utils as hu
hu.assert_deadline_disabled()


class TestQATBackward(TestCase):

    @given(quantize_forward=st.booleans(),
           quantize_backward=st.booleans(),
           device=st.sampled_from(['cpu']),
           N=st.integers(1000, 1000),
           D_in=st.sampled_from([10, 100, 1000]),
           D_out=st.sampled_from([10, 100, 1000]))
    def test_qat_backward(
            self,
            quantize_forward,
            quantize_backward,
            device,
            N,
            D_in,
            D_out,
    ):
        def fake_quantize_tensor(X):
            scale, zero_point = torch._choose_qparams_per_tensor(X, reduce_range=False)
            return torch.fake_quantize_per_tensor_affine(X, scale, zero_point, 0, 255)

        x = torch.randn(N, D_in).to(device)
        y = torch.randn(N, D_out).to(device)

        base_net = nn.Linear(D_in, D_out)

        net = nn.Linear(D_in, D_out)
        net.weight = torch.nn.Parameter(base_net.weight.clone().detach()).to(device)
        net.bias = torch.nn.Parameter(base_net.bias.clone().detach()).to(device)

        net.qconfig = _get_default_qat_qconfig_backward(quantize_forward, quantize_backward)
        torch.quantization.prepare_qat(net, inplace=True)

        net_ref = nn.Linear(D_in, D_out)
        net_ref.weight = torch.nn.Parameter(base_net.weight.clone().detach()).to(device)
        net_ref.bias = torch.nn.Parameter(base_net.bias.clone().detach()).to(device)

        if quantize_forward:
            net_ref.qconfig = default_qat_qconfig
            net_ref = torch.quantization.prepare_qat(base_net, inplace=False)

        loss_fn = torch.nn.MSELoss(reduction='sum')
        learning_rate = 1e-4

        for _ in range(1):
            loss = loss_fn(net(x), y)
            loss_ref = loss_fn(net_ref(x), y)

            print(net(x))
            print(net_ref(x))

            net.zero_grad()
            net_ref.zero_grad()

            loss.backward()
            loss_ref.backward()

            print(_, loss.item(), loss_ref.item())
            self.assertEqual(loss.item(), loss_ref.item(), atol=1e-5, rtol=0)

            with torch.no_grad():
                for param in net.parameters():
                    param -= learning_rate * param.grad
                for param in net_ref.parameters():
                    if quantize_backward:
                        param.grad = fake_quantize_tensor(param.grad)
                    param -= learning_rate * param.grad
