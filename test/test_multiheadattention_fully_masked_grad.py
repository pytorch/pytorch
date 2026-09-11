import torch
from torch import nn
from torch.testing._internal.common_utils import TestCase


class TestMultiheadAttentionFullyMaskedGrad(TestCase):
    def test_need_weights_fully_masked_row_has_no_nan_grad(self):
        torch.manual_seed(0)
        device = "cpu"
        mha = nn.MultiheadAttention(4, 2, device=device)
        x = torch.rand(4, 2, 4, device=device)
        key_padding_mask = torch.tensor(
            [[False, False, False, False], [False, False, True, True]],
            device=device,
        )
        neg_inf = float("-inf")
        attn_mask = torch.tensor(
            [
                [0.0, neg_inf, neg_inf, neg_inf],
                [0.0, 0.0, neg_inf, neg_inf],
                [neg_inf, 0.0, 0.0, neg_inf],
                [neg_inf, neg_inf, 0.0, 0.0],
            ],
            device=device,
        )

        def run(need_weights):
            local = nn.MultiheadAttention(4, 2, device=device)
            local.load_state_dict(mha.state_dict())
            out, _ = local(
                x,
                x,
                x,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                need_weights=need_weights,
            )
            out[:2].sum().backward()
            return out.detach(), [p.grad.clone() for p in local.parameters()]

        out_weights, grads_weights = run(True)
        out_fast, grads_fast = run(False)

        self.assertFalse(torch.isnan(out_weights).any())
        for grad in grads_weights:
            self.assertFalse(torch.isnan(grad).any())
        torch.testing.assert_close(out_weights, out_fast, atol=1e-5, rtol=1e-4)
        for grad_weights, grad_fast in zip(grads_weights, grads_fast):
            torch.testing.assert_close(grad_weights, grad_fast, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    torch.testing._internal.common_utils.run_tests()
