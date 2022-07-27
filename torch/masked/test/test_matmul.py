# Copyright (c) Meta Platforms, Inc. and affiliates
# flake8: noqa

import unittest

import torch
from maskedtensor import masked_bmm, masked_tensor
from torch.testing._internal.common_utils import TestCase


class TestMaskedTensorMatmul(TestCase):
    def test_bmm(self):
        x = torch.rand(3, 2, 1)
        key_padding_mask = torch.as_tensor(
            [
                [False, False, False],
                [False, True, True],
            ]
        )
        x_mt = masked_tensor(
            x, ~(key_padding_mask.transpose(0, 1).unsqueeze(-1).expand_as(x))
        )
        x = x.masked_fill(~x_mt.mask(), 0)
        attn_2 = torch.bmm(x, x.transpose(-2, -1))
        attn_3 = torch.bmm(x_mt, x_mt.transpose(-2, -1))
        self.assertEqual(attn_3.masked_data.masked_fill(~attn_3.mask(), 0), attn_2)  # type: ignore[attr-defined]

    def test_bmm_2(self):
        x = torch.arange(3 * 2 * 2).reshape(3, 2, 2).float()
        x_t = x.transpose(-2, -1) + x.sum()
        key_padding_mask = torch.as_tensor(
            [
                [False, False, False],
                [False, True, True],
            ]
        )
        x_mt = masked_tensor(
            x, ~(key_padding_mask.transpose(0, 1).unsqueeze(-1).expand_as(x))
        )
        y = torch.bmm(x, x_t)
        y = torch.bmm(x, x_mt.transpose(-2, -1) + x.sum())

    def test_masked_bmm(self):
        key_padding_mask = torch.as_tensor(
            [
                [False, False, False, True],
                [False, True, True, True],
                [False, True, False, True],
            ]
        )
        x = torch.arange(4 * 3 * 2).reshape(4, 3, 2).float().requires_grad_()
        x_mt = masked_tensor(
            x,
            ~(key_padding_mask.transpose(0, 1).unsqueeze(-1).expand_as(x)),
            requires_grad=True,
        )
        attn_mask_bool = torch.as_tensor(
            [
                [False, True, True],
                [False, False, True],
                [True, False, False],
            ]
        )
        attn_mask = attn_mask_bool.float().masked_fill_(attn_mask_bool, float("-inf"))
        v = masked_bmm(x, x_mt.transpose(1, 2), attn_mask)
        v.sum().backward()
        x = torch.arange(4 * 3 * 2).reshape(4, 3, 2).float().requires_grad_()
        x0 = torch.arange(4 * 3 * 2).reshape(4, 3, 2).float().requires_grad_()
        y = torch.bmm(x, x0.transpose(-2, -1))
        y = y * (~attn_mask_bool).float()
        y.sum().backward()

    def test_linear(self):
        x = torch.arange(4 * 3 * 2).reshape(4, 3, 2)
        w_x = torch.arange(10).reshape(5, 2) + x.amax()
        linear = torch.nn.functional.linear
        key_padding_mask = torch.as_tensor(
            [
                [False, False, False, True],
                [False, True, True, True],
                [False, True, False, True],
            ]
        )
        x_mt = masked_tensor(
            x, ~(key_padding_mask.transpose(0, 1).unsqueeze(-1).expand_as(x))
        )


if __name__ == "__main__":
    unittest.main()
