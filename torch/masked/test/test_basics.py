# Copyright (c) Meta Platforms, Inc. and affiliates

import unittest

import torch

from common_utils import _compare_mt_t, _compare_mts, _generate_sample_data
from maskedtensor import masked_tensor
from torch.testing._internal.common_utils import TestCase


class TestMaskedTensor(TestCase):
    def test_add(self):
        data = torch.arange(5.0)
        mask = torch.tensor([True, True, False, True, False])
        m0 = masked_tensor(data, mask)
        m1 = masked_tensor(data, ~mask)
        self.assertRaises(ValueError, lambda: m0 + m1)

    def test_softmax(self):
        x = torch.randn(3, 4) * 0.1
        m = torch.tensor(
            [
                [True, True, True, False],
                [False, True, False, True],
                [True, True, False, False],
            ]
        )
        mx = masked_tensor(x, m, requires_grad=True)
        ts = torch.softmax(mx, -1)
        ts.sum().backward()
        xinf = x.masked_fill(~m, float("-inf")).detach().clone().requires_grad_()
        torch.softmax(xinf, -1)

    def test_where(self):
        # http://pytorch.org/maskedtensor/main/notebooks/nan_grad.html
        x = torch.tensor(
            [-10.0, -5, 0, 5, 10, 50, 60, 70, 80, 90, 100], requires_grad=True
        )
        mask = x < 0
        mx = masked_tensor(x, mask, requires_grad=True)
        my = masked_tensor(torch.ones_like(x), ~mask, requires_grad=True)
        y = torch.where(mask, torch.exp(mx), my)
        s = y.sum()
        s.backward()

    def test_mha_issue_41508(self):
        # https://github.com/pytorch/pytorch/issues/41508
        import torch

        torch.manual_seed(0)
        attn_nn = torch.nn.MultiheadAttention(1, 1, bias=False)
        attn_mt = torch.nn.MultiheadAttention(1, 1, bias=False)
        for (na, a), (nb, b) in zip(
            attn_nn.named_parameters(), attn_mt.named_parameters()
        ):
            a.data.copy_(b.data)

        x = torch.rand(3, 2, 1)
        key_padding_mask = torch.as_tensor(
            [
                [False, False, False],
                [False, True, True],
            ]
        )
        attn_mask = torch.as_tensor(
            [
                [False, True, True],
                [False, False, True],
                [True, False, False],
            ]
        )
        output, scores = attn_nn(
            x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        loss0 = output[0, :].sum()

        x_mt = masked_tensor(
            x, ~(key_padding_mask.transpose(0, 1).unsqueeze(-1).expand_as(x))
        )

        output, scores = attn_mt(x, x_mt, x, attn_mask=attn_mask)
        loss1 = output[0, :].sum()
        self.assertEqual(loss0, loss1.masked_data)

    def test_chunk(self):
        return
        # This breaks because split_backward allocates
        # Tensors using zero and then cats them together.
        # I don't know why the masks are coming into play here.
        # It's an autograd thing.
        k_data = torch.tensor([4.0])
        k_mask = torch.tensor([True])
        k = masked_tensor(k_data[0], k_mask[0], requires_grad=True)
        w = torch.tensor([1.0, 2.0], requires_grad=True)
        w_q, w_k = w.chunk(2)
        o0 = k + w_k
        o0.backward()
        return

    def test_to_sparse(self):
        for sample in _generate_sample_data():
            data = sample.input
            mask = sample.kwargs["mask"]
            mt = masked_tensor(data.clone().detach(), mask, requires_grad=True)

            sparse_mt = mt.to_sparse()
            data.to_sparse().to_dense().sum().backward()
            sparse_mt.to_dense().sum().backward()

            _compare_mt_t(sparse_mt, data)
            _compare_mt_t(mt.grad, data.grad)

    def test_to_dense(self):
        samples = _generate_sample_data(
            layout=torch.sparse_coo
        ) + _generate_sample_data(layout=torch.sparse_csr)
        for sample in samples:
            data = sample.input
            mask = sample.kwargs["mask"]
            mt = masked_tensor(data.clone().detach(), mask, requires_grad=True)

            dense_data = data.to_dense().clone().detach().requires_grad_(True)
            dense_mt = mt.to_dense()
            dense_data.sum().backward()
            dense_mt.sum().backward()

            _compare_mt_t(dense_mt, dense_data)
            _compare_mt_t(mt.grad.to_dense(), dense_data.grad)

    def test_to_dense_and_sparse_coo(self):
        for sample in _generate_sample_data(layout=torch.strided):
            data = sample.input
            mask = sample.kwargs["mask"]
            ms = mask.to_sparse_coo().coalesce()

            t1 = data.clone().detach().requires_grad_(True)
            t1s = data.sparse_mask(ms).clone().detach().requires_grad_(True)
            mt = masked_tensor(t1, mask, requires_grad=True)
            mts = masked_tensor(t1s, ms, requires_grad=True)

            converted = mt.to_sparse().to_dense().requires_grad_(True)
            converted.sum().backward()

            converted2 = mts.to_dense().requires_grad_(True)
            converted2.sum().backward()

            _compare_mts(mt.grad, mts.grad.to_dense())

    def test_to_dense_and_sparse_csr(self):
        for sample in _generate_sample_data(layout=torch.strided):
            data = sample.input
            mask = sample.kwargs["mask"]
            if data.ndim != 2:
                continue
            ms = mask.to_sparse_csr()

            t1 = data.clone().detach().requires_grad_(True)
            t1s = data.sparse_mask(ms).clone().detach().requires_grad_(True)
            mt = masked_tensor(t1, mask, requires_grad=True)
            mts = masked_tensor(t1s, ms, requires_grad=True)

            converted = mt.to_sparse_csr().to_dense()
            converted.sum().backward()

            converted2 = mts.to_dense()
            converted2.sum().backward()

            _compare_mts(mt.grad, mts.grad.to_dense())

    def test_contiguous(self):
        data = torch.randn(3, 3)

        contiguous_data = data.clone()
        mask1 = (contiguous_data > 0).bool()
        not_contiguous_data = torch.as_strided(data.clone(), (2, 2), (1, 2))
        mask2 = (not_contiguous_data > 0).bool()

        contiguous_mt = masked_tensor(contiguous_data, mask1)
        not_contiguous_mt = masked_tensor(not_contiguous_data, mask2)

        contiguous_mt_sparse = masked_tensor(
            contiguous_data.to_sparse_coo(), mask1.to_sparse_coo()
        )
        not_contiguous_mt_sparse = masked_tensor(
            not_contiguous_data.to_sparse_coo(), mask2.to_sparse_coo()
        )

        self.assertEqual(contiguous_data.is_contiguous(), True)
        self.assertEqual(not_contiguous_data.is_contiguous(), False)

        self.assertEqual(contiguous_mt.is_contiguous(), True)
        self.assertEqual(not_contiguous_mt.is_contiguous(), False)

        error_msg = "MaskedTensors with sparse data do not have is_contiguous"
        for t in [contiguous_mt_sparse, not_contiguous_mt_sparse]:
            with self.assertRaisesRegex(ValueError, error_msg):
                t.is_contiguous()
            with self.assertRaisesRegex(ValueError, error_msg):
                t.contiguous()

        now_contiguous_mt = not_contiguous_mt.contiguous()

        self.assertEqual(now_contiguous_mt.is_contiguous(), True)
        self.assertEqual(now_contiguous_mt.masked_data.is_contiguous(), True)
        self.assertEqual(now_contiguous_mt.is_contiguous(), True)


if __name__ == "__main__":
    unittest.main()
