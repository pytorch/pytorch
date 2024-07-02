# Owner(s): ["module: nn"]
import contextlib
import random
import unittest
import unittest.mock as mock

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_NUMPY,
    TEST_WITH_CROSSREF,
    TestCase,
)

import random
import torch.nn.utils.rnn as rnn_utils



if TEST_NUMPY:
    import numpy as np


# WARNING: If you add a new top-level test case to this file, you MUST
# update test/run_test.py to list it, otherwise it will NOT be run in
# CI.

class TestMultiheadAttentionNNDeviceType(NNTestCase):
    
    @torch.no_grad()
    @unittest.skipIf(
        TEST_WITH_CROSSREF,
        "CrossRef turns on TorchFunctionMode, and so disables fastpath.",
    )
    def test_multihead_self_attn_two_masks_fast_path_mock(self, device):
        """
        Multihead self-attention should take fast path when both attention mask (mask type 0)
        and key padding mask (mask type 1) are provided at the same time on CPU and CUDA and PrivateUse1
        """
        device = device.rstrip(":0123456789")
        if device not in ["cpu", "cuda", "xpu", torch._C._get_privateuse1_backend_name()]:
            self.skipTest("Fastpath only runs on CPU and CUDA and PrivateUse1.")

        with torch.autocast(device_type=device, enabled=False):
            embed_dim = 16
            num_heads = 8
            batch_size = 8
            src_len = 5

            query = value = key = torch.rand(batch_size, src_len, embed_dim).to(device)
            # Create masks of two different types
            attn_mask = torch.randint(0, 2, (src_len, src_len)).bool().to(device)
            key_padding_mask = (
                torch.randint(0, 2, (batch_size, src_len)).bool().to(device)
            )

            with mock.patch(
                "torch._native_multi_head_attention",
                new=mock.MagicMock(return_value=(torch.Tensor(), torch.Tensor())),
            ) as fastpath_mock:
                # Compute attention on the fast path
                mta_model = torch.nn.MultiheadAttention(
                    embed_dim, num_heads, batch_first=True, device=device
                ).eval()
                mta_model.training = False
                mta_model(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                )
                # If mock was called, fastpath was taken
                self.assertTrue(fastpath_mock.called)



class PackedSequenceTest(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = 5
        self.max_length = 6

    def _ordered_sequence(self, tensor_type):
        """Create ordered list of random sequences"""
        seqs = [
            tensor_type(random.randint(1, self.max_length))
            for _ in range(self.batch_size)
        ]
        if tensor_type == torch.ByteTensor:
            seqs = [s.random_(0, 256) for s in seqs]
        else:
            seqs = [s.random_(-128, 128) for s in seqs]
        ordered = sorted(seqs, key=len, reverse=True)
        return ordered

    def _padded_sequence(self, tensor_type):
        """Create Tensor of random padded sequences"""
        ordered = self._ordered_sequence(tensor_type)
        lengths = [len(i) for i in ordered]
        padded_tensor = rnn_utils.pad_sequence(ordered)
        return padded_tensor, lengths

    def test_to(self):
        for enforce_sorted in (True, False):
            padded, lengths = self._padded_sequence(torch.IntTensor)
            a = rnn_utils.pack_padded_sequence(
                padded, lengths, enforce_sorted=enforce_sorted
            ).cpu()

            self.assertIs(a, a.to("cpu"))
            self.assertIs(a, a.cpu())
            self.assertIs(a, a.to("cpu", dtype=torch.int32))
            self.assertEqual(a.long(), a.to(torch.int64))

            if torch.xpu.is_available():
                for xpu in [
                    "xpu",
                    "xpu:0" if torch.xpu.device_count() == 1 else "xpu:1",
                ]:
                    b = a.xpu(device=xpu)
                    self.assertIs(b, b.to(xpu))
                    self.assertIs(b, b.xpu())
                    self.assertEqual(a, b.to("cpu"))
                    self.assertEqual(b, a.to(xpu))
                    self.assertEqual(a, b.to("cpu", dtype=torch.int32))
                    self.assertIs(b, b.to(dtype=torch.int32))
                    self.assertEqual(b.long(), b.to(dtype=torch.int64))


instantiate_device_type_tests(TestMultiheadAttentionNNDeviceType, globals(), allow_xpu=True)


if __name__ == "__main__":
    run_tests()
