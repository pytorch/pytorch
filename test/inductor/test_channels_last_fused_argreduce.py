# Owner(s): ["module: inductor"]

import unittest

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.inductor_utils import HAS_CPU


class TestChannelsLastFusedArgreduce(TestCase):
    @unittest.skipUnless(HAS_CPU, "requires CPU")
    def test_mean_argmax_channels_last(self):
        # https://github.com/pytorch/pytorch/issues/193751
        torch.manual_seed(1234)
        x = torch.randn(4, 4, 8, 8).to(memory_format=torch.channels_last)

        def fn(t):
            return torch.argmax(torch.mean(t, dim=-1))

        self.assertEqual(fn(x), torch.compile(fn)(x))

    @unittest.skipUnless(HAS_CPU, "requires CPU")
    def test_mean_argmax_channels_last_contiguous(self):
        torch.manual_seed(1234)
        x = torch.randn(4, 4, 8, 8)

        def fn(t):
            return torch.argmax(torch.mean(t, dim=-1))

        self.assertEqual(fn(x), torch.compile(fn)(x))

    @unittest.skipUnless(HAS_CPU, "requires CPU")
    def test_mean_argmax_channels_last_3d(self):
        torch.manual_seed(1234)
        x = torch.randn(2, 3, 4, 5, 6).to(memory_format=torch.channels_last_3d)

        def fn(t):
            return torch.argmax(torch.mean(t, dim=-1))

        self.assertEqual(fn(x), torch.compile(fn)(x))


if __name__ == "__main__":
    run_tests()
