# -*- coding: utf-8 -*-
# torch
import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.functional as F

# Testing utils
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_utils import run_tests

# TODO: remove this global setting
# JIT tests use double as the default dtype
torch.set_default_dtype(torch.double)

class TestSerialization(TestCase):
    def test_conv(self):
        data = torch.randn(1, 3, 10, 10, dtype=torch.float)
        data = torch.quantize_per_tensor(data, 0.5, 2, torch.quint8)
        # quantized conv module
        qconv = nnq.Conv2d(3, 3, kernel_size=3, stride=1, padding=0, dilation=1,
                                  groups=1, bias=True, padding_mode="zeros")
        # torch.save(qconv.state_dict(), 'qconv.state_dict')
        state_dict = torch.load('qconv.state_dict')
        qconv.load_state_dict(state_dict)

        # qconv_scripted = torch.jit.script(qconv)
        # torch.jit.save(qconv_scripted, 'qconv.scripted')
        # qconv_traced = torch.jit.trace(qconv, data)
        # torch.jit.save(qconv_traced, 'qconv.traced')
        # traced/scripted quantized conv module
        qconv_scripted = torch.jit.load('qconv.scripted')
        qconv_traced = torch.jit.load('qconv.traced')

        # TODO: graph mode quantized conv module

if __name__ == "__main__":
    run_tests()
