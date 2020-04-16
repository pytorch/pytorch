# -*- coding: utf-8 -*-

import sys
import os
import unittest

# torch
import torch
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
import torch.nn.intrinsic.quantized as nniq

# Testing utils
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_utils import run_tests

# TODO: remove this global setting
# JIT tests use double as the default dtype
torch.set_default_dtype(torch.double)

class TestSerialization(TestCase):
    # Copy and modified from TestCase.assertExpected
    def _test_op(self, qmodule, subname=None, prec=None):
        r""" Test quantized modules serialized previously can be loaded
        with current code, make sure we don't break backward compatibility for the
        serialization of quantized modules
        """
        def remove_prefix(text, prefix):
            if text.startswith(prefix):
                return text[len(prefix):]
            return text
        # NB: we take __file__ from the module that defined the test
        # class, so we place the expect directory where the test script
        # lives, NOT where test/common_utils.py lives.
        module_id = self.__class__.__module__
        munged_id = remove_prefix(self.id(), module_id + ".")
        test_file = os.path.realpath(sys.modules[module_id].__file__)
        # TODO: change to quantization/serialized after we add test_quantization.py
        # under pytorch/test folder
        base_name = os.path.join(os.path.dirname(test_file),
                                 "serialized",
                                 munged_id)

        subname_output = ""
        if subname:
            base_name += "_" + subname
            subname_output = " ({})".format(subname)

        input_file = base_name + ".input.pt"
        state_dict_file = base_name + ".state_dict.pt"
        scripted_module_file = base_name + ".scripted.pt"
        traced_module_file = base_name + ".traced.pt"
        expected_file = base_name + ".expected.pt"

        # saving code : uncomment when adding a new test
        # and run the new test, e.g.
        # python test/quantization/test_backward_compatibility.py TestSerialization.test_conv3d
        data = torch.randn(2, 3).float()
        data = torch.quantize_per_tensor(data, 0.5, 2, torch.quint8)
        torch.save(data, input_file)
        torch.save(qmodule.state_dict(), state_dict_file)
        torch.jit.save(torch.jit.script(qmodule), scripted_module_file)
        torch.jit.save(torch.jit.trace(qmodule, data), traced_module_file)
        torch.save(qmodule(data), expected_file)

        data = torch.load(input_file)
        qmodule.load_state_dict(torch.load(state_dict_file))
        qmodule_scripted = torch.jit.load(scripted_module_file)
        qmodule_traced = torch.jit.load(traced_module_file)

        expected = torch.load(expected_file)
        self.assertEqual(qmodule(data), expected, prec=prec)
        self.assertEqual(qmodule_scripted(data), expected, prec=prec)
        # self.assertEqual(qmodule_traced(data), expected, prec=prec)

    @unittest.skipUnless(
        'fbgemm' in torch.backends.quantized.supported_engines or
        'qnnpack' in torch.backends.quantized.supported_engines,
        " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
        " with instruction set support avx2 or newer.",
    )
    def test_linear(self):
        module_qint8 = nnq.Linear(3, 1, bias_=True, dtype=torch.qint8)
        self._test_op(module_qint8, "qint8")

    @unittest.skipUnless(
        'fbgemm' in torch.backends.quantized.supported_engines or
        'qnnpack' in torch.backends.quantized.supported_engines,
        " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
        " with instruction set support avx2 or newer.",
    )
    def test_linear_relu(self):
        module_qint8 = nniq.LinearReLU(3, 1, bias=True, dtype=torch.qint8)
        self._test_op(module_qint8, "qint8")

    @unittest.skipUnless(
        'fbgemm' in torch.backends.quantized.supported_engines or
        'qnnpack' in torch.backends.quantized.supported_engines,
        " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
        " with instruction set support avx2 or newer.",
    )
    def test_linear_dynamic(self):
        module_qint8 = nnqd.Linear(3, 1, bias_=True, dtype=torch.qint8)
        module_float16 = nnqd.Linear(3, 1, bias_=True, dtype=torch.float16)
        self._test_op(module_qint8, "qint8")
        self._test_op(module_float16, "float16")

    @unittest.skipUnless(
        'fbgemm' in torch.backends.quantized.supported_engines or
        'qnnpack' in torch.backends.quantized.supported_engines,
        " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
        " with instruction set support avx2 or newer.",
    )
    def test_conv2d(self):
        module = nnq.Conv2d(3, 3, kernel_size=3, stride=1, padding=0, dilation=1,
                            groups=1, bias=True, padding_mode="zeros")
        self._test_op(module)
        # TODO: graph mode quantized conv2d module

    @unittest.skipUnless(
        'fbgemm' in torch.backends.quantized.supported_engines or
        'qnnpack' in torch.backends.quantized.supported_engines,
        " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
        " with instruction set support avx2 or newer.",
    )
    def test_conv2d_relu(self):
        module = nniq.ConvReLU2d(3, 3, kernel_size=3, stride=1, padding=0, dilation=1,
                                 groups=1, bias=True, padding_mode="zeros")
        self._test_op(module)
        # TODO: graph mode quantized conv2d module

    @unittest.skipUnless(
        'fbgemm' in torch.backends.quantized.supported_engines or
        'qnnpack' in torch.backends.quantized.supported_engines,
        " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
        " with instruction set support avx2 or newer.",
    )
    def test_conv3d(self):
        module = nnq.Conv3d(3, 3, kernel_size=3, stride=1, padding=0, dilation=1,
                            groups=1, bias=True, padding_mode="zeros")
        self._test_op(module)
        # TODO: graph mode quantized conv3d module

    @unittest.skipUnless(
        'fbgemm' in torch.backends.quantized.supported_engines or
        'qnnpack' in torch.backends.quantized.supported_engines,
        " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
        " with instruction set support avx2 or newer.",
    )
    def test_conv3d_relu(self):
        module = nniq.ConvReLU3d(3, 3, kernel_size=3, stride=1, padding=0, dilation=1,
                                 groups=1, bias=True, padding_mode="zeros")
        self._test_op(module)
        # TODO: graph mode quantized conv3d module



if __name__ == "__main__":
    run_tests()
