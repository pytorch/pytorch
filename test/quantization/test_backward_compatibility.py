# -*- coding: utf-8 -*-

import sys
import os
import unittest

# torch
import torch
import torch.nn.quantized as nnq

# Testing utils
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_utils import run_tests

# TODO: remove this global setting
# JIT tests use double as the default dtype
torch.set_default_dtype(torch.double)

ACCEPT = os.getenv('EXPECTTEST_ACCEPT')

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

        state_dict_file = base_name + ".state_dict.pt"
        scripted_module_file = base_name + ".scripted.pt"
        traced_module_file = base_name + ".traced.pt"
        input_file = base_name + ".input.pt"
        expected_file = base_name + ".expected.pt"

        data = torch.load(input_file)
        qmodule.load_state_dict(torch.load(state_dict_file))
        qmodule_scripted = torch.jit.load(scripted_module_file)
        qmodule_traced = torch.jit.load(traced_module_file)

        expected = torch.load(expected_file)
        self.assertEqual(qmodule(data), expected, prec=prec)
        self.assertEqual(qmodule_scripted(data), expected, prec=prec)
        self.assertEqual(qmodule_traced(data), expected, prec=prec)

    @unittest.skipUnless(
        'fbgemm' in torch.backends.quantized.supported_engines,
        " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
        " with instruction set support avx2 or newer.",
    )
    def test_conv(self):
        # quantized conv module
        qconv = nnq.Conv2d(3, 3, kernel_size=3, stride=1, padding=0, dilation=1,
                           groups=1, bias=True, padding_mode="zeros")
        self._test_op(qconv)
        # TODO: graph mode quantized conv module

if __name__ == "__main__":
    run_tests()
