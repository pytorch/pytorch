# -*- coding: utf-8 -*-

import sys
import os

# torch
import torch
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
import torch.nn.intrinsic.quantized as nniq

# Testing utils
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_quantized import override_quantized_engine, supported_qengines

class TestSerialization(TestCase):
    """ Test backward compatiblity for serialization and numerics
    """
    # Copy and modified from TestCase.assertExpected
    def _test_op(self, qmodule, subname=None, input_size=None, input_quantized=True, generate=False, prec=None, iter=0):
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

        # only generate once.
        if generate and iter == 0:
            input_tensor = torch.rand(*input_size).float()
            if input_quantized:
                input_tensor = torch.quantize_per_tensor(input_tensor, 0.5, 2, torch.quint8)
            torch.save(input_tensor, input_file)
            torch.save(qmodule.state_dict(), state_dict_file)
            torch.jit.save(torch.jit.script(qmodule), scripted_module_file)
            torch.jit.save(torch.jit.trace(qmodule, input_tensor), traced_module_file)
            torch.save(qmodule(input_tensor), expected_file)

        input_tensor = torch.load(input_file)
        qmodule.load_state_dict(torch.load(state_dict_file))
        qmodule_scripted = torch.jit.load(scripted_module_file)
        qmodule_traced = torch.jit.load(traced_module_file)

        expected = torch.load(expected_file)
        self.assertEqual(qmodule(input_tensor), expected, atol=prec)
        self.assertEqual(qmodule_scripted(input_tensor), expected, atol=prec)
        self.assertEqual(qmodule_traced(input_tensor), expected, atol=prec)

    def test_linear(self):
        for i, qengine in enumerate(supported_qengines):
            with override_quantized_engine(qengine):
                module = nnq.Linear(3, 1, bias_=True, dtype=torch.qint8)
                self._test_op(module, input_size=[1, 3], generate=False, iter=i)

    def test_linear_relu(self):
        for i, qengine in enumerate(supported_qengines):
            with override_quantized_engine(qengine):
                module = nniq.LinearReLU(3, 1, bias=True, dtype=torch.qint8)
                self._test_op(module, input_size=[1, 3], generate=False, iter=i)

    def test_linear_dynamic(self):
        for i, qengine in enumerate(supported_qengines):
            with override_quantized_engine(qengine):
                module_qint8 = nnqd.Linear(3, 1, bias_=True, dtype=torch.qint8)
                self._test_op(module_qint8, "qint8", input_size=[1, 3], input_quantized=False, generate=False, iter=i)
                if qengine == 'fbgemm':
                    module_float16 = nnqd.Linear(3, 1, bias_=True, dtype=torch.float16)
                    self._test_op(module_float16, "float16", input_size=[1, 3], input_quantized=False, generate=False)

    def test_conv2d(self):
        for i, qengine in enumerate(supported_qengines):
            with override_quantized_engine(qengine):
                module = nnq.Conv2d(3, 3, kernel_size=3, stride=1, padding=0, dilation=1,
                                    groups=1, bias=True, padding_mode="zeros")
                self._test_op(module, input_size=[1, 3, 6, 6], generate=False, iter=i)
                # TODO: graph mode quantized conv2d module

    def test_conv2d_relu(self):
        for i, qengine in enumerate(supported_qengines):
            with override_quantized_engine(qengine):
                module = nniq.ConvReLU2d(3, 3, kernel_size=3, stride=1, padding=0, dilation=1,
                                         groups=1, bias=True, padding_mode="zeros")
                self._test_op(module, input_size=[1, 3, 6, 6], generate=False, iter=i)
                # TODO: graph mode quantized conv2d module

    def test_conv3d(self):
        if 'fbgemm' in supported_qengines:
            with override_quantized_engine('fbgemm'):
                module = nnq.Conv3d(3, 3, kernel_size=3, stride=1, padding=0, dilation=1,
                                    groups=1, bias=True, padding_mode="zeros")
                self._test_op(module, input_size=[1, 3, 6, 6, 6], generate=False)
                # TODO: graph mode quantized conv3d module

    def test_conv3d_relu(self):
        if 'fbgemm' in supported_qengines:
            with override_quantized_engine('fbgemm'):
                module = nniq.ConvReLU3d(3, 3, kernel_size=3, stride=1, padding=0, dilation=1,
                                         groups=1, bias=True, padding_mode="zeros")
                self._test_op(module, input_size=[1, 3, 6, 6, 6], generate=False)
                # TODO: graph mode quantized conv3d module
