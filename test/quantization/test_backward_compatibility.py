# -*- coding: utf-8 -*-

import sys
import os

# torch
import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
import torch.nn.intrinsic.quantized as nniq

# Testing utils
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_quantized import override_qengines, qengine_is_fbgemm

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def get_filenames(self, subname):
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

    return input_file, state_dict_file, scripted_module_file, traced_module_file, expected_file

class TestSerialization(TestCase):
    """ Test backward compatiblity for serialization and numerics
    """
    # Copy and modified from TestCase.assertExpected
    def _test_op(self, qmodule, subname=None, input_size=None, input_quantized=True,
                 generate=False, prec=None, new_zipfile_serialization=False):
        r""" Test quantized modules serialized previously can be loaded
        with current code, make sure we don't break backward compatibility for the
        serialization of quantized modules
        """
        input_file, state_dict_file, scripted_module_file, traced_module_file, expected_file = \
            get_filenames(self, subname)

        # only generate once.
        if generate and qengine_is_fbgemm():
            input_tensor = torch.rand(*input_size).float()
            if input_quantized:
                input_tensor = torch.quantize_per_tensor(input_tensor, 0.5, 2, torch.quint8)
            torch.save(input_tensor, input_file)
            # Temporary fix to use _use_new_zipfile_serialization until #38379 lands.
            torch.save(qmodule.state_dict(), state_dict_file, _use_new_zipfile_serialization=new_zipfile_serialization)
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

    def _test_op_graph(self, qmodule, subname=None, input_size=None, input_quantized=True,
                       generate=False, prec=None, new_zipfile_serialization=False):
        r"""
        Input: a floating point module

        If generate == True, traces and scripts the module and quantizes the results with
        PTQ, and saves the results.

        If generate == False, traces and scripts the module and quantizes the results with
        PTQ, and compares to saved results.
        """
        input_file, state_dict_file, scripted_module_file, traced_module_file, expected_file = \
            get_filenames(self, subname)

        # only generate once.
        if generate and qengine_is_fbgemm():
            input_tensor = torch.rand(*input_size).float()
            torch.save(input_tensor, input_file)

            # convert to TorchScript
            scripted = torch.jit.script(qmodule)
            traced = torch.jit.trace(qmodule, input_tensor)

            # quantize

            def _eval_fn(model, data):
                model(data)

            qconfig_dict = {'': torch.quantization.default_qconfig}
            scripted_q = torch.quantization.quantize_jit(
                scripted, qconfig_dict, _eval_fn, [input_tensor])
            traced_q = torch.quantization.quantize_jit(
                traced, qconfig_dict, _eval_fn, [input_tensor])

            torch.jit.save(scripted_q, scripted_module_file)
            torch.jit.save(traced_q, traced_module_file)
            torch.save(scripted_q(input_tensor), expected_file)

        input_tensor = torch.load(input_file)
        qmodule_scripted = torch.jit.load(scripted_module_file)
        qmodule_traced = torch.jit.load(traced_module_file)
        expected = torch.load(expected_file)
        self.assertEqual(qmodule_scripted(input_tensor), expected, atol=prec)
        self.assertEqual(qmodule_traced(input_tensor), expected, atol=prec)

    @override_qengines
    def test_linear(self):
        module = nnq.Linear(3, 1, bias_=True, dtype=torch.qint8)
        self._test_op(module, input_size=[1, 3], generate=False)

    @override_qengines
    def test_linear_relu(self):
        module = nniq.LinearReLU(3, 1, bias=True, dtype=torch.qint8)
        self._test_op(module, input_size=[1, 3], generate=False)

    @override_qengines
    def test_linear_dynamic(self):
        module_qint8 = nnqd.Linear(3, 1, bias_=True, dtype=torch.qint8)
        self._test_op(module_qint8, "qint8", input_size=[1, 3], input_quantized=False, generate=False)
        if qengine_is_fbgemm():
            module_float16 = nnqd.Linear(3, 1, bias_=True, dtype=torch.float16)
            self._test_op(module_float16, "float16", input_size=[1, 3], input_quantized=False, generate=False)

    @override_qengines
    def test_conv2d(self):
        module = nnq.Conv2d(3, 3, kernel_size=3, stride=1, padding=0, dilation=1,
                            groups=1, bias=True, padding_mode="zeros")
        self._test_op(module, input_size=[1, 3, 6, 6], generate=False)

    @override_qengines
    def test_conv2d_nobias(self):
        module = nnq.Conv2d(3, 3, kernel_size=3, stride=1, padding=0, dilation=1,
                            groups=1, bias=False, padding_mode="zeros")
        self._test_op(module, input_size=[1, 3, 6, 6], generate=False)

    @override_qengines
    def test_conv2d_graph(self):
        module = nn.Sequential(
            torch.quantization.QuantStub(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=0, dilation=1,
                      groups=1, bias=True, padding_mode="zeros"),
        )
        self._test_op_graph(module, input_size=[1, 3, 6, 6], generate=False)

    @override_qengines
    def test_conv2d_nobias_graph(self):
        module = nn.Sequential(
            torch.quantization.QuantStub(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=0, dilation=1,
                      groups=1, bias=False, padding_mode="zeros"),
        )
        self._test_op_graph(module, input_size=[1, 3, 6, 6], generate=False)

    @override_qengines
    def test_conv2d_graph_v2(self):
        # tests the same thing as test_conv2d_graph, but for version 2 of
        # ConvPackedParams{n}d
        module = nn.Sequential(
            torch.quantization.QuantStub(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=0, dilation=1,
                      groups=1, bias=True, padding_mode="zeros"),
        )
        self._test_op_graph(module, input_size=[1, 3, 6, 6], generate=False)

    @override_qengines
    def test_conv2d_nobias_graph_v2(self):
        # tests the same thing as test_conv2d_nobias_graph, but for version 2 of
        # ConvPackedParams{n}d
        module = nn.Sequential(
            torch.quantization.QuantStub(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=0, dilation=1,
                      groups=1, bias=False, padding_mode="zeros"),
        )
        self._test_op_graph(module, input_size=[1, 3, 6, 6], generate=False)

    @override_qengines
    def test_conv2d_relu(self):
        module = nniq.ConvReLU2d(3, 3, kernel_size=3, stride=1, padding=0, dilation=1,
                                 groups=1, bias=True, padding_mode="zeros")
        self._test_op(module, input_size=[1, 3, 6, 6], generate=False)
        # TODO: graph mode quantized conv2d module

    @override_qengines
    def test_conv3d(self):
        if qengine_is_fbgemm():
            module = nnq.Conv3d(3, 3, kernel_size=3, stride=1, padding=0, dilation=1,
                                groups=1, bias=True, padding_mode="zeros")
            self._test_op(module, input_size=[1, 3, 6, 6, 6], generate=False)
            # TODO: graph mode quantized conv3d module

    @override_qengines
    def test_conv3d_relu(self):
        if qengine_is_fbgemm():
            module = nniq.ConvReLU3d(3, 3, kernel_size=3, stride=1, padding=0, dilation=1,
                                     groups=1, bias=True, padding_mode="zeros")
            self._test_op(module, input_size=[1, 3, 6, 6, 6], generate=False)
            # TODO: graph mode quantized conv3d module

    @override_qengines
    def test_lstm(self):
        class LSTMModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nnqd.LSTM(input_size=3, hidden_size=7, num_layers=1).to(dtype=torch.float)

            def forward(self, x):
                x = self.lstm(x)
                return x
        if qengine_is_fbgemm():
            mod = LSTMModule()
            self._test_op(mod, input_size=[4, 4, 3], input_quantized=False, generate=False, new_zipfile_serialization=True)
