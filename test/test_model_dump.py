#!/usr/bin/env python3
import sys
import io
import unittest

import torch
import torch.utils.model_dump
import torch.utils.mobile_optimizer
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_quantized import supported_qengines


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(16, 64)
        self.relu1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(64, 8)
        self.relu2 = torch.nn.ReLU()

    def forward(self, features):
        act = features
        act = self.layer1(act)
        act = self.relu1(act)
        act = self.layer2(act)
        act = self.relu2(act)
        return act


class QuantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.core = SimpleModel()

    def forward(self, x):
        x = self.quant(x)
        x = self.core(x)
        x = self.dequant(x)
        return x


class ModelWithLists(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rt = [torch.zeros(1)]
        self.ot = [torch.zeros(1), None]

    def forward(self, arg):
        arg = arg + self.rt[0]
        o = self.ot[0]
        if o is not None:
            arg = arg + o
        return arg


class TestModelDump(TestCase):
    @unittest.skipIf(sys.version_info < (3, 7), "importlib.resources was new in 3.7")
    def test_inline_skeleton(self):
        skel = torch.utils.model_dump.get_inline_skeleton()
        assert "unpkg.org" not in skel
        assert "src=" not in skel

    def do_dump_model(self, model, extra_files=None):
        # Just check that we're able to run successfully.
        buf = io.BytesIO()
        torch.jit.save(model, buf, _extra_files=extra_files)
        info = torch.utils.model_dump.get_model_info(buf)
        assert info is not None

    def test_scripted_model(self):
        model = torch.jit.script(SimpleModel())
        self.do_dump_model(model)

    def test_traced_model(self):
        model = torch.jit.trace(SimpleModel(), torch.zeros(2, 16))
        self.do_dump_model(model)

    def get_quant_model(self):
        fmodel = QuantModel().eval()
        fmodel = torch.quantization.fuse_modules(fmodel, [
            ["core.layer1", "core.relu1"],
            ["core.layer2", "core.relu2"],
        ])
        fmodel.qconfig = torch.quantization.get_default_qconfig("qnnpack")
        prepped = torch.quantization.prepare(fmodel)
        prepped(torch.randn(2, 16))
        qmodel = torch.quantization.convert(prepped)
        return qmodel

    @unittest.skipUnless("qnnpack" in supported_qengines, "QNNPACK not available")
    def test_quantized_model(self):
        qmodel = self.get_quant_model()
        self.do_dump_model(torch.jit.script(qmodel))

    @unittest.skipUnless("qnnpack" in supported_qengines, "QNNPACK not available")
    def test_optimized_quantized_model(self):
        qmodel = self.get_quant_model()
        smodel = torch.jit.trace(qmodel, torch.zeros(2, 16))
        omodel = torch.utils.mobile_optimizer.optimize_for_mobile(smodel)
        self.do_dump_model(omodel)

    def test_model_with_lists(self):
        model = torch.jit.script(ModelWithLists())
        self.do_dump_model(model)

    def test_invalid_json(self):
        model = torch.jit.script(SimpleModel())
        self.do_dump_model(model, extra_files={"foo.json": "{"})


if __name__ == '__main__':
    run_tests()
