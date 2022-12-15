#!/usr/bin/env python3
# Owner(s): ["oncall: mobile"]

import sys
import os
import io
import functools
import tempfile
import urllib
import unittest

import torch
import torch.backends.xnnpack
import torch.utils.model_dump
import torch.utils.mobile_optimizer
from torch.testing._internal.common_utils import TestCase, run_tests, IS_WINDOWS, skipIfNoXNNPACK
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
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
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


def webdriver_test(testfunc):
    @functools.wraps(testfunc)
    def wrapper(self, *args, **kwds):
        self.needs_resources()

        if os.environ.get("RUN_WEBDRIVER") != "1":
            self.skipTest("Webdriver not requested")
        from selenium import webdriver

        for driver in [
                "Firefox",
                "Chrome",
        ]:
            with self.subTest(driver=driver):
                wd = getattr(webdriver, driver)()
                testfunc(self, wd, *args, **kwds)
                wd.close()

    return wrapper


class TestModelDump(TestCase):
    def needs_resources(self):
        if sys.version_info < (3, 7):
            self.skipTest("importlib.resources was new in 3.7")

    def test_inline_skeleton(self):
        self.needs_resources()
        skel = torch.utils.model_dump.get_inline_skeleton()
        assert "unpkg.org" not in skel
        assert "src=" not in skel

    def do_dump_model(self, model, extra_files=None):
        # Just check that we're able to run successfully.
        buf = io.BytesIO()
        torch.jit.save(model, buf, _extra_files=extra_files)
        info = torch.utils.model_dump.get_model_info(buf)
        assert info is not None

    def open_html_model(self, wd, model, extra_files=None):
        buf = io.BytesIO()
        torch.jit.save(model, buf, _extra_files=extra_files)
        page = torch.utils.model_dump.get_info_and_burn_skeleton(buf)
        wd.get("data:text/html;charset=utf-8," + urllib.parse.quote(page))

    def open_section_and_get_body(self, wd, name):
        container = wd.find_element_by_xpath(f"//div[@data-hider-title='{name}']")
        caret = container.find_element_by_class_name("caret")
        if container.get_attribute("data-shown") != "true":
            caret.click()
        content = container.find_element_by_tag_name("div")
        return content

    def test_scripted_model(self):
        model = torch.jit.script(SimpleModel())
        self.do_dump_model(model)

    def test_traced_model(self):
        model = torch.jit.trace(SimpleModel(), torch.zeros(2, 16))
        self.do_dump_model(model)

    def test_main(self):
        self.needs_resources()
        if IS_WINDOWS:
            # I was getting tempfile errors in CI.  Just skip it.
            self.skipTest("Disabled on Windows.")

        with tempfile.NamedTemporaryFile() as tf:
            torch.jit.save(torch.jit.script(SimpleModel()), tf)
            # Actually write contents to disk so we can read it below
            tf.flush()

            stdout = io.StringIO()
            torch.utils.model_dump.main(
                [
                    None,
                    "--style=json",
                    tf.name,
                ],
                stdout=stdout)
            self.assertRegex(stdout.getvalue(), r'\A{.*SimpleModel')

            stdout = io.StringIO()
            torch.utils.model_dump.main(
                [
                    None,
                    "--style=html",
                    tf.name,
                ],
                stdout=stdout)
            self.assertRegex(
                stdout.getvalue().replace("\n", " "),
                r'\A<!DOCTYPE.*SimpleModel.*componentDidMount')

    def get_quant_model(self):
        fmodel = QuantModel().eval()
        fmodel = torch.ao.quantization.fuse_modules(fmodel, [
            ["core.layer1", "core.relu1"],
            ["core.layer2", "core.relu2"],
        ])
        fmodel.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
        prepped = torch.ao.quantization.prepare(fmodel)
        prepped(torch.randn(2, 16))
        qmodel = torch.ao.quantization.convert(prepped)
        return qmodel

    @unittest.skipUnless("qnnpack" in supported_qengines, "QNNPACK not available")
    def test_quantized_model(self):
        qmodel = self.get_quant_model()
        self.do_dump_model(torch.jit.script(qmodel))

    @skipIfNoXNNPACK
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

    @webdriver_test
    def test_memory_computation(self, wd):
        def check_memory(model, expected):
            self.open_html_model(wd, model)
            memory_table = self.open_section_and_get_body(wd, "Tensor Memory")
            device = memory_table.find_element_by_xpath("//table/tbody/tr[1]/td[1]").text
            self.assertEqual("cpu", device)
            memory_usage_str = memory_table.find_element_by_xpath("//table/tbody/tr[1]/td[2]").text
            self.assertEqual(expected, int(memory_usage_str))

        simple_model_memory = (
            # First layer, including bias.
            64 * (16 + 1) +
            # Second layer, including bias.
            8 * (64 + 1)
            # 32-bit float
        ) * 4

        check_memory(torch.jit.script(SimpleModel()), simple_model_memory)

        # The same SimpleModel instance appears twice in this model.
        # The tensors will be shared, so ensure no double-counting.
        a_simple_model = SimpleModel()
        check_memory(
            torch.jit.script(
                torch.nn.Sequential(a_simple_model, a_simple_model)),
            simple_model_memory)

        # The freezing process will move the weight and bias
        # from data to constants.  Ensure they are still counted.
        check_memory(
            torch.jit.freeze(torch.jit.script(SimpleModel()).eval()),
            simple_model_memory)

        # Make sure we can handle a model with both constants and data tensors.
        class ComposedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = torch.zeros(1, 2)
                self.w2 = torch.ones(2, 2)

            def forward(self, arg):
                return arg * self.w2 + self.w1

        check_memory(
            torch.jit.freeze(
                torch.jit.script(ComposedModule()).eval(),
                preserved_attrs=["w1"]),
            4 * (2 + 4))


if __name__ == '__main__':
    run_tests()
