# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_global.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

"""Verify that warnings are issued for global statements following use."""

from test.support import check_syntax_error
from test.support.warnings_helper import check_warnings
import unittest
import warnings


class GlobalTests(CPythonTestCase):

    def setUp(self):
        super().setUp()
        self.enterContext(check_warnings())
        warnings.filterwarnings("error", module="<test string>")

    def test1(self):
        prog_text_1 = """\
def wrong1():
    a = 1
    b = 2
    global a
    global b
"""
        check_syntax_error(self, prog_text_1, lineno=4, offset=5)

    def test2(self):
        prog_text_2 = """\
def wrong2():
    print(x)
    global x
"""
        check_syntax_error(self, prog_text_2, lineno=3, offset=5)

    def test3(self):
        prog_text_3 = """\
def wrong3():
    print(x)
    x = 2
    global x
"""
        check_syntax_error(self, prog_text_3, lineno=4, offset=5)

    def test4(self):
        prog_text_4 = """\
global x
x = 2
"""
        # this should work
        compile(prog_text_4, "<test string>", "exec")


def setUpModule():
    unittest.enterModuleContext(warnings.catch_warnings())
    warnings.filterwarnings("error", module="<test string>")


if __name__ == "__main__":
    run_tests()
