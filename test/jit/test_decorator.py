# Owner(s): ["oncall: jit"]
# flake8: noqa

import sys
import unittest
from enum import Enum
from typing import List, Optional

import torch
from torch.testing._internal.jit_utils import JitTestCase

from jit.myfunction_a import my_function_a


class TestDecorator(JitTestCase):
    def test_decorator(self):
        # Note: JitTestCase.checkScript() does not work with decorators
        # self.checkScript(my_function_a, (1.0,))
        # Error:
        #   RuntimeError: expected def but found '@' here:
        #   @my_decorator
        #   ~ <--- HERE
        #   def my_function_a(x: float) -> float:
        # Do a simple torch.jit.script() test instead
        fn = my_function_a
        fx = torch.jit.script(fn)
        self.assertEqual(fn(1.0), fx(1.0))
