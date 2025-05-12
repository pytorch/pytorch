# Owner(s): ["module: dynamo"]

# ruff: noqa: TRY002
# flake8: noqa

import itertools
import types
import unittest
import weakref
from typing import Any, Optional, Tuple

import torch
import torch._dynamo.config
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._functorch.config
import torch.nn
import torch.utils.checkpoint
from torch._dynamo.testing import same
from torch._dynamo.utils import dict_items
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase



class SetTests(torch._dynamo.test_case.TestCase):
    def test_set_recompile_on_key_pop(self):
        s = {
            torch._C._set_grad_enabled,
            torch.amp._enter_autocast,
            torch.amp._exit_autocast,
        }

        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, s):
            if torch.amp._exit_autocast in s:
                return x.sin()
            return x.cos()

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x, s)
        opt_fn(x, s)
        self.assertEqual(res, fn(x, s))
        # No recompilation
        self.assertEqual(cnts.frame_count, 1)

        # Pop a value
        s.remove(torch.amp._exit_autocast)

        res = opt_fn(x, s)
        # Check recompilation
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(res, fn(x, s))

    def test_set_recompile_on_key_change(self):
        s = {
            torch._C._set_grad_enabled,
            torch.amp._enter_autocast,
            torch.amp._exit_autocast,
        }

        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, s):
            if torch.amp._exit_autocast in s:
                return x.sin()
            return x.cos()

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x, s)
        opt_fn(x, s)
        self.assertEqual(res, fn(x, s))
        # No recompilation
        self.assertEqual(cnts.frame_count, 1)

        # Pop a value
        s.remove(torch.amp._exit_autocast)
        # Add a different value
        s.add(torch._C._set_autograd_fallback_mode)

        res = opt_fn(x, s)
        # Check recompilation
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(res, fn(x, s))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
