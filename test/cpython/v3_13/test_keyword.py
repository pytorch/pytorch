# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_keyword.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_TORCHDYNAMO,
)

# ======= END DYNAMO PATCH =======

import keyword
import unittest


class Test_iskeyword(CPythonTestCase):
    def test_true_is_a_keyword(self):
        self.assertTrue(keyword.iskeyword('True'))

    def test_uppercase_true_is_not_a_keyword(self):
        self.assertFalse(keyword.iskeyword('TRUE'))

    def test_none_value_is_not_a_keyword(self):
        self.assertFalse(keyword.iskeyword(None))

    def test_changing_the_kwlist_does_not_affect_iskeyword(self):
        old_list = keyword.kwlist
        self.addCleanup(setattr, keyword, 'kwlist', old_list)
        keyword.kwlist = ['eggs', 'spam']
        self.assertFalse(keyword.iskeyword('eggs'))
        self.assertTrue(keyword.iskeyword('True'))

    def test_changing_the_softkwlist_does_not_affect_issoftkeyword(self):
        old_list = keyword.softkwlist
        self.addCleanup(setattr, keyword, 'softkwlist', old_list)
        keyword.softkwlist = ['eggs', 'spam']
        self.assertFalse(keyword.issoftkeyword('eggs'))
        self.assertTrue(keyword.issoftkeyword('_'))

    @unittest.skipIf(TEST_WITH_TORCHDYNAMO, "exec() unsupported")
    def test_all_keywords_fail_to_be_used_as_names(self):
        for key in keyword.kwlist:
            with self.assertRaises(SyntaxError):
                exec(f"{key} = 42")

    @unittest.skipIf(TEST_WITH_TORCHDYNAMO, "exec() unsupported")
    def test_all_soft_keywords_can_be_used_as_names(self):
        for key in keyword.softkwlist:
            exec(f"{key} = 42")

    def test_keywords_are_sorted(self):
        self.assertListEqual(sorted(keyword.kwlist), keyword.kwlist)

    def test_softkeywords_are_sorted(self):
        self.assertListEqual(sorted(keyword.softkwlist), keyword.softkwlist)

    def test_soft_keywords(self):
        self.assertIn('_', keyword.softkwlist)
        self.assertIn('case', keyword.softkwlist)
        self.assertIn('match', keyword.softkwlist)

    def test_async_and_await_are_keywords(self):
        self.assertIn("async", keyword.kwlist)
        self.assertIn("await", keyword.kwlist)


if __name__ == "__main__":
    run_tests()
