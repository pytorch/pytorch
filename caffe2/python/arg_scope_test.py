from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
from caffe2.python.layers.arg_scope import (
    arg_scope,
    add_arg_scope,
)


@add_arg_scope
def simple_function1(required_param, optional_param_1=None, optional_param_2=None):
    """Some doc block."""
    return (required_param, optional_param_1, optional_param_2)


@add_arg_scope
def simple_function2(required_param, optional_param_1=None, optional_param_2=None):
    return (required_param, optional_param_1, optional_param_2)


def simple_function_no_arg_scope(optional_param=None):
    return (optional_param)


class ArgScopeTest(unittest.TestCase):
    def test_raise_value_error_no_arg_scope(self):
        with self.assertRaises(ValueError):
            with arg_scope([simple_function_no_arg_scope], optional_param=2):
                pass

    def test_raise_value_error_kwargs_arg_scope(self):
        with arg_scope([simple_function1], optional_param_1=1) as scope:
            pass

        with self.assertRaises(ValueError):
            with arg_scope(scope, optional_param_2=2):
                pass

    def test_raise_value_error_overriding_invalid_param(self):
        with self.assertRaises(ValueError):
            with arg_scope([simple_function1], optional_param_3=2):
                simple_function1(3)

    def test_raise_value_error_overriding_required_param(self):
        with self.assertRaises(ValueError):
            with arg_scope([simple_function1], required_param=2):
                simple_function1()

    def test_correct_documentation(self):
        self.assertEquals(simple_function1.__doc__, "Some doc block.")
        self.assertEquals(simple_function1.__name__, "simple_function1")
        self.assertEquals(simple_function1.__module__,
                          "caffe2.caffe2.python.arg_scope_test")

    def test_simple_param_override(self):
        with arg_scope([simple_function1], optional_param_1=1):
            self.assertEquals(simple_function1(2), (2, 1, None))

    def test_simple_param_provided(self):
        with arg_scope([simple_function1], optional_param_1=1):
            self.assertEquals(simple_function1(2, optional_param_1=2), (2, 2, None))

    def test_nested_override(self):
        with arg_scope([simple_function1], optional_param_1=1):
            with arg_scope([simple_function1], optional_param_2=2):
                with arg_scope([simple_function1], optional_param_1=3):
                    self.assertEquals(
                        simple_function1(2), (2, 3, 2))
                self.assertEquals(
                    simple_function1(4), (4, 1, 2))
            self.assertEquals(
                simple_function1(5), (5, 1, None))

    def test_multiple_override(self):
        with arg_scope([simple_function1, simple_function2],
                       optional_param_1=1, optional_param_2=4):
            with arg_scope([simple_function1], optional_param_1=3):
                self.assertEquals(simple_function1(2), (2, 3, 4))
                self.assertEquals(simple_function2(3), (3, 1, 4))

    def test_reusing_scope(self):
        with arg_scope([simple_function1, simple_function2],
                       optional_param_1=1, optional_param_2=4):
            with arg_scope([simple_function1], optional_param_1=3) as scope:
                pass
        with arg_scope(scope):
            self.assertEquals(simple_function1(2), (2, 3, 4))
            self.assertEquals(simple_function2(3), (3, 1, 4))
