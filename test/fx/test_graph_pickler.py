# Owner(s): ["module: fx"]

#
# Tests the graph pickler by using pickling on all the inductor tests.
#

import contextlib
import importlib
import os
import sys
from unittest.mock import patch

import torch
import torch.library
from torch._dynamo.testing import make_test_cls_with_patches
from torch._inductor.test_case import TestCase
from torch.testing._internal.inductor_utils import HAS_CPU


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model,
    CommonTemplate,
    copy_tests,
    TestFailure,
)


importlib.import_module("filelock")

# xfail by default, set is_skip=True to skip
test_failures = {
    # TypeError: cannot pickle 'generator' object
    "test_layer_norm_graph_pickler": TestFailure(("cpu"), is_skip=True),
}


def make_test_cls(cls, xfail_prop="_expected_failure_graph_pickler"):
    return make_test_cls_with_patches(
        cls,
        "GraphPickler",
        "_graph_pickler",
        (
            torch._inductor.compile_fx,
            "fx_compile_mode",
            torch._inductor.compile_fx.FxCompileMode.SERIALIZE,
        ),
        xfail_prop=xfail_prop,
    )


GraphPicklerCommonTemplate = make_test_cls(CommonTemplate)


if HAS_CPU:

    class GraphPicklerCpuTests(TestCase):
        common = check_model
        device = "cpu"

    copy_tests(GraphPicklerCommonTemplate, GraphPicklerCpuTests, "cpu", test_failures)


class TestGraphPickler(TestCase):
    def setUp(self):
        torch._dynamo.reset()
        super().setUp()

        self._stack = contextlib.ExitStack()
        self._stack.enter_context(
            patch(
                "torch._inductor.compile_fx.fx_compile_mode",
                torch._inductor.compile_fx.FxCompileMode.SERIALIZE,
            )
        )

    def tearDown(self):
        self._stack.close()
        TestCase.tearDown(self)
        torch._dynamo.reset()

    def test_simple(self):
        # Make sure that compiling works when we pass the input + output from
        # fx_codegen_and_compile() through serde.

        def fn(a, b):
            return a + b

        check_model(self, fn, (torch.tensor([False, True]), torch.tensor([True, True])))


class TestDebugDumps(TestCase):
    """Tests for GraphPickler.debug_dumps debugging utility."""

    def setUp(self):
        super().setUp()
        from torch.fx._graph_pickler import GraphPickler

        self.GraphPickler = GraphPickler

    def test_picklable_object_returns_none_equivalent(self):
        """
        When an object is fully picklable, debug_dumps should return
        "root" since there's no unpicklable leaf.
        """
        result = self.GraphPickler.debug_dumps([1, 2, 3], verbose=False)
        self.assertIsNone(result)

    def test_simple_unpicklable_lambda(self):
        """
        A lambda at the root should return "root" as the unpicklable path.
        """
        bad_obj = lambda x: x  # noqa: E731
        result = self.GraphPickler.debug_dumps(bad_obj, verbose=False)
        self.assertIn("root", result)

    def test_nested_unpicklable_in_list(self):
        """
        When a lambda is nested in a list, debug_dumps should find the path
        to it (e.g., "root[1]").
        """
        bad_obj = [1, lambda x: x, 3]  # noqa: E731
        result = self.GraphPickler.debug_dumps(bad_obj, verbose=False)
        self.assertIn("root[1]", result)

    def test_nested_unpicklable_in_dict(self):
        """
        When a lambda is nested in a dict, debug_dumps should find the path
        to it (e.g., "root['bad_key']").
        """
        bad_obj = {"good": 1, "bad": lambda x: x}  # noqa: E731
        result = self.GraphPickler.debug_dumps(bad_obj, verbose=False)
        self.assertIn("root['bad']", result)

    def test_deeply_nested_unpicklable(self):
        """
        debug_dumps should find unpicklables even when deeply nested.
        """
        bad_obj = {"level1": {"level2": {"level3": [1, 2, lambda x: x]}}}  # noqa: E731
        result = self.GraphPickler.debug_dumps(bad_obj, verbose=False)
        self.assertIn("level3", result)
        self.assertIn("[2]", result)

    def test_unpicklable_in_tuple(self):
        """
        debug_dumps should handle tuples correctly.
        """
        bad_obj = (1, 2, lambda x: x)  # noqa: E731
        result = self.GraphPickler.debug_dumps(bad_obj, verbose=False)
        self.assertIn("root[2]", result)

    def test_unpicklable_in_set(self):
        """
        Sets containing unhashable items can't be tested, but we can test
        sets with objects that pickle might choke on differently. We'll test
        a frozenset to ensure container traversal works.
        """
        good_obj = frozenset([1, 2, 3])
        result = self.GraphPickler.debug_dumps(good_obj, verbose=False)
        self.assertIsNone(result)

    def test_max_depth_limit(self):
        """
        When max_depth is reached, the path should include "(depth_limit)".
        """

        def build_nested(depth):
            if depth == 0:
                return lambda x: x  # noqa: E731
            return [build_nested(depth - 1)]

        deeply_nested = build_nested(100)
        result = self.GraphPickler.debug_dumps(
            deeply_nested, max_depth=5, verbose=False
        )
        self.assertIn("depth_limit", result)

    def test_object_with_unpicklable_attribute(self):
        """
        An object with an unpicklable attribute in __dict__ should be found.
        """

        class Container:
            def __init__(self):
                self.good = 1
                self.bad = lambda x: x  # noqa: E731

        obj = Container()
        result = self.GraphPickler.debug_dumps(obj, verbose=False)
        self.assertIn("bad", result)

    def test_dataclass_with_unpicklable_field(self):
        """
        A dataclass with an unpicklable field should be found.
        """
        import dataclasses

        @dataclasses.dataclass
        class MyData:
            good: int
            bad: object

        obj = MyData(good=1, bad=lambda x: x)  # noqa: E731
        result = self.GraphPickler.debug_dumps(obj, verbose=False)
        self.assertIn("bad", result)

    def test_verbose_output(self):
        """
        When verbose=True, output should be printed (captured via capsys or similar).
        We just verify it doesn't crash.
        """
        import io
        import sys

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            self.GraphPickler.debug_dumps([1, lambda x: x], verbose=True)  # noqa: E731
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        self.assertIn("Walking:", output)

    def test_generator_object(self):
        """
        Generator objects are not picklable. debug_dumps should identify them.
        """

        def gen():
            yield 1

        bad_obj = {"data": gen()}
        result = self.GraphPickler.debug_dumps(bad_obj, verbose=False)
        self.assertIn("data", result)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
