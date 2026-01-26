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


# Module-level classes for FSDP pickle tests (needed for proper qualname resolution)
class FSDPModule:
    """Mock FSDPModule class for testing. Name must be exactly 'FSDPModule'."""

    pass


class _TestSimpleLayer(torch.nn.Module):
    """A simple layer to wrap with FSDP for testing."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class TestDynamicFSDPClassPickle(TestCase):
    """
    Tests for dynamic FSDP class pickling support in GraphPickler.

    FSDP creates wrapper classes at runtime using type(), which cannot be
    pickled normally. These tests verify that GraphPickler correctly handles
    these dynamic classes.
    """

    def _create_dynamic_fsdp_class(self, orig_cls):
        """Create a dynamic FSDP class like the real FSDP implementation does."""
        dct = {"__deepcopy__": lambda *args, **kwargs: None}
        dynamic_cls = type(f"FSDP{orig_cls.__name__}", (FSDPModule, orig_cls), dct)
        return dynamic_cls

    def test_is_dynamic_fsdp_class_detection(self):
        """Test that _is_dynamic_fsdp_class correctly identifies FSDP classes."""
        from torch.fx._graph_pickler import _is_dynamic_fsdp_class

        # Create a dynamic FSDP class
        dynamic_cls = self._create_dynamic_fsdp_class(_TestSimpleLayer)

        # Should be detected as an FSDP class
        self.assertTrue(_is_dynamic_fsdp_class(dynamic_cls))

        # Regular classes should not be detected
        self.assertFalse(_is_dynamic_fsdp_class(_TestSimpleLayer))
        self.assertFalse(_is_dynamic_fsdp_class(torch.nn.Module))
        self.assertFalse(_is_dynamic_fsdp_class(object))

        # Non-type objects should not be detected
        self.assertFalse(_is_dynamic_fsdp_class("string"))
        self.assertFalse(_is_dynamic_fsdp_class(123))
        self.assertFalse(_is_dynamic_fsdp_class(_TestSimpleLayer()))

    def test_dynamic_fsdp_class_pickle_data(self):
        """Test _DynamicFSDPClassPickleData serialization."""
        from torch.fx._graph_pickler import _DynamicFSDPClassPickleData

        dynamic_cls = self._create_dynamic_fsdp_class(_TestSimpleLayer)

        # Create pickle data
        data = _DynamicFSDPClassPickleData(dynamic_cls)

        # Check that the data captured the right information
        self.assertEqual(data.fsdp_mixin_qualname, "FSDPModule")
        self.assertEqual(data.orig_cls_qualname, "_TestSimpleLayer")
        self.assertEqual(data.class_name, "FSDP_TestSimpleLayer")
        self.assertIn("__deepcopy__", data.class_dict_keys)

    def test_unpickle_reconstructs_class(self):
        """Test that unpickling reconstructs the dynamic class correctly."""
        from torch.fx._graph_pickler import (
            _DynamicFSDPClassPickleData,
            _unpickle_dynamic_fsdp_class,
        )

        dynamic_cls = self._create_dynamic_fsdp_class(_TestSimpleLayer)

        # Simulate the pickle/unpickle cycle
        data = _DynamicFSDPClassPickleData(dynamic_cls)
        args = data.as_tuple()

        # Reconstruct the class
        reconstructed = _unpickle_dynamic_fsdp_class(*args)

        # Verify the reconstructed class
        self.assertEqual(reconstructed.__name__, dynamic_cls.__name__)
        self.assertEqual(len(reconstructed.__mro__), len(dynamic_cls.__mro__))
        self.assertEqual(
            [c.__name__ for c in reconstructed.__mro__],
            [c.__name__ for c in dynamic_cls.__mro__],
        )

    def test_full_pickle_cycle_with_graph_pickler(self):
        """Test that GraphPickler can serialize and deserialize FSDP classes."""
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx._graph_pickler import GraphPickler

        dynamic_cls = self._create_dynamic_fsdp_class(_TestSimpleLayer)

        # Pickle using GraphPickler
        serialized = GraphPickler.dumps(dynamic_cls)

        # Unpickle
        with FakeTensorMode() as fake_mode:
            deserialized = GraphPickler.loads(serialized, fake_mode)

        # Verify
        self.assertEqual(deserialized.__name__, dynamic_cls.__name__)
        self.assertEqual(
            [c.__name__ for c in deserialized.__mro__],
            [c.__name__ for c in dynamic_cls.__mro__],
        )


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
