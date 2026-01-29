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


class TestHigherOrderOperatorPickle(TestCase):
    """Tests for HigherOrderOperator pickling support in GraphPickler."""

    def setUp(self):
        super().setUp()
        from torch.fx._graph_pickler import _OpPickleData, GraphPickler, Options

        self.GraphPickler = GraphPickler
        self.Options = Options
        self._OpPickleData = _OpPickleData

    def test_higher_order_operator_pickle_roundtrip(self):
        """
        Test that a HigherOrderOperator (e.g., cond) can be pickled and
        unpickled correctly through the _OpPickleData.
        """
        from torch._higher_order_ops.cond import cond_op

        options = self.Options(ops_filter=None)
        pickle_data = self._OpPickleData.pickle(cond_op, options)

        from torch.fx._graph_pickler import _HigherOrderOperatorPickleData

        self.assertIsInstance(pickle_data, _HigherOrderOperatorPickleData)
        self.assertEqual(pickle_data.name, "cond")

    def test_higher_order_operator_unpickle(self):
        """
        Test that a pickled HigherOrderOperator can be unpickled back
        to the same operator.
        """
        from torch._higher_order_ops.cond import cond_op
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx._graph_pickler import (
            _HigherOrderOperatorPickleData,
            _UnpickleState,
        )
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        pickle_data = _HigherOrderOperatorPickleData("cond")
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        unpickle_state = _UnpickleState(fake_mode)
        unpickled_op = pickle_data.unpickle(unpickle_state)
        self.assertIs(unpickled_op, cond_op)

    def test_higher_order_operator_not_found_error(self):
        """
        Test that unpickling a non-existent HigherOrderOperator raises
        a helpful error message.
        """
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx._graph_pickler import (
            _HigherOrderOperatorPickleData,
            _UnpickleState,
        )
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        pickle_data = _HigherOrderOperatorPickleData("non_existent_op")
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        unpickle_state = _UnpickleState(fake_mode)
        with self.assertRaises(RuntimeError) as cm:
            pickle_data.unpickle(unpickle_state)
        self.assertIn("non_existent_op", str(cm.exception))
        self.assertIn("not found", str(cm.exception))


class TestHopSchemaPickle(TestCase):
    """Tests for HopSchema pickling support."""

    def test_hop_schema_pickle_roundtrip(self):
        """
        Test that a HopSchema can be pickled and unpickled correctly.
        The schema should be reconstructed from its string representation.
        """
        import pickle

        from torch._higher_order_ops.schema import HopSchema

        schema = HopSchema(
            name="test_op",
            overload_name="",
            arguments=[],
            returns=[],
            is_vararg=False,
            is_varret=False,
            schema_tree_spec=None,
        )
        pickled = pickle.dumps(schema)
        unpickled = pickle.loads(pickled)
        self.assertIsInstance(unpickled, HopSchema)
        self.assertEqual(unpickled.name, "test_op")
        self.assertEqual(unpickled.is_vararg, False)
        self.assertEqual(unpickled.is_varret, False)
        self.assertIsNone(unpickled.tree_spec)

    def test_hop_schema_pickle_with_arguments(self):
        """
        Test that a HopSchema with arguments can be pickled and unpickled.
        """
        import pickle

        from torch._higher_order_ops.schema import HopSchema

        tensor_type = torch._C.TensorType.get()
        arg = torch._C.Argument("x", tensor_type, None, None, False, None)
        ret = torch._C.Argument("", tensor_type, None, None, False, None)
        schema = HopSchema(
            name="test_op_with_args",
            overload_name="",
            arguments=[arg],
            returns=[ret],
            is_vararg=True,
            is_varret=True,
            schema_tree_spec=None,
        )
        pickled = pickle.dumps(schema)
        unpickled = pickle.loads(pickled)
        self.assertIsInstance(unpickled, HopSchema)
        self.assertEqual(unpickled.name, "test_op_with_args")
        self.assertEqual(len(unpickled.arguments), 1)
        self.assertEqual(len(unpickled.returns), 1)
        self.assertEqual(unpickled.is_vararg, True)
        self.assertEqual(unpickled.is_varret, True)


class TestSerializedGraphModule(TestCase):
    """Tests for SerializedGraphModule using GraphPickler."""

    def test_simple_graph_module_roundtrip(self):
        """
        Test that a simple GraphModule can be serialized and deserialized.
        """
        from torch._functorch._aot_autograd.aot_autograd_result import (
            SerializedGraphModule,
        )

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        gm = torch.fx.symbolic_trace(SimpleModule())
        serialized = SerializedGraphModule(gm)
        deserialized = serialized.deserialize()
        self.assertIsInstance(deserialized, torch.fx.GraphModule)
        test_input = torch.tensor([1.0, 2.0, 3.0])
        original_output = gm(test_input)
        deserialized_output = deserialized(test_input)
        self.assertEqual(original_output, deserialized_output)

    def test_graph_module_with_multiple_ops(self):
        """
        Test serialization of a GraphModule with multiple operations.
        """
        from torch._functorch._aot_autograd.aot_autograd_result import (
            SerializedGraphModule,
        )

        class MultiOpModule(torch.nn.Module):
            def forward(self, x, y):
                z = x + y
                w = z * 2
                return w - 1

        gm = torch.fx.symbolic_trace(MultiOpModule())
        serialized = SerializedGraphModule(gm)
        deserialized = serialized.deserialize()
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([3.0, 4.0])
        self.assertEqual(gm(x, y), deserialized(x, y))


class TestGraphModuleGetState(TestCase):
    """Tests that _GraphModulePickleData respects custom __getstate__ methods."""

    def test_graph_module_getstate_is_called(self):
        """
        When a GraphModule subclass defines __getstate__, the pickler should
        use it instead of copying __dict__ directly. This ensures that custom
        serialization logic (e.g., filtering out unpicklable attributes) is
        respected.
        """
        from unittest.mock import patch

        from torch.fx._graph_pickler import _GraphModulePickleData, Options

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        gm = torch.fx.symbolic_trace(SimpleModule())

        getstate_called = []

        original_getstate = gm.__getstate__

        def mock_getstate():
            getstate_called.append(True)
            return original_getstate()

        with patch.object(gm, "__getstate__", mock_getstate):
            _GraphModulePickleData(gm, Options())

        self.assertEqual(
            len(getstate_called),
            1,
            "__getstate__ should be called exactly once during serialization",
        )

    def test_graph_module_custom_getstate_filters_attributes(self):
        """
        Test that a custom __getstate__ that filters attributes is respected.
        The filtered attribute should not appear in the pickled data.
        """
        from torch.fx._graph_pickler import _GraphModulePickleData, Options

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        gm = torch.fx.symbolic_trace(SimpleModule())
        gm.unpicklable_attr = lambda: None

        original_getstate = gm.__getstate__

        def custom_getstate():
            state = original_getstate()
            if "unpicklable_attr" in state:
                del state["unpicklable_attr"]
            return state

        gm.__getstate__ = custom_getstate

        pickle_data = _GraphModulePickleData(gm, Options())

        self.assertNotIn(
            "unpicklable_attr",
            pickle_data.gm_dict,
            "Custom __getstate__ should filter out unpicklable_attr",
        )


class TestNodeMetadataKeyFilter(TestCase):
    """Tests for the node_metadata_key_filter option in GraphPickler."""

    def test_default_filter_excludes_known_unserializable_keys(self):
        """
        The default _node_metadata_key_filter_safe should exclude known
        unserializable metadata keys like source_fn_stack, nn_module_stack,
        and fwd_source_fn_stack.
        """
        from torch.fx._graph_pickler import _node_metadata_key_filter_safe

        self.assertFalse(_node_metadata_key_filter_safe("source_fn_stack"))
        self.assertFalse(_node_metadata_key_filter_safe("nn_module_stack"))
        self.assertFalse(_node_metadata_key_filter_safe("fwd_source_fn_stack"))

    def test_default_filter_allows_standard_keys(self):
        """
        The default _node_metadata_key_filter_safe should allow standard
        metadata keys that are normally serializable.
        """
        from torch.fx._graph_pickler import _node_metadata_key_filter_safe

        self.assertTrue(_node_metadata_key_filter_safe("val"))
        self.assertTrue(_node_metadata_key_filter_safe("tensor_meta"))
        self.assertTrue(_node_metadata_key_filter_safe("stack_trace"))
        self.assertTrue(_node_metadata_key_filter_safe("example_value"))

    def test_node_pickle_data_uses_metadata_filter(self):
        """
        _NodePickleData should use the node_metadata_key_filter from Options
        to filter node metadata during serialization.
        """
        from torch.fx._graph_pickler import _NodePickleData, Options

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        gm = torch.fx.symbolic_trace(SimpleModule())

        for node in gm.graph.nodes:
            node.meta["source_fn_stack"] = "should_be_filtered"
            node.meta["nn_module_stack"] = "should_be_filtered"
            node.meta["keep_this"] = "should_remain"

        options = Options()
        node_mapping: dict[torch.fx.Node, _NodePickleData] = {}
        for node in gm.graph.nodes:
            node_mapping[node] = _NodePickleData(node, node_mapping, options)

        for node in gm.graph.nodes:
            pickle_data = node_mapping[node]
            self.assertNotIn("source_fn_stack", pickle_data.meta)
            self.assertNotIn("nn_module_stack", pickle_data.meta)
            self.assertIn("keep_this", pickle_data.meta)
            self.assertEqual(pickle_data.meta["keep_this"], "should_remain")

    def test_custom_metadata_filter(self):
        """
        A custom node_metadata_key_filter should be respected. When provided,
        only keys that pass the filter (return True) should be included.
        """
        from torch.fx._graph_pickler import _NodePickleData, Options

        def custom_filter(key: str) -> bool:
            return key.startswith("allowed_")

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        gm = torch.fx.symbolic_trace(SimpleModule())

        for node in gm.graph.nodes:
            node.meta["allowed_key"] = "included"
            node.meta["blocked_key"] = "excluded"
            node.meta["val"] = "excluded_too"

        options = Options(node_metadata_key_filter=custom_filter)
        node_mapping: dict[torch.fx.Node, _NodePickleData] = {}
        for node in gm.graph.nodes:
            node_mapping[node] = _NodePickleData(node, node_mapping, options)

        for node in gm.graph.nodes:
            pickle_data = node_mapping[node]
            self.assertIn("allowed_key", pickle_data.meta)
            self.assertNotIn("blocked_key", pickle_data.meta)
            self.assertNotIn("val", pickle_data.meta)

    def test_none_filter_includes_all_keys(self):
        """
        When node_metadata_key_filter is None, all metadata keys should be
        included without filtering.
        """
        from torch.fx._graph_pickler import _NodePickleData, Options

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        gm = torch.fx.symbolic_trace(SimpleModule())

        for node in gm.graph.nodes:
            node.meta["source_fn_stack"] = "normally_filtered"
            node.meta["nn_module_stack"] = "normally_filtered"
            node.meta["custom_key"] = "included"

        options = Options(node_metadata_key_filter=None)
        node_mapping: dict[torch.fx.Node, _NodePickleData] = {}
        for node in gm.graph.nodes:
            node_mapping[node] = _NodePickleData(node, node_mapping, options)

        for node in gm.graph.nodes:
            pickle_data = node_mapping[node]
            self.assertIn("source_fn_stack", pickle_data.meta)
            self.assertIn("nn_module_stack", pickle_data.meta)
            self.assertIn("custom_key", pickle_data.meta)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
