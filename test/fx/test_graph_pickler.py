# Owner(s): ["module: fx"]

#
# Tests the graph pickler by using pickling on all the inductor tests.
#

import contextlib
import importlib
import os
import sys
import unittest
from unittest.mock import patch

import torch
import torch.library
from torch._dynamo.testing import make_test_cls_with_patches
from torch._inductor.test_case import TestCase
from torch.testing._internal.inductor_utils import HAS_CPU
from torch.utils._import_utils import import_dill


dill = import_dill()
HAS_DILL = dill is not None


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


if HAS_CPU and HAS_DILL:

    class GraphPicklerCpuTests(TestCase):
        common = check_model
        device = "cpu"

    copy_tests(GraphPicklerCommonTemplate, GraphPicklerCpuTests, "cpu", test_failures)


@unittest.skipUnless(HAS_DILL, "dill not available")
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


@unittest.skipUnless(HAS_DILL, "dill not available")
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


@unittest.skipUnless(HAS_DILL, "dill not available")
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


@unittest.skipUnless(HAS_DILL, "dill not available")
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


@unittest.skipUnless(HAS_DILL, "dill not available")
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


@unittest.skipUnless(HAS_DILL, "dill not available")
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


@unittest.skipUnless(HAS_DILL, "dill not available")
class TestDillSerializationFeatures(TestCase):
    """
    Tests for dill-enabled serialization features in GraphPickler.
    These test cases would have failed with standard pickle because pickle
    relies on module-level name lookup and cannot handle inner functions,
    lambdas, locally defined classes, and closures capturing runtime state.
    """

    def setUp(self):
        super().setUp()
        from torch.fx._graph_pickler import GraphPickler, Options

        self.GraphPickler = GraphPickler
        self.Options = Options

    def test_inner_function_in_graph_metadata(self):
        """
        Test that a graph with an inner function in node metadata can be
        serialized and deserialized. Standard pickle would fail because
        inner functions are not defined at module level.
        """
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        gm = torch.fx.symbolic_trace(SimpleModule())

        def inner_helper(x):
            return x * 2

        for node in gm.graph.nodes:
            node.meta["inner_fn"] = inner_helper

        options = self.Options(node_metadata_key_filter=None)
        serialized = self.GraphPickler.dumps(gm, options)
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        deserialized = self.GraphPickler.loads(serialized, fake_mode)

        self.assertIsInstance(deserialized, torch.fx.GraphModule)
        for node in deserialized.graph.nodes:
            self.assertIn("inner_fn", node.meta)
            self.assertEqual(node.meta["inner_fn"](5), 10)

    def test_lambda_in_graph_metadata(self):
        """
        Test that a graph with a lambda in node metadata can be serialized
        and deserialized. Standard pickle cannot handle lambdas because they
        are anonymous functions without module-level names.
        """
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        gm = torch.fx.symbolic_trace(SimpleModule())

        for node in gm.graph.nodes:
            node.meta["lambda_fn"] = lambda x: x * 3  # noqa: E731

        options = self.Options(node_metadata_key_filter=None)
        serialized = self.GraphPickler.dumps(gm, options)
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        deserialized = self.GraphPickler.loads(serialized, fake_mode)

        self.assertIsInstance(deserialized, torch.fx.GraphModule)
        for node in deserialized.graph.nodes:
            self.assertIn("lambda_fn", node.meta)
            self.assertEqual(node.meta["lambda_fn"](7), 21)

    def test_closure_capturing_runtime_state(self):
        """
        Test that a graph with a closure capturing runtime state can be
        serialized and deserialized. Standard pickle cannot handle closures
        because it cannot capture the closed-over variables.
        """
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        gm = torch.fx.symbolic_trace(SimpleModule())

        captured_value = 42
        multiplier = 10

        def closure_fn(x):
            return x + captured_value * multiplier

        for node in gm.graph.nodes:
            node.meta["closure_fn"] = closure_fn

        options = self.Options(node_metadata_key_filter=None)
        serialized = self.GraphPickler.dumps(gm, options)
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        deserialized = self.GraphPickler.loads(serialized, fake_mode)

        self.assertIsInstance(deserialized, torch.fx.GraphModule)
        for node in deserialized.graph.nodes:
            self.assertIn("closure_fn", node.meta)
            self.assertEqual(node.meta["closure_fn"](8), 8 + 42 * 10)

    def test_locally_defined_class_in_metadata(self):
        """
        Test that a graph with locally defined class instances in node
        metadata can be serialized and deserialized. Standard pickle cannot
        handle classes not defined at module level.
        """
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        gm = torch.fx.symbolic_trace(SimpleModule())

        class LocalConfig:
            def __init__(self, value, name):
                self.value = value
                self.name = name

            def compute(self):
                return self.value * 2

        local_instance = LocalConfig(value=100, name="test_config")
        for node in gm.graph.nodes:
            node.meta["local_class"] = local_instance

        options = self.Options(node_metadata_key_filter=None)
        serialized = self.GraphPickler.dumps(gm, options)
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        deserialized = self.GraphPickler.loads(serialized, fake_mode)

        self.assertIsInstance(deserialized, torch.fx.GraphModule)
        for node in deserialized.graph.nodes:
            self.assertIn("local_class", node.meta)
            obj = node.meta["local_class"]
            self.assertEqual(obj.value, 100)
            self.assertEqual(obj.name, "test_config")
            self.assertEqual(obj.compute(), 200)

    def test_nested_closures_with_multiple_captures(self):
        """
        Test that deeply nested closures with multiple captured variables
        can be serialized and deserialized.
        """
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        gm = torch.fx.symbolic_trace(SimpleModule())

        outer_val = 5

        def outer_fn(a):
            inner_val = a * 2

            def inner_fn(b):
                return b + outer_val + inner_val

            return inner_fn

        nested_closure = outer_fn(10)

        for node in gm.graph.nodes:
            node.meta["nested_closure"] = nested_closure

        options = self.Options(node_metadata_key_filter=None)
        serialized = self.GraphPickler.dumps(gm, options)
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        deserialized = self.GraphPickler.loads(serialized, fake_mode)

        self.assertIsInstance(deserialized, torch.fx.GraphModule)
        for node in deserialized.graph.nodes:
            self.assertIn("nested_closure", node.meta)
            self.assertEqual(node.meta["nested_closure"](3), 3 + 5 + 20)

    def test_node_with_slice(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        def foo(x):
            return x[0 : x.shape[0]]

        gm = torch.fx.symbolic_trace(foo)

        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x):
    getattr_1 = x.shape
    getitem = getattr_1[0];  getattr_1 = None
    getitem_1 = x[slice(0, getitem, None)];  x = getitem = None
    return getitem_1""",
        )
        options = self.Options(node_metadata_key_filter=None)
        serialized = self.GraphPickler.dumps(gm, options)

        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        deserialized = self.GraphPickler.loads(serialized, fake_mode)
        deserialized.recompile()

        self.assertEqual(gm.code, deserialized.code)

    def test_lambda_with_default_arguments(self):
        """
        Test that lambdas with default arguments can be serialized. Standard
        pickle has trouble with default argument values that aren't simple
        literals.
        """
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        gm = torch.fx.symbolic_trace(SimpleModule())

        default_list = [1, 2, 3]
        fn_with_defaults = lambda x, y=default_list: x + sum(y)  # noqa: E731

        for node in gm.graph.nodes:
            node.meta["fn_with_defaults"] = fn_with_defaults

        options = self.Options(node_metadata_key_filter=None)
        serialized = self.GraphPickler.dumps(gm, options)
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        deserialized = self.GraphPickler.loads(serialized, fake_mode)

        self.assertIsInstance(deserialized, torch.fx.GraphModule)
        for node in deserialized.graph.nodes:
            self.assertIn("fn_with_defaults", node.meta)
            self.assertEqual(node.meta["fn_with_defaults"](10), 10 + 6)


@unittest.skipUnless(HAS_DILL, "dill not available")
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


class TestNodeStateSerialization(TestCase):
    def test_type_entry_preserved_in_getstate(self):
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.neg(x)
                return y + 1

        gm = torch.fx.symbolic_trace(M())
        node = next(n for n in gm.graph.nodes if n.op == "call_function")
        node.type = torch.Tensor
        state = node.__getstate__()
        self.assertIs(state["type"], torch.Tensor)


@unittest.skipUnless(HAS_DILL, "dill not available")
class TestIgnoreRawNode(TestCase):
    """Tests for the ignore_raw_node option in GraphPickler.Options."""

    def setUp(self):
        super().setUp()
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx._graph_pickler import GraphPickler, Options
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        self.GraphPickler = GraphPickler
        self.Options = Options
        self.fake_mode = FakeTensorMode(shape_env=ShapeEnv())

    def _make_graph_with_raw_node_in_meta(self):
        """Return a graph module whose first call_function node has a raw
        torch.fx.Node stored in its metadata under the key 'raw_ref'."""

        class M(torch.nn.Module):
            def forward(self, x):
                return x + 1

        gm = torch.fx.symbolic_trace(M())
        call_node = next((n for n in gm.graph.nodes if n.op == "call_function"), None)
        self.assertIsNotNone(call_node)
        # Store a raw Node reference in meta – this is the problematic case.
        call_node.meta["raw_ref"] = call_node
        return gm

    def test_raw_node_in_meta_raises_by_default(self):
        """Pickling should raise AssertionError when a raw Node is in metadata
        and ignore_raw_node is False (the default)."""
        gm = self._make_graph_with_raw_node_in_meta()
        with self.assertRaises(AssertionError) as cm:
            self.GraphPickler.dumps(gm)
        self.assertIn("raw Node", str(cm.exception))

    def test_raw_node_in_meta_with_ignore_raw_node(self):
        """With ignore_raw_node=True, pickling should succeed and the raw Node
        should be replaced with None after round-trip deserialization."""
        gm = self._make_graph_with_raw_node_in_meta()
        options = self.Options(ignore_raw_node=True)
        data = self.GraphPickler.dumps(gm, options)
        restored = self.GraphPickler.loads(data, self.fake_mode)
        self.assertIsInstance(restored, torch.fx.GraphModule)
        call_node = next(
            (n for n in restored.graph.nodes if n.op == "call_function"), None
        )
        self.assertIsNotNone(call_node)
        self.assertIsNone(call_node.meta.get("raw_ref"))


class _WeakrefTarget:
    """A simple picklable class that supports weak references, for use in tests.

    Plain dicts do not support weak references in Python, so tests must use
    instances of a regular class instead.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@unittest.skipUnless(HAS_DILL, "dill not available")
class TestWeakrefPickle(TestCase):
    """Tests that weakref objects are properly serialized and reconstructed."""

    def setUp(self):
        super().setUp()
        import weakref

        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx._graph_pickler import GraphPickler, Options
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        self.weakref = weakref
        self.GraphPickler = GraphPickler
        self.Options = Options
        self.fake_mode = FakeTensorMode(shape_env=ShapeEnv())

    def _make_graph_with_weakref_in_meta(self, ref_obj):
        """Return a graph module with a weakref stored in node metadata."""

        class M(torch.nn.Module):
            def forward(self, x):
                return x + 1

        gm = torch.fx.symbolic_trace(M())
        call_node = next((n for n in gm.graph.nodes if n.op == "call_function"), None)
        self.assertIsNotNone(call_node)
        call_node.meta["weak_ref"] = ref_obj
        return gm

    def test_alive_weakref_in_meta_is_reconstructed(self):
        """An alive weakref.ref in node metadata should be reconstructed as a weakref."""
        target = _WeakrefTarget(key="value")
        weak = self.weakref.ref(target)
        gm = self._make_graph_with_weakref_in_meta(weak)
        # Also store a strong ref so the referent survives after unpickling
        call_node = next((n for n in gm.graph.nodes if n.op == "call_function"), None)
        call_node.meta["strong_ref"] = target

        options = self.Options(node_metadata_key_filter=None)
        data = self.GraphPickler.dumps(gm, options)
        restored = self.GraphPickler.loads(data, self.fake_mode)

        self.assertIsInstance(restored, torch.fx.GraphModule)
        call_node = next(
            (n for n in restored.graph.nodes if n.op == "call_function"), None
        )
        self.assertIsNotNone(call_node)
        restored_ref = call_node.meta.get("weak_ref")
        self.assertIsInstance(restored_ref, self.weakref.ref)
        self.assertEqual(restored_ref().key, "value")

    def test_dead_weakref_in_meta_unpickles_as_callable_none(self):
        """A dead weakref should unpickle as a callable that returns None."""
        target = _WeakrefTarget()
        weak = self.weakref.ref(target)
        gm = self._make_graph_with_weakref_in_meta(weak)
        # Kill the referent so the weakref is dead at pickle time
        del target

        options = self.Options(node_metadata_key_filter=None)
        data = self.GraphPickler.dumps(gm, options)
        restored = self.GraphPickler.loads(data, self.fake_mode)

        self.assertIsInstance(restored, torch.fx.GraphModule)
        call_node = next(
            (n for n in restored.graph.nodes if n.op == "call_function"), None
        )
        self.assertIsNotNone(call_node)
        restored_ref = call_node.meta.get("weak_ref")
        # Should be callable and return None, like a dead weakref
        self.assertIsNotNone(restored_ref)
        self.assertIsNone(restored_ref())

    def test_keyed_ref_in_meta_is_reconstructed(self):
        """A weakref.KeyedRef (from WeakValueDictionary) should be reconstructed."""
        wvd = self.weakref.WeakValueDictionary()
        target = _WeakrefTarget(val=42)
        wvd["k"] = target
        keyed_ref = wvd.data["k"]
        self.assertIsInstance(keyed_ref, self.weakref.KeyedRef)

        gm = self._make_graph_with_weakref_in_meta(keyed_ref)
        # Also store a strong ref so the referent survives after unpickling
        call_node = next((n for n in gm.graph.nodes if n.op == "call_function"), None)
        call_node.meta["strong_ref"] = target

        options = self.Options(node_metadata_key_filter=None)
        data = self.GraphPickler.dumps(gm, options)
        restored = self.GraphPickler.loads(data, self.fake_mode)

        self.assertIsInstance(restored, torch.fx.GraphModule)
        call_node = next(
            (n for n in restored.graph.nodes if n.op == "call_function"), None
        )
        self.assertIsNotNone(call_node)
        restored_ref = call_node.meta.get("weak_ref")
        self.assertIsInstance(restored_ref, self.weakref.ref)
        self.assertEqual(restored_ref().val, 42)

    def test_weakref_in_module_dict_is_reconstructed(self):
        """A weakref stored in the graph module's __dict__ should be reconstructed."""

        class M(torch.nn.Module):
            def forward(self, x):
                return x + 1

        gm = torch.fx.symbolic_trace(M())
        target = _WeakrefTarget(key="value")
        gm._weak_cache = self.weakref.ref(target)
        # Also store a strong ref so the referent survives after unpickling
        gm._strong_cache = target

        options = self.Options(node_metadata_key_filter=None)
        data = self.GraphPickler.dumps(gm, options)
        restored = self.GraphPickler.loads(data, self.fake_mode)

        self.assertIsInstance(restored, torch.fx.GraphModule)
        restored_ref = restored._weak_cache
        self.assertIsInstance(restored_ref, self.weakref.ref)
        self.assertEqual(restored_ref().key, "value")

    def test_weakref_and_strong_ref_share_same_object(self):
        """When a weakref and a strong ref point to the same object, pickle's
        memo should deduplicate them so they share identity after unpickling.
        This also covers the case where the weakref is the first reference
        pickle encounters — the memo must still work correctly."""

        class M(torch.nn.Module):
            def forward(self, x):
                return x + 1

        gm = torch.fx.symbolic_trace(M())
        target = _WeakrefTarget(key="value")
        weak = self.weakref.ref(target)

        call_node = next((n for n in gm.graph.nodes if n.op == "call_function"), None)
        self.assertIsNotNone(call_node)
        # Put weakref first so it's the first reference pickle encounters
        call_node.meta["weak_ref"] = weak
        call_node.meta["strong_ref"] = target

        options = self.Options(node_metadata_key_filter=None)
        data = self.GraphPickler.dumps(gm, options)
        restored = self.GraphPickler.loads(data, self.fake_mode)

        self.assertIsInstance(restored, torch.fx.GraphModule)
        restored_node = next(
            (n for n in restored.graph.nodes if n.op == "call_function"), None
        )
        self.assertIsNotNone(restored_node)

        restored_weak = restored_node.meta["weak_ref"]
        restored_strong = restored_node.meta["strong_ref"]

        self.assertIsInstance(restored_weak, self.weakref.ref)
        # The weakref's referent and the strong ref should be the same object
        self.assertIs(restored_weak(), restored_strong)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
