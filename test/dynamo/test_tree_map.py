# Owner(s): ["module: dynamo"]

try:
    import optree
except ImportError:  # pragma: no cover
    optree = None

import torch
import torch._dynamo
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils import _pytree as pytree


try:
    import torch.utils._cxx_pytree as cxx_pytree
except ImportError:  # pragma: no cover
    cxx_pytree = None


def _tensor_leaf(*values):
    first = values[0].clone()
    for other in values[1:]:
        first = first + other
    return first


def _combine_leaves(*values):
    first = values[0]
    if isinstance(first, torch.Tensor):
        return _tensor_leaf(*values)
    if first is None:
        return None
    if isinstance(first, tuple):
        # When tuples are marked as leaves, keep the structure from
        # the leading tree so that specs remain aligned.
        return first
    total = first
    for other in values[1:]:
        total = total + other
    return total


def _tuple_is_leaf(node):
    return isinstance(node, tuple)


def _require_optree(test_case):
    if optree is None:
        test_case.skipTest("optree is unavailable")


TREE_MAP_IMPLEMENTATIONS = []
if optree is not None:
    TREE_MAP_IMPLEMENTATIONS.append(("optree", optree.tree_map))
TREE_MAP_IMPLEMENTATIONS.append(("pytree_python", pytree.tree_map))
if cxx_pytree is not None:
    TREE_MAP_IMPLEMENTATIONS.append(("pytree_cxx", cxx_pytree.tree_map))


KWARG_CASES = [
    ("default", {}, None),
    ("none_is_leaf", {"none_is_leaf": True}, {"optree"}),
    ("is_leaf", {"is_leaf": _tuple_is_leaf}, None),
    ("namespace", {"namespace": "torch"}, {"optree"}),
    (
        "namespace_and_none_is_leaf",
        {"namespace": "torch", "none_is_leaf": True},
        {"optree"},
    ),
    (
        "namespace_none_is_leaf_predicate",
        {"namespace": "torch", "none_is_leaf": True, "is_leaf": _tuple_is_leaf},
        {"optree"},
    ),
]


_NONE_IS_LEAF_UNSET = object()


def _build_tree(offset: int) -> dict[str, object]:
    base = torch.arange(4, dtype=torch.float32).reshape(2, 2) + offset
    nested = base + 5
    return {
        "tensor": base,
        "list": [
            base + 1,
            {
                "inner": base + 2,
                "none": None,
            },
        ],
        "tuple": (3 + offset, (nested, None)),
        "const_dict": {"leaf": base + 3},
    }


def _assert_trees_allclose(test_case: TestCase, ref, res) -> None:
    ref_flat, ref_spec = pytree.tree_flatten(ref)
    res_flat, res_spec = pytree.tree_flatten(res)
    test_case.assertEqual(ref_spec, res_spec)
    for expected, actual in zip(ref_flat, res_flat):
        if isinstance(expected, torch.Tensor):
            test_case.assertTrue(torch.allclose(expected, actual))
        else:
            test_case.assertEqual(expected, actual)


@instantiate_parametrized_tests
class TreeMapCompileTests(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def _run_tree_map(self, tree_map_impl, kwargs):
        lhs = _build_tree(0)
        rhs = _build_tree(7)

        def fn(a, b):
            return tree_map_impl(_combine_leaves, a, b, **kwargs)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        expected = fn(lhs, rhs)
        result = compiled(lhs, rhs)
        _assert_trees_allclose(self, expected, result)

    @parametrize("tree_map_name,tree_map_impl", TREE_MAP_IMPLEMENTATIONS)
    @parametrize("kwargs_name,kwargs,allowed_impls", KWARG_CASES)
    def test_tree_map_variants(
        self,
        tree_map_name: str,
        tree_map_impl,
        kwargs_name: str,
        kwargs: dict,
        allowed_impls,
    ) -> None:
        if tree_map_name == "pytree_cxx" and cxx_pytree is None:
            self.skipTest("torch.utils._cxx_pytree is unavailable")
        if allowed_impls is not None and tree_map_name not in allowed_impls:
            self.skipTest("kwargs unsupported for implementation")
        self._run_tree_map(tree_map_impl, kwargs)

    def test_tree_map_rejects_mismatched_container_types(self) -> None:
        def fn(a, b):
            return pytree.tree_map(lambda u, v: u + v, a, b)

        lhs = [torch.ones(2), torch.ones(2)]
        rhs = (torch.ones(2), torch.ones(2))

        with self.assertRaises(ValueError):
            fn(lhs, rhs)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            (ValueError, torch._dynamo.exc.Unsupported),
            "Node type mismatch",
        ):
            compiled(lhs, rhs)

    def test_tree_map_is_leaf_handles_tensor_nodes(self) -> None:
        def fn(tree):
            return pytree.tree_map(
                lambda pair: torch.stack(pair).sum(dim=0),
                tree,
                is_leaf=lambda node: isinstance(node, tuple),
            )

        tree = [(torch.ones(2), torch.ones(2) * 4)]
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        expected = fn(tree)
        result = compiled(tree)
        _assert_trees_allclose(self, expected, result)

    def test_tree_map_only_applies_to_tensor_nodes(self) -> None:
        tree = {"tensor": torch.ones(2), "int": 3}

        def mapper(node):
            if not isinstance(node, torch.Tensor):
                raise AssertionError("mapper should only see tensors")
            return node + 2

        def fn(arg):
            return pytree.tree_map_only(torch.Tensor, mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        expected = fn(tree)
        result = compiled(tree)
        _assert_trees_allclose(self, expected, result)

    def test_tree_map_only_multiple_trees_falls_back(self) -> None:
        lhs = {"a": torch.ones(2), "b": torch.ones(2) * 2}
        rhs = {"a": torch.ones(2) * 3, "b": torch.ones(2) * 4}

        def fn(a, b):
            return pytree.tree_map_only(torch.Tensor, lambda x, y: x + y, a, b)

        with self.assertRaisesRegex(TypeError, "callable"):
            fn(lhs, rhs)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            (TypeError, torch._dynamo.exc.Unsupported),
            r"(callable|Unsupported function call)",
        ):
            compiled(lhs, rhs)

    def test_tree_map_only_handles_multiple_types(self) -> None:
        tree = {"int": 7, "tuple": (1, 2), "tensor": torch.ones(2)}

        def mapper(node):
            if isinstance(node, int):
                return node + 1
            if isinstance(node, tuple):
                return tuple(val + 10 for val in node)
            raise AssertionError("unexpected node passed to mapper")

        def fn(arg):
            return pytree.tree_map_only((int, tuple), mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        expected = fn(tree)
        result = compiled(tree)
        _assert_trees_allclose(self, expected, result)

    def test_tree_map_is_leaf_non_constant_fallback(self) -> None:
        tree = {"a": torch.arange(2.0), "b": torch.arange(2.0) + 1}

        def is_leaf(node):
            if isinstance(node, torch.Tensor):
                # Depends on runtime tensor value; cannot be folded to a constant.
                return (node.sum() > 1).item()
            return False

        def mapper(node):
            return node * 2 if isinstance(node, torch.Tensor) else node

        def fn(arg):
            return pytree.tree_map(mapper, arg, is_leaf=is_leaf)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        expected = fn(tree)
        result = compiled(tree)
        _assert_trees_allclose(self, expected, result)

    def test_tree_map_only_predicate_selector_skips_fastpath(self) -> None:
        tree = {"keep": torch.ones(2), "other": (1, 2)}

        def selector(node):
            return isinstance(node, torch.Tensor) and node.shape == (2,)

        def mapper(node):
            return node + 5 if isinstance(node, torch.Tensor) else node

        def fn(arg):
            return pytree.tree_map_only(selector, mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        expected = fn(tree)
        result = compiled(tree)
        _assert_trees_allclose(self, expected, result)

    def test_tree_map_none_nodes_reject_mismatched_siblings(self) -> None:
        _require_optree(self)

        def fn(a, b):
            return optree.tree_map(lambda u, v: (u, v), a, b)

        lhs = {"k": None}
        rhs = {"k": torch.ones(2)}

        with self.assertRaisesRegex(ValueError, "Expected None"):
            fn(lhs, rhs)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            (ValueError, torch._dynamo.exc.Unsupported),
            r"(Expected None|expected <class 'NoneType'>)",
        ):
            compiled(lhs, rhs)

    @parametrize("tree_map_name,tree_map_impl", TREE_MAP_IMPLEMENTATIONS)
    def test_tree_map_none_nodes_default_behavior(
        self, tree_map_name: str, tree_map_impl
    ) -> None:
        if tree_map_name == "optree":
            self.skipTest("optree treats None as an internal node by default")

        def fn(a, b):
            return tree_map_impl(lambda u, v: (u, v), a, b)

        tree = {"k": None}
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        expected = fn(tree, tree)
        result = compiled(tree, tree)

        self.assertEqual(result["k"], (None, None))
        self.assertEqual(result, expected)

    def test_constantvariable_handles_none_is_leaf_kwarg(self) -> None:
        _require_optree(self)

        tree = {"none": None}

        def run_case(none_is_leaf_flag):
            def fn(arg):
                def mapper(node):
                    if node is None:
                        return "visited"
                    return node

                kwargs = {}
                if none_is_leaf_flag is not _NONE_IS_LEAF_UNSET:
                    kwargs["none_is_leaf"] = none_is_leaf_flag
                return optree.tree_map(mapper, arg, **kwargs)

            compiled = torch.compile(fn, backend="eager", fullgraph=True)
            expected = fn(tree)
            result = compiled(tree)
            self.assertEqual(result, expected)
            return result["none"]

        self.assertEqual(run_case(_NONE_IS_LEAF_UNSET), None)
        self.assertEqual(run_case(False), None)
        self.assertEqual(run_case(True), "visited")

    def test_constantvariable_handles_python_and_dtype_leaves(self) -> None:
        _require_optree(self)

        tree = {
            "int": 7,
            "nested": {"string": "foo", "dtype": torch.float32},
        }

        def fn(arg):
            def mapper(node):
                if isinstance(node, int):
                    return node + 1
                if isinstance(node, str):
                    return node.upper()
                if isinstance(node, torch.dtype):
                    return torch.float64
                return node

            return optree.tree_map(mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        expected = fn(tree)
        result = compiled(tree)
        self.assertEqual(result["int"], 8)
        self.assertEqual(result["nested"]["string"], "FOO")
        self.assertIs(result["nested"]["dtype"], torch.float64)
        self.assertEqual(result, expected)

    @parametrize("tree_map_name,tree_map_impl", TREE_MAP_IMPLEMENTATIONS)
    def test_user_defined_object_treated_as_leaf(
        self, tree_map_name: str, tree_map_impl
    ) -> None:
        """User-defined objects (not registered in pytree) should be treated as leaves."""

        class MyCustomClass:
            def __init__(self, value):
                self.value = value

        obj1 = MyCustomClass(torch.ones(2))
        tree = {"custom": obj1, "tensor": torch.zeros(2)}

        visited_types = []

        def mapper(node):
            visited_types.append(type(node))
            if isinstance(node, MyCustomClass):
                return MyCustomClass(node.value * 3)
            if isinstance(node, torch.Tensor):
                return node + 1
            return node

        def fn(arg):
            return tree_map_impl(mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)

        # Run eager first to establish expected behavior
        visited_types.clear()
        fn(tree)
        eager_types = visited_types.copy()

        # Run compiled
        visited_types.clear()
        result = compiled(tree)

        # Verify the custom object was visited as a leaf
        self.assertIn(MyCustomClass, eager_types)

        # Verify results match
        self.assertIsInstance(result["custom"], MyCustomClass)
        self.assertTrue(torch.allclose(result["custom"].value, obj1.value * 3))
        self.assertTrue(torch.allclose(result["tensor"], torch.ones(2)))

    @parametrize("tree_map_name,tree_map_impl", TREE_MAP_IMPLEMENTATIONS)
    def test_user_defined_object_multiple_trees(
        self, tree_map_name: str, tree_map_impl
    ) -> None:
        """User-defined objects should work correctly with multiple input trees."""

        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        tree1 = {"point": Point(1, 2), "val": 10}
        tree2 = {"point": Point(3, 4), "val": 20}

        def mapper(a, b):
            if isinstance(a, Point) and isinstance(b, Point):
                return Point(a.x + b.x, a.y + b.y)
            return a + b

        def fn(t1, t2):
            return tree_map_impl(mapper, t1, t2)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        result = compiled(tree1, tree2)

        self.assertIsInstance(result["point"], Point)
        self.assertEqual(result["point"].x, 4)
        self.assertEqual(result["point"].y, 6)
        self.assertEqual(result["val"], 30)

    @parametrize("tree_map_name,tree_map_impl", TREE_MAP_IMPLEMENTATIONS)
    def test_dict_subclass_treated_as_leaf(
        self, tree_map_name: str, tree_map_impl
    ) -> None:
        """Dict subclasses (not registered in pytree) should be treated as leaves."""

        class MyDict(dict):
            def custom_method(self):
                return sum(self.values())

        my_dict = MyDict({"a": 1, "b": 2})
        tree = {"custom_dict": my_dict, "regular": {"x": 3}}

        def mapper(node):
            if isinstance(node, MyDict):
                result = MyDict()
                for k, v in node.items():
                    result[k] = v * 2
                return result
            return node

        def fn(arg):
            return tree_map_impl(mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        result = compiled(tree)

        # The MyDict should be treated as a leaf, not traversed into
        self.assertIsInstance(result["custom_dict"], MyDict)
        self.assertEqual(result["custom_dict"]["a"], 2)
        self.assertEqual(result["custom_dict"]["b"], 4)
        # Regular dict should still be traversed
        self.assertEqual(result["regular"]["x"], 3)

    @parametrize("tree_map_name,tree_map_impl", TREE_MAP_IMPLEMENTATIONS)
    def test_list_subclass_treated_as_leaf(
        self, tree_map_name: str, tree_map_impl
    ) -> None:
        """List subclasses (not registered in pytree) should be treated as leaves."""

        class MyList(list):
            def custom_sum(self):
                return sum(self)

        my_list = MyList([1, 2, 3])
        tree = {"custom_list": my_list, "regular": [4, 5, 6]}

        def mapper(node):
            if isinstance(node, MyList):
                return MyList([x * 2 for x in node])
            if isinstance(node, int):
                return node + 10
            return node

        def fn(arg):
            return tree_map_impl(mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        result = compiled(tree)

        # MyList should be treated as a leaf
        self.assertIsInstance(result["custom_list"], MyList)
        self.assertEqual(list(result["custom_list"]), [2, 4, 6])
        # Regular list should be traversed
        self.assertEqual(result["regular"], [14, 15, 16])

    @parametrize("tree_map_name,tree_map_impl", TREE_MAP_IMPLEMENTATIONS)
    def test_tuple_subclass_treated_as_leaf(
        self, tree_map_name: str, tree_map_impl
    ) -> None:
        """Tuple subclasses (not registered in pytree) should be treated as leaves."""

        class MyTuple(tuple):  # noqa: SLOT001
            pass

        my_tuple = MyTuple((1, 2, 3))
        tree = {"custom_tuple": my_tuple, "regular": (4, 5, 6)}

        def mapper(node):
            if isinstance(node, MyTuple):
                return MyTuple(tuple(x * 2 for x in node))
            if isinstance(node, int):
                return node + 10
            return node

        def fn(arg):
            return tree_map_impl(mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        result = compiled(tree)

        # MyTuple should be treated as a leaf
        self.assertIsInstance(result["custom_tuple"], MyTuple)
        self.assertEqual(tuple(result["custom_tuple"]), (2, 4, 6))
        # Regular tuple should be traversed
        self.assertEqual(result["regular"], (14, 15, 16))

    @parametrize("tree_map_name,tree_map_impl", TREE_MAP_IMPLEMENTATIONS)
    def test_user_defined_object_nested_in_containers(
        self, tree_map_name: str, tree_map_impl
    ) -> None:
        """User-defined objects nested inside containers should be leaves."""

        class Wrapper:
            def __init__(self, value):
                self.value = value

        tree = {
            "list_of_wrappers": [Wrapper(1), Wrapper(2)],
            "nested": {"wrapper": Wrapper(3)},
        }

        def mapper(node):
            if isinstance(node, Wrapper):
                return Wrapper(node.value * 10)
            return node

        def fn(arg):
            return tree_map_impl(mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        result = compiled(tree)

        self.assertEqual(result["list_of_wrappers"][0].value, 10)
        self.assertEqual(result["list_of_wrappers"][1].value, 20)
        self.assertEqual(result["nested"]["wrapper"].value, 30)

    @parametrize("tree_map_name,tree_map_impl", TREE_MAP_IMPLEMENTATIONS)
    def test_user_defined_object_with_is_leaf_predicate(
        self, tree_map_name: str, tree_map_impl
    ) -> None:
        """Test that is_leaf predicate interacts correctly with user-defined objects."""

        class Container:
            def __init__(self, items):
                self.items = items

        container = Container([1, 2, 3])
        tree = {"container": container, "list": [4, 5]}

        def is_leaf_fn(node):
            # Make lists be treated as leaves too
            return isinstance(node, (Container, list))

        def mapper(node):
            if isinstance(node, Container):
                return Container([x * 2 for x in node.items])
            if isinstance(node, list):
                return [x + 10 for x in node]
            return node

        def fn(arg):
            return tree_map_impl(mapper, arg, is_leaf=is_leaf_fn)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        result = compiled(tree)

        self.assertEqual(result["container"].items, [2, 4, 6])
        self.assertEqual(result["list"], [14, 15])

    def test_registered_custom_type_falls_back_pytree(self) -> None:
        """Custom types registered with pytree should fall back to tracing."""

        class RegisteredContainer:
            def __init__(self, items):
                self.items = list(items)

        # Register with pytree
        pytree.register_pytree_node(
            RegisteredContainer,
            lambda x: (x.items, None),
            lambda items, _: RegisteredContainer(items),
        )

        try:
            tree = RegisteredContainer([torch.ones(2), torch.zeros(2)])

            def mapper(node):
                if isinstance(node, torch.Tensor):
                    return node + 1
                return node

            def fn(arg):
                return pytree.tree_map(mapper, arg)

            compiled = torch.compile(fn, backend="eager", fullgraph=True)
            result = compiled(tree)

            # Since it's registered, the container should be traversed
            self.assertIsInstance(result, RegisteredContainer)
            self.assertTrue(torch.allclose(result.items[0], torch.ones(2) + 1))
            self.assertTrue(torch.allclose(result.items[1], torch.zeros(2) + 1))
        finally:
            # Clean up registration
            pytree._deregister_pytree_node(RegisteredContainer)

    def test_registered_custom_type_falls_back_optree(self) -> None:
        """Custom types registered with optree should fall back to tracing."""
        _require_optree(self)
        import optree.registry as optree_registry

        global_namespace = getattr(
            optree_registry, "_TreeMapCompileTests__GLOBAL_NAMESPACE", None
        )
        if global_namespace is None:
            # Handle Python name mangling - the attribute has double underscore prefix
            global_namespace = getattr(optree_registry, "__GLOBAL_NAMESPACE", None)
        if global_namespace is None:
            # Fallback: access via module internals
            for name in dir(optree_registry):
                if "GLOBAL_NAMESPACE" in name:
                    global_namespace = getattr(optree_registry, name)
                    break

        if global_namespace is None:
            self.skipTest("Could not find optree global namespace")

        class OptreeRegisteredContainer:
            def __init__(self, items):
                self.items = list(items)

        # Register with optree in global namespace
        # Note: optree unflatten_func signature is (metadata, children)
        optree.register_pytree_node(
            OptreeRegisteredContainer,
            lambda x: (x.items, None),
            lambda _, children: OptreeRegisteredContainer(children),
            namespace=global_namespace,
        )

        try:
            tree = OptreeRegisteredContainer([torch.ones(2), torch.zeros(2)])

            def mapper(node):
                if isinstance(node, torch.Tensor):
                    return node + 1
                return node

            def fn(arg):
                return optree.tree_map(mapper, arg)

            compiled = torch.compile(fn, backend="eager", fullgraph=True)
            result = compiled(tree)

            # Since it's registered, the container should be traversed
            self.assertIsInstance(result, OptreeRegisteredContainer)
            self.assertTrue(torch.allclose(result.items[0], torch.ones(2) + 1))
            self.assertTrue(torch.allclose(result.items[1], torch.zeros(2) + 1))
        finally:
            # Clean up registration
            optree.unregister_pytree_node(
                OptreeRegisteredContainer, namespace=global_namespace
            )

    def test_optree_namespaced_registration_with_namespace_arg(self) -> None:
        """Types registered with a namespace should be traversed when namespace is provided."""
        _require_optree(self)

        class NamespacedContainer:
            def __init__(self, items):
                self.items = list(items)

        # Register with a specific namespace
        optree.register_pytree_node(
            NamespacedContainer,
            lambda x: (x.items, None),
            lambda _, children: NamespacedContainer(children),
            namespace="test_namespace",
        )

        try:
            tree = NamespacedContainer([torch.ones(2), torch.zeros(2)])

            def mapper(node):
                if isinstance(node, torch.Tensor):
                    return node + 1
                return node

            def fn(arg):
                # Call with matching namespace - should traverse
                return optree.tree_map(mapper, arg, namespace="test_namespace")

            compiled = torch.compile(fn, backend="eager", fullgraph=True)
            result = compiled(tree)

            # Since namespace matches, the container should be traversed
            self.assertIsInstance(result, NamespacedContainer)
            self.assertTrue(torch.allclose(result.items[0], torch.ones(2) + 1))
            self.assertTrue(torch.allclose(result.items[1], torch.zeros(2) + 1))
        finally:
            optree.unregister_pytree_node(
                NamespacedContainer, namespace="test_namespace"
            )

    def test_optree_namespaced_registration_without_namespace_arg(self) -> None:
        """Types registered with a namespace should be leaves when no namespace is provided."""
        _require_optree(self)

        class NamespacedContainer:
            def __init__(self, items):
                self.items = list(items)

        # Register with a specific namespace
        optree.register_pytree_node(
            NamespacedContainer,
            lambda x: (x.items, None),
            lambda _, children: NamespacedContainer(children),
            namespace="test_namespace",
        )

        try:
            tree = NamespacedContainer([torch.ones(2), torch.zeros(2)])

            mapper_calls = []

            def mapper(node):
                mapper_calls.append(type(node))
                if isinstance(node, NamespacedContainer):
                    # Should be called as a leaf
                    return NamespacedContainer([t * 2 for t in node.items])
                if isinstance(node, torch.Tensor):
                    return node + 1
                return node

            def fn(arg):
                # Call WITHOUT namespace - container should be treated as leaf
                return optree.tree_map(mapper, arg)

            compiled = torch.compile(fn, backend="eager", fullgraph=True)

            # Clear and run
            mapper_calls.clear()
            result = compiled(tree)

            # Container should be treated as a leaf (mapper sees NamespacedContainer)
            self.assertIn(NamespacedContainer, mapper_calls)
            self.assertIsInstance(result, NamespacedContainer)
            # The container was mapped directly, so items are doubled tensors
            self.assertTrue(torch.allclose(result.items[0], torch.ones(2) * 2))
            self.assertTrue(torch.allclose(result.items[1], torch.zeros(2) * 2))
        finally:
            optree.unregister_pytree_node(
                NamespacedContainer, namespace="test_namespace"
            )

    def test_optree_namespace_mismatch(self) -> None:
        """Types registered with one namespace should be leaves when different namespace is used."""
        _require_optree(self)

        class NamespacedContainer:
            def __init__(self, items):
                self.items = list(items)

        # Register with a specific namespace
        optree.register_pytree_node(
            NamespacedContainer,
            lambda x: (x.items, None),
            lambda _, children: NamespacedContainer(children),
            namespace="namespace_a",
        )

        try:
            tree = NamespacedContainer([torch.ones(2), torch.zeros(2)])

            mapper_calls = []

            def mapper(node):
                mapper_calls.append(type(node))
                if isinstance(node, NamespacedContainer):
                    return NamespacedContainer([t * 2 for t in node.items])
                if isinstance(node, torch.Tensor):
                    return node + 1
                return node

            def fn(arg):
                # Call with DIFFERENT namespace - container should be leaf
                return optree.tree_map(mapper, arg, namespace="namespace_b")

            compiled = torch.compile(fn, backend="eager", fullgraph=True)

            mapper_calls.clear()
            result = compiled(tree)

            # Container should be treated as a leaf
            self.assertIn(NamespacedContainer, mapper_calls)
            self.assertIsInstance(result, NamespacedContainer)
            self.assertTrue(torch.allclose(result.items[0], torch.ones(2) * 2))
        finally:
            optree.unregister_pytree_node(NamespacedContainer, namespace="namespace_a")

    @parametrize("tree_map_name,tree_map_impl", TREE_MAP_IMPLEMENTATIONS)
    def test_dataclass_treated_as_leaf(self, tree_map_name: str, tree_map_impl) -> None:
        """Dataclasses should be treated as leaves (not registered by default)."""
        import dataclasses

        @dataclasses.dataclass
        class DataPoint:
            x: int
            y: int

        tree = {"point": DataPoint(1, 2), "val": 3}

        def mapper(node):
            if isinstance(node, DataPoint):
                return DataPoint(node.x * 2, node.y * 2)
            return node + 10

        def fn(arg):
            return tree_map_impl(mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        result = compiled(tree)

        self.assertIsInstance(result["point"], DataPoint)
        self.assertEqual(result["point"].x, 2)
        self.assertEqual(result["point"].y, 4)
        self.assertEqual(result["val"], 13)

    @parametrize("tree_map_name,tree_map_impl", TREE_MAP_IMPLEMENTATIONS)
    def test_user_defined_object_with_tensor_attribute(
        self, tree_map_name: str, tree_map_impl
    ) -> None:
        """User-defined objects containing tensors should be treated as leaves."""

        class TensorWrapper:
            def __init__(self, tensor):
                self.tensor = tensor

        wrapper = TensorWrapper(torch.ones(2, 2))
        tree = {"wrapper": wrapper, "direct_tensor": torch.zeros(2)}

        def mapper(node):
            if isinstance(node, TensorWrapper):
                return TensorWrapper(node.tensor * 2)
            if isinstance(node, torch.Tensor):
                return node + 1
            return node

        def fn(arg):
            return tree_map_impl(mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        result = compiled(tree)

        self.assertIsInstance(result["wrapper"], TensorWrapper)
        self.assertTrue(torch.allclose(result["wrapper"].tensor, torch.ones(2, 2) * 2))
        self.assertTrue(torch.allclose(result["direct_tensor"], torch.ones(2)))

    @parametrize("tree_map_name,tree_map_impl", TREE_MAP_IMPLEMENTATIONS)
    def test_user_defined_object_no_fallback(
        self, tree_map_name: str, tree_map_impl
    ) -> None:
        """Verify user-defined objects use fastpath without triggering fallback."""
        import logging

        class SimpleObj:
            def __init__(self, x):
                self.x = x

        tree = {"obj": SimpleObj(42), "val": 10}

        def mapper(node):
            if isinstance(node, SimpleObj):
                return SimpleObj(node.x * 2)
            return node + 1

        def fn(arg):
            return tree_map_impl(mapper, arg)

        # Capture debug logs to ensure no fallback is triggered
        log_records = []
        handler = logging.Handler()
        handler.emit = lambda record: log_records.append(record)
        logger = logging.getLogger("torch._dynamo.variables.base")
        old_level = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        try:
            compiled = torch.compile(fn, backend="eager", fullgraph=True)
            result = compiled(tree)

            # Verify results
            self.assertIsInstance(result["obj"], SimpleObj)
            self.assertEqual(result["obj"].x, 84)
            self.assertEqual(result["val"], 11)

            # Verify no fallback was triggered for UserDefinedObjectVariable
            fallback_messages = [
                r.getMessage()
                for r in log_records
                if "tree_map fastpath fallback triggered" in r.getMessage()
                and "UserDefinedObjectVariable" in r.getMessage()
            ]
            self.assertEqual(
                len(fallback_messages),
                0,
                f"Fallback was triggered unexpectedly: {fallback_messages}",
            )
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)


if __name__ == "__main__":  # pragma: no cover
    run_tests()
