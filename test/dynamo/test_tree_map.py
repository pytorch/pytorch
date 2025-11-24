# Owner(s): ["module: dynamo"]

import optree

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


TREE_MAP_IMPLEMENTATIONS = [
    ("optree", optree.tree_map),
    ("pytree_python", pytree.tree_map),
]
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


if __name__ == "__main__":  # pragma: no cover
    run_tests()
