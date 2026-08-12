# mypy: allow-untyped-defs
from dataclasses import dataclass
from typing import Any

import torch
from torch import SymBool, SymFloat, SymInt
from torch.types import py_sym_types
from torch.utils._ordered_set import OrderedSet


_ASSERT_TARGETS = (
    torch._check,
    torch._assert_scalar,
    torch.ops.aten._assert_scalar.default,
)


@dataclass
class _SymExprHash:
    """
    Hash for a py_sym_types that will use the underlying sympy expression
    """

    sym_obj: SymInt | SymFloat | SymBool

    def __hash__(self) -> int:
        return hash((type(self.sym_obj), self.sym_obj.node.expr))

    def __eq__(self, value) -> bool:
        if not isinstance(value, _SymExprHash):
            return False
        return self.sym_obj.node.expr == value.sym_obj.node.expr


class _SymHashingDict:
    """
    Wrapper around a dictionary that will convert sym types to hash with _SymExprHash and reuse
    existing sym proxies.

    SymPy hash is not always reliable so optimistically hash sympy expression, and if those fail,
    fallback to symnodes.
    """

    def __init__(self):
        self.sym_hash_dict = {}

    def __setitem__(self, key, value):
        self.sym_hash_dict.__setitem__(self._wrap_to_sym_expr_hash(key), value)

    def __getitem__(self, key):
        return self.sym_hash_dict[self._wrap_to_sym_expr_hash(key)]

    def __contains__(self, key):
        return self._wrap_to_sym_expr_hash(key) in self.sym_hash_dict

    def get(self, key, default=None):
        return self.sym_hash_dict.get(self._wrap_to_sym_expr_hash(key), default)

    def _wrap_to_sym_expr_hash(self, key):
        return _SymExprHash(key) if isinstance(key, py_sym_types) else key


def _assertion_condition(node: torch.fx.Node) -> Any:
    if node.args:
        return node.args[0]
    return node.kwargs.get("cond", node.kwargs.get("self"))


def _runtime_assert_condition_nodes(graph: torch.fx.Graph) -> OrderedSet[torch.fx.Node]:
    protected: OrderedSet[torch.fx.Node] = OrderedSet()

    def add_ancestors(node: torch.fx.Node) -> None:
        pending = [node]
        while pending:
            current = pending.pop()
            if current in protected:
                continue
            protected.add(current)
            pending.extend(current.all_input_nodes)

    for node in graph.nodes:
        if node.target not in _ASSERT_TARGETS:
            continue
        condition = _assertion_condition(node)
        if isinstance(condition, torch.fx.Node):
            add_ancestors(condition)

    return protected


def dedupe_symints(graph: torch.fx.Graph):
    """
    Dedupes sym ints in the graph to nodes are resolvable to symint graph inputs.

    We only dedupe from graph inputs to avoid adding a potential dependency in the forward
    from the backward.

    """

    sym_dict = _SymHashingDict()
    resolvable_from_input_symints = OrderedSet[Any]()
    runtime_assert_condition_nodes = _runtime_assert_condition_nodes(graph)

    for node in graph.nodes:
        val = node.meta.get("val", None)
        if val is None or not isinstance(val, py_sym_types):
            continue

        if node.op == "placeholder":
            resolvable_from_input_symints.add(node)
            sym_dict[val] = node
        elif existing_node := sym_dict.get(val):
            if node in runtime_assert_condition_nodes:
                # Preserve the runtime provenance used by the assertion. The
                # existing canonical node proves this symbolic value is
                # resolvable, so descendants can still participate in CSE.
                resolvable_from_input_symints.add(node)
            else:
                node.replace_all_uses_with(existing_node)
                graph.erase_node(node)
        elif all(n in resolvable_from_input_symints for n in node.all_input_nodes):
            sym_dict[val] = node
            resolvable_from_input_symints.add(node)
