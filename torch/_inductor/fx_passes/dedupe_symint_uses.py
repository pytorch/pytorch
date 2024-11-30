# mypy: allow-untyped-defs
from dataclasses import dataclass
from typing import Union

import torch
from torch import SymBool, SymFloat, SymInt
from torch.types import py_sym_types


@dataclass
class _SymExprHash:
    """
    Hash for a py_sym_types that will use the underlying sympy expression
    """

    sym_obj: Union[SymInt, SymFloat, SymBool]

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


def dedupe_symints(graph: torch.fx.Graph):
    """
    Dedupes sym ints in the graph to nodes are resolvable to symint graph inputs.

    We only dedupe from graph inputs to avoid adding a potential dependency in the forward
    from the backward.

    """

    sym_dict = _SymHashingDict()
    resolvable_from_input_symints = set()

    for node in graph.nodes:
        val = node.meta.get("val", None)
        if val is None or not isinstance(val, py_sym_types):
            continue

        if node.op == "placeholder":
            resolvable_from_input_symints.add(node)
            sym_dict[val] = node
        elif existing_node := sym_dict.get(val):
            node.replace_all_uses_with(existing_node)
            graph.erase_node(node)
        elif all(n in resolvable_from_input_symints for n in node.all_input_nodes):
            sym_dict[val] = node
            resolvable_from_input_symints.add(node)
