# mypy: allow-untyped-defs
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Union

import torch
import torch.utils._pytree as pytree
from torch._functorch import config
from torch.fx.experimental.proxy_tensor import py_sym_types, SymBool, SymFloat, SymInt


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
        if config.append_backward:
            self.sym_hash_dict = defaultdict(list)
        else:
            self.sym_hash_dict = {}

    def __setitem__(self, key, value):
        self.sym_hash_dict.__setitem__(self._wrap_to_sym_expr_hash(key), value)

    def __getitem__(self, key):
        return self.sym_hash_dict[self._wrap_to_sym_expr_hash(key)]

    def __contains__(self, key):
        return self._wrap_to_sym_expr_hash(key) in self.sym_hash_dict

    def __str__(self):
        return f"_SymHashingDict={self.sym_hash_dict}"

    def __repr__(self):
        return f"_SymHashingDict={self.sym_hash_dict}"

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

    if config.append_backward:
        # The graph can contain duplicated symbolic integers, for example:
        # "return [mul, mul_1, sym_1, sym_2, mul_2, mul_3, None, None, sym_1, sym_2]".
        # If we simply remove duplicates like "sym_1", it may cause the forward or
        # backward graph to reference nonexistent arguments. To prevent such issues,
        # we disambiguate these duplicates by assigning unique names to each instance.

        def compute_nodes_in_forward_graph(forward_inputs, forward_outputs):
            queue = [*forward_inputs, *forward_outputs]
            s = set()
            while len(queue) > 0:
                last = queue.pop()
                if last in s:
                    continue
                s.add(last)
                for node in last.all_input_nodes:
                    if node not in s:
                        queue.append(node)
            return s

        [output] = graph.find_nodes(op="output")
        new_args = list(output.args[0])
        for node, cnt in Counter(output.args[0]).items():
            if node and cnt > 1:
                # copy and replace one occurrence on list
                with graph.inserting_after(node):
                    copy = graph.node_copy(node)
                    new_args[new_args.index(node)] = copy

        output.args = (new_args,)
        # print(graph)

        if hasattr(graph._codegen, "pytree_info"):
            # instance of _PyTreeCodegen
            forward_inputs, _ = pytree.tree_unflatten(
                graph.find_nodes(op="placeholder"), graph._codegen.pytree_info.in_spec
            )
            forward_outputs, _ = pytree.tree_unflatten(
                new_args, graph._codegen.pytree_info.out_spec
            )
            forward_nodes = compute_nodes_in_forward_graph(
                forward_inputs, forward_outputs
            )
        else:
            forward_nodes = ()

        for node in graph.nodes:
            val = node.meta.get("val", None)
            if val is None or not isinstance(val, py_sym_types):
                continue

            # print(f"[{node.op}] {node} - {val}")
            sym_dict_vals = sym_dict.get(val, [])
            existing_nodes = (
                [n for n in sym_dict_vals if n in forward_nodes]
                if node in forward_nodes
                else sym_dict_vals
            )

            if node.op == "placeholder":
                resolvable_from_input_symints.add(node)
                # sym_dict[val] = node
                sym_dict[val].append(node)
            elif len(sym_dict_vals) > 0 and len(existing_nodes) > 0:
                # skip if node in forward_nodes but no replacement found in forward_nodes
                node.replace_all_uses_with(existing_nodes[0])
                graph.erase_node(node)
            # elif existing_node := sym_dict.get(val):
            #     node.replace_all_uses_with(existing_node)
            #     graph.erase_node(node)
            elif all(n in resolvable_from_input_symints for n in node.all_input_nodes):
                # sym_dict[val] = node
                sym_dict[val].append(node)
                resolvable_from_input_symints.add(node)

        # print(graph)

    else:
        # print(graph)

        for node in graph.nodes:
            val = node.meta.get("val", None)
            if val is None or not isinstance(val, py_sym_types):
                continue

            # print(f"[{node.op}] {node} - {val}")

            if node.op == "placeholder":
                resolvable_from_input_symints.add(node)
                sym_dict[val] = node
            elif existing_node := sym_dict.get(val):
                node.replace_all_uses_with(existing_node)
                graph.erase_node(node)
            elif all(n in resolvable_from_input_symints for n in node.all_input_nodes):
                sym_dict[val] = node
                resolvable_from_input_symints.add(node)

        # print(graph)
