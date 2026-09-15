# ruff: noqa: S101

import inspect
import operator
from types import SimpleNamespace

import torch
import torch.utils._pytree as pytree
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.node import Node


class FxPattern:
    def __init__(self, fn, graph_module, names, arg_names, return_names):
        self.fn = fn
        self.graph_module = graph_module
        self.names = names
        self.arg_names = arg_names
        self.return_names = return_names


class FxReplacement:
    def __init__(self, fn, graph_module, arg_names):
        self.fn = fn
        self.graph_module = graph_module
        self.arg_names = arg_names


class FxPatternCaptures:
    def __init__(self):
        self.names = []

    def __setattr__(self, name, value):
        if name != "names":
            self.names.append(name)
        object.__setattr__(self, name, value)


class FxGraphEditor:
    def __init__(self, gm: GraphModule):
        self.gm = gm
        self.graph = gm.graph

    def before(self, node: Node):
        return self.graph.inserting_before(node)

    def after(self, node: Node):
        return self.graph.inserting_after(node)

    def emit(self, target, *args, unpack=True):
        target = self._resolve_target(target, args)
        example_inputs = tuple(self._example_input(arg) for arg in args)
        graph_module = make_fx(lambda *xs: target(*xs), tracing_mode="real")(*example_inputs)
        placeholders = [n for n in graph_module.graph.nodes if n.op == "placeholder"]
        val_map = dict(zip(placeholders, _flatten_args(args), strict=True))
        result = self.graph.graph_copy(graph_module.graph, val_map)
        output_value = _graph_output_value(graph_module)
        if isinstance(result, list):
            result = tuple(result)
        if _is_single_node_output(output_value, result):
            result = next(iter(result))
        if (
            not unpack
            and isinstance(result, tuple)
            and all(_is_getitem(n, result[0].args[0], i) for i, n in enumerate(result))
        ):
            producer = result[0].args[0]
            for node in reversed(result):
                self.graph.erase_node(node)
            return producer
        return result

    def replace_uses_between(
        self,
        *,
        old: Node,
        new: Node,
        start: Node | None = None,
        end: Node | None = None,
    ) -> None:
        in_range = start is None
        nodes = set()
        for node in self.graph.nodes:
            if node is start:
                in_range = True
            if in_range:
                nodes.add(node)
            if node is end:
                break

        for user in list(old.users):
            if user in nodes:
                user.replace_input_with(old, new)

    @staticmethod
    def _resolve_target(target, args):
        if isinstance(target, torch._ops.OpOverload):
            return target
        if target is torch.ops.aten.size:
            return target.int if len(args) == 2 else target.default
        if isinstance(target, torch._ops.OpOverloadPacket):
            overloads = target.overloads()
            if overloads == ["default"]:
                return target.default
            raise RuntimeError(f"ambiguous operator overload {target}: {overloads}")
        return target

    @staticmethod
    def _example_input(arg):
        if isinstance(arg, tuple):
            return tuple(FxGraphEditor._example_input(item) for item in arg)
        if isinstance(arg, list):
            return [FxGraphEditor._example_input(item) for item in arg]
        if isinstance(arg, dict):
            return {key: FxGraphEditor._example_input(value) for key, value in arg.items()}
        if not isinstance(arg, Node):
            return arg

        val = arg.meta.get("val")
        if val is not None:
            if hasattr(val, "shape") and hasattr(val, "dtype"):
                return torch.empty(tuple(val.shape), dtype=val.dtype)
            return val

        tensor_meta = arg.meta.get("tensor_meta")
        if tensor_meta is not None:
            return torch.empty(tensor_meta.shape, dtype=tensor_meta.dtype)

        raise RuntimeError(f"missing example input metadata for {arg}")


def fx_pattern(*example_inputs):
    def decorator(fn):
        captures = FxPatternCaptures()
        graph_module = make_fx(lambda *args: fn(captures, *args), tracing_mode="real")(
            *example_inputs
        )
        return FxPattern(
            fn,
            graph_module,
            captures.names,
            list(inspect.signature(fn).parameters)[1:],
            _pattern_return_names(graph_module, captures.names),
        )

    return decorator


def fx_replacement(*example_inputs):
    def decorator(fn):
        graph_module = make_fx(fn, tracing_mode="real")(*example_inputs)
        return FxReplacement(fn, graph_module, list(inspect.signature(fn).parameters))

    return decorator


def find_pattern_matches(gm, pattern):
    pattern_gm = pattern.graph_module
    pattern_names = pattern.names
    arg_names = pattern.arg_names
    placeholders = [
        node for node in pattern_gm.graph.nodes if node.op == "placeholder"
    ]
    placeholder_names = dict(zip(placeholders, arg_names, strict=True))
    pattern_call_nodes = _pattern_call_nodes(pattern_gm)
    if len(pattern_names) > len(pattern_call_nodes):
        raise RuntimeError("expected no more pattern captures than pattern calls")
    pattern_calls = pattern_call_nodes[: len(pattern_names)]
    pattern_call_names = {
        node: name
        for node, name in zip(pattern_calls, pattern_names, strict=True)
    }

    def match_value(pattern_value, graph_value, nodes):
        if isinstance(pattern_value, Node):
            if pattern_value.op == "placeholder":
                name = placeholder_names[pattern_value]
                existing = nodes.get(name)
                if existing is not None:
                    return existing is graph_value
                if not isinstance(graph_value, Node):
                    return False
                nodes[name] = graph_value
                return True
            existing = nodes.get(pattern_value.name)
            return existing is graph_value

        if isinstance(pattern_value, tuple):
            return (
                isinstance(graph_value, tuple)
                and len(pattern_value) == len(graph_value)
                and all(
                    match_value(pattern_item, graph_item, nodes)
                    for pattern_item, graph_item in zip(pattern_value, graph_value, strict=True)
                )
            )

        if isinstance(pattern_value, list):
            return (
                isinstance(graph_value, list)
                and len(pattern_value) == len(graph_value)
                and all(
                    match_value(pattern_item, graph_item, nodes)
                    for pattern_item, graph_item in zip(pattern_value, graph_value, strict=True)
                )
            )

        if isinstance(pattern_value, dict):
            return (
                isinstance(graph_value, dict)
                and pattern_value.keys() == graph_value.keys()
                and all(
                    match_value(pattern_value[key], graph_value[key], nodes)
                    for key in pattern_value
                )
            )

        return pattern_value == graph_value

    matches = []

    def search(pattern_idx, nodes):
        if pattern_idx == len(pattern_calls):
            matches.append(SimpleNamespace(**nodes))
            return

        pattern_node = pattern_calls[pattern_idx]
        for graph_node in gm.graph.nodes:
            if graph_node.op != pattern_node.op or graph_node.target != pattern_node.target:
                continue
            if len(graph_node.args) != len(pattern_node.args):
                continue
            if pattern_node.kwargs and not pattern_node.kwargs.keys() <= graph_node.kwargs.keys():
                continue

            next_nodes = dict(nodes)
            next_nodes[pattern_node.name] = graph_node
            next_nodes[pattern_call_names[pattern_node]] = graph_node
            args_match = all(
                match_value(pattern_arg, graph_arg, next_nodes)
                for pattern_arg, graph_arg in zip(pattern_node.args, graph_node.args, strict=True)
            )
            kwargs_match = all(
                match_value(pattern_node.kwargs[key], graph_node.kwargs[key], next_nodes)
                for key in pattern_node.kwargs
            )
            if args_match and kwargs_match:
                search(pattern_idx + 1, next_nodes)

    search(0, {})
    return matches


def replace_pattern(gm, pattern, replacement):
    editor = FxGraphEditor(gm)
    for m in find_pattern_matches(gm, pattern):
        old_values = tuple(getattr(m, name) for name in pattern.return_names)
        matched_nodes = {value for value in vars(m).values() if isinstance(value, Node)}
        if isinstance(replacement, FxReplacement):
            placeholders = [
                node
                for node in replacement.graph_module.graph.nodes
                if node.op == "placeholder"
            ]
            val_map = {
                placeholder: getattr(m, name)
                for placeholder, name in zip(placeholders, replacement.arg_names, strict=True)
            }
            with editor.before(_first_external_user(gm, old_values, matched_nodes)):
                new_values = gm.graph.graph_copy(replacement.graph_module.graph, val_map)
            output_value = _graph_output_value(replacement.graph_module)
        else:
            new_values = replacement(editor, m)
            output_value = None
        if isinstance(new_values, list):
            new_values = tuple(new_values)
        if _is_single_node_output(output_value, new_values):
            new_values = next(iter(new_values))
        if not isinstance(new_values, tuple):
            new_values = (new_values,)
        for old, new in zip(old_values, new_values, strict=True):
            for user in list(old.users):
                if user not in matched_nodes:
                    user.replace_input_with(old, new)
    return gm


def _first_external_user(gm, old_values, matched_nodes):
    external_users = {
        user
        for old_value in old_values
        for user in old_value.users
        if user not in matched_nodes
    }
    return next(
        node
        for node in gm.graph.nodes
        if node.op == "output" or node in external_users
    )


def _is_getitem(node: Node, source: Node, idx: int) -> bool:
    return (
        node.op == "call_function"
        and node.target is operator.getitem
        and node.args == (source, idx)
    )


def _flatten_args(value):
    yield from pytree.tree_leaves(value)


def _graph_output_value(gm):
    output = next(node for node in reversed(gm.graph.nodes) if node.op == "output")
    return output.args[0]


def _is_single_node_output(output_value, value):
    return (
        (isinstance(output_value, Node) or _single_node_container(output_value))
        and not isinstance(value, Node)
        and hasattr(value, "__len__")
        and len(value) == 1
    )


def _single_node_container(value):
    return (
        not isinstance(value, Node)
        and hasattr(value, "__len__")
        and len(value) == 1
        and isinstance(next(iter(value)), Node)
    )


def _pattern_call_nodes(pattern_gm):
    return [
        node
        for node in pattern_gm.graph.nodes
        if node.op == "call_function"
        and not (node.target is operator.getitem and len(node.users) == 0)
    ]


def _pattern_return_names(pattern_gm, pattern_names):
    pattern_calls = _pattern_call_nodes(pattern_gm)[: len(pattern_names)]
    node_to_name = dict(zip(pattern_calls, pattern_names, strict=True))
    output = next(node for node in reversed(pattern_gm.graph.nodes) if node.op == "output")
    return [
        node_to_name[node]
        for node in _flatten_args(output.args[0])
        if isinstance(node, Node) and node in node_to_name
    ]
