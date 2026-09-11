"""A parsed access path into a call's arguments: a root parameter name plus a
sequence of item/attr steps. Used to point mutates_args at specific tensor
leaves, e.g. "state['buf']" or "buffers[0]" or "box.tensor"."""

import ast
from typing import Any


class ArgPath:
    def __init__(self, source: str, root: str, steps: tuple[tuple[str, Any], ...]):
        self.source = source
        self.root = root
        self.steps = steps

    def resolve(self, values: dict[str, Any]) -> Any:
        if self.root not in values:
            raise ValueError(
                f"mutation path {self.source!r} refers to a missing argument"
            )
        value = values[self.root]
        for kind, key in self.steps:
            value = value[key] if kind == "getitem" else getattr(value, key)
        return value


def parse_arg_path(path: str) -> ArgPath:
    """Parse an access-path string into an ArgPath (root + getitem/getattr steps).

    Examples (path -> root, steps):
        "x"               -> "x",       ()
        "buffers[0]"      -> "buffers", (("getitem", 0),)
        "state['buf']"    -> "state",   (("getitem", "buf"),)
        "box.tensor"      -> "box",     (("getattr", "tensor"),)
        "state['buf'][0]" -> "state",   (("getitem", "buf"), ("getitem", 0))

    Only a name followed by constant subscripts / attribute accesses is allowed;
    anything else (calls, slices, variable keys) raises ValueError.
    """

    def parse_node(node: ast.AST) -> tuple[str, list[tuple[str, Any]]]:
        if isinstance(node, ast.Name):
            return node.id, []
        if isinstance(node, ast.Attribute):
            root, steps = parse_node(node.value)
            steps.append(("getattr", node.attr))
            return root, steps
        if isinstance(node, ast.Subscript):
            root, steps = parse_node(node.value)
            index = node.slice
            if isinstance(index, ast.Constant) and isinstance(index.value, (int, str)):
                steps.append(("getitem", index.value))
                return root, steps
        raise ValueError(f"unsupported mutation path: {path!r}")

    expr = ast.parse(path, mode="eval").body
    root, steps = parse_node(expr)
    return ArgPath(path, root, tuple(steps))
