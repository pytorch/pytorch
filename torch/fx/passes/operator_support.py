from typing import Dict, Any

import torch
import torch.fx
from torch.fx.node import _get_qualified_name

from .tools_common import CALLABLE_NODE_OPS


def get_node_target(submodules: Dict[str, torch.nn.Module], node: torch.fx.Node) -> str:
    """
    Given a `node` returns its target typename.

    For "call_method" node, return node.target which is the name of that method being called.
    This could potential lead to conflict but should be okay because normally it's on a tensor.

    For "call_function" node, return typename of node.target.

    For "call_module" node, return typename of the module that node.target point to.

    If seeing "_VariableFunctionsClass" in the target name string, it will be replaced by
    "torch". e.g. _VariableFunctionsClass.relu would become torch.relu.
    """

    assert node.op in CALLABLE_NODE_OPS, (
        "Expect op types of " + ", ".join(CALLABLE_NODE_OPS) + f", but found {node.op}"
    )

    if node.op == "call_module":
        assert isinstance(node.target, str)
        return torch.typename(submodules[node.target])
    elif node.op == "call_function":
        target: Any = node.target
        return (
            f"acc_ops.{target.__name__}"
            if target.__module__ == "glow.fb.fx.acc_ops"
            else _get_qualified_name(target)
        )
    else:
        assert isinstance(node.target, str)
        return node.target


class OperatorSupport:
    """
    `_support_dict` maps node.target to supported inputs dtypes.

    node.target is retrived using helper function `get_node_target()`

    If supported inputs dtypes is None, it means any dtype is supported, else
    we should see a tuple like (([dtypes], ...), {"name":[dtypes], ...}).

    The first tuple ([dtypes], ...) indicates what dtypes are supported for
    inputs in node.args and the second dict {"name": [dtypes], ...} indicates
    what dtypes are supported for inputs in node.kwargs.

    For inputs in args, if we don't want to check it, we can put None there,
    e.g. (None, [torch.float]) indicates that we don't care about the type of
    the first input in args. And for inputs in kwargs, if not listed, will not
    be checked.
    """

    _support_dict: Dict = {}

    def is_node_supported(
        self, submodules: Dict[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        """
        Args:
            `sumodules`: mapping from module name to the module. This can be
                         retrieved by calling model.named_modules().

            `node`: a Fx node that we want to determine whether it's supported.

        Returns:
            `is_supported`: whether the arg `node` is supported.
        """
        if node.op not in CALLABLE_NODE_OPS:
            return True

        target = get_node_target(submodules, node)

        # Target not found in _support_dict meaning that we don't support this op at all
        if target not in self._support_dict:
            return False

        # The rule for target is None meaning that we accept any dtype
        if self._support_dict[target] is None:
            return True

        args_dtypes, kwargs_dtypes = self._support_dict[target]

        # Check args dtypes
        for i, dtypes in enumerate(args_dtypes):
            if len(node.args) <= i:
                break

            # None indicates we don't care about the dtype of args[i]
            if dtypes is None:
                continue

            # If arg is not a node then we don't check it
            if not isinstance(node.args[i], torch.fx.Node):
                continue

            arg_tensor_meta = node.args[i].meta.get("tensor_meta")  # type: ignore[union-attr]
            arg_dtype = arg_tensor_meta.dtype if arg_tensor_meta else None

            if arg_dtype not in dtypes:
                return False

        # Check kwargs dtypes
        for k, dtypes in kwargs_dtypes.items():
            if k not in node.kwargs:
                continue

            # If arg is not a node then we don't check it
            if not isinstance(node.kwargs[k], torch.fx.Node):
                continue

            kwarg_tensor_meta = node.kwargs[k].meta.get("tensor_meta")  # type: ignore[union-attr]
            kwarg_dtype = kwarg_tensor_meta.dtype if kwarg_tensor_meta else None

            if kwarg_dtype not in dtypes:
                return False

        return True
