import itertools
import operator
from collections.abc import Iterable
from typing import Set

import torch
from functorch.experimental import control_flow
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import GraphModule
from torch.fx._compatibility import compatibility


PRESERVED_META_KEYS: Set[str] = {
    "val",
    "stack_trace",
}


@compatibility(is_backward_compatible=False)
class SpecViolationError(Exception):
    pass


@compatibility(is_backward_compatible=False)
def is_functional(op: OpOverload) -> bool:
    return not op._schema.is_mutable


@compatibility(is_backward_compatible=False)
def _check_has_fake_tensor(node: torch.fx.Node) -> None:
    def _check_is_fake_tensor(val):
        if isinstance(val, FakeTensor):
            return True
        if isinstance(val, Iterable):
            return all(_check_is_fake_tensor(x) for x in val)
        return False

    val = node.meta.get("val", None)
    if val is None or not _check_is_fake_tensor(val):
        raise SpecViolationError("Node.meta {} is missing val field.".format(node.name))


@compatibility(is_backward_compatible=False)
def check_valid(gm: GraphModule) -> None:  # noqa: C901

    for node in gm.graph.nodes:
        # TODO(T140410192): should have fake tensor for all dialects
        if node.op in {"call_module", "call_method"}:
            raise SpecViolationError(
                "call_module is not valid: got a class '{}' ".format(node.target),
            )

        if node.op == "call_function":
            _check_has_fake_tensor(node)
            op_name = (
                node.target.name
                if hasattr(node.target, "name")
                else node.target.__name__
            )
            is_builtin_func = node.target in [
                'while_loop',
                operator.getitem,
                'cond',
                control_flow.cond,
                control_flow.map,
            ]
            if not isinstance(node.target, OpOverload) and not is_builtin_func:
                raise SpecViolationError(
                    "Operator '{}' is not a registered Op".format(op_name),
                )

            # All ops functional
            if not is_builtin_func and not is_functional(node.target):
                raise SpecViolationError(
                    f"operator '{op_name}' is not functional"
                )

            if isinstance(node.target, OpOverload):
                # Check preserved metadata
                for meta in PRESERVED_META_KEYS:
                    if node.meta.get(meta, None) is None:
                        raise SpecViolationError(
                            f"node {node} is missing metadata {meta}"
                        )


@compatibility(is_backward_compatible=False)
def is_valid(gm: GraphModule) -> bool:
    try:
        check_valid(gm)
        return True
    except SpecViolationError:
        return False


@compatibility(is_backward_compatible=False)
def _check_tensors_are_contiguous(gm: GraphModule) -> None:
    # Tensors be of contiguous format
    for name, param in itertools.chain(gm.named_parameters(), gm.named_buffers()):
        if isinstance(param, torch.Tensor):
            if not param.is_contiguous():
                raise SpecViolationError(
                    f"Tensors in Aten dialect must be contiguous, {name} is not contiguous"
                )


@compatibility(is_backward_compatible=False)
def check_valid_aten_dialect(gm: GraphModule) -> None:
    """Raises exception if gm is not in aten dialect.

    Args:
        gm: GraphModule
    """
    # need to be first valid
    check_valid(gm)

    # Operators be aten cannonical
    for n in gm.graph.nodes:
        if n.op == "call_function" and isinstance(n.target, OpOverload):
            if (
                torch.Tag.core not in n.target.tags  # type: ignore[attr-defined]
                and torch.Tag.view_copy not in n.target.tags  # type: ignore[attr-defined]
            ):
                # NOTE(qihan): whether view_copy operators are marked as canonical is still under
                #            discussion.
                raise SpecViolationError(
                    "Operator {}.{} is not Aten Canonical.".format(
                        n.target.__module__, n.target.__name__
                    )
                )

    _check_tensors_are_contiguous(gm)


@compatibility(is_backward_compatible=False)
def is_valid_aten_dialect(gm: GraphModule) -> bool:
    try:
        check_valid_aten_dialect(gm)
        return True
    except SpecViolationError:
        return False
