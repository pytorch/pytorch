import itertools
import operator
from collections.abc import Iterable

import torch
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import GraphModule


ALLOWED_META_KEYS = {"spec", "stack_trace"}

@torch.fx._compatibility.compatibility(is_backward_compatible=False)
class SpecViolationError(Exception):
    pass

@torch.fx._compatibility.compatibility(is_backward_compatible=False)
def is_functional(op: OpOverload) -> bool:
    return not op._schema.is_mutable


@torch.fx._compatibility.compatibility(is_backward_compatible=False)
def _check_has_fake_tensor(node: torch.fx.Node) -> None:
    def _check_is_fake_tensor(val):
        if isinstance(val, FakeTensor):
            return True
        if isinstance(val, Iterable):
            return all(_check_is_fake_tensor(x) for x in val)
        return False

    val = node.meta.get("val")
    if not _check_is_fake_tensor(val):
        raise SpecViolationError("Node.meta {} is missing val field.".format(node.name))


@torch.fx._compatibility.compatibility(is_backward_compatible=False)
def check_valid(gm: GraphModule) -> None:  # noqa: C901

    for node in gm.graph.nodes:
        # TODO(T140410192): should have fake tensor for all dialects
        if node.op == "call_method":
            # what is delegates in ATen dialect?
            raise SpecViolationError(
                "call_module can only be used for delegates, got a object of class '{}.{}' instead".format(
                    type(node.args[0]).__module__, type(node.args[0]).__name__
                ),
            )

        if node.op == "call_module":
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
            is_builtin_func = (node.target == operator.getitem or node.target.__name__ in [
                'while_loop',
                'cond',
            ])
            if not isinstance(node.target, OpOverload) and not is_builtin_func:
                raise SpecViolationError(
                    "Operator '{}' is not a registered Op".format(op_name),
                )
            # All ops functional
            # TODO(qihan): use node.target.is_functional: when PR/83134 lands
            if not is_builtin_func and not is_functional(node.target):
                raise SpecViolationError(
                    "operator '{}' is not functional".format(op_name),
                )

            if isinstance(node.target, OpOverload):
                stacktrace = node.meta.get("stack_trace")

                if stacktrace is None:
                    raise SpecViolationError(
                        "node of name '{}' for operator '{}' is missing stackstrace".format(
                            node.name, op_name
                        ),
                    )


@torch.fx._compatibility.compatibility(is_backward_compatible=False)
def is_valid(gm: GraphModule) -> bool:
    try:
        check_valid(gm)
        return True
    except SpecViolationError:
        return False


@torch.fx._compatibility.compatibility(is_backward_compatible=False)
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

    # Tensors be of contiguous format
    for name, param in itertools.chain(gm.named_parameters(), gm.named_buffers()):
        if isinstance(param, torch.Tensor):
            if not param.is_contiguous():
                raise SpecViolationError(
                    f"Tensors in Aten dialect must be contiguous, {name} is not contiguous"
                )


@torch.fx._compatibility.compatibility(is_backward_compatible=False)
def is_valid_aten_dialect(gm: GraphModule) -> bool:
    try:
        check_valid_aten_dialect(gm)
        return True
    except SpecViolationError:
        return False


@torch.fx._compatibility.compatibility(is_backward_compatible=False)
def check_valid_edge_dialect(gm: GraphModule) -> None:
    check_valid_aten_dialect(gm)

    # Additionally, edge dialect's operator must have same input dtype
    for n in gm.graph.nodes:
        if n.op == "call_function" and isinstance(n.target, OpOverload):
            _check_has_fake_tensor(n)
            dtypes = set()
            for arg in n.args:
                if isinstance(arg, torch.Tensor):
                    dtypes.add(arg.dtype)
                if isinstance(arg, torch.fx.Node):
                    dtypes.add(arg.meta["val"].dtype)
            if len(dtypes) > 1:
                raise SpecViolationError(
                    "Operators of Edge dialect in should work on tensors of same dtype"
                )


@torch.fx._compatibility.compatibility(is_backward_compatible=False)
def is_valid_edge_dialect(gm: GraphModule) -> bool:
    try:
        check_valid_edge_dialect(gm)
        return True
    except SpecViolationError:
        return False
