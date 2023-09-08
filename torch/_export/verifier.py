import itertools
import operator
from collections.abc import Iterable
from typing import Set

import torch
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
        raise SpecViolationError(f"Node.meta {node.name} is missing val field.")


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
class Verifier:
    def __call__(self, gm: GraphModule) -> None:
        self.check_valid(gm)

    @compatibility(is_backward_compatible=False)
    def valid_builtin_funcs(self):
        return [
            operator.getitem,
            torch.ops.higher_order.cond,
            torch.ops.map_impl,
        ]

    @compatibility(is_backward_compatible=False)
    def check_valid_op(self, op):
        op_name = op.name if hasattr(op, "name") else op.__name__

        if not isinstance(op, OpOverload):
            raise SpecViolationError(
                f"Operator '{op_name}' is not a registered Op",
            )

        # All ops functional
        if not is_functional(op):
            raise SpecViolationError(
                f"operator '{op_name}' is not functional"
            )

    @compatibility(is_backward_compatible=False)
    def check_valid(self, gm: GraphModule) -> None:  # noqa: C901

        for node in gm.graph.nodes:
            # TODO(T140410192): should have fake tensor for all dialects
            if node.op in {"call_module", "call_method"}:
                raise SpecViolationError(
                    f"call_module is not valid: got a class '{node.target}' ",
                )

            if node.op == "call_function":
                _check_has_fake_tensor(node)

                if node.target not in self.valid_builtin_funcs():
                    self.check_valid_op(node.target)

                if isinstance(node.target, OpOverload):
                    # Check preserved metadata
                    for meta in PRESERVED_META_KEYS:
                        if node.meta.get(meta, None) is None:
                            raise SpecViolationError(
                                f"node {node} is missing metadata {meta}"
                            )

    @compatibility(is_backward_compatible=False)
    def is_valid(self, gm: GraphModule) -> bool:
        try:
            self.check_valid(gm)
            return True
        except SpecViolationError:
            return False


class ATenDialectVerifier(Verifier):
    @compatibility(is_backward_compatible=False)
    def check_valid_op(self, op) -> None:
        super().check_valid_op(op)

        op_name = op.name if hasattr(op, "name") else op.__name__

        if not isinstance(op, OpOverload):
            raise SpecViolationError(
                f"Operator '{op_name}' is not a registered Op",
            )

        if (
            torch.Tag.core not in op.tags
            and torch.Tag.view_copy not in op.tags
        ):
            # NOTE(qihan): whether view_copy operators are marked as canonical is still under
            #            discussion.
            raise SpecViolationError(
                f"Operator {op.__module__}.{op.__name__} is not Aten Canonical."
            )

    @compatibility(is_backward_compatible=False)
    def check_valid(self, gm: GraphModule) -> None:
        super().check_valid(gm)
        _check_tensors_are_contiguous(gm)
