# mypy: allow-untyped-defs
from typing import Callable

import torch
from torch._export.utils import (
    _collect_all_valid_cia_ops,
    _collect_all_valid_cia_ops_for_aten_namespace,
    _get_decomp_for_cia,
    _is_aten_op,
)


__all__ = ["CustomDecompTable"]


"""
Core ATen ops with Composite Implicit Autograd dispatch that should be excluded from decomposition
by default. The decomposition logic should eventually exclude all core-tagged CIA ops, but until all
backends are ready, this list allows opt-in one at a time.
"""
PRESERVED_ATEN_CIA_OPS = {
    torch.ops.aten.upsample_bilinear2d.vec,
    torch.ops.aten.upsample_nearest2d.vec,
}


class CustomDecompTable(dict[torch._ops.OperatorBase, Callable]):
    """
    This is a custom dictionary that is specifically used for handling decomp_table in export.
    The reason we need this is because in the new world, you can only *delete* an op from decomp
    table to preserve it. This is problematic for custom ops because we don't know when the custom
    op will actually be loaded to the dispatcher. As a result, we need to record the custom ops operations
    until we really need to materialize it (which is when we run decomposition pass.)

    Invariants we hold are:
     1. All aten decomp is loaded at the init time
     2. We materialize ALL ops when user ever reads from the table to make it more likely
        that dispatcher picks up the custom op.
     3. If it is write operation, we don't necessarily materialize
     4. We load the final time during export, right before calling run_decompositions()

    """

    def __init__(self):
        super().__init__()
        from torch._decomp import _core_aten_decompositions_post_autograd

        # For aten ops, we load them up in the beginning
        self.decomp_table = _core_aten_decompositions_post_autograd()

        for op in _collect_all_valid_cia_ops_for_aten_namespace():
            if op not in PRESERVED_ATEN_CIA_OPS:
                self.decomp_table[op] = _get_decomp_for_cia(op)

        # This is to track the *pending* deleted custom ops that haven't been materialized yet
        self.deleted_custom_ops = set()
        # When this is true, there shouldn't be any pending operations in the table.
        self.has_materialized = False

    def __getitem__(self, key):
        self._materialize_if_needed()
        return self.decomp_table.__getitem__(key)

    def __setitem__(self, key, value) -> None:
        self.decomp_table.__setitem__(key, value)

        if key in self.deleted_custom_ops:
            self.deleted_custom_ops.remove(key)

    def keys(self):
        self._materialize_if_needed()
        return self.decomp_table.keys()

    def __delitem__(self, key) -> None:
        self.pop(key)

    def update(self, other_dict):  # type: ignore[override]
        for k, v in other_dict.items():
            self.decomp_table.__setitem__(k, v)

    def __missing__(self, key) -> bool:
        return not self.__contains__(key)

    def __contains__(self, key) -> bool:
        self._materialize_if_needed()
        return self.decomp_table.__contains__(key)

    def __len__(self) -> int:
        self._materialize_if_needed()
        return self.decomp_table.__len__()

    def __iter__(self):
        self._materialize_if_needed()
        return self.decomp_table.__iter__()

    def __reversed__(self):
        self._materialize_if_needed()
        return self.decomp_table.__reversed__()

    def copy(self) -> "CustomDecompTable":
        new_dict = CustomDecompTable()
        new_dict.decomp_table = self.decomp_table.copy()
        new_dict.deleted_custom_ops = self.deleted_custom_ops.copy()
        new_dict.has_materialized = self.has_materialized
        return new_dict

    def pop(self, *args):
        def _pop_if_can(key):
            if _is_aten_op(key):
                return self.decomp_table.pop(key)

            if key in self.decomp_table:
                # Even if we materialized it, we should add it to the deleted
                # custom ops list so that when we materialize next time,
                # we should respect user's intention.
                self.deleted_custom_ops.add(key)
                return self.decomp_table.pop(key)

            if key in self.deleted_custom_ops:
                raise KeyError(f"{key} doesn't exist in the table")

            self.deleted_custom_ops.add(key)
            # We would come here when user pops off something that is
            # not in the table. In this case, we just pretend that it
            # was in the table.
            return _get_decomp_for_cia(key)

        if len(args) == 1:
            return _pop_if_can(args[0])

        if len(args) == 2:
            try:
                return _pop_if_can(args[0])
            except KeyError:
                return args[1]

    def items(self):
        self._materialize_if_needed()
        return self.decomp_table.items()

    def materialize(self) -> dict[torch._ops.OperatorBase, Callable]:
        for op in _collect_all_valid_cia_ops():
            if _is_aten_op(op):
                continue
            elif op in self.decomp_table:
                continue
            elif op not in self.deleted_custom_ops:
                self.decomp_table[op] = _get_decomp_for_cia(op)

        self.has_materialized = True
        self.deleted_custom_ops = set()
        return {**self.decomp_table}

    def _materialize_if_needed(self) -> None:
        if not self.has_materialized:
            self.materialize()
