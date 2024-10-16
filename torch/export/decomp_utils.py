import itertools
from typing import Callable, Dict

import torch
from torch._export.utils import (
    _collect_all_valid_cia_ops_for_aten_namespace,
    _get_decomp_for_cia,
    _is_aten_op,
    _is_custom_op,
)


class CustomDecompTable(Dict[torch._ops.OperatorBase, Callable]):
    """
    This is a custom dictionary that is specifically used for handling decomp_table in export.
    The reason we need this is because in the new world, you can only *delete* an op from decomp
    table to preserve it. This is problematic for custom ops because we don't know when the custom
    op will actually be loaded to the dispatcher. As a result, we need to record the custom ops operations
    until we really need to materialize it (which is when we run decomposition pass.)

    Invariant we hold is:
    1. If it is read operation on this dict, we always materialize
    2. If it is write operation, we don't necessarily materialize
    """

    def __init__(self):
        super().__init__()
        from torch._decomp import _core_aten_decompositions_post_autograd

        # For aten ops, we load them up in the beginning
        self.aten_decomp_table = _core_aten_decompositions_post_autograd()

        for op in _collect_all_valid_cia_ops_for_aten_namespace():
            self.aten_decomp_table[op] = _get_decomp_for_cia(op)

        # This is to track the *pending* deleted custom ops that haven't been materialized yet
        self.deleted_custom_ops = set()
        # When we can materialize a custom op, we use this table
        self.additional_custom_op_decomp = {}
        # When this is true, there shouldn't be any pending operations in the table.
        self.has_materialized = False

    def __getitem__(self, key):
        self._materialize_if_needed()
        if key in self.aten_decomp_table:
            return self.aten_decomp_table[key]

        if _is_aten_op(key):
            raise KeyError(f"Op {key} is not in the decomposition table")

        assert _is_custom_op(key)

        if key in self.additional_custom_op_decomp:
            return self.additional_custom_op_decomp[key]

        raise KeyError(f"Op {key} is not in the decomposition table")

    def __setitem__(self, key, value):
        if _is_aten_op(key):
            self.aten_decomp_table[key] = value
            return

        assert _is_custom_op(key)

        self.additional_custom_op_decomp[key] = value

        if self.has_materialized:
            return

        if key in self.deleted_custom_ops:
            self.deleted_custom_ops.remove(key)

    def keys(self):
        self._materialize_if_needed()

        return itertools.chain(
            self.aten_decomp_table.keys(), self.additional_custom_op_decomp.keys()
        )

    def __delitem__(self, key):
        self.pop(key)

    def update(self, other_dict):  # type: ignore[override]
        for k, v in other_dict.items():
            if _is_aten_op(k):
                self.aten_decomp_table[k] = v
            else:
                self.additional_custom_op_decomp[k] = v
                if not self.has_materialized:
                    if k in self.deleted_custom_ops:
                        self.deleted_custom_ops.remove(k)

    def __missing__(self, key):
        return not self.__contains__(key)

    def __contains__(self, key):
        self._materialize_if_needed()
        if _is_aten_op(key):
            return key in self.aten_decomp_table

        if key in self.additional_custom_op_decomp:
            return True

        return False

    def __len__(self):
        self._materialize_if_needed()
        return len(self.aten_decomp_table) + len(self.additional_custom_op_decomp)

    def __iter__(self):
        self._materialize_if_needed()
        return itertools.chain(
            self.aten_decomp_table.__iter__(),
            self.additional_custom_op_decomp.__iter__(),
        )

    def __reverse__(self):
        raise RuntimeError("Cannot call reverse() on custom decomp table")

    def copy(self):
        new_dict = CustomDecompTable()
        new_dict.aten_decomp_table = self.aten_decomp_table.copy()
        new_dict.additional_custom_op_decomp = self.additional_custom_op_decomp.copy()
        new_dict.deleted_custom_ops = self.deleted_custom_ops.copy()
        new_dict.has_materialized = self.has_materialized
        return new_dict

    def pop(self, *args):
        def _pop_if_can(key):
            from torch._export.utils import _is_preservable_cia_op

            if _is_aten_op(key):
                if key in self.aten_decomp_table:
                    return self.aten_decomp_table.pop(key)

            if key in self.additional_custom_op_decomp:
                # Even if we materialized it, we should add it to the deleted
                # custom ops list so that when we materialize next time,
                # we should respect user's intention.
                self.deleted_custom_ops.add(key)
                return self.additional_custom_op_decomp.pop(key)

            if self.has_materialized:
                raise KeyError(f"{key} doesn't exist in the table")

            if key in self.deleted_custom_ops:
                raise KeyError(f"{key} doesn't exist in the table")

            if not _is_preservable_cia_op(key):
                raise KeyError(f"{key} doesn't exist in the table")

            self.deleted_custom_ops.add(key)
            return None

        if len(args) == 1:
            return _pop_if_can(args[0])

        if len(args) == 2:
            try:
                return _pop_if_can(args[0])
            except KeyError:
                return args[1]

    def items(self):
        self._materialize_if_needed()
        return itertools.chain(
            self.aten_decomp_table.items(), self.additional_custom_op_decomp.items()
        )

    def materialize(self):
        from torch._export.utils import _collect_all_valid_cia_ops, _get_decomp_for_cia

        for op in _collect_all_valid_cia_ops():
            if _is_aten_op(op):
                continue
            elif op in self.additional_custom_op_decomp:
                continue
            elif op not in self.deleted_custom_ops:
                self.additional_custom_op_decomp[op] = _get_decomp_for_cia(op)

        self.has_materialized = True
        self.deleted_custom_ops = set()
        return {**self.aten_decomp_table, **self.additional_custom_op_decomp}

    def _materialize_if_needed(self):
        if not self.has_materialized:
            self.materialize()
