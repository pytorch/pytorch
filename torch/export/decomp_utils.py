# mypy: allow-untyped-defs
import functools
from collections.abc import Callable
from contextlib import contextmanager

import torch
from torch._export.utils import (
    _collect_all_valid_cia_ops,
    _collect_all_valid_cia_ops_for_aten_namespace,
    _get_decomp_for_cia,
    _is_aten_op,
    _is_preservable_cia_op,
    _special_op_to_preserve_cia,
)
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._subclasses.fake_impls import (
    _deregister_op_impl,
    _is_op_registered_to_fake_rule,
    register_op_impl,
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
    # NB: don't use the C++ decomp, because it is not functional!
    torch.ops.aten.silu_backward.default,
    torch.ops.aten.mish_backward.default,
    torch.ops.aten._fused_rms_norm.default,
}


# This list is compiled from DispatchKey.cpp.
# The idea is that we use these keys to override CIA decomp in export.
_AUTOGRAD_ALIAS_BACKEND_KEYS_TO_OVERRIDE = [
    torch._C.DispatchKey.AutogradCPU,
    torch._C.DispatchKey.AutogradCUDA,
    torch._C.DispatchKey.AutogradMeta,
    torch._C.DispatchKey.AutogradXLA,
    torch._C.DispatchKey.AutogradLazy,
    torch._C.DispatchKey.AutogradIPU,
    torch._C.DispatchKey.AutogradXPU,
    torch._C.DispatchKey.AutogradMPS,
    torch._C.DispatchKey.AutogradHPU,
    torch._C.DispatchKey.AutogradPrivateUse1,
    torch._C.DispatchKey.AutogradPrivateUse2,
    torch._C.DispatchKey.AutogradPrivateUse3,
]


# This list is compiled from DispatchKey.cpp.
# The idea is that we use these keys to add python kernels that directly use
# default CIA decomp. See NOTE Registering old CIA to Backend kernel.
_BACKEND_KEYS_TO_OVERRIDE = [
    torch._C.DispatchKey.CPU,
    torch._C.DispatchKey.CUDA,
    torch._C.DispatchKey.Meta,
    torch._C.DispatchKey.XLA,
    torch._C.DispatchKey.Lazy,
    torch._C.DispatchKey.IPU,
    torch._C.DispatchKey.XPU,
    torch._C.DispatchKey.MPS,
    torch._C.DispatchKey.HPU,
]


@contextmanager
def _override_composite_implicit_decomp(cia_ops_to_callable):
    # This function overrides CompositeImplicitAutograd decomp for
    # functional composite ops that user specified. Ideally we want to not-decompose
    # ALL composite ops but today's C++ functinalization relies on
    # the fact that it is working with the opset after decomp is run.
    # Hence we can only do it for functional ops. One caveat is that
    # there are some composite ops that lie about their schema (claimed to be
    # functional but not really aka dropout), for these cases, we just decompose.
    saved_tables = {}
    patched_ops = set()
    for op_overload, decomp_callable in cia_ops_to_callable.items():
        saved_tables[op_overload] = op_overload.py_kernels.copy()
        patched_ops.add(op_overload)
        for override_dispatch_key in _AUTOGRAD_ALIAS_BACKEND_KEYS_TO_OVERRIDE:
            if override_dispatch_key not in op_overload.py_kernels:
                # TODO (tmanlaibaatar)https://github.com/pytorch/pytorch/issues/129430
                op_overload.py_impl(override_dispatch_key)(
                    autograd_not_implemented(op_overload, deferred_error=True)
                )
        # See NOTE: Registering old CIA to Backend kernel
        # It is important that we cache this before we override py_kernels.
        orig_cia_callable = _get_decomp_for_cia(op_overload)
        if torch._C.DispatchKey.CompositeImplicitAutograd in op_overload.py_kernels:
            del op_overload.py_kernels[torch._C.DispatchKey.CompositeImplicitAutograd]

        op_overload.py_impl(torch._C.DispatchKey.CompositeImplicitAutograd)(
            decomp_callable
        )

        # [NOTE] Directly registering fake tensor rule to CIA ops
        # The problem we are facing here is if your CIA custom rule
        # says we want to preserve the op, we will return NotImplemented.
        # Unfortunately, this will invoke meta device tracing in fake tensor
        # resulting in divergent behaviour for CIA kernels that has device based
        # branching (one case is torch.ops.aten.scaled_dot_product.attention)
        # To get around this issue, we register direct fake impl so that we
        # run the kernel before we actually try to decompose the op in FakeTensorMode.
        # Note that is a no-op in most cases, because:
        #   1) In post dispatch tracing, CIA would have already decomposed
        #   2) Most CIA impl are device agnostic.
        def _force_dispatch_to_orig_cia_callable(fake_tensor_mode, op, *args, **kwargs):
            orig_cia_callable = kwargs["original_callable"]
            del kwargs["original_callable"]
            with fake_tensor_mode:
                return orig_cia_callable(*args, **kwargs)

        if not _is_op_registered_to_fake_rule(op_overload):
            register_op_impl(op_overload)(
                functools.partial(
                    _force_dispatch_to_orig_cia_callable,
                    original_callable=orig_cia_callable,
                )
            )

        for key in _BACKEND_KEYS_TO_OVERRIDE:
            if key not in op_overload.py_kernels:
                # [NOTE] Registering old CIA to Backend kernel
                # We always register original CIA behavior to the backend keys kernel
                # The reason is when we are fake tensor prop-ing or executing real kernel,
                # we end up calling an operator on respective backend, which in python dispatcher,
                # will resolve into CIA key. (see resolve_key in torch/_ops.py)
                # As a result, this CIA now will call into the custom user defined
                # CIA which can cause a problem.
                # To make it more concrete, the case we are handling is:
                #  (1) there is a tensor constant we are performing constant propagation
                #      on during tracing
                #  (2) we invoke an op underneath autograd (either because we are below autograd,
                #      or we are tracing in inference mode), so one of the backend keys gets hit
                #  (3) the op we are invoking has a CIA impl that normally runs in eager mode
                #      (and the user wants to tweak this CIA impl during tracing, but during
                #      const-prop we want the original CIA to run
                op_overload.py_impl(key)(orig_cia_callable)

    try:
        yield
    finally:
        for op in patched_ops:
            op.py_kernels.clear()
            op.py_kernels.update(saved_tables[op])
            op._dispatch_cache.clear()
            _deregister_op_impl(op)


def _split_decomp_table_to_cia_and_python_decomp(
    decomp_table: dict[torch._ops.OperatorBase, Callable],
) -> tuple[dict[torch._ops.OperatorBase, Callable], ...]:
    all_preservable_cia_ops = set(_collect_all_valid_cia_ops())
    cia_ops_to_callable = {}

    for op in list(decomp_table.keys()):
        # TODO we are silently allowing non-safe(non-functional) ops through a crack
        # due to core aten decomp table having non-functional entries. Once we have
        # a tighter check around core aten decomp, we should warn users about them.
        # Tracking issue: (https://github.com/pytorch/pytorch/issues/135759)

        # if it is a valid CIA op we can mess with in export, we check if it is:
        #  1. Has been marked as to be decomposed. Example:
        #        decomp_table = decomp_table_to_core_aten()
        #        del decomp_table[aten.linear]
        #     In this case, user says decompose everything except for aten.linear
        #  2. Has been marked with custom decomp behaviour. Example:
        #        decomp_table = {aten.linear: some_op}
        # For (1), we want to remove all the CIA ops that weren't handled by user as
        # it suggests they are safe to decompose, so we should remove from preservable_list.
        # for (2), we just plumb the custom decomp to AOTDIspatcher.
        # In both cases, we want to remove this CIA op from the decomp_table as it is special
        # handled.
        if op in all_preservable_cia_ops:
            cia_ops_to_callable[op] = decomp_table[op]
            all_preservable_cia_ops.remove(op)
            del decomp_table[op]
        # If it is a custom op, we want to still preserve or do whatever
        # with it if it is a functional CIA. The reason we don't remove
        # from CIA list is because we don't query custom ops.
        elif _is_preservable_cia_op(op):
            op_name = op.name()
            if op_name.startswith("aten"):
                raise AssertionError(
                    f"This should be a custom op, got aten op: {op_name}"
                )
            cia_ops_to_callable[op] = decomp_table[op]

    # If we reached here, it means user intentionally deleted these CIA ops from
    # decomp table.
    for k in all_preservable_cia_ops:
        cia_ops_to_callable[k] = _special_op_to_preserve_cia

    return cia_ops_to_callable, decomp_table


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
            if op not in PRESERVED_ATEN_CIA_OPS and op not in self.decomp_table:
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
