import weakref

import torch
import torch.utils._pytree as pytree
from torch._C import _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet
from torch._ops import OpOverload
from torch.library import Library
from torchgen.model import (
    BaseTy,
    BaseType,
    FunctionSchema,
    OperatorName,
    OptionalType,
    SchemaKind,
)

from .autograd import autograd_not_implemented


def register_functional_op(
    lib: Library,
    new_op_name: str,
    mutable_op: OpOverload,
) -> None:
    """Given a mutable operator, registers the functional variant.

    This API also correctly links the functional variant with the mutable
    operator for the purposes of functionalization.

    All of the new registrations are performed on the ``lib`` passed in.

    Arguments:
        lib (Library): Should be a torch.library.Library object that has
            the same namespace as ``mutable_op``'s namespace.
            lib will be used to register the new functional op as well
            as a functionalization kernel for the ``mutable_op``
            If you don't have a library handy, use
            ``torch.library.Library(ns, 'FRAGMENT')`` to construct one.
        new_op_name (str): The name of the functional operator (without the
            namespace). If no namespace, the new functional variant will be
            accessible under ``torch.ops.{lib.ns}.new_op_name``.
        mutable_op (OpOverload): The mutable custom operator. Note
            that you may need to add a `.default` to it, like
            `torch.ops.aten.abs_.default`.

    """
    validate(mutable_op)
    schema = functional_schema(new_op_name, mutable_op)
    lib.define(schema)

    functional_impl = construct_functional_impl(mutable_op)
    lib.impl(new_op_name, functional_impl, 'CompositeExplicitAutograd')

    functional_op = getattr(getattr(torch.ops, lib.ns), new_op_name).default

    # There's no easy way for us to generate the autograd kernel, so we
    # use autograd_not_implemented. Also, this makes it so that the user
    # is unable to register an autograd formula themselves. This shouldn't
    # be a problem if the user doesn't use the functional op direclty
    # in their program, but we may need to revist this in the future.
    lib.impl(new_op_name, autograd_not_implemented(functional_op), 'Autograd')

    f_kernel = construct_functionalization_kernel(weakref.proxy(mutable_op), functional_op)

    lib.impl(mutable_op, f_kernel, 'Functionalize')


def construct_functional_impl(mutable_op):
    def functional_impl(*args):
        # Strategy:
        # - clone args that would have been mutated
        # - run mutable_op
        # - return the cloned args as additional outputs
        new_args = []
        extra_rets = []
        for is_write, arg in zip(mutable_args(mutable_op), args):
            if is_write:
                cloned = arg.clone() if arg is not None else None
                new_args.append(cloned)
                extra_rets.append(cloned)
            else:
                new_args.append(arg)
        result = mutable_op(*new_args)
        if result is None:
            return tuple(extra_rets)
        if isinstance(result, tuple):
            return (*result, *extra_rets)
        return (result, *extra_rets)
    return functional_impl


def construct_functionalization_kernel(mutable_op, functional_op):
    def kernel(*args):
        # There's nothing to be functionalized!
        # We can still end up here because DispatchKey::Functionalize is a mode key
        if pytree.tree_all_only(torch.Tensor, lambda x: not torch._is_functional_tensor(x), args):
            with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Functionalize)):
                return mutable_op(*args)

        # NB: This differs from the codegen -- codegen handles cases where there
        # are mixed FunctionalTensorWrapper and non-FunctionalTensorWrapper.
        # This only really matters for XLA (mixed CPU-XLA tensors) and
        # running functionalization without the PT2 stack (which guarantees to us that
        # all tensors are FunctionalTensorWrapper).
        if not pytree.tree_all_only(torch.Tensor, torch._is_functional_tensor, args):
            raise RuntimeError("{mutable_op}: expected all args to be FunctionalTensorWrapper")

        unwrapped_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and torch._is_functional_tensor(arg):
                torch._sync(arg)
                unwrapped = torch._from_functional_tensor(arg)
                unwrapped_args.append(unwrapped)
            else:
                unwrapped_args.append(arg)

        with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Functionalize)):
            output = functional_op(*unwrapped_args)

        num_actual_output = len(mutable_op._schema.returns)
        actual_output = pytree.tree_map(
            torch._to_functional_tensor, output[:num_actual_output])

        new_values_to_propagate = output[num_actual_output:]
        inputs_to_replace = [arg for is_write, arg in zip(mutable_args(mutable_op), args)
                             if is_write]
        assert len(new_values_to_propagate) == len(inputs_to_replace)
        for new_value, arg in zip(new_values_to_propagate, inputs_to_replace):
            if (arg is None and new_value is None) or (arg is not None and new_value is not None):
                continue
            torch._C._propagate_xla_data(arg, new_value)
            torch._C._replace_(arg, new_value)
            torch._C._commit_update(arg)
            torch._sync(arg)

        if len(actual_output) == 1:
            return actual_output[0]
        elif len(actual_output) == 0:
            return None
        return actual_output

    return kernel


def validate(mutable_op: OpOverload):
    if not isinstance(mutable_op, OpOverload):
        raise TypeError(
            f"register_functional_op(mutable_op): expected mutable_op to be instance of "
            f"OpOverload but got {type(mutable_op)}")

    # There are generally three types of "in-place" or "mutable" ops.
    # Each of them have their own conventions:
    # - inplace (first input modified in-place and returned as only output)
    # - out= (some args modified in-place and returned as outputs)
    # - mutable (some args modified in-place but none of those returned as outputs)
    # In theory we can support all three, but we'll just support the last
    # option right now for simplicity.
    schema = FunctionSchema.parse(str(mutable_op._schema))
    if not schema.kind() == SchemaKind.mutable:
        raise RuntimeError("Expected op to be mutable (as opposed to functional, inplace or out)")
    for ret in schema.returns:
        # construct_functionalization_kernel assumes this for simplicity
        if ret.annotation is not None:
            raise NotImplementedError(
                "NYI: register_functional_op(op) where op returns a mutated or aliased value. "
                "Please file an issue (and as a workaround, modify your operator to "
                "not return the mutated value or aliases)")
    for arg in schema.arguments.flat_all:
        # construct_functionalization_kernel assumes this for simplicity
        if arg.type.is_tensor_like() and (
            arg.type != BaseType(BaseTy.Tensor)
            and arg.type != OptionalType(BaseType(BaseTy.Tensor))
        ):
            raise NotImplementedError(
                "NYI: register_functional_op(op) where op has a List[Tensor] input."
                "Please file an issue.")


def functional_schema(new_op_name, op: OpOverload):
    schema = FunctionSchema.parse(str(op._schema))
    schema = schema.signature().with_name(OperatorName.parse(new_op_name))
    return str(schema)


def mutable_args(op: OpOverload):
    return tuple(False if arg.alias_info is None else arg.alias_info.is_write
                 for arg in op._schema.arguments)
