# mypy: allow-untyped-defs
"""
This file contains utilities related to functionalization in AOTAutograd:
1. converting to/from functional tensors
2. detecting Tensor mutations - both metadata and Tensor value
3. regenerating/replaying views from their base
4. checking if a graph is functional i.e. whether it contains any mutation ops
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch._C import _functionalization
from torch._logging import getArtifactLogger
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor
from torch._subclasses.meta_utils import is_sparse_any
from torch.fx.experimental.symbolic_shapes import SymIntEqByExpr
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    transform_subclass,
)


aot_joint_log = getArtifactLogger(__name__, "aot_joint_graph")


def to_fun(t):
    if isinstance(t, Tensor):
        if is_traceable_wrapper_subclass(t):
            # See Note [Functionalization always runs last]
            # This means that if we want to "functionalize" a subclass, we need to ensure that the functional wrapper
            # goes at the bottom.
            # recurse here, so we can support nested wrapper subclasses
            out = transform_subclass(t, lambda _, inner_t: to_fun(inner_t))
            torch._mirror_autograd_meta_to(t, out)  # type: ignore[attr-defined]
            return out
        else:
            return FunctionalTensor.to_functional(t)
    else:
        return t


def sync_functional_tensor(t):
    if is_traceable_wrapper_subclass(t):
        attrs, _ctx = t.__tensor_flatten__()  # type: ignore[attr-defined]
        for attr in attrs:
            sync_functional_tensor(getattr(t, attr))
    else:
        torch._sync(t)


# When subclasses are involved, t here will usually look something like:
# SubclassA(SubclassB(FunctionalTensor(_to_fun_tensor(FakeTensor))))
def from_fun(t):
    if isinstance(t, Tensor) and is_traceable_wrapper_subclass(t):
        # See Note [Functionalization always runs last]
        # This means that if we want to "functionalize" a subclass, we need to ensure that the functional wrapper
        # goes at the bottom.
        # recurse here, so we can support nested wrapper subclasses
        out = transform_subclass(t, lambda _, inner_t: from_fun(inner_t))
        torch._mirror_autograd_meta_to(t, out)  # type: ignore[attr-defined]
        return out

    if not isinstance(t, FunctionalTensor):
        # quick sanity assert
        if isinstance(t, torch.Tensor):
            assert not torch._is_functional_tensor(t)  # type: ignore[attr-defined]
        return t
    sync_functional_tensor(t)
    return torch._from_functional_tensor(t.elem)


def is_fun(t):
    if isinstance(t, Tensor) and is_traceable_wrapper_subclass(t):
        # See Note [Functionalization always runs last]
        # This means that if we want to "functionalize" a subclass, we need to ensure that the functional wrapper
        # goes at the bottom.
        # recurse here, so we can support nested wrapper subclasses
        t_attrs, _ = t.__tensor_flatten__()  # type: ignore[attr-defined]
        t_inners = [getattr(t, attr) for attr in t_attrs]
        any_fun = any(is_fun(x) for x in t_inners)
        all_fun = all(is_fun(x) for x in t_inners)
        assert any_fun == all_fun
        return any_fun

    return isinstance(t, FunctionalTensor)


# t here is either
# (1) A FunctionalTensor(_to_functional_tensor(FakeTensor))
# (2) A traceable tensor subclass that holds a FunctionalTensor
# (3) Not a tensor
def has_data_mutation(t):
    if is_traceable_wrapper_subclass(t):
        attrs, _ = t.__tensor_flatten__()
        # A tensor subclass was updated if any of its inner elements were updated
        return any(has_data_mutation(getattr(t, attr)) for attr in attrs)
    else:
        if isinstance(t, torch.Tensor):
            assert isinstance(t, FunctionalTensor)
            return torch._functionalize_has_data_mutation(t.elem)  # type: ignore[attr-defined]
        return False


def are_all_mutations_hidden_from_autograd(t):
    if is_traceable_wrapper_subclass(t):
        attrs, _ = t.__tensor_flatten__()
        # If all inner elements are mutations hidden from autograd, then it is a mutation hidden from autograd.
        return all(
            are_all_mutations_hidden_from_autograd(getattr(t, attr)) for attr in attrs
        )
    elif isinstance(t, torch.Tensor):
        assert isinstance(t, FunctionalTensor)
        return torch._functionalize_are_all_mutations_hidden_from_autograd(t.elem)
    else:
        return False


def are_all_mutations_under_no_grad_or_inference_mode(t):
    if is_traceable_wrapper_subclass(t):
        attrs, _ = t.__tensor_flatten__()
        return all(
            are_all_mutations_under_no_grad_or_inference_mode(getattr(t, attr))
            for attr in attrs
        )
    else:
        assert isinstance(t, FunctionalTensor)
        return torch._functionalize_are_all_mutations_under_no_grad_or_inference_mode(
            t.elem
        )


def was_inductor_storage_resized(t):
    if is_traceable_wrapper_subclass(t):
        attrs, _ = t.__tensor_flatten__()
        if any(was_inductor_storage_resized(getattr(t, attr)) for attr in attrs):
            raise RuntimeError(
                f"storage resizing is not supported on tensor subclass: {type(t)}"
            )
    elif not isinstance(t, torch.Tensor):
        return False
    else:
        assert isinstance(t, FunctionalTensor)
        return torch._functionalize_was_inductor_storage_resized(t.elem)


# f_arg here is either
# (1) A FunctionalTensor(_to_functional_tensor(FakeTensor))
# (2) A traceable tensor subclass that holds a FunctionalTensor
# (3) Not a tensor
# Assumption: arg promises to be the "original" tensor wrapped by f_arg
# Note: "storage mutations" coming from set_() are a type of metadata mutation. So:
# - check_only_storage_mutation=True: only return true if there was a storage mutation
# - check_only_storage_mutation=Flse: return true if there was any metadata mutation (including a storage mutation)
def has_metadata_mutation(f_arg, arg, *, check_only_storage_mutation: bool):
    if is_traceable_wrapper_subclass(f_arg):
        attrs, _ = f_arg.__tensor_flatten__()
        # A tensor subclass was updated if any of its inner elements were updated
        f_inner_ts = [getattr(f_arg, attr) for attr in attrs]
        inner_ts = [getattr(arg, attr) for attr in attrs]
        return any(
            has_metadata_mutation(
                f_inner_t,
                inner_t,
                check_only_storage_mutation=check_only_storage_mutation,
            )
            for f_inner_t, inner_t in zip(f_inner_ts, inner_ts)
        )
    else:
        if not isinstance(f_arg, torch.Tensor):
            assert not isinstance(arg, torch.Tensor)
            return False
        assert isinstance(f_arg, FunctionalTensor)
        assert isinstance(arg, FakeTensor)

        arg_after = torch._from_functional_tensor(f_arg.elem)
        # This is true if the current tensor experienced at least one set_() call
        maybe_storage_changed = torch._functionalize_was_storage_changed(f_arg.elem)  # type: ignore[attr-defined]
        # However, multiple set_() calls can cancel out. So we also check whether the
        # storage of the tensor has changed.
        # Note: if an input experienced two set_() calls that cancel out, **and**
        # it experiences an data mutation, we pessimistically think that the set_()
        # call is necessary here. We could in theory fix this, but this will
        # hopefully never happen in user code, and is not needed for fsdp.
        if is_sparse_any(arg):
            # TODO:add sparse tensors support to functionalization
            same_storages = False
        else:
            same_storages = StorageWeakRef(arg.untyped_storage()) == StorageWeakRef(
                arg_after.untyped_storage()
            )
        has_storage_metadata_mutation = maybe_storage_changed and not same_storages
        if check_only_storage_mutation:
            return has_storage_metadata_mutation

        # storage metadata mutation is a type of metadata mutation, so return true if we saw one
        if has_storage_metadata_mutation:
            return True

        maybe_metadata_mutated = torch._functionalize_has_metadata_mutation(f_arg.elem)  # type: ignore[attr-defined]
        # This is true if the current tensor experienced at least one metadata mutation.
        # So if false, we know there was no metadata mutation
        if not maybe_metadata_mutated:
            return False

        # However, multi metadata mutations can cancel out.
        # So we also check if the concrete sizes/strides on the tensor have changed.
        same_sizes = arg.shape == arg_after.shape
        same_strides = arg.stride() == arg_after.stride()
        same_offsets = arg.storage_offset() == arg_after.storage_offset()
        has_metadata_mutation_ = maybe_metadata_mutated and not (
            same_sizes and same_strides and same_offsets
        )
        # We consider a tensor to have been metadata mutated if its storage was mutated through a set_() call.
        return has_metadata_mutation_


def gen_alias_from_base(
    aliased_base_tensor,
    target_meta_tensor,
    target_requires_grad,
    target_view_meta_sequence: Optional[ViewMetaSequence] = None,
    *,
    replay_views: bool,
):
    # Patch the correct requires_grad field of the output tensor, depending on whether:
    # (i) the reconstructed output (out) was came from a tensor that requires grad or not;
    # and (ii) the concrete returned output does require grad or not.
    def patch_requires_grad(out):
        if aliased_base_tensor.requires_grad and not target_requires_grad:
            out = out.detach()
        elif not aliased_base_tensor.requires_grad and target_requires_grad:
            out.requires_grad_(True)
        return out

    # If provided, use the target functional tensor for replaying the views.
    #
    # In summary, we use the fact that FunctionalTensorWrapper saves the view
    # functions applied to itself (collected during functionalization) so as
    # to replay them (view functions) on the aliased_base_tensor.
    if (
        replay_views
        and target_view_meta_sequence is not None
        and not any(vm.has_symbolic_inputs for vm in target_view_meta_sequence.sequence)
    ):
        out = _functionalization.apply_view_meta_sequence(
            aliased_base_tensor, target_view_meta_sequence.sequence
        )
        # If re-applying the ViewMeta sequence succeeded, there should be no more
        # problems going forward. We just check we got to the target shape and
        # patch requires_grad flag.
        assert out.shape == target_meta_tensor.shape, (
            "incorrect out shape after application of ViewMeta sequence: "
            f"{tuple(out.shape)} (actual) vs {tuple(target_meta_tensor.shape)} (expected)"
        )
        return patch_requires_grad(out)

    # Try to do view-replay if possible.
    # fall back to .as_strided() if we can't.
    if target_meta_tensor._base is not None:
        # The base that we want to replay our view off of might have a different shape than the view's original base.
        b = target_meta_tensor._base
        abt = aliased_base_tensor
        # Don't unnecessarily call as_strided if nothing changed; as_strided's
        # backward is poorly implemented and slow
        if abt is not b and (
            abt.size() != b.size()
            or abt.stride() != b.stride()
            or abt.storage_offset() != b.storage_offset()
        ):
            reshaped_base_tensor = aliased_base_tensor.as_strided(
                b.size(), b.stride(), b.storage_offset()
            )
        else:
            reshaped_base_tensor = aliased_base_tensor
        out = target_meta_tensor._view_func(reshaped_base_tensor)
        # This shape mismatch can happen due to a bug in inplace/view handling in autograd.
        # Try putting a breakpoint here and running
        # `test/functorch/test_aotdispatch TestAOTAutograd.test_output_all_alias_types`
        # Also, https://github.com/pytorch/pytorch/issues/49825
        #
        # As a stopgap, we'll fall back to as_strided.
        if out is not None and out.shape == target_meta_tensor.shape:
            return patch_requires_grad(out)

    size = target_meta_tensor.size()
    stride = target_meta_tensor.stride()
    storage_offset = target_meta_tensor.storage_offset()
    if aliased_base_tensor.is_complex() and not target_meta_tensor.is_complex():
        aliased_out = torch.view_as_real(aliased_base_tensor).as_strided(
            size, stride, storage_offset
        )
    elif not aliased_base_tensor.is_complex() and target_meta_tensor.is_complex():
        aliased_out = torch.view_as_complex(aliased_base_tensor).as_strided(
            size, stride, storage_offset
        )
    else:
        aliased_out = aliased_base_tensor.as_strided(size, stride, storage_offset)
    # For outputs aliasing inputs, we need to check if the requires-gradness has changed.
    aliased_out = patch_requires_grad(aliased_out)
    # For outputs aliasing inputs, we need to check if the dtype has changed.
    # as_strided() is the "most generic" view, but it does not cover cross-dtype views
    if aliased_out.dtype != target_meta_tensor.dtype:
        aliased_out = aliased_out.view(target_meta_tensor.dtype)
    return aliased_out


@dataclass(frozen=True)
class MetadataKey:
    size: Tuple[SymIntEqByExpr, ...]
    layout: torch.layout
    is_sparse: bool
    # these are empty when is_sparse
    stride: Optional[Tuple[SymIntEqByExpr, ...]]
    storage_offset: Optional[SymIntEqByExpr]
    is_conj: bool
    is_neg: bool

    @staticmethod
    def make(t):
        is_sparse = is_sparse_any(t)
        return MetadataKey(
            size=tuple(SymIntEqByExpr(s) for s in t.size()),
            layout=t.layout,
            is_sparse=is_sparse,
            stride=None if is_sparse else tuple(SymIntEqByExpr(s) for s in t.stride()),
            storage_offset=None if is_sparse else SymIntEqByExpr(t.storage_offset()),
            is_conj=t.is_conj(),
            is_neg=t.is_neg(),
        )


# ViewMeta sequence wrapper for equality comparisons.
#
# Even though we can compare each ViewMeta instance, we compare the resulting
# tensor metadata, instead. That's because the creation of synthetic bases + the
# re-generation of input views might end-up creating a different sequence of
# ViewMeta that is semantically equivalent. i.e. gets to a tensor with the same
# metadata.
#
# Therefore, we store what the end result should look like as serializable
# metadata.
#
# When logging, this class should look like:
#
#     ViewMetaSequence(view, select_int, slice_Tensor)
#
# i.e. a parenthesized list of view operations within that ViewMeta sequence.
class ViewMetaSequence:
    def __init__(self, tensor: FunctionalTensor) -> None:
        assert torch._is_functional_tensor(tensor.elem)
        self.sequence = _functionalization.get_view_meta_sequence(tensor.elem)
        self.metadata = MetadataKey.make(tensor)

    def __repr__(self) -> str:
        suffix = len("_ViewMeta")
        types = ", ".join(type(vm).__name__[:-suffix] for vm in self.sequence)
        return f"ViewMetaSequence({types})"

    def __eq__(self, other: object) -> bool:
        # If other is None, then it probably means that we weren't able to recreate
        # the ViewMeta sequence. One example is when we update the view metadata by
        # calling: create_synthetic_base_metadata.
        if other is None:
            return True

        # Comparison against any other type is not implemented.
        if not isinstance(other, ViewMetaSequence):
            return NotImplemented

        return self.metadata == other.metadata


# new_arg and arg here are either:
# (1) both a FakeTensor
# (2) both a traceable tensor subclass that holds a FakeTensor
# Pre-condition: the two args are the "old" and "new" inputs from running functionalization.
# When we run functionalization and wrap our inputs into FunctionalTensors,
# we can detect whether or not an input was mutated by checking to see if the inner tensor has changed
#
# Normally it would be enough just to check if arg is new_arg, which is normally enough for functionalization
# to confirm that inputs were not mutated when running the user's model with functionalization on.
# But when we have subclass inputs, we can't rely on that:
# `from_fun(to_fun(x)) is x` will return False, because the call to `from_fun` constructs
# a brand new subclass instance: we are calling __tensor_unflatten__, and going
# from Subclass(FakeTensor) to Subclass(FunctionalTensor(FakeTensor))
def was_tensor_updated(arg, new_arg):
    if is_traceable_wrapper_subclass(arg):
        assert is_traceable_wrapper_subclass(new_arg)
        attrs, _ = arg.__tensor_flatten__()
        new_attrs, _ = new_arg.__tensor_flatten__()
        assert attrs == new_attrs
        # A tensor subclass was updated if any of its inner elements were updated
        return any(
            was_tensor_updated(getattr(arg, attr), getattr(new_arg, attr))
            for attr in attrs
        )
    else:
        return arg is not new_arg


# new_arg and arg here are either:
# (1) both a FakeTensor
# (2) both a traceable tensor subclass that holds a FakeTensor
# Pre-condition: the two args are the "old" and "new" inputs from running functionalization.
# When we run functionalization and wrap our inputs into FunctionalTensors,
# we can detect whether or not an input was mutated by checking to see if the inner tensor has changed,
# but shares storage with the old input
def was_tensor_metadata_updated(arg, new_arg):
    if is_traceable_wrapper_subclass(arg):
        assert is_traceable_wrapper_subclass(new_arg)
        attrs, _ = arg.__tensor_flatten__()
        new_attrs, _ = new_arg.__tensor_flatten__()
        assert attrs == new_attrs
        # A tensor subclass was updated if any of its inner elements were updated
        return any(
            was_tensor_metadata_updated(getattr(arg, attr), getattr(new_arg, attr))
            for attr in attrs
        )
    else:
        return arg is not new_arg and StorageWeakRef(
            arg.untyped_storage()
        ) == StorageWeakRef(new_arg.untyped_storage())


# Returns the number of detected copy_
def assert_functional_graph(fx_g: torch.fx.Graph) -> int:
    allowed_mutation_ops = [
        torch.ops.aten.copy_.default,
        torch.ops.aten.set_.source_Tensor,
    ]
    if hasattr(torch.ops.fsdp, "copy_"):
        allowed_mutation_ops.append(torch.ops.fsdp.copy_.default)

    placeholders = set()
    mutation_count = 0
    # NB: It would also be nice to verify that the mutations all happen at the
    # end, but we also do some administrative views after mutations so this
    # isn't actually true.  (TODO: Could this cause problems for Inductor?)
    for n in fx_g.nodes:
        if n.op == "placeholder":
            placeholders.add(n)
        if isinstance(n.target, torch._ops.OpOverload):
            if n.target in allowed_mutation_ops:
                # Can only copy_/set_ into an input
                # this is mostly a hack to avoid failing XLA tests.
                # See https://github.com/pytorch/pytorch/pull/122434#issuecomment-2101012113
                if "set_buffer_donor_" not in str(n.args[0]):
                    assert (
                        n.args[0] in placeholders
                    ), f"n={str(n)}, n.args[0]={str(n.args[0])}, placeholders={str(placeholders)}, graph={str(fx_g)}"
                mutation_count += 1
            else:
                assert (
                    not n.target._schema.is_mutable
                ), f"aot_autograd expected to have an entirely functional graph, but found {n.format_node()}"
    return mutation_count


def propagate_input_mutation_stacktraces(fx_g: torch.fx.Graph) -> None:
    placeholders = set()
    for n in fx_g.nodes:
        if n.op == "placeholder":
            placeholders.add(n)
        if isinstance(n.target, torch._ops.OpOverload):
            if n.target is torch.ops.aten.copy_.default:
                # Can only copy_ into an input, and can only do so once
                if "set_buffer_donor_" not in str(n.args[0]):
                    assert (
                        n.args[0] in placeholders
                    ), f"n={str(n)}, n.args[0]={str(n.args[0])}, placeholders={str(placeholders)}, graph={str(fx_g)}"
                    placeholders.remove(n.args[0])
                copy_from_node = n.args[1]
                # Pre-condition: every node has a "stack_trace" field in its meta,
                # but copy_() nodes do not (since we manually added them during functionalization).
                # Instead, we manually propagate here.
                if "stack_trace" in copy_from_node.meta:
                    n.meta["stack_trace"] = copy_from_node.meta["stack_trace"]


def _check_if_mutation_can_be_in_graph(
    keep_input_mutations: bool,
    mutates_data,
    mutates_metadata,
    mutations_hidden_from_autograd,
    mutations_under_no_grad_or_inference_mode,
    mutates_storage_metadata,
    mutation_inductor_storage_resize,
    requires_grad,
):
    if keep_input_mutations:
        in_graph = (
            mutates_data or mutates_storage_metadata or mutation_inductor_storage_resize
        ) and (
            (not mutates_metadata and not requires_grad)
            or mutations_hidden_from_autograd
            or mutations_under_no_grad_or_inference_mode
        )
    else:
        in_graph = False
    # See Note [set_() Input Mutations in AOTAutograd]
    # If there was a `set_()`, we require that all mutations were under no_grad,
    # so we can (safely) emit the set_() in the graph at runtime
    # resize_() gets the same treatment
    if mutation_inductor_storage_resize or mutates_storage_metadata:
        op_name = "resize_" if mutation_inductor_storage_resize else "set_"
        assert in_graph, f"""\
Encountered a {op_name} on a graph input, but the input has other mutations that we cannot
keep in the graph. This is not supported today. Current state:
  keep_input_mutations={keep_input_mutations}
  mutates_data={mutates_data}
  mutates_metadata={mutates_metadata}
  mutations_hidden_from_autograd={mutations_hidden_from_autograd}
  mutations_under_no_grad_or_inference_mode={mutations_under_no_grad_or_inference_mode}
  mutation_inductor_storage_resize={mutation_inductor_storage_resize}
  requires_grad={requires_grad}"""
    return in_graph
