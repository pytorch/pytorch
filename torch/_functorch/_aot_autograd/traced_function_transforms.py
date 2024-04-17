"""
This module is responsible for transforming functions to be traced into a form
that is easier for the downstream infra (e.g. Autograd, FX, AOTAutograd analysis)
to handle.

It does so by:
1. functionalization (including RNG functionalzation)
2. creating a joint graph when required
3. transforming mutations into extra outputs
4. dispatching subclasses
"""

import warnings
from contextlib import nullcontext
from functools import wraps
from typing import Any, Callable, List, Tuple, Union
from unittest.mock import patch

import torch
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch import Tensor
from torch._decomp.decompositions_for_rng import PhiloxStateTracker
from torch._guards import detect_fake_mode
from torch._prims_common import CUDARngStateHelper
from torch.fx.experimental.symbolic_shapes import (
    definitely_false,
    rebind_unbacked,
    sym_eq,
)
from torch.nn.utils import stateless

from .. import config
from .collect_metadata_analysis import run_functionalized_fw_and_collect_metadata
from .functional_utils import (
    from_fun,
    has_data_mutation,
    has_metadata_mutation,
    is_fun,
    sync_functional_tensor,
    to_fun,
)
from .logging_utils import setup_stacktrace_preservation_hooks
from .schemas import (
    AOTConfig,
    MutationType,
    OutputType,
    SubclassMeta,
    SubclassTracingInfo,
    ViewAndMutationMeta,
)
from .subclass_utils import (
    create_subclass_meta,
    requires_subclass_dispatch,
    unwrap_tensor_subclasses,
    wrap_tensor_subclasses_maybe_joint,
)
from .utils import maybe_to_fresh_input


# This function returns a new function that returns mutated inputs as outputs.
# if keep_data_input_mutations is set, then we assume that data-only mutations
# will be left in the graph, and we only return metadata-mutated inputs as outputs.
def fn_input_mutations_to_outputs(
    fn: Callable,
    meta: ViewAndMutationMeta,
    keep_data_input_mutations: bool,
) -> Any:
    @wraps(fn)
    def inner_fn(*args):
        outs = fn(*args)
        assert len(meta.output_info) == len(outs)
        # The compiled fw will return mutated input tensors, *including* metadata-only mutation.
        # However, if keep_data_input_mutations is set, the compiled fw only needs to return metadata-mutated inputs.
        # (because data-only input mutations are handled directly in the compiled graph)
        mutated_inputs_to_return = [
            x for (i, x) in enumerate(args) if i in meta.mutated_inp_runtime_indices
        ]
        return *mutated_inputs_to_return, *outs

    return inner_fn


# This function takes in a fn with external aliasing and mutation,
# and returns a new fn with no external aliasing and mutation,
# as needed for autograd.
# The main transformations are:
# - Return mutated inputs as extra outputs
# - Clone mutated inputs that require gradients,
#   because autograd will require us to pass the pre-mutated inputs into autograd.grad
# - Return intermediate bases of outputs as additional outputs,
#   needed to appease autograd.Function
# The new function returns:
# (1) The updated outputs
# (2) A boolean mask of len(new_fn_outputs),
#     that can be used to tell autograd.grad which outputs should get tangents
#     if we trace the backward.
def fn_prepped_for_autograd(
    fn: Callable,
    meta: ViewAndMutationMeta,
) -> Any:
    @wraps(fn)
    def inner_fn(*args):
        args_maybe_cloned = [
            maybe_to_fresh_input(i, t, meta) for i, t in enumerate(args)
        ]

        outs = fn(*args_maybe_cloned)
        assert isinstance(outs, (tuple, list))
        outs = list(outs)
        assert len(meta.output_info) == len(outs)

        mutated_inputs_to_return = [
            x
            for (i, x) in enumerate(args_maybe_cloned)
            if i in meta.mutated_inp_runtime_indices
        ]

        intermediate_bases = []
        for i, (o, info) in enumerate(zip(outs, meta.output_info)):
            if info.output_type == OutputType.alias_of_intermediate_save_as_output:
                intermediate_bases.append(o._base)

        assert meta.num_intermediate_bases == len(intermediate_bases)

        # the compiled forward should return (mutated_inputs, user_outs, intermediate_bases)
        fw_outs_to_return = *mutated_inputs_to_return, *outs, *intermediate_bases

        # Also return a boolean mask specifying which outputs to this function will be used as tangents
        mutated_inputs_grad_mask = [
            meta.input_info[meta.mutated_inp_runtime_indices[i]].mutates_data
            and meta.input_info[meta.mutated_inp_runtime_indices[i]].requires_grad
            for (i, x) in enumerate(mutated_inputs_to_return)
        ]

        # Pass any (non-aliased) outputs in as tangents, since they'll be returned as outputs in the fw
        # For outputs that are aliases of intermediates, we will have returned the output's _base as an output in the graph instead,
        # which we *should* send to grad()
        output_grad_mask = [
            meta.output_info[i].output_type
            in [
                OutputType.non_alias,
                OutputType.unsafe_view_alias,
                OutputType.custom_function_view,
            ]
            # Also, only tensor outputs should participate in the backward
            # (in particular, Symint outputs in the forward graph shouldn't get tangents)
            and issubclass(meta.output_info[i].raw_type, Tensor)
            and meta.output_info[i].requires_grad
            for (i, x) in enumerate(outs)
        ]

        intermediate_base_grad_mask = [True for _ in range(len(intermediate_bases))]

        out_grad_mask = (
            mutated_inputs_grad_mask + output_grad_mask + intermediate_base_grad_mask
        )
        assert len(out_grad_mask) == len(fw_outs_to_return)

        # Take care to grab and sync the updated inputs from primals_after_cloning (the inputs we actually mutate!)
        # and not primals (the preserved inputs, pre-mutation, that we pass to grad())
        # This is annoying: our joint function needs to be aware of functionalization
        # (syncing mutated inputs before calling autograd.grad())
        # In theory, we could make the autograd engine do this automatically, although that probably isn't any cleaner.
        for arg in args_maybe_cloned:
            if not isinstance(arg, Tensor):
                continue
            sync_functional_tensor(arg)

        return fw_outs_to_return, out_grad_mask

    return inner_fn


# Given a fn, computes the joint.
# NOTE: fn is expects the following behavior:
# (1) fn() needs to return a tuple of (outs, mask),
#     where `mask` tells us which outputs are meant to have tangents.
#     we don't know this info automatically, because we don't actually want to blindly
#     compute tangents for every output that requires grad.
#     Specifically, outputs that alias inputs won't participate in the backward and get tangents.
# (2) fn() cannot mutate any inputs that require gradient.
#     otherwise, when we compute autograd.grad(), we will not take those input mutations into account
#     (the way this is handled is that we ensure any inputs that normally get mutated are cloned first)
def create_joint(fn: Callable, *, aot_config: AOTConfig) -> Any:
    def inner_fn(primals: List[Any], tangents: List[Any]):
        outs, tangent_mask = fn(*primals)
        assert len(tangent_mask) == len(outs)
        outs_to_grad = [
            o for needs_tangent, o in zip(tangent_mask, outs) if needs_tangent
        ]
        assert len(outs_to_grad) == len(tangents)

        # Get the inputs that need gradients
        grad_primals = []
        inputs_needs_grads = []
        # Note that we're not using primals here,
        # being carefully not to pass any mutated inputs into autograd.grad()
        for p in primals:
            is_grad_tensor = isinstance(p, Tensor) and p.requires_grad
            inputs_needs_grads.append(is_grad_tensor)
            if is_grad_tensor:
                grad_primals.append(p)

        # Get the outputs that need gradients
        needed_outs = []
        needed_tangents = []
        for out, tangent in zip(outs_to_grad, tangents):
            if isinstance(out, Tensor) and out.requires_grad:
                # A bit sketchy, but fixes e.g. test_aot_autograd_exhaustive_matmul_cpu_float32
                # The issue is that we are sensitive to decomps that don't accurately maintain
                # their output's _base.shape compared to eager mode, and this helps mitigate a bit.
                # The not definitely_false is also sketchy; if unbacked
                # symints are involved, we're just going to assume that the
                # decomps setup the base shape correctly
                needed_outs.append(
                    out
                    if not definitely_false(sym_eq(out.shape, tangent.shape))
                    else out.view(tangent.shape)
                )
                needed_tangents.append(tangent)

        setup_stacktrace_preservation_hooks([out.grad_fn for out in needed_outs])

        if config.functionalize_rng_ops:
            PhiloxStateTracker.mark_beginning_of_backward()
        backward_out: Tuple[Tensor, ...] = tuple()
        # Call the backwards pass
        if grad_primals:
            with fx_traceback.preserve_node_meta():
                # for full graph export, we always export a joint graph where we assume no tangents are needed.
                if aot_config.no_tangents:
                    assert len(needed_tangents) == 1 and needed_tangents[0].numel() == 1
                    backward_out = torch.autograd.grad(
                        needed_outs,
                        grad_primals,
                        allow_unused=True,
                    )
                else:
                    backward_out = torch.autograd.grad(
                        needed_outs,
                        grad_primals,
                        grad_outputs=needed_tangents,
                        allow_unused=True,
                    )
        backward_out_iter = iter(backward_out)
        return outs, [
            next(backward_out_iter) if i else None for i in inputs_needs_grads
        ]

    def inner_fn_with_anomaly(*args):
        with fx_traceback.preserve_node_meta(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Anomaly Detection has been enabled.")
            with torch.autograd.detect_anomaly(check_nan=False):
                return inner_fn(*args)

    return inner_fn_with_anomaly


def create_functionalized_rng_ops_wrapper(func, args, trace_joint=True) -> Any:
    # Functionalization of rng ops changes the calling convention of the joint graph.
    # It goes from (primals, tangents) to (seed, offset, primals, tangents)
    # At runtime, we pass on the current seed and offset. This is hidden from
    # the user.
    fake_mode = detect_fake_mode()
    if fake_mode is None:
        fake_mode = nullcontext()

    def override_get_rng_state(device: Union[int, str, torch.device] = "cuda"):
        out = PhiloxStateTracker.get_state_as_tensor()
        return out

    def override_set_rng_state(x, device: Union[int, str, torch.device] = "cuda"):
        PhiloxStateTracker.set_state_from_tensor(x)

    def append_rng_offsets(args):
        if trace_joint:
            # args signature before: Tuple(fwd_outputs), Tuple(bwd_outputs)
            # args signature after: Tuple(fwd_outputs, new_fwd_rng_offset), Tuple(bwd_offset, new_bwd_rng_offset)
            return (
                (*args[0], PhiloxStateTracker.get_updated_fwd_offset()),
                (*args[1], PhiloxStateTracker.get_updated_bwd_offset()),
            )
        else:
            # args signature before: Tuple(fwd_outputs)
            # args signature after: Tuple(fwd_outputs, new_fwd_rng_offset)
            return (*args, PhiloxStateTracker.get_updated_fwd_offset())

    def traced_joint(
        primals, tangents, fwd_seed, fwd_base_offset, bwd_seed, bwd_base_offset
    ):
        with patch("torch.cuda.get_rng_state", override_get_rng_state), patch(
            "torch.cuda.set_rng_state", override_set_rng_state
        ):
            return append_rng_offsets(func(primals, tangents))

    def traced_forward(*primals_fwd_seed_fwd_base_offset):
        # The signature is (*primals, seed, offset)
        with patch("torch.cuda.get_rng_state", override_get_rng_state), patch(
            "torch.cuda.set_rng_state", override_set_rng_state
        ):
            return append_rng_offsets(func(*primals_fwd_seed_fwd_base_offset[:-2]))

    if trace_joint:
        # Get the current seed and offset to setup tracing.
        fwd_seed, fwd_base_offset = CUDARngStateHelper.get_torch_state_as_tuple(
            fake_mode
        )
        bwd_seed, bwd_base_offset = CUDARngStateHelper.get_torch_state_as_tuple(
            fake_mode
        )
        PhiloxStateTracker.record_state(fwd_seed, fwd_base_offset, "forward")
        PhiloxStateTracker.record_state(bwd_seed, bwd_base_offset, "backward")
        return traced_joint, (
            *args,
            fwd_seed,
            fwd_base_offset,
            bwd_seed,
            bwd_base_offset,
        )
    else:
        # Get the current seed and offset to setup tracing.
        fwd_seed, fwd_base_offset = CUDARngStateHelper.get_torch_state_as_tuple(
            fake_mode
        )
        PhiloxStateTracker.record_state(fwd_seed, fwd_base_offset, "forward")
        return traced_forward, (*args, fwd_seed, fwd_base_offset)


# This creates the final function that we want to trace using make_fx(),
# in both aot_dispatch_autograd and aot_dispatch_base.
# Preconditions:
# - fn corresponds to the user's fw function
# - fn arguments have been flattened, duplicate arguments have been handled
# - In the returned function, the "primals" arguments *includes* synthetic bases.
# This function does the work of functionalizing the input function,
# and performing copy_() calls at the end of the function if `keep_input_mutations` is set.
# The function returned has signature that is either:
# (1) "traced_fn(primals: List[Any])" if trace_joint is False
# (2) "traced_fn(primals: List[Any], tangents: List[Any])" if trace_joint is True
# Returns a new (functionalized) function, and updated arguments to call it with.
def create_functionalized_fn(
    fn,
    args,
    *,
    meta: ViewAndMutationMeta,
    aot_config: AOTConfig,
    trace_joint: bool,
) -> Any:
    @wraps(fn)
    def _functionalized_f_helper(*args):
        # See Note [Disabling Functionalize TLS Above Python Functionalization]
        disable_above = torch._C._ExcludeDispatchKeyGuard(
            torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
        )

        # See Note [Side-Effectful Tokens in AOTAutograd]
        if trace_joint:
            assert (
                isinstance(args, tuple)
                and len(args) == 2
                and isinstance(args[0], (list, tuple))
            )
            tokens = args[0][: len(meta.tokens)]
            actual_args = args[0][len(meta.tokens) :]
            args = (actual_args, args[1])
        else:
            tokens = args[: len(meta.tokens)]
            args = args[len(meta.tokens) :]
        assert all(token.numel() == 0 for token in tokens)

        with disable_above:
            # Wrap inputs into functional wrappers
            f_args = pytree.tree_map(to_fun, args)
            f_tokens = pytree.tree_map(to_fun, tokens)

            # Populate the current FunctionalTensorMode with the tokens per
            # operator. See Note [FunctionalTensorMode is Stateful]
            functional_tensor_mode = (
                torch.utils._python_dispatch._detect_functional_mode()
            )
            assert functional_tensor_mode is not None
            for i, k in enumerate(meta.tokens.keys()):
                functional_tensor_mode._tokens[k] = f_tokens[i]

            # Run the joint
            f_outs = fn(*f_args)

            # Return both the tokens and the outputs
            # See Note [Side-Effectful Tokens in AOTAutograd]
            f_outs = (*functional_tensor_mode._tokens.values(), *f_outs)

        if trace_joint:
            # We support a limited amount of mutation of graph inputs during the backward pass.
            # (This is used e.g. by Float8, which needs to update buffers during the backward pass)
            # Here, we perform extra checks for primals that were mutated in the **backward**
            # We're doing the checks here instead of doing them with the rest of the input mutation handling because:
            # - We need to detect inputs that were mutated in the backward **separately** from mutations that happened
            #   during the forward, because the handling is different: some input mutations from the the forward
            #   can be only handled in a fw-only runtime epilogue, and in theory if we wanted to handle those same
            #   types of mutations in the backward we would need a bw-only runtime epilogue.
            # - We could in theory have our analysis pass differentiate mutations in the fw from mutations in
            #   the bw by running our analysis first on the fw-only graph, and then on the joint graph. This would
            #   require an extra round of tracing though, so it's more efficient to do in-line here.
            assert (
                isinstance(args, tuple)
                and len(args) == 2
                and isinstance(args[0], (list, tuple))
            )
            # Only look at mutations that happened to forward inputs (e.g. fw buffers that were saved for bw)
            primals_before = args[0]
            primals_after = pytree.tree_map(from_fun, f_args[0])
            for f_inpt, before, after, inpt_info in zip(
                f_args[0], primals_before, primals_after, meta.input_info
            ):
                # Ban metadata mutations on fw inputs during the bw
                if not inpt_info.mutates_metadata:
                    assert not has_metadata_mutation(
                        f_inpt, before, check_only_storage_mutation=False
                    ), "Found a graph input that had its metadata mutated in the backward. This is not supported"
                # Allow data mutations on fw inputs during the bw, but only if they do not require grad
                # So we can guarantee that we can keep the mutations in the graph
                if (
                    has_data_mutation(f_inpt)
                    and not inpt_info.mutates_data
                    and not inpt_info.mutates_storage_metadata
                ):
                    assert (
                        not inpt_info.requires_grad
                    ), "Found a graph input that requires_grad and was mutated in the backward. This is not supported"
                    # Otherwise, put the mutation in the graph
                    before.copy_(after)
            # Now that we covered mutations to *forward* inputs during the backward,
            # we also need to cover mutations to *backward-only* inputs during the backward (e.g. mutation to a grad_out).
            # Today, we will just error in all cases of this happening unless someone needs us to support it.
            tangents_before = args[1]
            tangents_after = pytree.tree_map(from_fun, f_args[1])
            for f_inpt, before, after in zip(
                f_args[1], tangents_before, tangents_after
            ):
                assert not has_metadata_mutation(
                    f_inpt, before, check_only_storage_mutation=False
                ) and not has_data_mutation(
                    f_inpt
                ), "Found an input to the backward that was mutated during the backward pass. This is not supported"

        if aot_config.keep_inference_input_mutations:
            # Note: This is a bit annoying. There's a layering issue here, where:
            # (1) functionalization needs to operate on **synthetic base** inputs, before unpacking them into the "real" inputs.
            # (2) For keep_input_mutations, we support tracing a call to copy_() directly on mutated inputs.
            #     However, we **only** want to support this for inputs that have data-only (and no metadata) mutations,
            #     because inductor (and backends in generally) would prefer not to see these (e.g. as_strided_(), resize_()).
            #     This makes it pretty difficult for this logic to operate on synthetic bases.
            # (3) In addition, there are cases where it's significantly cheaper to perform the copy on the individual
            #     (unpacked) input aliases, instead of the synthetic base.
            # Example case where (3) could be important:
            #
            #     def f(x, y):
            #         x.mul_(2)
            #         y.mul_(3)
            #         return x, y
            #    a = torch.ones(1'000'000)
            #    x, y = out(a[0:9], a[1:10])
            #
            # It would be much better to add copy_() calls into the graph for the two tiny slices, instead of materializing
            # a giant "updated synthetic base" and copying into a's entire storage.
            #
            # For now, we are pessimistically not performing the optimization from (3);
            # we will materialize an "updated" synthetic base, and copy it back to the synthetic input base.
            # This allows us to factor aot autograd much more nicely, since only one area of the code needs to worry
            # about synthetic bases.
            for i, (inpt_old, inpt_f) in enumerate(
                zip(args, f_args) if not trace_joint else zip(args[0], f_args[0])
            ):
                if not isinstance(inpt_f, torch.Tensor):
                    continue
                assert is_fun(inpt_f)
                inpt_new = from_fun(inpt_f)
                if meta.input_info[i].mutation_type == MutationType.MUTATED_IN_GRAPH:
                    # See Note [set_() Input Mutations in AOTAutograd]
                    # all mutations on the input must be under no_grad, so it is safe to put in the graph
                    # Here, we're saying that if an input experienced a set call, inp.set_(other),
                    # then we can effectively not have to worry about whether its data was mutated.
                    # There are 3 cases:
                    # (1) We mutate inp *after* the set_() call. other is a graph intermediate.
                    #     In this case, we're not really mutating the input storage of "inp";
                    #     we're mutating the storage of an intermdiate value (other),
                    #     and slamming that storage into the input tensor. So no data mutation is necessary.
                    # (2) We mutate inp *after* the set_() call. other is a graph *input*.
                    #     In this case, the data mutation will be properly handled in the runtime
                    #     epilogue during the processing of "other"
                    # (3) We mutate inp *before* the set_() call.
                    #     This case is *not* currently handled.
                    if meta.input_info[i].mutates_storage_metadata:
                        with torch.no_grad():
                            inpt_old.set_(inpt_new)

                    # We found an input that had a (data-only) mutation.
                    # Since keep_input_mutations is set, we need to faithfully apply a copy_()
                    # so the compiler will see the input mutation in the graph.
                    if (
                        meta.input_info[i].mutates_data
                        and meta.input_info[i].mutations_hidden_from_autograd
                    ):
                        # Hidden from autograd = run under no_grad, **and** don't bump VC
                        with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(
                            inpt_old
                        ):
                            inpt_old.copy_(inpt_new)
                    elif (
                        meta.input_info[i].mutates_data
                        and meta.input_info[i].mutations_under_no_grad_or_inference_mode
                    ):
                        # Under no_grad = run under no_grad (we still bump the VC though)
                        # (inference_mode will also bump the VC, as long as the tensor in question
                        # was created outside of inference_mode)
                        with torch.no_grad():
                            inpt_old.copy_(inpt_new)
                    elif meta.input_info[i].mutates_data:
                        inpt_old.copy_(inpt_new)

            # When an output tensor is a functionalized mutated input, and we
            # were able to move the mutation in to the graph then we can return
            # the mutated input directly. This prevents duplicating the
            # tensors contents.
            flat_outs, outs_spec = pytree.tree_flatten(f_outs)
            flat_outs = [from_fun(o) for o in flat_outs]
            num_outs = len(meta.output_info)

            for i, outp in enumerate(flat_outs[:num_outs]):
                info = meta.output_info[i]
                if info.output_type != OutputType.is_input:
                    continue

                assert info.base_idx is not None
                if (
                    meta.input_info[info.base_idx].mutation_type
                    == MutationType.MUTATED_IN_GRAPH
                ):
                    fw_args = args[0] if trace_joint else args
                    flat_outs[i] = fw_args[info.base_idx]
            return pytree.tree_unflatten(flat_outs, outs_spec)

        return pytree.tree_map(from_fun, f_outs)

    # Kinda annoying, but needed to make sure that the fx graph we trace out has "primals"
    # and "tangents" as its input names (which are special-cased by the partitioner)
    # TODO (tmanlaibaatar) revisit this if we ever need to turn on non-strict joint graph export
    def joint_helper(primals, tangents):
        return _functionalized_f_helper(primals, tangents)

    helper = joint_helper if trace_joint else _functionalized_f_helper
    if config.functionalize_rng_ops:
        # Setup the wrapper for functionalization of rng ops
        helper, args = create_functionalized_rng_ops_wrapper(helper, args, trace_joint)

    # Additionally pass in tokens as inputs
    # See Note [Side-Effectful Tokens in AOTAutograd]
    additional_token_inputs = [torch.tensor([])] * len(meta.tokens)
    if trace_joint:
        args = ([*additional_token_inputs, *args[0]], *args[1:])
    else:
        args = [*additional_token_inputs, *args]

    return helper, args


# Given a function operating on Subclass -> Subclass, returns an function that operates on Tensor -> Tensor
# Also returns:
# - the new set of arguments to pass into this function (now that tensor subclasses have been eliminated)
# - the updated ViewAndMutationMeta for this dense -> dense function.
# The other important arguments are:
# - flat_fn_maybe_joint: when is_joint_structure=True, this is the joint fw-bw function.
#                        when is_joint_structure=False, this is just the forward function.
# - fw_only: this is *always* the forward-only function.
#   Why do we need this? We need to collect updated ViewAndMutationMeta on our new dense -> dense functions.
#   In particular, we need this to tell the partitioner how many dense forward outputs there are.
def aot_dispatch_subclass(
    flat_fn_maybe_joint,
    args: List[Any],
    *,
    is_joint_structure: bool,
    meta: ViewAndMutationMeta,
    fw_only: Callable,
) -> SubclassTracingInfo:
    # Skip logic if we don't need to trace through any subclasses
    req_subclass_dispatch = requires_subclass_dispatch(args, meta)
    if not req_subclass_dispatch:
        return SubclassTracingInfo(
            plain_tensor_trace_fn=flat_fn_maybe_joint,
            plain_tensor_args=args,
            maybe_subclass_meta=None,
        )

    # TODO: add subclass guards (later PR).

    # What's going on here? We need to compute subclass metadata about the outputs of the joint (grad_inputs).
    # Annoying: we don't know the grad input metas until we're in the middle of tracing the joint,
    # so we set it later, while we're tracing the joint (see inner_fn() below).
    # Another option would be to run our run_functionalized_fw_and_collect_metadata() function
    # directly on the joint, but this would hurt compile time (adding yet another pass through the joint).
    subclass_meta = SubclassMeta()

    def inner_fn(fn, args, *, use_trace_joint: bool):
        # Step 1: wrap tensor inputs into subclasses if necessary
        all_args = wrap_tensor_subclasses_maybe_joint(
            args, is_joint_structure=use_trace_joint, meta=meta
        )

        # Step 2: call the inner function, with our (maybe subclass) inputs
        wrapped_outs = fn(*all_args)

        if use_trace_joint:
            # See Note: [Computing Subclass Metadata about grad_inputs]
            # We also stash subclass info on our grad_inputs, if we're tracing the joint.
            nonlocal subclass_meta
            assert isinstance(wrapped_outs, tuple) and len(wrapped_outs) == 2
            # Don't need fw outs since we already have subclass metadata on them
            grad_inputs = wrapped_outs[1]
            subclass_meta.grad_input_metas = create_subclass_meta(grad_inputs)

        # Step 3: Unwrap any subclass outputs back into dense tensors
        unwrapped_outs = unwrap_tensor_subclasses(
            wrapped_outs, is_joint_structure=use_trace_joint
        )
        return unwrapped_outs

    def joint_fn(primals, tangents):
        return inner_fn(flat_fn_maybe_joint, (primals, tangents), use_trace_joint=True)

    def fw_fn(*primals):
        return inner_fn(flat_fn_maybe_joint, primals, use_trace_joint=False)

    def metadata_fn(*primals):
        return inner_fn(fw_only, primals, use_trace_joint=False)

    args_unwrapped = unwrap_tensor_subclasses(
        args, is_joint_structure=is_joint_structure
    )

    if is_joint_structure:
        primals_unwrapped = args_unwrapped[0]
        fn_to_trace = joint_fn
    else:
        primals_unwrapped = args_unwrapped
        fn_to_trace = fw_fn

    # Note: [Partitioner handling for Subclasses, Part 1]
    # The way the partitioner works is that:
    # (1) we pass is a single graph containing the joint fw/bw,
    #     where the # of graph outputs corresponds to # fw_outputs + # grad_inputs
    # (2) The partitioner accepts an arguments, num_fwd_outputs,
    #     and assumes that the first "num_fwd_outputs" graph outputs correspond
    #     to outputs of the forward graph.
    # How do tensor subclasses enter the picture?
    # the num_fwd_outputs in the final graph is actually non-trivial to compute,
    # because it can be influenced by input mutations and intermediate bases.
    # So we compute it by inspecting the current ViewAndMutationMeta object.
    # However, the original ViewAndMutationMeta that we computed was created
    # on the subclass -> subclass graph,
    # which can have a different number of outputs than the dense -> dense graph.
    # That's why we createa a fresh metadata object on the dense -> dense function here,
    # and plumb it back up to the partitioner.
    # See Note: [Partitioner handling for Subclasses, Part 2] for more info.
    meta_updated = run_functionalized_fw_and_collect_metadata(
        metadata_fn,
        keep_input_mutations=meta.keep_input_mutations,
        is_train=meta.is_train,
    )(*primals_unwrapped)

    subclass_meta.fw_metadata = meta_updated

    return SubclassTracingInfo(
        plain_tensor_trace_fn=fn_to_trace,
        plain_tensor_args=args_unwrapped,
        maybe_subclass_meta=subclass_meta,
    )


class PropagateUnbackedSymInts(torch.fx.Interpreter):
    def run_node(self, n: torch.fx.Node):
        result = super().run_node(n)
        rebind_unbacked(detect_fake_mode().shape_env, n, result)
        return result


def create_functional_call(mod, params_spec, params_len, store_orig_mod=False):
    # Redundant with dynamo, but worth having in case this gets invoked elsewhere.
    # https://github.com/pytorch/pytorch/issues/103569

    def functional_call(*args, **kwargs):
        with stateless._reparametrize_module(
            mod, pytree.tree_unflatten(args[:params_len], params_spec)
        ):
            if isinstance(mod, torch.fx.GraphModule):
                with fx_traceback.preserve_node_meta(), warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", "Anomaly Detection has been enabled."
                    )
                    with torch.autograd.detect_anomaly(check_nan=False):
                        out = PropagateUnbackedSymInts(mod).run(
                            *args[params_len:], **kwargs
                        )
            else:
                out = mod(*args[params_len:], **kwargs)

        if not isinstance(out, (tuple, list)):
            raise RuntimeError(
                "Graph output must be a tuple(). This is so that we can avoid "
                "pytree processing of the outputs. Please change the module to "
                "have tuple outputs or use aot_module instead."
            )
        return out

    # Note [Preserving the nn module stack metadata during export non-strict mode]
    # This path is currently only used by the non-strict export flow,
    # where we cannot rely on dynamo to preserve nn stack metadata in our captured graph.
    # Instead, we stash the original user nn module here, and rely on `make_fx` to grab
    # this stashed module and use it to track nn module stack metadata
    if store_orig_mod and not hasattr(functional_call, "_orig_mod"):
        functional_call._orig_mod = mod  # type: ignore[attr-defined]

    return functional_call
