# mypy: allow-untyped-defs
"""
This module is one of the analysis modules - it takes as input a function or graph
and some preexisting properties, and returns some data that is useful for deciding
how to further proceed with compilation or construct runtime wrappers.

In particular, the analysis here constructs view and mutation metadata from running
a functionalized version of the graph under compilation.
"""

import collections
import logging
from functools import wraps
from typing import Callable, DefaultDict, Dict, List

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._guards import detect_fake_mode
from torch._subclasses.functional_tensor import FunctionalTensor, FunctionalTensorMode
from torch._subclasses.meta_utils import safe_is_leaf
from torch.fx.experimental.symbolic_shapes import is_concrete_int
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    transform_subclass,
)
from .functional_utils import (
    are_all_mutations_hidden_from_autograd,
    are_all_mutations_under_no_grad_or_inference_mode,
    from_fun,
    has_data_mutation,
    has_metadata_mutation,
    has_same_metadata,
    to_fun,
    was_inductor_storage_resized,
)
from .schemas import (
    FunctionalTensorMetadataEq,
    InputAliasInfo,
    MutationType,
    OutputAliasInfo,
    OutputType,
    ViewAndMutationMeta,
)
from .subclass_utils import create_subclass_meta

from .utils import _get_autocast_states, KNOWN_TYPES, strict_zip

zip = strict_zip

log = logging.getLogger(__name__)


# Note [Tangents must be contiguous]
# We force tangents to be contiguous today.
# The idea is that we are technically making a guess about the strides of our tangents,
# while we trace out the joint.
# Today, we force this guess to be correct by additioanlly calling contiguous()
# on all tangents at runtime.
# In the future, you could imagine lifting this restriction, since these contiguous()
# calls can have noticeable perf overhead depending on the model.
def coerce_tangent(x):
    if not isinstance(x, Tensor):
        return x
    out = x.detach().contiguous()
    # Note [Tangents must be contiguous, Part 2]
    # In the same way that "what strides do we assigns to our tangents" is a question
    # that we can not answer (and therefore have to guess) as we trace the backward ahead-of-time,
    # The same applies to any tensor subclass metadata, when we have tangents that are subclasses.
    # To handle this situation, we have two new methods that a tensor subclass can implement:
    # (1) __coerce_tangent_metadata__(self)
    #     Given a subclass with "non-standard" metadata, turn it into a new subclass with "normal" metadata.
    #     The main example here is a DTensor with the "_Partial" placement.
    #     If we have a forward output with a _Partial placement, and corresponding tangent
    #     with a Replicate/Shard placement, we have no way to convert the tangent "back" to a _Partial placement.
    #     This method lets us avoid the problem entirely by allowing subclasses to ensure that we can never
    #     have a tangent with "problematic" metadata, that we cannot convert to.
    # (1) __coerce_same_metadata_as_tangent__(self, metadata)
    #     Given a subclass, and a target differing metadata,
    #     convert self to have the same metadata as the target.
    #     With DTensor being the main example, we can use this to convert a DTensor with a Replicate()
    #     placement into one with a Shard() placement, in the case that we "guessed wrong",
    #     and traced tangents with a Shard() placement at compile time.
    #
    if is_traceable_wrapper_subclass(out) and hasattr(
        out, "__coerce_tangent_metadata__"
    ):
        out = out.__coerce_tangent_metadata__()
    # It's possible to have a subclass that advertises as contiguous,
    # but has noncontiguous inner tensors.
    # Force these to be conntiguous too
    if is_traceable_wrapper_subclass(out):
        for attr in out.__tensor_flatten__()[0]:  # type: ignore[attr-defined]
            elem = getattr(out, attr)
            if not elem.is_contiguous():
                elem_contig = elem.contiguous()
                setattr(out, attr, elem_contig)
    return out


# This is a version of functionalization that is specifically designed
# for the AOTAutograd use case.
#
# Unlike functorch's variant, this doesn't use the functorch level system,
# instead it directly uses PyTorch's conventional dispatcher to hit the
# functionalization key.  In particular, this means that FunctionalTensorWrapper
# can have autograd data stored directly on it.
#
# In typical AOTAutograd usage, the dispatch key order will look like:
#
#   Autograd - Functionalization ~~~~> Proxy Mode - Fake Tensor
#       outer tensor                        inner tensor
#
# Returns:
# - ViewAndMutationMeta, telling us metadata about the inputs and outputs, and
#   The list of outputs from the forward, but **only** the outputs that we need
#   to pass in as tangents into the backward.
#   Specifically, aliased outputs from the forward get regenerated, and don't participate
#   in the compiled backward function.
def run_functionalized_fw_and_collect_metadata(
    f,
    *,
    keep_input_mutations: bool,
    # TODO: refactor to kill this flag
    is_train: bool = False,
    pre_dispatch: bool = False,
) -> Callable[..., ViewAndMutationMeta]:
    memo: Dict[Tensor, Tensor] = {}

    def _to_fun(t):
        if isinstance(t, Tensor):
            if t in memo:
                return memo[t]
            r = to_fun(t)
            memo[t] = r
            return r
        else:
            return t

    @wraps(f)
    def inner(*flat_args):
        # This function is meant to be run with the forward, which expects a flat list of tensor/symint/other args.
        assert all(isinstance(a, tuple(KNOWN_TYPES)) for a in flat_args)

        input_info: List[InputAliasInfo] = []
        output_info: List[OutputAliasInfo] = []

        prior_grad_enabled = torch.is_grad_enabled()
        prior_autocast_states = _get_autocast_states()

        # See Note [Disabling Functionalize TLS Above Python Functionalization]
        disable_above = torch._C._ExcludeDispatchKeyGuard(
            torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
        )

        # It doesn't matter if we run this under predispatch or not because it is
        # only for figuring out metadata
        mode = FunctionalTensorMode(_allow_token_discovery=True)
        with disable_above, mode:
            # precondition: The passed in function already handles unflattening inputs + flattening outputs
            flat_f_args = pytree.tree_map(_to_fun, flat_args)
            flat_f_outs = f(*flat_f_args)
            # We didn't do any tracing, so we don't need to process the
            # unbacked symbols, they will just disappear into the ether.
            # Also, prevent memoization from applying.
            if (fake_mode := detect_fake_mode()) and (shape_env := fake_mode.shape_env):
                shape_env.pending_fresh_unbacked_symbols.clear()
                fake_mode.epoch += 1

        if prior_autocast_states != _get_autocast_states():
            raise RuntimeError(
                "AOTAutograd does not support tracing graphs that mutate the autocast state. "
                "Dynamo will only insert autocast context managers (e.g. with torch.autocast(..)) into the graph, "
                "which will unwind all of their mutations to autocast state before the graph exits. "
                "If you encounter this error while using torch.compile, please file a bug."
            )

        # Inspect the state of the input tensor functional wrapper to detect input mutation info
        # If inp[i] has a metadata-only mutation, then maybe_inputs_with_mutated_metadata[i] contains the updated version
        for i, (arg, f_arg) in enumerate(zip(flat_args, flat_f_args)):
            # NB: Mutation of non-contiguous tensor subclass input can result in a mismatch in
            # strides between the functionalized arg inner tensors and non-functionalized arg inner
            # tensors. This is a problem as the inner tensor stride change may not be reflected
            # correctly in the outer tensor, so disallow this for now.
            mutates_data = has_data_mutation(f_arg)
            if (
                mutates_data
                and not arg.is_contiguous()
                and is_traceable_wrapper_subclass(arg)
            ):
                raise RuntimeError(
                    "Mutations on non-contiguous inputs are currently not allowed on "
                    "tensor subclasses"
                )

            if not isinstance(arg, Tensor):
                new_arg = arg
            else:
                new_arg = from_fun(f_arg)
            mutates_metadata = has_metadata_mutation(
                f_arg, arg, check_only_storage_mutation=False
            )
            if mutates_metadata and is_traceable_wrapper_subclass(arg):
                raise RuntimeError(
                    "Metadata mutations are currently not allowed on tensor subclasses"
                )
            mutates_storage_metadata = has_metadata_mutation(
                f_arg, arg, check_only_storage_mutation=True
            )
            mutations_hidden_from_autograd = are_all_mutations_hidden_from_autograd(
                f_arg
            )
            mutations_under_no_grad_or_inference_mode = (
                mutates_data
                and are_all_mutations_under_no_grad_or_inference_mode(f_arg)
            )
            mutation_inductor_storage_resize = was_inductor_storage_resized(f_arg)

            if mutates_storage_metadata:
                mutates_data = False

            requires_grad = isinstance(f_arg, torch.Tensor) and f_arg.requires_grad

            input_info.append(
                InputAliasInfo(
                    is_leaf=isinstance(arg, Tensor) and safe_is_leaf(arg),
                    mutates_data=mutates_data,
                    mutates_metadata=mutates_metadata,
                    mutations_hidden_from_autograd=mutations_hidden_from_autograd,
                    mutates_storage_metadata=mutates_storage_metadata,
                    mutations_under_no_grad_or_inference_mode=mutations_under_no_grad_or_inference_mode,
                    mutation_inductor_storage_resize=mutation_inductor_storage_resize,
                    requires_grad=requires_grad,
                    keep_input_mutations=keep_input_mutations,
                )
            )

        # If a function involves creating a tensor, and returning a view of it, such that its _base is the intermediate,
        # We need to make sure our graph returns the _base as a graph output, and we manually recreate the view
        # to return to the user. Why? The backend compiler is free to (incorrectly) not set requires_grad
        # on the base tensor, but we are obligated to properly set requires-gradness on the real output.

        inp_storage_refs = {
            StorageWeakRef(inpt.untyped_storage()): idx
            for idx, inpt in enumerate(flat_f_args)
            if isinstance(inpt, Tensor)
        }

        # We need inp tensor id's to be able to tell if an outputs **are** inputs.
        inp_tensor_ids = {id(inpt) for inpt in flat_f_args if isinstance(inpt, Tensor)}
        # We need output tensor id's to tell if any output._base` attributes **are** other outputs.
        # (This is also a dict because we need to know that output's index, so we can regenerate
        # the alias from it).
        out_tensor_ids = {id(o): i for i, o in enumerate(flat_f_outs)}

        # Keep track of which outputs alias other outputs
        out_tensor_alias_counts: DefaultDict = collections.defaultdict(int)
        # This tells us, for a given group of outputs that alias each other,
        # whether they e.g. all came from an unbind call
        num_aliased_tensors_that_are_multi_output_views: DefaultDict = (
            collections.defaultdict(int)
        )
        out_storage_to_tensors: DefaultDict = collections.defaultdict(set)
        curr_storage = None
        for o in flat_f_outs:
            if isinstance(o, torch.Tensor):
                curr_storage = StorageWeakRef(o.untyped_storage())
                out_tensor_alias_counts[curr_storage] += 1
                # Note: [AOTAutograd: differentiable outputs that alias each other from a multi-output view call]
                # This is an optimization on top of the "alias of intermediates" logic,
                # which you can read more about under Note [AOT Autograd: outputs aliasing inputs or intermediates!]
                #
                # Before describing the optimization: this is important for AOTAutograd to have good
                # perf around, multi-output views. HOWEVER:
                # - There is a more generic change to AOTAutograd that we'd like to make, that subsumes this case,
                #   around using pre-dispatch tracing to partition out a graph so we can faithfully replay all
                #   views without having to regenerate them at runtime.
                # - It's loosely described in this doc (more details will be added soon):
                #   https://docs.google.com/document/d/1DlfFq8TKbuAn2zyJxLfoW-X1qkkm5PLdHFtySo03QAk/edit
                # - Once that change lands, we should just rip out this "optimization", since:
                #   (1) It will be fully unnecessary
                #   (2) Although it is only a few lines of code, it is a bit difficult to reason about
                #       its correctness with the autograd engine in all cases.
                #
                #
                # What is this optimization? Consider the below case:
                # def f(x):
                #     intermediate = x.mul(2)
                #     # x and intermediate here require grad
                #     o1, o2, ... o10 = intermediate.unbind(-1)
                #     return intermediate, o1, o2, ... o10
                # Now, the "intermediate base" handling in AOTAutograd implies that we must do the following:
                #   (1) return "intermediate as an extra output of the compiled graph
                #   (2) regenerate each aliased output off of "intermediate", **outside** of the autograd.Function.
                # The reason AOTAutograd ordinarily does this is for safety: the autograd engine needs to know
                # that o1 through o10 are all aliased, and if we blindly return o1 through o10 from the autograd.Function,
                # this information will be hidden.
                # In particular, mutating one alias might require autograd to update autograd metadata on the other aliases
                # (like their grad_fn, for example, when the autograd engine needs to do view-replay).
                #
                # However, intermediate_base logic can be bad for backward performance (we sometimes generate
                # as_strided calls during the intermediate base logic, which can have a slow backward formula).
                # Is it possible to find a set of conditions where it is **safe** to hide the output aliasing from autograd?
                #
                # For a set of outputs of the graph that alias each other, o_1...o_k, consider:
                # (1) They came from the same multi-output view op, e.g. o_1, ..., o_k = intermediate.unbind(0)
                # (2) If there are any other aliases of o_1 through o_k (in the example above, intermediate),
                #     **at most** 1 can escape from the graph (e.g. there is not some other graph input/output
                #     o_other, that aliases these outputs)
                # (3) o_1...o_k all require_grad, they all share the same ._base, and their ._base requires grad.
                #     This condition is important because it's what causes slowness in the intermediate_base
                #     codepath of aot_autograd. Ordinarily, o_1...o_k would all get a grad_fn, and
                #     aot_autograd's view-replay might give each output an AsStridedBackward as its grad_fn.
                #     "K" AsStridedBackward calls will be *much* slower than a single UnbindBackward.
                # In this setup, is it possible to mutate one of the outputs o_i in a way that would affect the autograd meta
                # of the other aliases?
                #
                # Claim: No! Consider a few example (which I'm pretty sure cover all cases of mutation w.r.t. autograd):
                # (a) What happens if we mutate any of o_1 through o_k directly?
                #     Autograd raises an error:
                #     "RuntimeError: Output 0 of UnbindBackward0 is a view and is being modified inplace. This view is
                #      the output of a function that returns multiple views. Such functions do not allow the output
                #      views to be modified inplace. You should replace the inplace operation by an out-of-place one."
                # (b) What if we take a view of o_k and mutate it, o_k.view(o_k.shape).mul_(2)?
                #     Autograd raises the same error- the "multi-output-view"ness of an alias propagates to future views.
                # (c) What if we mutate o_k under no_grad?
                #     Autograd raises the same error
                # (d) What if we detach and mutate, e.g. o_k.detach().mul_(2)?
                #     Autograd allows this, *but* autograd updates all alias's grad_fn's to be error functions when accessed.
                #     Autograd raises the same error
                # (e) What if we try to mutate another alias of o_1...o_k, that was **not** created from a multi-output view?
                #     We promised that there is at most **one** such alias, e.g. intermediate in the example above.
                #     You can mutate intermediate, but in eager mode this will change the grad_fn of o_1...o_k
                #     to be error fn's.
                #     Since intermediate was the *only* non-multi-output-alias, there are no other aliases
                #     of `intermediate` around that were produced by the compiled fn and have a valid grad_fn.
                #
                # Coming back to this optimization:
                # Given that it is not possible for mutating one of these aliases to affect the autograd metadata of another alias
                # without causing an error in eager mode, we will simple hide the aliasing from autograd during torch.compile
                # if all of the above conditions are met.
                # This has the slight downside that it's possible to write some "bad" code that autograd will raise an error on
                # in eager but fail to during torch.compile, but it has the benefit that this code has much better performance.
                # NOTE: if and when we eventually update AOTAutograd to do the "view graph slicing" defined here:
                # https://docs.google.com/document/d/1DlfFq8TKbuAn2zyJxLfoW-X1qkkm5PLdHFtySo03QAk/edit,
                # then this optimization will probably matter less and might be ok to remove.
                is_cur_tensor_multi_out_view = isinstance(
                    o, FunctionalTensor
                ) and torch._functionalize_is_multi_output_view(  # type: ignore[attr-defined]
                    o.elem
                )
                if is_cur_tensor_multi_out_view:
                    num_aliased_tensors_that_are_multi_output_views[curr_storage] += 1
                out_storage_to_tensors[curr_storage].add(o)

        # maps the id of an intermediate base to its index in the output of the compiled forward
        intermediate_base_tensor_id_to_output_idx: Dict[int, int] = {}
        intermediate_bases: List[torch.Tensor] = []
        # Why Do We Care If Storage Changed?
        # It's important to understand the implications of storage changes in complex scenarios. Take this example:
        #
        # def f(x):
        #     x_storage = x.untyped_storage()
        #     non_leaf_tensor = torch.ones(4, requires_grad=True).clone()
        #
        #     # Using no_grad() and _unsafe_preserve_version_counter to simulate the .data = operation
        #     with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(x):
        #         x.set_(non_leaf_tensor.untyped_storage())
        #
        #     out = x.view(-1)
        #
        #     # Restoring x to its original storage, again simulating .data = operation
        #     with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(x):
        #         x.set_(x_storage)
        #
        #     return out
        #
        # In this scenario, 'x' and 'out' have different shapes and are stored at different memory addresses, aka no aliasing.
        # However, due to how set_() and more specificlaly, set is functionalized, is defined to preserve eager semantics,
        # the autograd engine mistakenly assumes that 'x' and 'out' are aliased, treating 'x' as 'out._base'.
        # This misinterpretation leads to an 'alias_of_input' flag, causing an unnecessary as_strided() call to be generated,
        # which could lead to issues later in the code.
        for o in flat_f_outs:
            functional_tensor_storage_changed = isinstance(
                o, FunctionalTensor
            ) and torch._functionalize_was_storage_changed(  # type: ignore[attr-defined]
                o.elem
            )
            curr_storage = (
                None
                if not isinstance(o, torch.Tensor)
                else StorageWeakRef(o.untyped_storage())
            )
            outs_with_identical_metadata_that_require_grad = (
                []
                if not isinstance(o, Tensor)
                else [
                    curr
                    for curr in out_storage_to_tensors[curr_storage]
                    if has_same_metadata(o, curr)
                    and curr.requires_grad
                    and o is not curr
                ]
            )

            # See Note [Accessing .grad_fn on FunctionalTensor]
            # In-place operations on views will trigger a lazy rebase of the autograd graph;
            # this runs during access to the .grad_fn. The rebase logic will invoke view ops
            # on FunctionalTensors, so we must enable a FunctionalTensorMode here to ensure
            # these op calls succeed.
            grad_fn = None
            if isinstance(o, Tensor):
                with FunctionalTensorMode():
                    grad_fn = o.grad_fn

            is_result_of_custom_autograd_fn = False
            # Need to check for both custom cpp (CppFunction) and python (BackwardCFunction)
            # autograd fns
            if type(grad_fn).__name__ == "CppFunction":
                is_result_of_custom_autograd_fn = True
            if isinstance(grad_fn, torch.autograd.function.BackwardCFunction):
                is_result_of_custom_autograd_fn = True

            if not isinstance(o, Tensor):
                output_type = OutputType.non_alias
                base_idx = None
            elif (
                curr_storage in inp_storage_refs
                and grad_fn is not None
                and is_result_of_custom_autograd_fn
            ):
                output_type = OutputType.custom_function_view
                base_idx = None
            elif (
                curr_storage in inp_storage_refs
                and not functional_tensor_storage_changed
            ):
                base_idx = inp_storage_refs[curr_storage]
                is_input_tensor = id(o) in inp_tensor_ids
                num_aliased_outs = out_tensor_alias_counts[curr_storage]
                num_multi_output_view_outs = (
                    num_aliased_tensors_that_are_multi_output_views[curr_storage]
                )
                num_aliased_outs_that_are_not_multi_output_views = (
                    num_aliased_outs - num_multi_output_view_outs
                )
                if (
                    grad_fn is not None
                    and num_aliased_outs_that_are_not_multi_output_views == 0
                ):
                    # See Note: [AOTAutograd: differentiable outputs that alias each other from a multi-output view call]
                    # In particular, given:
                    # def f(x):
                    #     return list(x.unbind(0))
                    # The main reason we ordinarily try to regenerate these output aliases outside of the
                    # compiled autograd.Function is because if any of the outputs are later mutated,
                    # autograd needs to perform view-replay to regenerate them.
                    # However, autograd does not allow users to mutate multi-output views
                    # in any way that can change the autograd metadata of other aliases.
                    # So we hide this aliasing from autograd here.
                    log.debug(
                        "Encountered AOTAutograd case: differentiable outputs that \
alias each other from a multi-output view call"
                    )
                    output_type = OutputType.non_alias
                elif is_input_tensor:
                    output_type = OutputType.is_input
                else:
                    output_type = OutputType.alias_of_input
            elif functional_tensor_storage_changed and id(o) in inp_tensor_ids:
                # When there is a set_() on an input, we cannot rely on checking storages
                # to detect if we are returning an input (since the inputs storage is different)
                assert curr_storage is not None
                base_idx = inp_storage_refs[curr_storage]
                output_type = OutputType.is_input

            # We only need to handle the intermediate base case when both
            # the intermediate base and the output require gradients.
            # See Note [AOT Autograd: outputs aliasing inputs or intermediates!]
            elif o._base is not None and o.requires_grad and o._base.requires_grad:
                num_aliased_outs = out_tensor_alias_counts[curr_storage]
                num_multi_output_view_outs = (
                    num_aliased_tensors_that_are_multi_output_views[curr_storage]
                )
                num_aliased_outs_that_are_not_multi_output_views = (
                    num_aliased_outs - num_multi_output_view_outs
                )
                # Note: [AOTAutograd: differentiable outputs that alias each other from a multi-output view call]
                if (
                    out_tensor_alias_counts[curr_storage] == 1
                    or num_aliased_outs_that_are_not_multi_output_views <= 1
                ):
                    # Note [Intermediate Bases Optimization]
                    # Normally if we have an output that aliases an intermediate,
                    # we need to add the extra "intermediate base" logic further down
                    # to prevent autograd from yelling at us if the user later tries to
                    # mutate that output.
                    # However, the common case here is if we have an output that aliases an intermediate,
                    # but doesn't alias any other outputs.
                    # In that case, autograd shouldn't have to worry about the aliasing at all
                    # (if that output is mutated, there are no other live aliases for autograd to worry about).
                    # The "intermediate bases" can hurt inductor perf by forcing more variables to become outputs.
                    # So as an optimization, we won't do intermediate base handling in this case.
                    # Instead, we'll hide the aliasing from autograd using aten._unsafe_view().
                    if (
                        out_tensor_alias_counts[curr_storage] != 1
                        and num_aliased_outs_that_are_not_multi_output_views <= 1
                    ):
                        log.debug(
                            "Encountered AOTAutograd case: differentiable outputs that alias each other \
from a multi-output view call"
                        )
                    output_type = OutputType.unsafe_view_alias
                    base_idx = None
                else:
                    # First, check if o's ._base is an existing output
                    maybe_existing_out_idx = out_tensor_ids.get(id(o._base), None)
                    if maybe_existing_out_idx is not None:
                        # Special case where the output is an alias of a graph intermediate, but that intermediate
                        # is itself also a user output.
                        output_type = (
                            OutputType.alias_of_intermediate_base_is_user_output
                        )
                        base_idx = maybe_existing_out_idx
                    else:
                        # Next, check if o's ._base is an intermediate base that we already returned
                        maybe_existing_base_output_idx = (
                            intermediate_base_tensor_id_to_output_idx.get(
                                id(o._base), None
                            )
                        )
                        if maybe_existing_base_output_idx is not None:
                            output_type = OutputType.alias_of_intermediate
                            base_idx = maybe_existing_base_output_idx
                        else:
                            # Otherwise, take o._base and explicitly return it as an output in the compiled graph
                            new_out_idx = len(intermediate_bases)
                            base_idx = new_out_idx
                            # Indicate to the logic later on (when we trace the joint)
                            # that this particular output should get it's ._base appended to the forward graph outputs
                            output_type = (
                                OutputType.alias_of_intermediate_save_as_output
                            )
                            intermediate_base_tensor_id_to_output_idx[
                                id(o._base)
                            ] = new_out_idx
                            intermediate_bases.append(o._base)
            elif (
                # See https://github.com/pytorch/pytorch/issues/100348 for this case.
                # This protects against the specific case where a user fn returns (output, output.detach())
                out_tensor_alias_counts[curr_storage] > 1
                and len(outs_with_identical_metadata_that_require_grad) > 0
                and not o.requires_grad
            ):
                # In theory we could use any of these tensors to regenerate the aliased outputs from,
                # since they all alias each other and have identical metatadata
                out_alias = outs_with_identical_metadata_that_require_grad[0]
                existing_out_idx = out_tensor_ids[id(out_alias)]
                output_type = OutputType.alias_of_intermediate_base_is_user_output
                base_idx = existing_out_idx
            else:
                output_type = OutputType.non_alias
                base_idx = None

            if isinstance(o, torch.Tensor):
                dynamic_dims = {
                    i for i, s in enumerate(o.shape) if not is_concrete_int(s)
                }
            else:
                dynamic_dims = None

            # Save the current FunctionalTensor output.
            #
            # This will be used at runtime for reconstructing output views from
            # their respective base tensors.
            #
            # The FunctionalTensor will be saved if one of the 2 conditions below
            # is true:
            functional_tensor = None
            if (
                # 1. If the output_type is either of:
                #    (i) alias_of_intermediate;
                #    (ii) alias_of_intermediate_save_as_output; or
                #    (iii) alias_of_intermediate_base_is_user_output.
                #
                # No need to worry about in-place view operations here, since
                # this functionalization step elimitates mutations.
                #
                # i.e. we have access to the actual base tensor, before the
                # in-place operation was applied.
                output_type
                in (
                    OutputType.alias_of_intermediate,
                    OutputType.alias_of_intermediate_save_as_output,
                    OutputType.alias_of_intermediate_base_is_user_output,
                )
            ) or (
                # 2. If the output_type is alias_of_input, and no in-place view
                #    operationthe was run on the input (base tensor).
                #
                # In this case, we need to check for metadata mutation because
                # the runtime explicitly reconstructs the inputs, before actually
                # reconstructing the outputs. Due to in-place view operations, the
                # fully reconstructed input may not be this output base tensor
                # anymore.
                output_type == OutputType.alias_of_input
                and base_idx is not None
                and not input_info[base_idx].mutates_metadata
            ):
                if isinstance(o, FunctionalTensor):
                    functional_tensor = FunctionalTensorMetadataEq(o.elem)

            out_info = OutputAliasInfo(
                output_type=output_type,
                raw_type=type(o),
                base_idx=base_idx,
                dynamic_dims=dynamic_dims,
                requires_grad=isinstance(o, torch.Tensor) and o.requires_grad,
                functional_tensor=functional_tensor,
            )
            output_info.append(out_info)

        # See Note [AOT Autograd: Views to avoid tangents aliasing inputs]
        def view_avoid_dupes_with_primals(t):
            if isinstance(t, Tensor) and is_traceable_wrapper_subclass(t):
                return transform_subclass(
                    t, lambda _, inner_t: view_avoid_dupes_with_primals(inner_t)
                )
            if isinstance(t, Tensor):
                return t.view(t.shape)
            return t

        # This analysis function returns *only* the outputs that are meant to be tangents to the backwards.
        # Anything that aliases (inputs returned in the fw due to metadata mutations, or outputs that alias inputs/intermediates)
        # are *regenerated* later, and not used directly in the autograd graph
        f_input_tangents = [
            inp
            for inp, info in zip(flat_f_args, input_info)
            if info.mutation_type == MutationType.MUTATED_OUT_GRAPH
            and info.mutates_data
            and info.requires_grad
        ]
        f_output_tangents = [
            o
            for o, info in zip(flat_f_outs, output_info)
            if info.output_type
            in [
                OutputType.non_alias,
                OutputType.unsafe_view_alias,
                OutputType.custom_function_view,
            ]
            and issubclass(info.raw_type, torch.Tensor)
            and info.requires_grad
        ]
        # intermediate bases are also included in the backward graph
        f_tangents = f_input_tangents + f_output_tangents + intermediate_bases
        traced_tangents = pytree.tree_map(from_fun, f_tangents)
        traced_tangents = pytree.tree_map(
            view_avoid_dupes_with_primals, traced_tangents
        )
        # See Note [Tangents must be contiguous]
        traced_tangents = pytree.tree_map(
            coerce_tangent,
            traced_tangents,
        )
        user_outs = pytree.tree_map(from_fun, f_output_tangents)

        if (
            torch._dynamo.config.inline_inbuilt_nn_modules
            or torch._dynamo.compiled_autograd.in_compiled_autograd_region
        ):
            static_parameter_input_indices = [
                i
                for i, arg in enumerate(flat_args)
                if isinstance(arg, torch.nn.Parameter)
            ]
        else:
            static_parameter_input_indices = []

        f_mutated_inputs = [
            inp
            for inp, info in zip(flat_f_args, input_info)
            if info.mutation_type == MutationType.MUTATED_OUT_GRAPH
        ]
        f_metadata_mutated_inputs = [
            inp for inp, info in zip(flat_f_args, input_info) if info.mutates_metadata
        ]
        # This logic (annoyingly) re-figures out exactly what the outputs to the compiled fw graph will be.
        # When handling subclasses, we need info about **all** outputs of compiled forward graph,
        # so we know precisely which graph outputs to wrap back into tensor subclasses
        # Ideally we would refactor this so not have an is_train flag, and have the separate
        # inference and training paths decide which inputs/output to ask for subclass info on.
        # However, we currently stash indexing information on each SubclassMeta about its order
        # in the graph outputs list.
        f_fw_graph_outs = list(flat_f_outs)
        if is_train or not keep_input_mutations:
            f_fw_graph_outs = f_mutated_inputs + f_fw_graph_outs
        else:
            # even when "keep_input_mutations" is True,
            # we never keep metadata-only mutations in the fw graph
            f_fw_graph_outs = f_metadata_mutated_inputs + f_fw_graph_outs
        if is_train:
            f_fw_graph_outs = f_fw_graph_outs + intermediate_bases
        fw_graph_outs = pytree.tree_map(from_fun, f_fw_graph_outs)

        grad_enabled_mutation = None
        if torch.is_grad_enabled() != prior_grad_enabled:
            grad_enabled_mutation = torch.is_grad_enabled()
            torch.set_grad_enabled(
                prior_grad_enabled
            )  # Restore the prior state after tracing it
            log.debug(
                (
                    "grad_mode mutation encountered in graph. "
                    "Will emit mutation epilogue, to set grad_mode=%s"
                ),
                grad_enabled_mutation,
            )

        metadata = ViewAndMutationMeta(
            input_info=input_info,
            output_info=output_info,
            num_intermediate_bases=len(intermediate_bases),
            keep_input_mutations=keep_input_mutations,
            traced_tangents=traced_tangents,
            subclass_inp_meta=create_subclass_meta(flat_args),
            subclass_fw_graph_out_meta=create_subclass_meta(fw_graph_outs),
            subclass_tangent_meta=create_subclass_meta(traced_tangents),
            is_train=is_train,
            grad_enabled_mutation=grad_enabled_mutation,
            static_parameter_indices=static_parameter_input_indices,
            tokens=mode._tokens,
        )
        return metadata

    return inner
