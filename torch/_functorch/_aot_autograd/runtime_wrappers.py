# mypy: allow-untyped-defs
"""
This module defines runtime wrappers, which, based on previous analysis attempts to:
1. process the inputs and outputs
2. apply mutations
3. handle functionalized randomness
4. deduplicate inputs and consolidate views into their bases (see input_output_analysis)
"""

import builtins
import collections
import contextlib
import copy
import functools
import itertools
import logging
import pprint
import typing
import warnings
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Optional, Union

import torch
import torch.fx as fx
import torch.utils.dlpack
from torch import Tensor
from torch._dynamo import config as dynamo_config
from torch._dynamo.callback import callback_handler, CallbackTrigger
from torch._dynamo.utils import CompileEventLogger, dynamo_timed, get_metrics_context
from torch._guards import (
    compile_context,
    CompileContext,
    detect_fake_mode,
    DuplicateInputs,
    tracing,
    TracingContext,
)
from torch._library.opaque_object import is_opaque_type
from torch._library.utils import is_builtin
from torch._logging import getArtifactLogger
from torch._prims_common import CUDARngStateHelper
from torch._subclasses import FakeTensor
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.proxy_tensor import HANDLED_TYPES
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    TorchDispatchMode,
)
from torch.utils._pytree import tree_flatten

from .. import config
from .collect_metadata_analysis import run_functionalized_fw_and_collect_metadata
from .descriptors import (
    AOTInput,
    AOTOutput,
    DummyAOTInput,
    MetadataMutationAOTOutput,
    SyntheticBaseAOTInput,
    ViewBaseAOTInput,
)
from .functional_utils import gen_alias_from_base
from .graph_capture_wrappers import aot_dispatch_subclass
from .input_output_analysis import (
    compute_overlapping_inputs,
    create_synthetic_base_metadata,
    remove_dupe_metadata,
)
from .logging_utils import describe_input, format_guard_bug_msg, track_graph_compiling
from .schemas import (
    AOTConfig,
    CompilerWrapper,
    FxValue,
    InductorWrapper,
    InputAliasInfo,
    MemoryFormatMeta,
    MutationType,
    OutputType,
    PlainTensorMeta,
    SubclassCreationMeta,
    SubclassMeta,
    TensorAlias,
    TraceFn,
    ViewAndMutationMeta,
)
from .subclass_utils import (
    requires_subclass_dispatch,
    runtime_unwrap_tensor_subclasses,
    wrap_tensor_subclasses,
)
from .utils import (
    call_and_expect_output_descs,
    call_func_at_runtime_with_args,
    make_boxed_func,
    partial_flatten_asdict,
    simple_wraps,
    strict_zip,
    without_output_descs,
)


zip = strict_zip

aot_graphs_log = getArtifactLogger(__name__, "aot_graphs")


def _describe_arg_for_logging(arg: object) -> str:
    from torch._library import opaque_object

    try:
        is_dtensor = isinstance(arg, torch.distributed._tensor.DTensor)
    except AttributeError:
        is_dtensor = False

    if is_dtensor:
        arg = typing.cast(torch.distributed._tensor.DTensor, arg)
        mesh = arg.device_mesh
        return (
            f"DTensor(shape={arg.shape}, dtype={arg.dtype}, "
            f"device={arg.device}, mesh_shape={mesh.shape}, "
            f"placements={arg.placements})"
        )
    elif isinstance(arg, torch.Tensor):
        return f"Tensor(shape={arg.shape}, dtype={arg.dtype}, device={arg.device})"
    elif opaque_object.is_opaque_type(type(arg)):
        return f"Opaque: {type(arg).__name__}"
    else:
        return f"{type(arg).__name__}: {arg}"


def _log_input_metadata(runtime_metadata: ViewAndMutationMeta) -> None:
    aot_graphs_log.debug(
        "Expected input metadata (count=%s):", len(runtime_metadata.subclass_inp_meta)
    )
    for i, meta in enumerate(runtime_metadata.subclass_inp_meta):
        aot_graphs_log.debug("  [%s] %s", i, meta)


def _log_args_list(args: Sequence[object], label: str) -> None:
    aot_graphs_log.debug("%s (count=%s):", label, len(args))
    for i, arg in enumerate(args):
        aot_graphs_log.debug("  [%s] %s", i, _describe_arg_for_logging(arg))


def _log_args_maybe_list(arg: object, label: str) -> None:
    if isinstance(arg, (list, tuple)):
        _log_args_list(arg, label)
    else:
        aot_graphs_log.debug("%s: %s", label, _describe_arg_for_logging(arg))


# The wrapper created by this function handles all of the runtime aliasing and mutation "epilogue" logic
# that needs to run after the compiled function.
#
# This function accepts a trace_joint flag, indicating whether or not we're generating the runtime
# epilogue for a forward-only inference graph, or for an autograd.Function.apply function.
# This is because there are some minor differences in how we treat these cases at runtime:
# - resize_() is currently handled in the inference case, but not fully handled in the autograd case.
# - the autograd cases inserts TensorAlias wrapper objects for outputs that alias inputs
@dataclass
class RuntimeWrapper(CompilerWrapper):
    indices_of_inps_to_detach: list[int]
    trace_joint: bool
    disable_amp: bool

    def post_compile(
        self,
        compiled_fn,
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ):
        return _create_runtime_wrapper(
            compiled_fn,
            runtime_metadata=runtime_metadata,
            indices_of_inps_to_detach=self.indices_of_inps_to_detach,
            trace_joint=self.trace_joint,
            keep_input_mutations=aot_config.keep_inference_input_mutations,
            disable_amp=self.disable_amp,
        )


class NoopAliasHandler:
    def __init__(self, info, runtime_metadata, trace_joint):
        pass

    def __call__(self, orig_inputs, fw_outs, out):
        return out


def _unwrap_tensoralias(x):
    assert isinstance(x, TensorAlias)
    return x.alias


def _identity(x):
    return x


class AliasOfInputHandler:
    def __init__(self, info, runtime_metadata, trace_joint):
        self.base_idx = info.base_idx
        self.unwrap_out = _unwrap_tensoralias if trace_joint else _identity
        self.requires_grad = info.requires_grad
        self.view_meta_sequence = info.view_meta_sequence
        self.replay_views = config.view_replay_for_aliased_outputs

    def __call__(self, orig_inputs, fw_outs, out):
        aliased_base_tensor = orig_inputs[self.base_idx]
        return gen_alias_from_base(
            aliased_base_tensor,
            self.unwrap_out(out),
            self.requires_grad,
            self.view_meta_sequence,
            replay_views=self.replay_views,
        )


class IsInputHandler:
    def __init__(self, info, runtime_metadata, trace_joint):
        self.base_idx = info.base_idx
        self.unwrap_out = _unwrap_tensoralias if trace_joint else _identity

    def __call__(self, orig_inputs, fw_outs, out):
        aliased_base_tensor = orig_inputs[self.base_idx]
        return aliased_base_tensor


class AliasOfIntermediateHandler:
    def __init__(self, info, runtime_metadata, trace_joint):
        self._unwrap_aliased_base_tensor = _identity
        if info.output_type in (
            OutputType.alias_of_intermediate,
            OutputType.alias_of_intermediate_save_as_output,
        ):
            num_user_outputs = len(runtime_metadata.output_info)
            self.base_idx = info.base_idx + num_user_outputs
        else:
            self.base_idx = info.base_idx
            if self.base_idx in runtime_metadata.aliased_out_indices:
                self._unwrap_aliased_base_tensor = _unwrap_tensoralias

        self.unwrap_out = _unwrap_tensoralias if trace_joint else _identity
        self.requires_grad = info.requires_grad
        self.view_meta_sequence = info.view_meta_sequence
        self.replay_views = config.view_replay_for_aliased_outputs

    def __call__(self, orig_inputs, fw_outs, out):
        aliased_base_tensor = fw_outs[self.base_idx]
        return gen_alias_from_base(
            self._unwrap_aliased_base_tensor(aliased_base_tensor),
            self.unwrap_out(out),
            self.requires_grad,
            self.view_meta_sequence,
            replay_views=self.replay_views,
        )


_HANDLER_MAP = {
    OutputType.non_alias: NoopAliasHandler,
    OutputType.unsafe_view_alias: NoopAliasHandler,
    OutputType.custom_function_view: NoopAliasHandler,
    OutputType.alias_of_input: AliasOfInputHandler,
    OutputType.is_input: IsInputHandler,
    OutputType.alias_of_intermediate: AliasOfIntermediateHandler,
    OutputType.alias_of_intermediate_save_as_output: AliasOfIntermediateHandler,
    OutputType.alias_of_intermediate_base_is_user_output: AliasOfIntermediateHandler,
}


def make_output_handler(info, runtime_metadata, trace_joint):
    handler_type = _HANDLER_MAP[info.output_type]
    return handler_type(info, runtime_metadata, trace_joint)


# not sure why AOTDispatcher needs to manually set this
def maybe_mark_dynamic_helper(t: torch.Tensor, dims: set[int]):
    if hasattr(t, "_dynamo_weak_dynamic_indices"):
        # pyrefly: ignore [missing-attribute]
        t._dynamo_weak_dynamic_indices |= dims
    else:
        t._dynamo_weak_dynamic_indices = dims.copy()  # type: ignore[attr-defined]


def _should_disable_saved_tensors_hooks():
    # Compiled autograd is not supported yet, to be added in future.
    if torch._dynamo.compiled_autograd.in_compiled_autograd_region:
        return False

    get_hooks = torch._functorch._aot_autograd.utils.top_saved_tensors_hooks
    are_inline_hooks = (
        torch._functorch._aot_autograd.utils.saved_tensors_hooks_are_inlineable
    )

    hooks = get_hooks()
    if are_inline_hooks(hooks):
        return True

    return False


def _schema_allows_aliasing(func) -> bool:
    schema = func._schema
    # View ops have non-write aliases declared in arguments
    if schema._is_view_op():
        return True
    # Handles cases like mkldnn::_convolution_pointwise_.binary
    # where the schema is Tensor(a!) other -> Tensor(a!) Y
    for ret in schema.returns:
        if ret.alias_info is not None:
            return True
    return False


def _check_custom_op_aliasing(name, args, kwargs, result):
    """
    Check if custom op outputs alias inputs or other outputs.
    If config.error_on_custom_op_aliasing is True, raises RuntimeError.
    Otherwise, emits a warning.
    """
    try:
        torch._library.utils._c_check_aliasing_constraint(
            name,
            args,
            kwargs,
            result,
        )
    except RuntimeError as e:
        if config.error_on_custom_op_aliasing:
            raise
        else:
            warnings.warn(str(e), UserWarning, stacklevel=3)


@functools.lru_cache(None)
def _is_fsdp_all_gather_copy_in(func) -> bool:
    """
    Check if func is torch.ops.fsdp.all_gather_copy_in.default by comparing
    namespace and name strings. This avoids accessing torch.ops.fsdp directly,
    which would fail on platforms where FSDP ops aren't registered (e.g., macOS
    builds with USE_DISTRIBUTED=0).
    """
    return (
        hasattr(func, "namespace")
        and func.namespace == "fsdp"
        and hasattr(func, "__name__")
        and func.__name__ == "all_gather_copy_in.default"
    )


class _AnalyzeCustomOpInputOutputMode(TorchDispatchMode):
    """
    Checks if inp/out of custom ops alias each other.
    If config.error_on_custom_op_aliasing is True, violations raise errors.
    Otherwise, violations emit warnings.
    """

    def __init__(self):
        super().__init__()
        self.supports_higher_order_operators = True

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if not kwargs:
            kwargs = {}

        flat_tensor_args = filter(
            lambda x: isinstance(x, torch.Tensor), tree_flatten((args, kwargs))[0]
        )

        # Defer this to subclass torchdispatch modes (probably shouldn't have fake tensor here tho)
        if not all(type(x) in HANDLED_TYPES for x in flat_tensor_args):
            return NotImplemented

        res = func(*args, **kwargs)
        # Only check aliasing for custom ops (non-aten/prim/prims/_c10d_functional/c10d)
        # that claim to be functional
        # Skip ops whose schema declares aliasing is allowed
        if (
            not isinstance(func, torch._ops.HigherOrderOperator)
            and not is_builtin(func)
            # TODO (https://github.com/pytorch/pytorch/issues/170986)
            and func.namespace not in ("_c10d_functional", "c10d", "onednn")
            # This op is quite important but has wrong schema, so lets skip for now
            and not _is_fsdp_all_gather_copy_in(func)
            and not _schema_allows_aliasing(func)
        ):
            _check_custom_op_aliasing(
                func.name(),
                args,
                kwargs,
                res,
            )
        return res

    @classmethod
    def ignore_compile_internals(cls):
        return True


class _FirstInvocationContext:
    """
    Context manager that tracks first invocation and conditionally enables _AnalyzeCustomOpInputOutputMode.
    This is useful when we have a custom op where we want to analyze its' input
    and output during cold start.
    """

    def __init__(self):
        self._is_first = True

    def __call__(self):
        """
        Returns a context manager: _AnalyzeCustomOpInputOutputMode on first invocation, nullcontext thereafter.
        Automatically updates state after first use.
        """
        if self._is_first and config.check_custom_op_aliasing:
            self._is_first = False
            return _AnalyzeCustomOpInputOutputMode()
        return nullcontext()


def _create_runtime_wrapper(
    compiled_fn,
    *,
    runtime_metadata: ViewAndMutationMeta,
    indices_of_inps_to_detach: list[int],
    trace_joint: bool,
    keep_input_mutations: bool,
    disable_amp: bool,
):
    if not getattr(compiled_fn, "_boxed_call", False):
        compiled_fn = make_boxed_func(compiled_fn)

    # We only want to run debugmode on custom ops at the first invocation of
    # runtime wrapper. For all subsequent uses, we should no-op for performance
    # See: https://github.com/pytorch/pytorch/issues/165349
    first_invocation_ctx = _FirstInvocationContext()

    # Note [Inputs needed in runtime epilogue after list clearing]
    # In Python functions, you can't free the input arguments of a function within the scope of that function. A workaround is to
    # wrap the input arguments in a list, and clear the list from within the function.
    # Here, this is implemented as `call_func_at_runtime_with_args(..., steal_args=True)`.
    #
    # This is needed for Compiled Autograd since some of the inputs (activations) should be freed early.
    # However, we cannot blindly clear the entire list, because AOTAutograd may need access to some of the graph inputs
    # **after** the compiled function has finished running. There are two main cases:
    #   (1) Input mutations: If there are an input mutations that we must run outside of the graph, we need access to the input.
    #   (2) Output aliasing: Outputs that aliases graph inputs generally must be regenerated outside of the `autograd.Function`,
    #       and doing so requires us accessing the corresponding input after the compiled artifact has run.
    epilogue_args_idx = []
    epilogue_args_idx.extend(runtime_metadata.mutated_inp_runtime_indices)
    for info in runtime_metadata.output_info:
        if (
            info.output_type == OutputType.alias_of_input
            or info.output_type == OutputType.is_input
        ):
            assert isinstance(info.base_idx, int)
            epilogue_args_idx.append(info.base_idx)

    if config.unlift_effect_tokens:
        assert len(runtime_metadata.tokens) == 0

    if runtime_metadata.num_outputs_aliased > 0:
        output_handlers = tuple(
            make_output_handler(info, runtime_metadata, trace_joint)
            for info in runtime_metadata.output_info
        )

    def record_runtime_wrapper_prologue_enter() -> Optional[
        AbstractContextManager[None]
    ]:
        if (
            torch.autograd.profiler._is_profiler_enabled
            and dynamo_config.record_runtime_overhead
        ):
            cm = torch._C._profiler._RecordFunctionFast(
                "AOTDispatcher Runtime Wrapper Prologue"
            )
            cm.__enter__()
            return cm
        return None

    def record_runtime_wrapper_prologue_exit(
        cm: Optional[AbstractContextManager[None]],
    ) -> None:
        if cm is not None:
            cm.__exit__(None, None, None)

    @simple_wraps(compiled_fn)
    def runtime_wrapper(args: list[Any]):
        # Create context manager for profiler
        cm = record_runtime_wrapper_prologue_enter()

        # stash a ref to each input tensor we plan to use after the compiled function
        orig_inputs = {i: args[i] for i in epilogue_args_idx}

        if keep_input_mutations:
            mutated_args = (
                args[i]
                for i in runtime_metadata.mutated_graph_handled_indices_seen_by_autograd
            )
            torch.autograd.graph.increment_version(mutated_args)

        # Enable _AnalyzeCustomOpInputOutputMode on first invocation to check aliasing constraints for custom ops
        with first_invocation_ctx():
            if trace_joint:
                args_ = list(args)
                # See Note [Detaching inputs that never need gradients]
                for idx in indices_of_inps_to_detach:
                    if isinstance(args_[idx], torch.Tensor):
                        args_[idx] = args_[idx].detach()

                # It's possible to have trace_joint inside user specified with no_grad() region,
                # if there is a nested with enable_grad(), that forces some outputs to require gradients.
                # Therefore, we unconditionally turn on enable_grad() for compiled_fn execution.
                with (
                    torch.autograd._force_original_view_tracking(True),
                    torch.enable_grad(),
                ):
                    record_runtime_wrapper_prologue_exit(cm)
                    all_outs = call_func_at_runtime_with_args(
                        compiled_fn, args_, disable_amp=disable_amp, steal_args=True
                    )
            else:
                # When we have an inference graph, we run with grad disabled.
                # It's possible to get an inference graph with inputs that require grad,
                # in which case we want to make sure autograd is disabled
                # (since e.g., inductor will generate aten.addmm.out calls which autograd will complain on)
                # NOTE: We use _set_grad_enabled directly to reduce runtime overhead
                grad_enabled = torch.is_grad_enabled()
                try:
                    if grad_enabled:
                        torch._C._set_grad_enabled(False)
                    record_runtime_wrapper_prologue_exit(cm)
                    all_outs = call_func_at_runtime_with_args(
                        compiled_fn, args, disable_amp=disable_amp, steal_args=True
                    )
                finally:
                    if grad_enabled:
                        torch._C._set_grad_enabled(True)

        del args

        num_mutated_runtime_inps = runtime_metadata.num_mutated_inp_runtime_indices
        num_intermediate_bases = runtime_metadata.num_intermediate_bases

        assert (
            len(all_outs)
            == num_mutated_runtime_inps
            + runtime_metadata.num_outputs
            + num_intermediate_bases
        )

        # Step 3: After running the compiled fw, apply updates to mutated inputs
        if num_mutated_runtime_inps > 0:
            updated_inputs = all_outs[:num_mutated_runtime_inps]
            fw_outs = all_outs[num_mutated_runtime_inps:]

            for i, inpt_idx in enumerate(runtime_metadata.mutated_inp_runtime_indices):
                meta = runtime_metadata.input_info[inpt_idx]
                if not meta.mutates_data and not meta.mutates_metadata:
                    continue
                original_inpt = orig_inputs[inpt_idx]
                updated_inpt = updated_inputs[i]
                if meta.mutates_storage_metadata:
                    # See Note [set_() Input Mutations in AOTAutograd]
                    # mutates_storage_metadata means our input saw a x.set_(y) call.
                    # What if x **also** saw a data and/or a metadata mutation?
                    # (1) If the [meta]data mutation occurred after the set_(),
                    #     then there is no need to copy_() the data.
                    #     When we perform x.set_(x_updated), we are guaranteed that
                    #     x_updated already has the final version of the data/metadata
                    # (2) If a data mutation occurred before the set_().
                    #     This case seems very difficult to support.
                    #     TODO: discuss on the PR and decide if we want to tr to
                    #     either support it, or detect and ban it.
                    if trace_joint:
                        assert isinstance(updated_inpt, TensorAlias)
                        updated_inpt = updated_inpt.alias
                    with torch.no_grad():
                        original_inpt.set_(updated_inpt)
                    continue
                if meta.mutates_metadata and not meta.mutates_data:
                    if trace_joint:
                        assert isinstance(updated_inpt, TensorAlias)
                        updated_inpt = updated_inpt.alias
                    # We need to grab the size/stride/storage_offset from the compiled forward,
                    # and use that to mutate the metadata of the input
                    original_inpt.as_strided_(
                        updated_inpt.size(),
                        updated_inpt.stride(),
                        updated_inpt.storage_offset(),
                    )
                else:
                    if meta.mutates_data and meta.mutates_metadata:
                        original_inpt.as_strided_(
                            updated_inpt.size(),
                            updated_inpt.stride(),
                            updated_inpt.storage_offset(),
                        )
                    else:
                        assert meta.mutates_data
                    if meta.is_leaf and original_inpt.requires_grad:
                        # We can hit this situation in this case:
                        #   def f(x):
                        #       x.detach().mul_(2)
                        #       return x + 1
                        # AOTAutograd will see a mutation in the above case, and try to
                        # apply a copy_() here, in the epilogue.
                        # But if x required gradients, and is a leaf, then autograd
                        # will yell at us for trying to mutate it.
                        # However, it's only possible to end up in this scenario (like the above)
                        # if all of the mutations to the leaf input were non-autograd-tracking mutations
                        # (aka mutations under no_grad(), or on detached views).
                        # In that case, we fully want to hide the mutation from autograd, so detaching is ok.
                        original_inpt.detach().copy_(updated_inpt)
                    else:
                        # Check if we have stream index information for this mutated input
                        if (
                            runtime_metadata.mutated_inp_stream_indices is not None
                            and i < len(runtime_metadata.mutated_inp_stream_indices)
                            and runtime_metadata.mutated_inp_stream_indices[i]
                            is not None
                        ):
                            raise RuntimeError(
                                "Mutations on inputs with user-specified streams are not yet supported. "
                                "See: https://github.com/pytorch/pytorch/issues/172522"
                            )
                        original_inpt.copy_(updated_inpt)
        else:
            fw_outs = all_outs

        # Step 4: Manually regenerate any outputs that are aliased to inputs, instead of
        # compiling them.
        if runtime_metadata.num_outputs_aliased > 0:
            # The compiled forward also returned intermediate bases. We don't want to return them to the user.
            expect_num_outputs = (
                len(output_handlers) + runtime_metadata.num_intermediate_bases
            )
            assert len(fw_outs) == expect_num_outputs
            ret_outs = [
                handler(orig_inputs, fw_outs, out)
                for out, handler in builtins.zip(fw_outs, output_handlers)
            ]
        else:
            ret_outs = fw_outs

        if runtime_metadata.dynamic_outputs:
            for t, o in zip(ret_outs, runtime_metadata.output_info):
                if o.dynamic_dims is None:
                    continue
                maybe_mark_dynamic_helper(t, o.dynamic_dims)
        if runtime_metadata.grad_enabled_mutation is not None:
            torch._C._set_grad_enabled(runtime_metadata.grad_enabled_mutation)
        return ret_outs

    if not (trace_joint and _should_disable_saved_tensors_hooks()):
        return runtime_wrapper

    # Disabling saved tensors hooks
    @simple_wraps(runtime_wrapper)
    def _runtime_wrapper(*args, **kwargs):
        with _disable_saved_tensors_hooks():
            return runtime_wrapper(*args, **kwargs)

    return _runtime_wrapper


# WARNING: this does NOT operate on TraceFn
@dataclass
class FunctionalizedRngRuntimeWrapper(InductorWrapper):
    # TODO: I would love to get rid of this argument, but it's
    # Wrapped pretty tightly around our aot_dispatch_autograd logic.
    # Specifically, tensors_saved_for_backwards_slice's value is both used for calculating indices
    # for setting placeholder strides(which is done before runtime, before this wrapper runs)
    # and for saving tensors for backward (which is done during runtime, after this wrapper runs)
    # So in aot_dispatch_autograd, this wrapper can't edit the set of outs without making one
    # of those two indices incorrect.
    return_new_outs: bool = True

    def pre_compile(
        self,
        flat_fn: torch.fx.GraphModule,
        flat_args,
        aot_config,
        *,
        fw_metadata,
    ) -> None:
        if config.functionalize_rng_ops:
            # Update example inputs for the fw_compiler
            fake_mode = detect_fake_mode()
            assert fake_mode is not None
            seed, offset = CUDARngStateHelper.get_torch_state_as_tuple(fake_mode)
            flat_args.extend([seed, offset])
            # We are not clearing flat_args here because
            # 1) There is a check in the debug compiler at the end
            # 2) It does not matter as these are fake tensors

    def post_compile(
        self,
        compiled_fn,
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ):
        @wraps(compiled_fn)
        def wrapper(runtime_args: list[Any]):
            if runtime_metadata.is_rng_op_functionalized:
                # Add the seed and offset to args
                seed, offset = CUDARngStateHelper.get_torch_state_as_tuple()
                runtime_args.extend([seed, offset])
                out = compiled_fn(runtime_args)
                out = self._functionalized_rng_runtime_epilogue(
                    runtime_metadata,
                    out,
                    # TODO: this won't be right for the backward when we convert the call_compiled_backward to use the wrapper
                    runtime_metadata.num_forward_returns,
                )
                return out
            return compiled_fn(runtime_args)

        return wrapper

    # Calling convention: If we are running functionalized RNG, then outs consists
    # of (user_outs, rng_offset)
    def _functionalized_rng_runtime_epilogue(
        self,
        metadata: ViewAndMutationMeta,
        outs,
        offset_index,
    ):
        if metadata.is_rng_op_functionalized:
            assert metadata.num_outputs_rng_offset == 1
            new_rng_offset = outs[offset_index]
            CUDARngStateHelper.set_new_offset(new_rng_offset)
            if self.return_new_outs:
                user_outs = outs[:offset_index] + outs[offset_index + 1 :]
                return user_outs
            else:
                return outs

        return outs


# WARNING: this does NOT operate on TraceFn
@dataclass
class FakifiedOutWrapper(InductorWrapper):
    out_metas: list[torch.Tensor] = field(default_factory=list)
    # TracingContext.fwd_output_strides
    # Generated from actually doing compile
    # NB: an entry is None if it's not a Tensor
    fwd_output_strides: Optional[list[Optional[list[int]]]] = None
    needs_post_compile: bool = True

    def pre_compile(
        self,
        fw_module: fx.GraphModule,  # Must be fw_module from aot_dispatch_*_graph
        flat_args,
        aot_config,
        *,
        fw_metadata,
    ) -> None:
        tracing_context = torch._guards.TracingContext.try_get()
        if tracing_context and tracing_context.fakify_first_call:
            self.out_metas = [
                n.meta["val"] for n in (list(fw_module.graph.nodes)[-1].args[0])
            ]
        else:
            self.needs_post_compile = False

    def _compute_output_meta_with_inductor_strides(self):
        out = self.out_metas
        fwd_output_strides = self.fwd_output_strides
        if not fwd_output_strides:
            return out

        from torch.fx.experimental.symbolic_shapes import statically_known_true

        for i in range(len(out)):
            if not isinstance(out[i], Tensor):
                continue
            strides = fwd_output_strides[i]
            # fwd_output_strides is best effort by Inductor.  When an output
            # Tensor has unbacked SymInts, Inductor may sometimes be unable
            # to compute what the output stride would be.  If Inductor doesn't
            # have any clear direction on the layout, we don't have to run
            # as_strided.  To repro without this, run:
            #
            # python test/distributed/test_dynamo_distributed.py
            # TestFakeDistributedSingleProc.test_unbacked_symbol_splitting_no_binding
            if strides is None:
                continue
            if all(
                statically_known_true(s1 == s2)
                for s1, s2 in zip(out[i].stride(), strides)
            ):
                continue
            out[i] = out[i].as_strided(out[i].shape, strides)
        return out

    # To be called post compile
    def set_fwd_output_strides(self, fwd_output_strides):
        self.fwd_output_strides = fwd_output_strides

    def post_compile(
        self,
        compiled_fn,
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ):
        if self.needs_post_compile:
            assert self.fwd_output_strides is not None
            fakified_out = self._compute_output_meta_with_inductor_strides()

            @wraps(compiled_fn)
            def wrapper(runtime_args):
                nonlocal fakified_out
                if fakified_out is not None:
                    out = fakified_out
                    fakified_out = None
                    return out
                return compiled_fn(runtime_args)

            return wrapper
        # If we don't need to fakify, we can just return the original compiled function
        return compiled_fn


# This wrapper handles the AOTDispatch runtime logic for tensor subclasses.
# At runtime, we have a compiled function that knows how to operate on the domain of DenseTensor -> DenseTensor,
# But the user might have passed us some tensor subclass inputs (or expect some subclass tensor outputs).
# This function handles the wrapping and unwrapping of tensor subclasses at runtime.
@dataclass
class AOTDispatchSubclassWrapper(CompilerWrapper):
    trace_joint: bool
    fw_only: Optional[Callable]  # Not cached, only used in pre_compile
    maybe_subclass_meta: Optional[SubclassMeta]
    num_fw_outs_saved_for_bw: Optional[int]

    def pre_compile(
        self,
        flat_fn: TraceFn,
        flat_args: list[FxValue],
        flat_args_descs: list[AOTInput],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ):
        (new_flat_fn, new_flat_args, new_flat_args_descs, subclass_meta) = (
            aot_dispatch_subclass(
                flat_fn,
                flat_args,
                flat_args_descs,
                is_joint_structure=self.trace_joint,
                meta=fw_metadata,
                fw_only=self.fw_only,  # type: ignore[arg-type]
            )
        )
        self.maybe_subclass_meta = subclass_meta
        return new_flat_fn, new_flat_args, new_flat_args_descs, fw_metadata

    def post_compile(
        self,
        compiled_fn,
        _aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ):
        if self.maybe_subclass_meta is None:
            return compiled_fn

        subclass_metas = runtime_metadata.subclass_fw_graph_out_meta

        @wraps(compiled_fn)
        def inner_fn(args: list[Any]):
            if aot_graphs_log.isEnabledFor(logging.DEBUG):
                aot_graphs_log.debug(
                    "=== AOTDispatchSubclassWrapper.inner_fn START ==="
                )
                _log_input_metadata(runtime_metadata)
                _log_args_list(args, "Incoming args")

            unwrapped_args = runtime_unwrap_tensor_subclasses(
                args,
                subclass_metas=runtime_metadata.subclass_inp_meta,
                append_symints=True,
            )

            if aot_graphs_log.isEnabledFor(logging.DEBUG):
                _log_args_list(unwrapped_args, "After unwrapping, unwrapped_args")

            args.clear()
            # expectation: runtime_fn is a boxed fn
            unwrapped_outs = compiled_fn(unwrapped_args)

            if aot_graphs_log.isEnabledFor(logging.DEBUG):
                _log_args_maybe_list(
                    unwrapped_outs, "After compiled_fn, unwrapped_outs"
                )

            wrapped_outs = wrap_tensor_subclasses(
                unwrapped_outs,
                subclass_metas=subclass_metas,
                num_fw_outs_saved_for_bw=self.num_fw_outs_saved_for_bw,
                is_runtime=True,
                included_subclass_symints=True,
            )

            if aot_graphs_log.isEnabledFor(logging.DEBUG):
                _log_args_maybe_list(wrapped_outs, "After wrapping, wrapped_outs")
                aot_graphs_log.debug("=== AOTDispatchSubclassWrapper.inner_fn END ===")

            return wrapped_outs

        # box it
        inner_fn._boxed_call = True  # type: ignore[attr-defined]
        return inner_fn


@dataclass
class EffectTokensWrapper(CompilerWrapper):
    def post_compile(
        self,
        compiled_fn,
        _aot_config,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ):
        num_tokens = len(runtime_metadata.tokens)

        @wraps(compiled_fn)
        def inner_fn(args: list[Any]):
            if num_tokens > 0:
                # Pass in forward effect tokens (See Note [Side-Effectful Tokens in AOTAutograd])
                old_args = args
                args = [*([None] * num_tokens), *args]
                old_args.clear()

            outs = compiled_fn(args)

            # Inductor cache DummyModule can return None
            if outs is None:
                return None
            # Toss out the effect tokens (See Note [Side-Effectful Tokens in AOTAutograd])
            return outs[num_tokens:] if num_tokens != 0 else outs

        # box it
        inner_fn._boxed_call = True  # type: ignore[attr-defined]
        return inner_fn


# MOTIVATION:
#
# When tracing functions for future execution, one must be careful not to pass
# in the same input tensor multiple times (e.g., f(x, x), as this can result
# in graphs that are ONLY valid if you later pass a new tensor in exactly the
# same way (e.g., f(y, y)).  (NB: we really mean duplicate; two distinct
# tensors that alias each other is a different situation that is covered by
# aot_dispatch_deduplicated_autograd). Here are two examples:
#
# (1) Suppose you have a function:
#
#   def f(x, y):
#       return x + y
#
# If you make_fx(f)(x, x), you will trace out:
#
#   def f(x, y):
#       return y + y
#
# Oops!
#
# (2) For most tensors x and y, you can compute f's gradient with respect to
# these to inputs by saying torch.autograd.grad(f(x, y), (x, y)).  However,
# if x is y, you will trace out a program that gets incorrect gradients:
#
#   >>> x = torch.randn(1, requires_grad=True)
#   >>> torch.autograd.grad(x + x, (x, x))
#   (tensor([2.]), tensor([2.]))
#
# In other words, the gradient is double-counted.  Deduplicating the arguments
# gives you an appropriate gradient:
#
#   >>> y = torch.randn(1, requires_grad=True)
#   >>> torch.autograd.grad(x + y, (x, y))
#   (tensor([1.]), tensor([1.]))
#
# HOW TO DEDUPLICATE:
#
# There are a few strategies, in order of preference:
#
# 1. For every duplicate argument to the function, detach it into
#    a separate leaf tensor, so that it is no longer duplicated.
#
#       PRO: The resulting compiled graph works for any configuration
#       of duplicated arguments.
#
#       CON: It does not (naively) work if you mutate the metadata of inputs:
#
#           def f(x, y):
#               x.transpose_(0, 1)
#               y.transpose_(0, 2)
#
#           x = torch.randn(2, 3, 4)
#           f(x, x)
#
#       The ordering of the transposes inside f dictates whether or not
#       you get [4, 2, 3] or [3, 4, 2].  This means that you cannot precompute
#       what metadata mutations should get applied to each input; you need to
#       assume they aren't duplicates (what we do today) or preserve
#       the original metadata mutations exactly in order, so that they work
#       for any duplicate configuration.
#
#       CON: It does not (naively) work if you mutate the data of inputs.
#       In particular, leaf tensors that require grad cannot be mutated,
#       this makes it impossible to differentiate with respect to the original
#       base.
#
# 2. For every duplicate argument to the function, remove it, so it is
#    no longer part of the "true" signature:
#
#       PRO: Implemented naively, it still works for metadata/data mutation.
#
#       CON: The resulting compiled graph is duplicate-specialized: it only
#       works if future calls duplicate arguments in exactly the same way.
#       Horribly, Dynamo doesn't guard on this at the moment.  But even if
#       it did, you could still end up recompiling a bunch of each duplicate.
#
# Our strategy is to do (1) if we can, and do (2) otherwise, erroring if
# Dynamo's guards are not enough.  In practice, this seems to cover
# everything.
#
@dataclass
class AOTDedupeWrapper(CompilerWrapper):
    keep_arg_mask: list[bool] = field(default_factory=list)
    add_dupe_map: list[int] = field(default_factory=list)
    old_input_metadata: list[InputAliasInfo] = field(default_factory=list)
    needs_post_compile: bool = True

    # NB: Hot path, avoid set lookups here
    # TODO: Can avoid the zip here too, probably
    def remove_dupe_args(self, args):
        return [t for t, keep in zip(args, self.keep_arg_mask) if keep]

    def add_dupe_args(self, args):
        return [args[i] for i in self.add_dupe_map]

    def pre_compile(
        self,
        flat_fn: TraceFn,
        flat_args: list[FxValue],
        flat_args_descs: list[AOTInput],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ) -> tuple[TraceFn, list[FxValue], list[AOTInput], ViewAndMutationMeta]:
        # Use information about whether or not flat_fn mutates its arguments
        # or not to handle dupe args

        # Strategy 1: For any input that is not mutated, we can leafify it if we
        # need to remove a duplicate.
        leaf_flat_args: list[FxValue] = []
        leaf_flat_args_descs: list[AOTInput] = []
        args_set = set()
        ok = True

        for i, (a, a_desc) in enumerate(zip(flat_args, flat_args_descs)):
            if not isinstance(a, torch.Tensor):
                leaf_flat_args.append(a)
                leaf_flat_args_descs.append(a_desc)
            elif a not in args_set:
                args_set.add(a)
                leaf_flat_args.append(a)
                leaf_flat_args_descs.append(a_desc)
            elif (
                not fw_metadata.input_info[i].mutates_data
                and not fw_metadata.input_info[i].mutates_metadata
            ):
                leaf_flat_args.append(a.detach().requires_grad_(a.requires_grad))
                leaf_flat_args_descs.append(a_desc)
            else:
                ok = False
                break

        if ok:
            self.needs_post_compile = False
            return flat_fn, leaf_flat_args, leaf_flat_args_descs, fw_metadata

        if requires_subclass_dispatch(leaf_flat_args, fw_metadata):
            raise RuntimeError(
                """\
        Encountered duplicate inputs that are mutated in the graph, but at least one input/output
        to the graph is a tensor subclass. This is not supported today. You can try to
        remove the aliasing yourself as a workaround, or otherwise file an issue on github."""
            )

        # export path: ban duplicate inputs for now, add later if requested.
        if aot_config.is_export:
            raise RuntimeError(
                f"""\
        Encountered duplicated inputs that are mutated in the graph you are trying to export.
        This functionality is currently not supported. If needed, please file a github issue.

        fw_metadata={str(fw_metadata)}
            """
            )

        # Strategy 2: Duplicate specialization
        #
        # When we have duplicate arguments in a function call, we need to handle them specially.
        # For example, if we have a function call f(a, b, a, c), we need to:
        #
        # 1. Remove duplicates to get a deduplicated list [a, b, c]
        # 2. Compile our function to work with this deduplicated list
        # 3. At runtime, convert incoming arguments with duplicates to the deduplicated form
        # 4. Pass the deduplicated arguments to our compiled function
        #
        # To do this, we need two helper functions:
        #
        # - remove_dupe_args: Converts [a, b, a, c] -> [a, b, c]
        # - add_dupe_args: Converts [a, b, c] -> [a, b, a, c]
        #
        # For our example [a, b, a, c], we track:
        #
        # - seen_args = {a: 0, b: 1, c: 2} (maps each unique arg to its first position)
        # - add_dupe_map = [0, 1, 0, 2] (tells us how to reconstruct the original list)
        # - keep_arg_mask = [True, True, False, True] (tells us which args to keep when deduplicating)

        seen_args: dict[Tensor, int] = {}
        # Implicitly map duped arg position (list index) to de-duped arg position
        keep_arg_mask: list[bool] = []
        add_dupe_map: list[int] = []
        duped_arg_len = len(flat_args)

        j = 0  # index into deduped_flat_args
        for t in flat_args:
            if isinstance(t, torch.Tensor):
                if t in seen_args:
                    keep_arg_mask.append(False)
                    add_dupe_map.append(seen_args[t])
                    continue
                seen_args[t] = j

            keep_arg_mask.append(True)
            add_dupe_map.append(j)
            j += 1
        assert len(add_dupe_map) == duped_arg_len, (
            f"Expects add_dupe_map to have length {duped_arg_len} but got {len(add_dupe_map)}"
        )

        self.keep_arg_mask = keep_arg_mask
        self.add_dupe_map = add_dupe_map

        deduped_flat_args = self.remove_dupe_args(flat_args)
        # TODO: instead of arbitrarily removing args, it might be useful to
        # have a record that these were duped, perhaps as a mutable attribute
        # on the kept arg?  Do this if someone needs it
        deduped_flat_args_descs = self.remove_dupe_args(flat_args_descs)

        # Update our input metadata to remove duped input metadata.
        updated_fw_metadata = remove_dupe_metadata(
            fw_metadata, keep_arg_mask, add_dupe_map
        )

        if (
            tracing_context := TracingContext.try_get()
            and aot_config.aot_autograd_arg_pos_to_source
        ):
            # TODO(voz): This structure is 1:1, we could consider an alternate structure like
            # kept_pos:[dupe_arg_pos], however, add_dupe_map is 1:1 so we would need a new structure there,
            # which feels like needless complexity for a tiny bit of efficiency at this point.
            for dupe_arg_pos, (kept_pos, keep_arg) in enumerate(
                zip(add_dupe_map, keep_arg_mask)
            ):
                if not keep_arg:
                    dupe_arg_source = aot_config.aot_autograd_arg_pos_to_source[
                        dupe_arg_pos
                    ]
                    kept_arg_source = aot_config.aot_autograd_arg_pos_to_source[
                        kept_pos
                    ]
                    tracing_context.guards_context.aotautograd_guards.append(  # type: ignore[attr-defined]
                        DuplicateInputs(kept_arg_source, dupe_arg_source)
                    )

        @simple_wraps(flat_fn)
        def wrapped_flat_fn(
            *args: FxValue,
        ) -> tuple[list[FxValue], list[AOTOutput]]:
            outs, out_descs = call_and_expect_output_descs(
                flat_fn, self.add_dupe_args(args)
            )
            return outs, out_descs

        if config.debug_assert:
            ref_fw_metadata = run_functionalized_fw_and_collect_metadata(
                without_output_descs(wrapped_flat_fn),
                flat_args_descs=deduped_flat_args_descs,
                static_input_indices=aot_config.static_input_indices,
                keep_input_mutations=fw_metadata.keep_input_mutations,
                is_train=fw_metadata.is_train,
            )(*deduped_flat_args)
            assert ref_fw_metadata == updated_fw_metadata, (
                f"ref_metadata={str(ref_fw_metadata)}, actual_metadata={str(updated_fw_metadata)}"
            )

        return (
            wrapped_flat_fn,
            deduped_flat_args,
            deduped_flat_args_descs,
            updated_fw_metadata,
        )

    def post_compile(
        self,
        compiled_fn,
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ):
        if not self.needs_post_compile:
            return compiled_fn

        @wraps(compiled_fn)
        def wrapped_compiled_fn(args: list[Any]):
            deduped_args = self.remove_dupe_args(args)
            args.clear()
            return compiled_fn(deduped_args)

        wrapped_compiled_fn._boxed_call = True  # type: ignore[attr-defined]

        # This can be uncommented when we properly guard for duplicates,
        # but right now we must not do it.
        # if not config.debug_assert:
        #     return wrapped_compiled_fn

        @wraps(wrapped_compiled_fn)
        def debugged_compiled_fn(args):
            # Test that the computed remove/add arg functions are an inverse
            new_args = self.add_dupe_args(self.remove_dupe_args(args))
            seen: dict[Any, None] = {}
            for i, (x, y) in enumerate(zip(new_args, args)):
                seen[y] = None
                assert x is y, format_guard_bug_msg(
                    aot_config,
                    f"{describe_input(i, aot_config)} would be a duplicate of "
                    f"{describe_input(self.add_dupe_map[i], aot_config)}",
                )
            # This is only an error if there is metadata mutation on both of
            # the duped arguments; in this case, we need to know what order
            # the metadata mutation applies in.  You'll get the correct result
            # otherwise, because a graph that assumes distinct inputs works if
            # you dupe the inputs (the gradient contributions from each input
            # will get summed up appropriately.)
            #
            # TODO: work out how to setup this assert correctly
            """
            assert len(seen) == unique_args, format_guard_bug_msg(aot_config,
                f"there would be {unique_args} distinct arguments"
            )
            """
            return wrapped_compiled_fn(args)

        debugged_compiled_fn._boxed_call = True  # type: ignore[attr-defined]

        return debugged_compiled_fn


# This layer handles the situation where you have two inputs that alias each other,
# and one of the inputs is mutated.
# We need to take special care to ensure that the mutation is applied to the other aliases in the graph.
#
# pre-condition: AOTDedupWrapper has already run.
# (This function will in theory work if there are duplicate args.
# However, the synthetic base code path is a bit sub-optimal, and running with dupe'd inputs
# would cause us to hit that path more frequently).
@dataclass
class AOTSyntheticBaseWrapper(CompilerWrapper):
    # Currently, the only reason we need to plumb this bool is because
    # the synthetic base code prohibits more cases in the autograd case than the inference case.
    trace_joint: bool  # TODO: refactor trace_joint
    needs_post_compile: bool = True
    aliased_arg_idx_with_metadata_mutations: list[int] = field(default_factory=list)

    def pre_compile(
        self,
        flat_fn: TraceFn,
        flat_args: list[FxValue],
        flat_args_descs: list[AOTInput],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ) -> tuple[Callable, list[FxValue], list[AOTInput], ViewAndMutationMeta]:
        is_inference = not self.trace_joint
        (
            flat_args_with_synthetic_bases,
            flat_args_descs_with_synthetic_bases,
            synthetic_base_info,
        ) = merge_view_inputs(
            aot_config,
            flat_args,
            flat_args_descs,
            fw_metadata.input_info,
            is_inference=is_inference,
        )

        # Happy path: we don't need synthetic bases
        if synthetic_base_info is None:
            self.needs_post_compile = False
            return flat_fn, flat_args, flat_args_descs, fw_metadata

        # export path: ban synthetic bases for now, add later if requested.
        if requires_subclass_dispatch(flat_args, fw_metadata):
            raise RuntimeError(
                """\
        Encountered aliased inputs that are mutated in the graph, but at least one input/output
        to the graph is a tensor subclass. This is not supported today. You can try to
        remove the aliasing yourself as a workaround, or otherwise file an issue on github."""
            )

        if aot_config.is_export:
            raise RuntimeError(
                f"""\
        Encountered aliased inputs that are mutated in the graph you are trying to export.
        This functionality is currently not supported. If needed, please file a github issue.

        synthetic_base_info={str(synthetic_base_info)}

        fw_metadata={str(fw_metadata)}
                """
            )

        assert len(fw_metadata.input_info) == len(synthetic_base_info)

        # Update our forward metadata to take synthetic bases into account
        (
            fw_metadata_updated,
            aliased_arg_idx_with_metadata_mutations,
        ) = create_synthetic_base_metadata(
            fw_metadata,
            synthetic_base_info,
            flat_args,
            flat_args_with_synthetic_bases,
            flat_args_descs_with_synthetic_bases,
        )
        # Save old input args for post-compile
        self.old_input_info = fw_metadata.input_info

        self.aliased_arg_idx_with_metadata_mutations = (
            aliased_arg_idx_with_metadata_mutations
        )
        replay_views = config.view_replay_for_aliased_outputs

        def _unpack_synthetic_bases(primals: tuple[Any, ...]) -> list[Any]:
            f_args_inner = []
            # pyrefly: ignore [not-iterable]
            for inner_idx_or_tuple in synthetic_base_info:
                if isinstance(inner_idx_or_tuple, int):
                    f_args_inner.append(primals[inner_idx_or_tuple])
                else:
                    inner_base_idx, view_tensor = inner_idx_or_tuple
                    base = primals[inner_base_idx]
                    view_arg = gen_alias_from_base(
                        base,
                        view_tensor,
                        view_tensor.requires_grad,
                        replay_views=replay_views,
                    )
                    f_args_inner.append(view_arg)
            return f_args_inner

        @simple_wraps(flat_fn)
        def wrapped_flat_fn(*args):
            unpacked_args = _unpack_synthetic_bases(args)
            # This is a bit subtle. The goal of this entire function (aot_dispatch_synthetic_bases)
            # is to relieve the downstream logic from having to reason about mutations on inputs that alias
            # each other, by replacing aliased inputs with a synthetic base.
            # One area where this breaks down a bit however is if one of those aliased inputs
            # experienced a metadata mutation.
            # We are now obligated to reapply the metadata mutation directly to the user's input;
            # it isn't enough to apply mutations back to the synthetic base in the downstream logic.
            #
            # The way we handle this is by pretending that those aliased inputs that experience metadata mutations
            # are additional outputs in the user's forward function.
            # The downstream logic will just treat these as "user outputs that alias inputs".
            # However, we will manually grab them at runtime here, use them to reapply the metadata mutation
            # to the user inputs, and not return them to the user.
            aliased_args_with_metadata_mutations = [
                x
                for i, x in enumerate(unpacked_args)
                if i in self.aliased_arg_idx_with_metadata_mutations
            ]
            out, out_descs = call_and_expect_output_descs(flat_fn, unpacked_args)
            if len(aliased_args_with_metadata_mutations) > 0:
                # TODO: record more detailed desc information here
                return (*out, *aliased_args_with_metadata_mutations), (
                    *out_descs,
                    *(
                        [
                            MetadataMutationAOTOutput(i)
                            for i in range(
                                len(self.aliased_arg_idx_with_metadata_mutations)
                            )
                        ]
                    ),
                )
            else:
                return out, out_descs

        if config.debug_assert:
            ref_fw_metadata = run_functionalized_fw_and_collect_metadata(
                without_output_descs(wrapped_flat_fn),
                flat_args_descs=flat_args_descs_with_synthetic_bases,
                static_input_indices=aot_config.static_input_indices,
                keep_input_mutations=fw_metadata.keep_input_mutations,
                is_train=fw_metadata.is_train,
            )(*flat_args_with_synthetic_bases)
            assert ref_fw_metadata == fw_metadata_updated, (
                f"ref_metadata={pprint.pformat(partial_flatten_asdict(ref_fw_metadata))}, "
                f"\nactual_metadata={pprint.pformat(partial_flatten_asdict(fw_metadata_updated))}"
            )
        return (
            wrapped_flat_fn,
            flat_args_with_synthetic_bases,
            flat_args_descs_with_synthetic_bases,
            fw_metadata_updated,
        )

    def post_compile(
        self,
        compiled_fn,
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ):
        if not self.needs_post_compile:
            return compiled_fn

        is_inference = not self.trace_joint

        @wraps(compiled_fn)
        def wrapped_compiled_fn(args):
            # TODO: this sure seems expensive to run at runtime (which
            # post_compile seems to imply it does?!)
            args_with_synthetic_bases, _, synthetic_base_info = merge_view_inputs(
                aot_config, args, None, self.old_input_info, is_inference=is_inference
            )
            assert synthetic_base_info is not None
            aliased_args_w_metadata_mutations = [
                args[i] for i in self.aliased_arg_idx_with_metadata_mutations
            ]
            num_aliased_args_with_metadata_mutations = len(
                aliased_args_w_metadata_mutations
            )
            args.clear()
            outs = compiled_fn(args_with_synthetic_bases)
            if num_aliased_args_with_metadata_mutations > 0:
                # This code does not handle **all** input metadata mutations.
                # Instead, it only handles metadata mutations on inputs that were converted into synthetic bases
                # (which only happens if at least one aliased input experienced a data mutation).
                # e.g:
                # def f(a, b):
                #     a.mul_(2)
                #     b.t_(1, 0)
                # f(x.view(2, 2), x.view(2, 2))
                mutated_metadata_inps = outs[-num_aliased_args_with_metadata_mutations:]
                user_outs = outs[:-num_aliased_args_with_metadata_mutations]
                for inp, mutated_inp in zip(
                    aliased_args_w_metadata_mutations, mutated_metadata_inps
                ):
                    inp.as_strided_(
                        mutated_inp.size(),
                        mutated_inp.stride(),
                        mutated_inp.storage_offset(),
                    )
                return user_outs
            return outs

        return wrapped_compiled_fn


# Note [Handling mutations on an input that aliases other inputs]
# The easiest example to show-case this edge case is here:
#
# def f(a, b):
#     a.mul_(2)
#     out = a + b
#     return out
# b = torch.ones(...)
# a = b.view(-1)
# f(a, b)
#
# In this situation, if a and b happened to be aliased, we need to trace something different!
# Suppose we had b = a.view(-1)
# (In this case, that means that `a._base is b`)
#
# We need to ensure that the aliasing relationship between a and b is preserved.
# We do that detecting the specific situation above (mutate an input that aliases another input),
# and when we do that, we create a synthetic base argument. Then inside of the traced forward,
# we regenerate a and b off of that base.
# The complete example of the transformed function looks like this:
#
# // The traced forward takes in a synthetic base, and regenerates the aliased inputs as views
# // We could consider getting view-replay support here to minimize as_strided_scatter ops in the graph
# def traced_forward(base):
#     a = base.as_strided(...)
#     b = base.as_strided(...)
#     a_updated = a.mul(2)
#     base_updated = torch.as_strided_scatter(base, a_updated, ...)
#     b_updated = base_updated.as_strided(...)
#     out = a_updated + b_updated
#     return a_updated, out
#
# def compiled_fn(a, b):
#     // we detect that a is the "differentiable base" here
#     base = a
#     // In other situations, we might do either:
#     // (1) a and b are both views off of some larger differentiable base
#     //     assert a._base is b._base and a._base is not None
#     //     base = a._base
#     // (2) a and b both don't require gradients. Create a base from the storage
#     //     assert a._base is None and b._base is None
#     //     base = torch.Tensor(a.storage())
#     a_updated, out = traced_forward(base)
#     a.copy_(a_updated)
#     return out
#
# This function:
# (1) Merges input views into a synthetic base argument, when any of those input views are mutated
# (2) Returns metadata telling the autograd.Function how to modify their arguments properly,
#     to respect the new calling convention.
#
# The calling convention is as follows.
# Any inputs that were originally views of one another get yanked, and replaced with a synthetic base.
# The argument list ordering goes [base1, ..., baseN], [arg1, ..., argN],
# Where the ordering of the bases is determined from the ordering of the original view args.
# baseA will come before baseB if the earliest original argument coming from baseA
# showed up earlier in the argument list than the earliest original argument coming from baseB.
#
# Example, given some tensors a, b, c, d
# call site:
#   f(a, c.view(-1), b.view(-1), b, c, d)
# Modified argument list:
#   c_base comes first because the first c view came earlier in arg list than the first b view
#   a and d still show up in the modified arg list, but b and c don't- they're regenerated from their bases
#   b_base = torch.Tensor(b.storage())
#   c_base = torch.Tensor(c.storage())
#   f(c_base, b_base, a, d)
def merge_view_inputs(
    aot_config: AOTConfig,
    fwd_inputs: list[Any],
    # This is None when called at runtime from post_compile closure
    fwd_inputs_descs: Optional[list[AOTInput]],
    mutated_input_info: list[InputAliasInfo],
    *,
    # The autograd case currently has more restrictions than the inference case.
    is_inference: bool,
) -> tuple[
    list[Any], list[AOTInput], Optional[list[Union[int, tuple[int, torch.Tensor]]]]
]:
    if fwd_inputs_descs is None:
        fwd_inputs_descs = [DummyAOTInput(i) for i in range(len(fwd_inputs))]

    def _are_differentiable_views(view1, view2):
        if view1 is view2:
            return True
        if view1._base is None and view2._base is None:
            return False
        if view1._base is view2._base or view1._base is view2 or view1 is view2._base:
            return True
        return False

    def _same_dtype_views(view1, view2):
        if view1.dtype != view2.dtype:
            return False
        if view1._base is not None and view1.dtype != view1._base.dtype:
            return False
        if view2._base is not None and view2.dtype != view2._base.dtype:
            return False
        return True

    assert len(fwd_inputs) == len(mutated_input_info)
    if not [info for info in mutated_input_info if info.mutates_data]:
        # Return early when there are no mutations.
        return fwd_inputs, fwd_inputs_descs, None

    storage_ref_to_idx: dict[StorageWeakRef, list[int]] = collections.defaultdict(list)
    base_args = []
    other_args = []
    base_args_descs = []
    other_args_descs = []
    for i, (inpt, source) in enumerate(zip(fwd_inputs, fwd_inputs_descs)):
        if isinstance(inpt, Tensor):
            storage_ref = StorageWeakRef(inpt.untyped_storage())
            storage_ref_to_idx[storage_ref].append(i)
        else:
            other_args.append(inpt)
            other_args_descs.append(source)
    # Note [Synthetic Base Info Metadata]
    # This list contains metadata that tells you what the i'th argument in the inner calling convention should be.
    # It's either:
    # - another int (corresponding to the index in the argument list of the element from the outer calling convention)
    # - idx, view_tensor, where we can generate the new output with view_tensor._view_func(old_args[idx])
    #   idx corresponds to which synthetic base from the outer calling context to view
    inner_calling_convention_meta: dict[int, Union[int, tuple[int, torch.Tensor]]] = {}
    for aliased_input_indices in storage_ref_to_idx.values():
        if len(aliased_input_indices) <= 1 or not any(
            # We only care about mutations that affect all aliases,
            # so metadata mutations on an input doesn't require us to do synthetic base handling.
            mutated_input_info[inpt_idx].mutates_data
            for inpt_idx in aliased_input_indices
        ):
            other_args.extend(
                fwd_inputs[curr_idx] for curr_idx in aliased_input_indices
            )
            other_args_descs.extend(
                fwd_inputs_descs[curr_idx] for curr_idx in aliased_input_indices
            )
            continue

        # Here, we attempt to do a more complicated check to detect false aliasing
        # (e.g. if all the tensors have the same storage, but don't actually overlap)
        # In theory, we could have a large group of tensors that all share storages, where only *some* of them
        # have overlapping memory.
        # I don't bother with that case for now: here, we only bail out earlier if we detect that **every** pair
        # of tensors in the current group that shares a storage is non-overlapping.
        aliased_input_indices_no_false_sharing = compute_overlapping_inputs(
            aot_config, fwd_inputs, aliased_input_indices
        )
        if len(aliased_input_indices_no_false_sharing) <= 1:
            other_args.extend(
                fwd_inputs[curr_idx] for curr_idx in aliased_input_indices
            )
            other_args_descs.extend(
                fwd_inputs_descs[curr_idx] for curr_idx in aliased_input_indices
            )
            continue

        # We detected an input that was mutated, AND aliases with another input.
        # we need to replace this set of aliased inputs with a single synthetic base.
        # For now, I'm banning a bunch of cases. We expect dynamo to properly detect these cases
        # and error out. We can fix them later.
        # These checks are transitive, so we don't need to check every pair.
        for idx1, idx2 in zip(
            aliased_input_indices, aliased_input_indices[1:], strict=False
        ):
            view1 = fwd_inputs[idx1]
            view2 = fwd_inputs[idx2]
            # The "inputs that are aliased but have different differentiable bases" case
            # is more complicated and hopefully pretty rare. Not currently handled.
            if not is_inference:
                assert _are_differentiable_views(view1, view2), (
                    "aot_autograd() does not yet handle non-differentiable view input mutations."
                )
            # Regenerating views when reinterpreting complex / real tensors seems non-trivial,
            # not handling for now
            assert _same_dtype_views(view1, view2), (
                "aot_autograd() does not yet handle input mutations on views with different dtypes."
            )
        non_none_bases = [
            (i, fwd_inputs[i]._base)
            for i in aliased_input_indices
            if fwd_inputs[i]._base is not None
        ]
        aliases_with_none_bases = [
            fwd_inputs[i] for i in aliased_input_indices if fwd_inputs[i]._base is None
        ]
        synthetic_base_desc: AOTInput
        if len(non_none_bases) == 0:
            # Case where none of the aliases have a ._base
            # we generate a synthetic base without gradients, and generate views off of it
            # We hit this case when we have input tensors to the graph that share a storage,
            # but do not have a ._base field.
            # Wondering when we hit this case?
            # The _base field simply says that autograd knows about the aliasing relationship,
            # but sometimes we create tensors which are aliased out of the same storage but guaranteed
            # to be disjoint. In these cases, we will skip setting up the _base relationship
            # for performance reasons (because the fact that the tensors share the same storage
            # is unobservable unless you (1) do naughty things with resize_/as_strided
            # or (2) look at the storage--as we are doing here.)
            # One particular example of this is optimizer steps on the LSTM module:
            # LSTM parameters are packed into a contiguous storage for efficiency reasons when
            # calling cuDNN kernels, so when these parameters get passed to the optimizer we will
            # find they share the same storage, but do not have _base set since they are all disjoint.
            #
            # NOTE: There is one case where this is unsafe:
            # torch.Tensor(storage) will ALWAYS create a 1D tensor, which is not necessarily
            # the same shape as the "actual" base that the tensor came from.
            # For the most part this is fine, because we always use as_strided()
            # to generate the original aliased inputs again.
            # If we were to use view-replay though, this could cause the aliased views
            # to have incorrect sizes.
            example_idx = aliased_input_indices[0]
            example_alias = fwd_inputs[example_idx]
            # Note that this function is reused at both trace time and runtime.
            # At trace time, we're under a FakeMode so synthetic_base becomes a FakeTensor.
            synthetic_base = torch.empty(
                (0,), dtype=example_alias.dtype, device=example_alias.device
            )
            # We don't actually have a convenient way of going from storage -> tensor,
            # So using set_() here (we suffer some minor overhead, but this case is rare).
            synthetic_base.set_(example_alias.untyped_storage())
            synthetic_base_desc = SyntheticBaseAOTInput(fwd_inputs_descs[example_idx])
        else:
            # Case where all of the aliases require gradients, and have the same _base.
            i, synthetic_base = non_none_bases[0]
            synthetic_base_desc = ViewBaseAOTInput(fwd_inputs_descs[i])
            for _, other_base in non_none_bases[1:]:
                assert other_base is synthetic_base, (
                    "aot_autograd() does not yet handle non-differentiable view input mutations."
                )
            for alias in aliases_with_none_bases:
                assert alias is synthetic_base, (
                    "aot_autograd() does not yet handle non-differentiable view input mutations."
                )
        base_args.append(synthetic_base)
        base_args_descs.append(synthetic_base_desc)
        for curr_view_idx in aliased_input_indices:
            curr_view = fwd_inputs[curr_view_idx]
            base_idx = len(base_args) - 1
            # We store just enough info here so that we can regenerate the view later.
            # Regeneration: curr_view._view_func(args[base_idx])
            inner_calling_convention_meta[curr_view_idx] = (base_idx, curr_view)
    if len(base_args) == 0:
        assert len(other_args) == len(fwd_inputs)
        # If no synthetic bases are necessary, just return the original inputs.
        return fwd_inputs, fwd_inputs_descs, None
    else:
        from torch.fx.experimental.symbolic_shapes import SymIntEqByExpr

        def make_hashable(arg):
            if isinstance(arg, torch.SymInt):
                # Since only nested SymInt objects can be hashed, we wrap them with
                # SymIntEqByExpr, which is a hashable wrapper of SymInts.
                return SymIntEqByExpr(arg)
            return arg

        # Otherwise, return:
        # (1) The new args according to the updated calling convention: (synthetic_bases, other_args)
        # (2) Metadata telling functionalization how to generate the inner argument list given the outer calling convention.
        #     We post-process it into a list, where meta[i] tells you info about the i'th argument in the inner calling convention.
        args_to_functionalization = base_args + other_args
        args_to_functionalization_descs = base_args_descs + other_args_descs

        # Map each argument into its old index.
        # There may be some repeated arguments, so we collect their indices in a list.
        arg_to_old_idx_map = collections.defaultdict(list)
        for i, arg in enumerate(fwd_inputs):
            arg_to_old_idx_map[make_hashable(arg)].append(i)
        # Reverse the list of each argument, so that we can easily pop them one-after-the-other in order.
        for hashable_arg in arg_to_old_idx_map:
            arg_to_old_idx_map[hashable_arg] = list(
                reversed(arg_to_old_idx_map[hashable_arg])
            )

        for i, other_arg in enumerate(other_args):
            new_idx = len(base_args) + i
            old_idx = arg_to_old_idx_map[make_hashable(other_arg)].pop()
            inner_calling_convention_meta[old_idx] = new_idx

        # post process into a list
        post_processed_calling_convention_meta: list[
            Union[int, tuple[int, torch.Tensor]]
        ] = [-1 for _ in range(len(inner_calling_convention_meta))]
        for k, v in inner_calling_convention_meta.items():
            post_processed_calling_convention_meta[k] = v
        # Quick assert: every argument in the inner calling convention should be accounted for.
        for x in post_processed_calling_convention_meta:
            assert x != -1
        return (
            args_to_functionalization,
            args_to_functionalization_descs,
            post_processed_calling_convention_meta,
        )


# Note: [Backward graph lazy lowering]
# After AOTDispatch traces the backward for graphs requiring autograd, we will lower the graph lazily,
# unless we suspect that inductor might specialize and insert additional guards. When we do lazy
# lowering, we stash the AOT backward graph (bw_module) in this class.
#
# Lowering passes are performed on a deepcopy of this bw_module due to compatibility
# with compiled autograd. See: https://github.com/pytorch/pytorch/pull/149229#discussion_r2002122645.
@dataclass
class AutogradLazyBackwardCompileInfo:
    bw_module: Callable
    placeholder_list: list[Any]
    saved_context: Optional[TracingContext]
    saved_compile_context: Optional[CompileContext]


# On an AOT Autograd cache hit, we already have a lowered backward, so there is usually
# no need to keep information around for a new lazy compilation. Except for compiled autograd,
# which wants to retrace this backward into a larger graph, and it needs the graph module to do so.
@dataclass
class CachedAutogradLazyBackwardCompileInfo:
    bw_module_fn: Callable


def _raise_if_functorch_active():
    # not ideal but prevent the user from seeing a nasty traceback - See #138422
    stack = torch._C._functorch.peek_interpreter_stack()
    torch._check(
        stack is None,
        lambda: (
            "It looks like you're trying to call a compiled backward function within vmap/grad/vjp, "
            "which isn't supported. Try wrapping vmap inside torch.compile, or skip compiling the "
            "backward function."
        ),
    )


# NOTE: this function must be torch._dynamo.allow_in_graph-able. Non tensor/symnode inputs must be constants.
def _backward_prologue_functional(
    ctx_saved_tensors,
    ctx_symints,
    ctx_opaque_objects,
    metadata,
    maybe_subclass_metadata,
    *flat_args,
):
    # Calling convention: we expect a grad_out passed to the backward:
    # - for every output of the fw that does *not* alias an input or graph intermediate
    # - for every updated_input generated by the fw that does *not* alias an input (aka only data-mutations)
    # - for every graph intermediate that we need to use to generate an output later.
    # The other outputs in the autograd.Function.forward that do *not* show up in the backward include:
    # - outputs that alias inputs or graph intermediates
    # - updated inputs due to metadata-only mutations.
    # We need to return them in the forward, but ensure that they all do not get gradients in the backward,
    # and we filter them out here before passing the remaining grad_outputs into the compiled backward.
    _raise_if_functorch_active()

    num_intermediate_bases = metadata.num_intermediate_bases
    num_mutated_runtime_inps = metadata.num_mutated_inp_runtime_indices
    expected_grad_outs = (
        metadata.num_outputs + num_mutated_runtime_inps + num_intermediate_bases
    )
    deterministic = metadata.deterministic
    global_deterministic = torch.are_deterministic_algorithms_enabled()
    if deterministic is not None:
        torch._check(
            not (not deterministic and global_deterministic),
            lambda: (
                "This compiled backward function is being run with "
                "torch.use_deterministic_algorithms(True), "
                "but it was previously generated during the forward function while "
                "torch.use_deterministic_algorithms(False) was set."
            ),
        )

    assert len(flat_args) == expected_grad_outs
    out_info = metadata.output_info

    inp_tangents, out_tangents, intermediate_base_tangents = (
        flat_args[:num_mutated_runtime_inps],
        flat_args[
            num_mutated_runtime_inps : num_mutated_runtime_inps + metadata.num_outputs
        ],
        flat_args[num_mutated_runtime_inps + metadata.num_outputs :],
    )
    # input_info contains info on *every* input,
    # But in the backward(), we are only given grad outputs for every mutated input
    # We then need to filter out the grad outputs that correspond to metadata-only mutations or don't require grad
    input_info = metadata.input_info
    inp_tangents_filtered = [
        x
        for x, info_idx in zip(
            inp_tangents,
            metadata.mutated_inp_runtime_indices,
        )
        if input_info[info_idx].mutates_data and input_info[info_idx].requires_grad
    ]
    # We also need to filter out grad outputs that correspond to outputs aliasing inputs/intermediates
    out_tangents_filtered = [
        x
        for x, info in zip(out_tangents, out_info)
        if info.output_type
        in [
            OutputType.non_alias,
            OutputType.unsafe_view_alias,
            OutputType.custom_function_view,
        ]
        and issubclass(info.raw_type, torch.Tensor)
        and info.requires_grad
    ]
    # intermediate bases always require gradients, and always participate in the backward graph.
    flat_bw_args_with_grads = [
        *inp_tangents_filtered,
        *out_tangents_filtered,
        *intermediate_base_tangents,
    ]
    num_flat_bw_args_with_grads = len(flat_bw_args_with_grads)

    # sanity asserts
    # metadata_only_inps = [
    #     x for x, info_idx in zip(inp_tangents, mutated_inp_indices)
    #     if not input_info[info_idx].mutates_data
    # ]
    # aliased_outputs = [
    #     x for x, info in zip(out_tangents, out_info) if info.output_type != OutputType.non_alias]
    # assert all(x is None for x in metadata_only_inps)
    # assert all(x is None for x in aliased_outputs)
    # TODO: replace this with FunctionalizedRngRuntimeWrapper
    rng_args = []
    if metadata.is_rng_op_functionalized:
        # Add the seed and offset to args
        rng_args = CUDARngStateHelper.get_torch_state_as_tuple()

    bw_tokens = [None] * metadata.num_backward_tokens

    # - note: donated buffer logic requires (*ctx.symints, *ctx.saved_tensors) showing up first
    #   in the bw output order.

    # Every dereference of ctx.saved_tensors incurs saved_tensors_hooks calls
    # There are tests that count these calls, saving to var.
    num_ctx_saved_tensors = len(ctx_saved_tensors)
    all_args = [
        *ctx_symints,
        *ctx_saved_tensors,
        *ctx_opaque_objects,
        *flat_bw_args_with_grads,
        *bw_tokens,
        *rng_args,
    ]
    del ctx_saved_tensors

    # Note: [AOTAutograd Backward Guards]
    # During AOTDispatch, we eagerly create and trace out a joint fw-bw graph.
    # Doing so requires us to "guess" about some of the metadata of our grad_outputs.
    #
    # In particular: if an output to the forward is a plain tensor or a subclass,
    # its corresponding grad_output in the backward **may or may not** be
    # a plain tensor or a subclass. The main cases are:
    # (1) If an output is a plain tensor, its grad_out will also be a plain tensor,
    #     *unless* the output is used in some subclass compute later in the forward graph,
    #     which will cause its grad_output to become a subclass
    # (2) If an output is a subclass, its grad_out will also be a subclass,
    #     *unless* the output of the forward did not actually participate in the gradient computation,
    #     in which case autograd will insert a plain tensor of zeros for the grad_output.
    #     We could avoid this case with `torch.autograd.Function.set_materialize_grads`,
    #     although this is not turned on today in AOTAutgrad and would require more work.
    #
    # Today, we make a guess on subclass-ness based on the above examples,
    # and hard-error in the backward if we guessed wrong.
    #
    # In the future, we should add backward guards that would allow us to
    # properly handle this case instead of erroring: we would need to retrace the backward graph,
    # since we might produce an entirely different trace if our grad_outputs are subclass or not.
    del flat_bw_args_with_grads

    tangents_start_idx = (
        len(all_args) - num_flat_bw_args_with_grads - len(rng_args) - len(bw_tokens)
    )
    assert tangents_start_idx == len(ctx_symints) + num_ctx_saved_tensors + len(
        ctx_opaque_objects
    )
    tangents_end_idx = len(all_args) - len(rng_args) - len(bw_tokens)

    # TODO: figure out how to refactor the backward properly
    # so I can use aot_dispatch_subclass_wrapper() here.
    if maybe_subclass_metadata is not None:
        tangents = all_args[tangents_start_idx:tangents_end_idx]

        if len(tangents) != len(metadata.subclass_tangent_meta):
            raise RuntimeError(
                "The grad inputs should be same number as forward output tangents"
            )

        flat_processed_tangents = list(
            itertools.chain.from_iterable(
                (
                    AOTDispatchAutograd.process_runtime_tangent(
                        t,
                        m,
                    )[1]
                )
                for t, m in zip(
                    tangents,
                    metadata.subclass_tangent_meta,
                )
            )
        )

        all_args = (
            runtime_unwrap_tensor_subclasses(
                all_args[:tangents_start_idx],
                # SymInts that are inputs to the backward graph are
                # already included in the "all_args" list.
                # Any symints coming from tensor subclasses should always
                # come from primals, and so they will show up as extra
                # arguments to the forward graph, and they will be saved
                # as activation in the backward graph.
                append_symints=False,
            )
            + flat_processed_tangents
            + runtime_unwrap_tensor_subclasses(
                all_args[tangents_end_idx:],
                append_symints=False,
            )
        )
    else:
        all_args = [
            (
                AOTDispatchAutograd.process_runtime_tangent(
                    t,
                    metadata.subclass_tangent_meta[i - tangents_start_idx],
                )[0]
                if (tangents_start_idx <= i < tangents_end_idx)
                else t
            )
            for i, t in enumerate(all_args)
        ]

    # Backward with forward inputs mutations is not supported in double backward.
    if (
        torch.is_grad_enabled()
        and metadata.indices_of_inputs_that_requires_grad_with_mutations_in_bw
    ):
        raise RuntimeError(
            "aot_autograd does not support input mutations with requires_grad in backward for create_graph=True"
        )

    return all_args


def initialize_rng_states(
    num_rng: int,
    graphsafe_idx: int,
    fwd_rng_states: list[torch.Generator],
    bwd_rng_states: list[torch.Generator],
):
    """
    Initialize the cudagraph safe rng states.

    Initialization of rng states should have a few properties:
    - the initialization for each rng state should be independent
    - the initialization should be deterministic
    - the initialization should be based off current rng state, so that independent graphs do not
    have equal rng behavior

    We defer initialization of rng states until runtime because compilation is wrapped
    with preserve_rng_states. Seed initialization should advance the rng states so consecutive compilations
    do not give equal randomness.
    """
    with torch.utils._python_dispatch._disable_current_modes():
        seeds = torch.randint(0, torch.iinfo(torch.int64).max, (num_rng,), device="cpu")
        fwd_rng_states.extend(
            [
                torch.cuda.default_generators[graphsafe_idx]
                .clone_state()
                .manual_seed(int(seeds[i]))
                for i in range(num_rng)
            ]
        )
        bwd_rng_states.extend(
            [
                torch.cuda.default_generators[graphsafe_idx]
                .clone_state()
                .manual_seed(int(seeds[i]))
                for i in range(num_rng)
            ]
        )


# NOTE: this function must be torch._dynamo.allow_in_graph-able. Non tensor/symnode inputs must be constants.
def _backward_epilogue_functional(
    metadata, maybe_subclass_metadata, out, *, make_subclass_override=None
):
    # Toss out the backward output tokens
    num_bw_tokens = metadata.num_backward_tokens
    if num_bw_tokens > 0:
        out = out[:-num_bw_tokens]

    # TODO: replace this with FunctionalizedRngRuntimeWrapper.post_compile
    out = FunctionalizedRngRuntimeWrapper()._functionalized_rng_runtime_epilogue(
        metadata, out, offset_index=len(out) - 1
    )
    out = tuple(out)

    # TODO: figure out how to refactor the backward properly so I can use aot_dispatch_subclass_wrapper() here.
    if maybe_subclass_metadata is not None:
        assert maybe_subclass_metadata.grad_input_metas is not None
        outs_wrapped = wrap_tensor_subclasses(
            out,
            subclass_metas=maybe_subclass_metadata.grad_input_metas,
            included_subclass_symints=True,
            is_runtime=True,
            make_subclass_override=make_subclass_override,
        )
        return outs_wrapped
    return out


def coerce_to_expected_memory_format(x: torch.Tensor, memory_format: MemoryFormatMeta):
    if memory_format.memory_format is not None:
        # Coerce to torch.memory_format
        if not x.is_contiguous(memory_format=memory_format.memory_format):
            x = x.contiguous(memory_format=memory_format.memory_format)
        return x

    expected_size = memory_format.size
    assert expected_size is not None
    expected_stride = memory_format.stride
    assert expected_stride is not None
    # Expected size and stride are static ints
    # ok to use == to compare runtime tensor strides and shapes

    if x.shape == expected_size and x.stride() == expected_stride:
        # Runtime tangent size and stride are the same as expected, no need to coerce
        return x

    # Empty_strided creates a raw Tensor.
    # We are guaranteed that only raw Tensors has expected size and stride.
    # Subclasses have only expected memory_format.
    restrided = torch.empty_strided(
        size=expected_size,
        stride=expected_stride,
        dtype=x.dtype,
        device=x.device,
        layout=x.layout,
        requires_grad=x.requires_grad,
    )
    restrided.copy_(x)
    return restrided


@contextlib.contextmanager
def _disable_saved_tensors_hooks():
    error_message = (
        "Saved tensors hooks were specialized as GraphModules."
        "In this case aot_autograd inlines them in forward and backward graph "
        "and disables them during runtime of aot_autograd compiled region."
        "If you see this error, that means that there is some unexpected push or pop manipulation "
        "during aot_autograd compiled region runtime."
        "Compilation with different hooks must result in recompilation."
    )
    fail_if_non_empty = False
    maybe_prev_message = None
    try:
        maybe_prev_message = (
            torch._C._autograd._saved_tensors_hooks_get_disabled_error_message()
        )
        torch._C._autograd._saved_tensors_hooks_disable(
            error_message, fail_if_non_empty
        )
        yield
    finally:
        if maybe_prev_message is None:
            torch._C._autograd._saved_tensors_hooks_enable()
        else:
            torch._C._autograd._saved_tensors_hooks_disable(
                maybe_prev_message, fail_if_non_empty
            )


@dataclass
class SerializableCompiledFunction:
    """
    Represents a result of AOTDispatch after calling the inner compiler
    that can be serialized
    """

    compiled_fn: Callable
    serialize_fn: Callable

    def __init__(self, compiled_fn: Callable, serialize_fn: Callable):
        self.compiled_fn = compiled_fn
        self.serialize_fn = serialize_fn
        # Equivalent to functools.wraps
        functools.update_wrapper(
            self,
            compiled_fn,
            assigned=("__doc__", "__annotations__", "__type_params__"),
        )

    def serialize(self) -> Any:
        return self.serialize_fn()

    def __call__(self, *args, **kwargs):
        return self.compiled_fn(*args, **kwargs)


# This is wrapped in a class just for namespacing purposes
# No need to make it into an actual CompilerWrapper because it doesn't fit the abstract as cleanly
class AOTDispatchAutograd:
    @staticmethod
    def process_runtime_tangent(x, meta: Union[PlainTensorMeta, SubclassCreationMeta]):
        if not isinstance(x, torch.Tensor):
            return x, [x]

        if isinstance(x, FakeTensor):
            assert meta.memory_format
            x = coerce_to_expected_memory_format(x, meta.memory_format)
            return x, [x]

        expected_type: Optional[type] = torch.Tensor
        expected_meta = None
        if isinstance(meta, SubclassCreationMeta):
            expected_type = meta.original_subclass_type
            expected_meta = meta.meta

        runtime_type = type(x)
        # When we're inside compiled autograd's AOTDispatcher step,
        # regular Tensors look like FunctionalTensors.
        # Tensor subclasses still look like Tensor subclasses though.
        if isinstance(x, torch._subclasses.functional_tensor.FunctionalTensor):
            runtime_type = torch.Tensor

        runtime_meta = None
        runtime_subclass_keys: Sequence[str] = []

        if is_traceable_wrapper_subclass(x):
            runtime_subclass_keys, runtime_meta = x.__tensor_flatten__()

        def maybe_coerce(x):
            same_type: bool = expected_type == runtime_type
            same_meta: bool = expected_meta == runtime_meta

            if same_type and same_meta:
                return x

            if not hasattr(x, "__coerce_same_metadata_as_tangent__"):
                return None

            if same_type:
                # Backward Compatibility, as some Subclass impls can have original 1-arg function.
                return x.__coerce_same_metadata_as_tangent__(expected_meta)

            return x.__coerce_same_metadata_as_tangent__(expected_meta, expected_type)

        # Coerce to expected type and metadata
        orig_x = x
        x = maybe_coerce(x)
        if x is None:
            raise RuntimeError(
                f"""
During the backward, we encountered a tensor subclass where we guessed its
metadata incorrectly.

Expected metadata: {str(expected_meta)}, expected type: {str(expected_type)}

Runtime metadata: {str(runtime_meta)}, runtime type: {str(runtime_type)}

shape: {str(orig_x.shape)}
To fix this, your tensor subclass must implement the dunder method __force_to_same_metadata__.
"""
            )

        # Coerce to expected memory format
        assert meta.memory_format
        x = coerce_to_expected_memory_format(x, meta.memory_format)

        if not is_traceable_wrapper_subclass(x):
            return x, [x]

        assert isinstance(meta, SubclassCreationMeta)
        if orig_x is not x:
            runtime_subclass_keys = x.__tensor_flatten__()[0]

        assert len(meta.attrs) == len(runtime_subclass_keys)
        leaves = []
        for attr, attr_meta in meta.attrs.items():
            elem = getattr(x, attr)
            new_elem, elem_leaves = AOTDispatchAutograd.process_runtime_tangent(
                elem, attr_meta
            )
            if new_elem is not elem:
                setattr(x, attr, new_elem)
            leaves.extend(elem_leaves)

        return x, leaves

    @staticmethod
    def post_compile(
        compiled_fw_func,  # fw_module after compilation + wrappers
        compiled_bw_func,  # bw_module after compilation + wrappers
        maybe_subclass_meta: Optional[SubclassMeta],
        num_symints_saved_for_bw_: int,
        backward_state_indices: list[int],
        disable_amp: bool,
        indices_of_inps_to_detach: list[int],
        lazy_backward_info: Optional[
            Union[
                AutogradLazyBackwardCompileInfo,
                CachedAutogradLazyBackwardCompileInfo,
            ]
        ],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,  # runtime metadata
        try_save_cache_entry: Optional[Callable],  # Serialization function
    ):
        # For additional context see Note [CUDA Graph Safe RNG Functionalization]
        # Each pair forward, backward rng states must be equal prior to its invocation on any
        # iteration of forward, backward. Because they are initialized equal, and are computing the same rng op,
        # running forward then backward advances them the same amount and keeps them equal.
        # However, a user may invoke multiple forwards, then backwards, such that they are not in sync.
        # Initially we have:
        # fwd_state0 == bwd_state0.
        # Lets say we run:
        # fwd0: fwd_state0 -> fwd_state1
        # fwd1: fwd_state1 -> fwd_state2
        # fwd2: fwd_state2 -> fwd_state3
        # If we now invoke bwd2,
        # we need to update bwd_state equal to the rng that was observed in fwd2.
        # we save the rng_state fwd_state2 in forward because we detect that it is not the
        # current backward state and therefore would not be accessible if we do not save it.
        # Similarly, if we are going to update the backward state to a new value, and there is a pending
        # forwards which needs its current state, we will save it.
        # Within the autograd context, we keep track of the curr iteration so that on backward
        # we know what the generator state must be before the backward is run.
        num_rng = fw_metadata.num_graphsafe_rng_states
        graphsafe_idx = fw_metadata.graphsafe_rng_state_index
        fwd_rng_states: list[torch.Generator] = []
        bwd_rng_states: list[torch.Generator] = []
        curr_fwd_iter = itertools.count(0)
        backward_state_position = 0
        pending_forwards: set[int] = set()
        saved_backward_tensor_states: dict[int, list[torch.Tensor]] = {}

        class CompiledFunction(torch.autograd.Function):
            compiled_fw = compiled_fw_func
            compiled_bw = compiled_bw_func
            metadata: ViewAndMutationMeta = fw_metadata  # type: ignore[assignment]
            maybe_subclass_metadata: Optional[SubclassMeta] = maybe_subclass_meta
            num_symints_saved_for_bw = num_symints_saved_for_bw_
            _aot_id = aot_config.aot_id
            _lazy_backward_info = lazy_backward_info

            @staticmethod
            def _compiled_autograd_key(ctx):
                return (ctx._autograd_function_id, *ctx.symints)

            @staticmethod
            # pyrefly: ignore [bad-override]
            def forward(ctx, *deduped_flat_tensor_args):
                args = deduped_flat_tensor_args
                if backward_state_indices:
                    bw_state = args[backward_state_indices[0]]
                    assert isinstance(bw_state, BackwardState)
                    ctx._compiled_autograd_backward_state = bw_state

                if num_rng:
                    if len(fwd_rng_states) == 0:
                        assert graphsafe_idx is not None
                        initialize_rng_states(
                            num_rng, graphsafe_idx, fwd_rng_states, bwd_rng_states
                        )

                    _curr_iter = next(curr_fwd_iter)
                    ctx._curr_iter = _curr_iter

                    # if this state is not contained in the backward,
                    # we need to save it for when its backward pass happens
                    if _curr_iter != backward_state_position:
                        saved_backward_tensor_states[_curr_iter] = [
                            rng_state.get_state() for rng_state in fwd_rng_states
                        ]

                    pending_forwards.add(_curr_iter)
                    args = (*args, *fwd_rng_states)

                # There is a pretty complicated calling convention around what the compiled fw returns.
                # The full list of outputs and their relative order is:
                # (*tokens, *mutated_inputs, *fw_outs, *fw_intermediate_bases, *saved_tensors, *saved_symints)
                # - Note that in the synthetic bases case, mutated_inputs will correspond to an updated version
                #   of the original view, and not the synthetic base
                # - Note that donated buffer logic requires (*saved_tensors, *saved_symints) showing up last
                #   in the fw output order.
                fw_outs = call_func_at_runtime_with_args(
                    CompiledFunction.compiled_fw,
                    # pyrefly: ignore [bad-argument-type]
                    args,
                    disable_amp=disable_amp,
                )

                num_outputs = CompiledFunction.metadata.num_outputs
                num_outputs_aliased = CompiledFunction.metadata.num_outputs_aliased
                num_mutated_runtime_inps = (
                    CompiledFunction.metadata.num_mutated_inp_runtime_indices
                )
                num_forward_returns = CompiledFunction.metadata.num_forward_returns

                # See Note [Activations with no version counter checks in eager]
                # Partitioners must put symint arguments at the end separate from tensor arguments
                # Split tensors into those that need VC checks (via save_for_backward)
                # and those that don't (stashed directly on ctx).
                # The partitioner sorts tensors so that no-VC-check tensors are at the end.
                tensors_saved_with_vc_check = fw_outs[
                    CompiledFunction.metadata.tensors_saved_for_backwards_with_vc_check_slice
                ]
                tensors_saved_no_vc_check = fw_outs[
                    CompiledFunction.metadata.tensors_saved_for_backwards_no_vc_check_slice
                ]
                assert all(
                    isinstance(x, torch.Tensor) for x in tensors_saved_with_vc_check
                )
                assert all(
                    isinstance(x, torch.Tensor) for x in tensors_saved_no_vc_check
                )

                # See Note [Detaching saved tensors in AOTAutograd]
                num_vc_check = len(tensors_saved_with_vc_check)
                tensors_to_save = [
                    x.detach() if x._is_view() else x
                    for x in tensors_saved_with_vc_check
                ]
                tensors_no_vc = [
                    x.detach() if x._is_view() else x for x in tensors_saved_no_vc_check
                ]

                # dynamic_saved_tensors_idxs has indices relative to all saved tensors
                # (vc_check + no_vc_check combined). Mark dynamics on the detached tensors.
                for (
                    idx,
                    dims,
                ) in CompiledFunction.metadata.dynamic_saved_tensors_idxs.items():
                    if idx < num_vc_check:
                        maybe_mark_dynamic_helper(tensors_to_save[idx], dims)
                    else:
                        maybe_mark_dynamic_helper(
                            tensors_no_vc[idx - num_vc_check], dims
                        )

                # Only save tensors that need VC checks via save_for_backward
                ctx.save_for_backward(*tensors_to_save)
                ctx._tensors_no_vc_check = tensors_no_vc

                symint_outs = fw_outs[
                    CompiledFunction.metadata.symints_saved_for_backwards_slice
                ]
                assert all(
                    isinstance(x, (int, float, torch.SymInt, torch.SymFloat))
                    for x in symint_outs
                ), str([type(x) for x in symint_outs])
                ctx.symints = symint_outs

                opaque_object_outs = fw_outs[
                    CompiledFunction.metadata.opaque_objects_saved_for_backwards_slice
                ]
                assert all(is_opaque_type(type(obj)) for obj in opaque_object_outs)
                ctx.opaque_objects = opaque_object_outs

                raw_returns = fw_outs[0:num_forward_returns]

                # Wrap all autograd.Function.forward() outputs that are aliases
                # so that autograd.Function doesn't treat them as tensors
                if num_mutated_runtime_inps > 0:
                    for i, idx in enumerate(
                        CompiledFunction.metadata.mutated_inp_runtime_indices
                    ):
                        # We could make this faster by only looping over inputs with metadata-only mutations
                        # (instead of looping over inputs with either data or metadata mutations), but there shouldn't be many.
                        info = CompiledFunction.metadata.input_info[idx]
                        if info.mutates_metadata and not info.mutates_data:
                            raw_return_idx = i
                            raw_returns[raw_return_idx] = TensorAlias(
                                raw_returns[raw_return_idx]
                            )

                    if config.debug_assert:
                        user_mutated_inputs_raw = raw_returns[
                            0:num_mutated_runtime_inps
                        ]
                        mut_inp_infos = [
                            x
                            for x in CompiledFunction.metadata.input_info
                            if x.mutates_data or x.mutates_metadata
                        ]
                        assert len(user_mutated_inputs_raw) == len(mut_inp_infos)

                if CompiledFunction.metadata.num_unsafe_view_outputs > 0:
                    for idx in CompiledFunction.metadata.unsafe_view_out_indices:
                        raw_return_idx = num_mutated_runtime_inps + idx
                        o = raw_returns[raw_return_idx]
                        raw_returns[raw_return_idx] = torch.ops.aten._unsafe_view(
                            o, o.shape
                        )

                if num_outputs_aliased > 0:
                    for idx in CompiledFunction.metadata.aliased_out_indices:
                        raw_return_idx = num_mutated_runtime_inps + idx
                        raw_returns[raw_return_idx] = TensorAlias(
                            raw_returns[raw_return_idx]
                        )

                    if config.debug_assert:
                        intermediates_raw = raw_returns[
                            num_mutated_runtime_inps + num_outputs :
                        ]
                        assert not any(
                            isinstance(x, TensorAlias) for x in intermediates_raw
                        )

                # invariant: intermediate bases always require gradients, so we don't have to
                # consider marking them as non-differentiable.
                raw_returns_not_including_intermediate_bases = raw_returns[
                    : num_mutated_runtime_inps + num_outputs
                ]
                raw_returns_meta = [
                    x
                    for x in CompiledFunction.metadata.input_info
                    if x.mutation_type == MutationType.MUTATED_OUT_GRAPH
                ] + CompiledFunction.metadata.output_info

                fw_outs_not_requiring_grad = [
                    x
                    for (i, x) in enumerate(
                        raw_returns_not_including_intermediate_bases
                    )
                    if isinstance(x, torch.Tensor)
                    and not raw_returns_meta[i].requires_grad
                ]
                ctx.mark_non_differentiable(*fw_outs_not_requiring_grad)
                ctx._materialize_non_diff_grads = False
                return tuple(raw_returns)

            @staticmethod
            def backward(ctx, *flat_args):
                # Combine tensors from both sources:
                # 1. ctx.saved_tensors - tensors that went through save_for_backward (with VC check)
                # 2. ctx._tensors_no_vc_check - tensors stashed directly on ctx (no VC check)
                all_args = _backward_prologue_functional(
                    (
                        list(ctx.saved_tensors) + ctx._tensors_no_vc_check
                        if len(ctx._tensors_no_vc_check) > 0
                        else ctx.saved_tensors
                    ),
                    ctx.symints,
                    ctx.opaque_objects,
                    CompiledFunction.metadata,
                    CompiledFunction.maybe_subclass_metadata,
                    *flat_args,
                )

                if num_rng:
                    nonlocal backward_state_position, bwd_rng_states
                    curr_backward_iter = ctx._curr_iter
                    retain_graph = (
                        torch._C._autograd._get_current_graph_task_keep_graph()
                    )

                    # Save current state if we have a pending forward that needs this state
                    # or this state may be needed again because of retain graph
                    if (
                        backward_state_position in pending_forwards
                        and backward_state_position not in saved_backward_tensor_states
                        and (
                            backward_state_position != curr_backward_iter
                            or retain_graph
                        )
                    ):
                        saved_backward_tensor_states[backward_state_position] = [
                            rng_state.get_state() for rng_state in bwd_rng_states
                        ]

                    # Restore saved states if needed
                    if curr_backward_iter in saved_backward_tensor_states:
                        if backward_state_position != curr_backward_iter:
                            for bwd_state, saved_state in zip(
                                bwd_rng_states,
                                saved_backward_tensor_states[curr_backward_iter],
                            ):
                                bwd_state.set_state(saved_state)
                        if not retain_graph:
                            del saved_backward_tensor_states[curr_backward_iter]
                    else:
                        assert backward_state_position == curr_backward_iter

                    backward_state_position = curr_backward_iter + 1
                    if not retain_graph:
                        pending_forwards.remove(curr_backward_iter)
                    all_args.extend(bwd_rng_states)

                def impl_fn(double_ctx=None):
                    out = CompiledFunction._backward_impl(ctx, all_args)
                    return _backward_epilogue_functional(
                        CompiledFunction.metadata,
                        CompiledFunction.maybe_subclass_metadata,
                        out,
                    )

                needs_grad = torch.is_grad_enabled() and any(
                    t.requires_grad for t in all_args if isinstance(t, torch.Tensor)
                )
                if needs_grad:
                    # double backward
                    return CompiledFunction._double_backward(ctx, impl_fn, all_args)
                else:
                    return impl_fn()

            @staticmethod
            def _double_backward(ctx, impl_fn, all_args):
                # Ensure that the graph is connected, and error if double backward is performed.
                # See comment for why once_differentiable is not sufficient:
                # https://github.com/pytorch/pytorch/pull/92348/files#r1072962107
                class CompiledFunctionBackward(torch.autograd.Function):
                    # CompiledFunctionBackward is not yet supported in dynamo skipfiles
                    _aot_id = aot_config.aot_id

                    @staticmethod
                    # pyrefly: ignore [bad-override]
                    def forward(double_ctx, *unused_args):
                        return impl_fn(double_ctx)

                    @staticmethod
                    def backward(double_ctx, *args):
                        raise RuntimeError(
                            "torch.compile with aot_autograd does not currently support double backward"
                        )

                CompiledFunctionBackward._compiled_autograd_key = (  # type: ignore[method-assign]
                    CompiledFunction._compiled_autograd_key
                )

                return CompiledFunctionBackward.apply(*all_args)

            @staticmethod
            def _backward_impl(ctx, all_args):
                # compiled autograd reimplements this function at proxy_call_aot_backward
                assert not backward_state_indices, (
                    "BackwardState requires CompiledAutograd"
                )
                ctx.maybe_clear_saved_tensors()

                saved_tensors_use_once = (
                    not torch._C._autograd._get_current_graph_task_keep_graph()
                )

                if CompiledFunction.compiled_bw is None:
                    assert lazy_backward_info is not None
                    assert isinstance(
                        lazy_backward_info, AutogradLazyBackwardCompileInfo
                    )

                    if (
                        hasattr(lazy_backward_info, "saved_context")
                        and lazy_backward_info.saved_context is not None
                    ):
                        assert isinstance(
                            lazy_backward_info.saved_context, TracingContext
                        )
                        ddp_ctx = lazy_backward_info.saved_context.ddp_optimizer_ctx
                        if ddp_ctx is not None:
                            assert ddp_ctx.curr_bucket >= 0, (
                                f"expected same # of fw and bw compiles, but found bucket {ddp_ctx.curr_bucket}"
                            )
                            curr_fw_meta = ddp_ctx.metadata_per_bucket[
                                ddp_ctx.curr_bucket
                            ]
                            # Note [DDPOptimizer and fw_metadata]
                            # When using the DDPOptimizer, we have a single dynamo graph (and TracingContext),
                            # but multiple AOTDispatcher graph.
                            #
                            # One consequence is that there will be **multiple** fw_metadata objects, one per AOT graph,
                            # which we stash the fw_metadata on the TracingContext.
                            #
                            # Normally what happens is that as we compile AOT graphs 1...N, we clobber the fw_metadata
                            # for graph i-1 when we start running AOT for graph i.
                            # Ordinarily this is fine, because inductor no longer needs the metadata from graph i-1.
                            #
                            # However, this is a problem for lazy compilation of the backward. During backward compilation,
                            # we compile the backward lazily at backward runtime, meaning that we will first compile
                            # backward graph N, N-1, ..., 1.
                            # We need to ensure that at the time inductor compiles bw graph N-1, it can access
                            # the corresponding fw_metadta for graph N-1.
                            #
                            # We do this by stashing a DDPOptimizerContext, which tracks:
                            # - the metadata of all N graphs
                            # - the graph we are currently compiling in our DDPOptimizer region.
                            ddp_ctx.curr_bucket -= 1
                            lazy_backward_info.saved_context.fw_metadata = curr_fw_meta

                    if not saved_tensors_use_once:
                        fw_metadata.bw_donated_idxs = []
                        # Update bw_donated_idxs if using lazy_backward_info from `aot_dispatch_autograd`
                        if (
                            hasattr(lazy_backward_info, "saved_context")
                            and hasattr(lazy_backward_info.saved_context, "fw_metadata")
                            and hasattr(
                                lazy_backward_info.saved_context.fw_metadata,  # type: ignore[union-attr]
                                "bw_donated_idxs",
                            )
                        ):
                            lazy_backward_info.saved_context.fw_metadata.bw_donated_idxs = (  # type: ignore[union-attr]
                                []
                            )

                    bw_module = lazy_backward_info.bw_module
                    placeholder_list = lazy_backward_info.placeholder_list
                    saved_context = lazy_backward_info.saved_context
                    saved_compile_context = lazy_backward_info.saved_compile_context

                    context = torch._C._DisableAutocast if disable_amp else nullcontext
                    metrics_context = get_metrics_context()
                    with (
                        tracing(saved_context),
                        compile_context(saved_compile_context),
                        context(),
                        track_graph_compiling(aot_config, "backward"),
                        metrics_context,
                        dynamo_timed(
                            "backward._backward_impl",
                            phase_name="entire_backward_compile",
                            log_pt2_compile_event=True,
                            dynamo_compile_column_us="backward_cumulative_compile_time_us",
                            log_waitcounter=True,
                            waitcounter_name_override="entire_backward_compile",
                        ),
                        callback_handler.install_callbacks(
                            CallbackTrigger.LAZY_BACKWARD,
                            str(CompileContext.current_compile_id()),
                        ),
                    ):
                        CompileEventLogger.compilation_metric(is_forward=False)
                        # See Note: [Backward graph lazy lowering]
                        CompiledFunction.compiled_bw = aot_config.bw_compiler(
                            copy.deepcopy(bw_module), placeholder_list
                        )
                        # Maybe save cache entry
                        if try_save_cache_entry is not None:
                            try_save_cache_entry(
                                CompiledFunction.compiled_bw,
                                bw_module,
                                fw_metadata,
                                aot_config,
                            )

                if (
                    torch._functorch.config.donated_buffer
                    and not saved_tensors_use_once
                    and fw_metadata.bw_donated_idxs != []
                ):
                    torch._check(
                        False,
                        lambda: (
                            "This backward function was compiled with non-empty donated "
                            "buffers which requires create_graph=False and retain_graph=False. "
                            "Please keep backward(create_graph=False, retain_graph=False) "
                            "across all backward() function calls, or set "
                            "torch._functorch.config.donated_buffer=False to disable "
                            "donated buffer."
                        ),
                    )

                out = call_func_at_runtime_with_args(
                    CompiledFunction.compiled_bw,
                    all_args,
                    steal_args=True,
                    disable_amp=disable_amp,
                )
                return out

        compiled_function = RuntimeWrapper(
            indices_of_inps_to_detach=indices_of_inps_to_detach,
            trace_joint=True,
            disable_amp=disable_amp,
        ).post_compile(
            CompiledFunction.apply,
            aot_config,
            runtime_metadata=fw_metadata,
        )

        return compiled_function


@dataclass
class DebugAssertWrapper(CompilerWrapper):
    flat_requires_grad: list[Optional[bool]] = field(default_factory=list)

    def post_compile(
        self,
        compiled_fn,
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ):
        @wraps(compiled_fn)
        def debug_compiled_function(args: list[Any]):
            # TODO: Check aliasing relationships
            # TODO: Check strides for metadata mutation
            # (NB: ideally, this logic is factored out of this function and
            # you move these debug checks there)

            # Check requires grad.  Bad case is when we compiled with
            # requires_grad = False, but input requires_grad = True
            # (vice versa is OK; we compute a gradient and then throw
            # it away when it hits the input.)
            for i, a in enumerate(args):
                can_require_grad = self.flat_requires_grad[i]
                if can_require_grad is None:
                    assert not isinstance(a, Tensor)
                elif not can_require_grad:
                    assert not a.requires_grad, format_guard_bug_msg(
                        aot_config,
                        f"{describe_input(i, aot_config)} would not require grad",
                    )

            return compiled_fn(args)

        return debug_compiled_function


def pre_compile(
    wrappers: list[CompilerWrapper],
    flat_fn: TraceFn,
    flat_args: list[FxValue],
    flat_args_descs: list[AOTInput],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
) -> tuple[TraceFn, list[FxValue], list[AOTInput], ViewAndMutationMeta]:
    """
    Runs a sequence of wrappers on the given function and arguments.
    Mutates wrappers in place.
    """
    for wrapper in wrappers:
        flat_fn, flat_args, flat_args_descs, fw_metadata = wrapper.pre_compile(
            flat_fn, flat_args, flat_args_descs, aot_config, fw_metadata=fw_metadata
        )
    return flat_fn, flat_args, flat_args_descs, fw_metadata


def post_compile(
    wrappers: list[CompilerWrapper],
    compiled_fn: Callable,
    aot_config: AOTConfig,
    *,
    runtime_metadata: ViewAndMutationMeta,
) -> tuple[Callable, ViewAndMutationMeta]:
    """
    Runs a sequence of wrappers on the given function. Should be called after pre_compile()
    """
    for wrapper in reversed(wrappers):
        compiled_fn = wrapper.post_compile(
            compiled_fn, aot_config, runtime_metadata=runtime_metadata
        )
    return compiled_fn, runtime_metadata


def make_runtime_safe(
    fw_metadata: ViewAndMutationMeta,
    maybe_subclass_meta: Optional[SubclassMeta],
):
    """
    Calls make_runtime_safe on all ViewAndMutationMetas.
    Modifies both arguments. Allows ViewAndMutationMetas to
    be safely cached in AOTAutogradCache.
    """
    fw_metadata.make_runtime_safe()
    if maybe_subclass_meta is not None:
        maybe_subclass_meta.fw_metadata.make_runtime_safe()
        if maybe_subclass_meta.grad_input_metas:
            for meta in maybe_subclass_meta.grad_input_metas:
                if isinstance(meta, SubclassCreationMeta):
                    meta.make_runtime_safe()
