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
import pprint
import typing
import warnings
from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

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
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import is_opaque_type
from torch._library.utils import is_builtin
from torch._logging import getArtifactLogger
from torch._opaque_base import OpaqueBase
from torch._ops import OpOverload
from torch._prims_common import CUDARngStateHelper
from torch._subclasses import FakeTensor
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.proxy_tensor import HANDLED_TYPES
from torch.multiprocessing.reductions import StorageWeakRef
from torch.types import IntLikeType
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
    OpaqueMeta,
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


def _unwrap_tensor_subclasses_no_symints(
    args: list[Any],
) -> list[Any]:
    return runtime_unwrap_tensor_subclasses(args, append_symints=False)  # type: ignore[arg-type]


zip = strict_zip

aot_graphs_log = getArtifactLogger(__name__, "aot_graphs")


def _unwrap_no_symints(args: list[Any]) -> list[Any]:
    return runtime_unwrap_tensor_subclasses(args, append_symints=False)


def _describe_arg_for_logging(arg: object) -> str:
    from torch._library import opaque_object

    try:
        is_dtensor = isinstance(arg, torch.distributed.tensor.DTensor)
    except AttributeError:
        is_dtensor = False

    if is_dtensor:
        arg = typing.cast(torch.distributed.tensor.DTensor, arg)
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
        compiled_fn: Callable[..., Any],
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ) -> Callable[..., Any]:
        return _create_runtime_wrapper(
            compiled_fn,
            runtime_metadata=runtime_metadata,
            indices_of_inps_to_detach=self.indices_of_inps_to_detach,
            trace_joint=self.trace_joint,
            keep_input_mutations=aot_config.keep_inference_input_mutations,
            disable_amp=self.disable_amp,
        )


class NoopAliasHandler:
    def __init__(
        self, info: Any, runtime_metadata: ViewAndMutationMeta, trace_joint: bool
    ) -> None:
        pass

    def __call__(self, orig_inputs: list[Any], fw_outs: list[Any], out: Any) -> Any:
        return out


def _unwrap_tensoralias(x: TensorAlias) -> torch.Tensor:
    if not isinstance(x, TensorAlias):
        raise AssertionError(f"expected TensorAlias, got {type(x)}")
    return x.alias


def _identity(x: Any) -> Any:
    return x


class AliasOfInputHandler:
    def __init__(
        self, info: Any, runtime_metadata: ViewAndMutationMeta, trace_joint: bool
    ) -> None:
        self.base_idx = info.base_idx
        self.unwrap_out = _unwrap_tensoralias if trace_joint else _identity
        self.requires_grad = info.requires_grad
        self.view_meta_sequence = info.view_meta_sequence
        self.replay_views = config.view_replay_for_aliased_outputs

    def __call__(
        self, orig_inputs: list[Any], fw_outs: list[Any], out: Any
    ) -> torch.Tensor:
        aliased_base_tensor = orig_inputs[self.base_idx]
        return gen_alias_from_base(
            aliased_base_tensor,
            self.unwrap_out(out),
            self.requires_grad,
            self.view_meta_sequence,
            replay_views=self.replay_views,
        )


class IsInputHandler:
    def __init__(
        self, info: Any, runtime_metadata: ViewAndMutationMeta, trace_joint: bool
    ) -> None:
        self.base_idx = info.base_idx
        self.unwrap_out = _unwrap_tensoralias if trace_joint else _identity

    def __call__(
        self, orig_inputs: list[Any], fw_outs: list[Any], out: Any
    ) -> torch.Tensor:
        aliased_base_tensor = orig_inputs[self.base_idx]
        return aliased_base_tensor


class AliasOfIntermediateHandler:
    def __init__(
        self, info: Any, runtime_metadata: ViewAndMutationMeta, trace_joint: bool
    ) -> None:
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

    def __call__(
        self, orig_inputs: list[Any], fw_outs: list[Any], out: Any
    ) -> torch.Tensor:
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


def make_output_handler(
    info: Any, runtime_metadata: ViewAndMutationMeta, trace_joint: bool
) -> Any:
    handler_type = _HANDLER_MAP[info.output_type]
    return handler_type(info, runtime_metadata, trace_joint)


# not sure why AOTDispatcher needs to manually set this
def maybe_mark_dynamic_helper(t: torch.Tensor, dims: set[int]) -> None:
    if hasattr(t, "_dynamo_weak_dynamic_indices"):
        # pyrefly: ignore [missing-attribute]
        t._dynamo_weak_dynamic_indices |= dims
    else:
        t._dynamo_weak_dynamic_indices = dims.copy()  # type: ignore[attr-defined]


def _should_disable_saved_tensors_hooks() -> bool:
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


def _schema_allows_aliasing(func: Any) -> bool:
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


def _check_custom_op_aliasing(
    name: str, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any
) -> None:
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
            msg = f"{e} This is deprecated and will become an error in PyTorch 2.12."
            warnings.warn(msg, UserWarning, stacklevel=3)


@functools.lru_cache(None)
def _is_fsdp_all_gather_copy_in(func: Any) -> bool:
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

    def __init__(self) -> None:
        super().__init__()
        self.supports_higher_order_operators = True

    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if not kwargs:
            kwargs = {}

        flat_tensor_args = filter(
            lambda x: isinstance(x, torch.Tensor), tree_flatten((args, kwargs))[0]
        )

        # Defer this to subclass torchdispatch modes (probably shouldn't have fake tensor here tho)
        # For Parameters, we need to check the underlying tensor type, not the Parameter itself
        for tensor in flat_tensor_args:
            underlying_tensor = tensor
            if isinstance(tensor, torch.nn.Parameter):
                underlying_tensor = tensor.data
            if type(underlying_tensor) not in HANDLED_TYPES:
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
    def ignore_compile_internals(cls) -> bool:
        return True


class _FirstInvocationContext:
    """
    Context manager that tracks first invocation and conditionally enables _AnalyzeCustomOpInputOutputMode.
    This is useful when we have a custom op where we want to analyze its' input
    and output during cold start.
    """

    def __init__(self) -> None:
        self._is_first = True

    def __call__(self) -> AbstractContextManager[Any]:
        """
        Returns a context manager: _AnalyzeCustomOpInputOutputMode on first invocation, nullcontext thereafter.
        Automatically updates state after first use.
        """
        # NB: Don't run the analyzer when you're forcing compile during FX
        # tracing, as the analyzer doesn't play nicely when it's being
        # make_fx'ed through
        if (
            self._is_first
            and config.check_custom_op_aliasing
            and not torch._dynamo.config.force_compile_during_fx_trace
        ):
            self._is_first = False
            return _AnalyzeCustomOpInputOutputMode()
        return nullcontext()


@dataclass
class _RuntimeCompiledFnInvoker:
    compiled_fn: Callable[..., Any]
    indices_of_inps_to_detach: list[int]
    trace_joint: bool
    disable_amp: bool
    first_invocation_ctx: _FirstInvocationContext = field(
        default_factory=_FirstInvocationContext
    )

    def __post_init__(self) -> None:
        if not getattr(self.compiled_fn, "_boxed_call", False):
            self.compiled_fn = make_boxed_func(self.compiled_fn)

    def run(self, args: list[Any], *, on_before_call: Callable[[], None]) -> list[Any]:
        with self.first_invocation_ctx():
            if self.trace_joint:
                args_ = list(args)
                # See Note [Detaching inputs that never need gradients]
                for idx in self.indices_of_inps_to_detach:
                    if isinstance(args_[idx], torch.Tensor):
                        args_[idx] = args_[idx].detach()

                # It's possible to have trace_joint inside user specified with no_grad() region,
                # if there is a nested with enable_grad(), that forces some outputs to require gradients.
                # Therefore, we unconditionally turn on enable_grad() for compiled_fn execution.
                with (
                    torch.autograd._force_original_view_tracking(True),
                    torch.enable_grad(),
                ):
                    on_before_call()
                    return call_func_at_runtime_with_args(
                        self.compiled_fn,
                        args_,
                        disable_amp=self.disable_amp,
                        steal_args=True,
                    )

            # When we have an inference graph, we run with grad disabled.
            # It's possible to get an inference graph with inputs that require grad,
            # in which case we want to make sure autograd is disabled
            # (since e.g., inductor will generate aten.addmm.out calls which autograd will complain on)
            # NOTE: We use _set_grad_enabled directly to reduce runtime overhead
            grad_enabled = torch.is_grad_enabled()
            try:
                if grad_enabled:
                    torch._C._set_grad_enabled(False)
                on_before_call()
                return call_func_at_runtime_with_args(
                    self.compiled_fn,
                    args,
                    disable_amp=self.disable_amp,
                    steal_args=True,
                )
            finally:
                if grad_enabled:
                    torch._C._set_grad_enabled(True)


@dataclass
class _RuntimeForwardEpilogue:
    runtime_metadata: ViewAndMutationMeta
    trace_joint: bool
    keep_input_mutations: bool
    epilogue_args_idx: tuple[int, ...] = field(init=False)
    output_handlers: tuple[Any, ...] = field(init=False)

    def __post_init__(self) -> None:
        epilogue_args_idx = list(self.runtime_metadata.mutated_inp_runtime_indices)
        for info in self.runtime_metadata.output_info:
            if (
                info.output_type == OutputType.alias_of_input
                or info.output_type == OutputType.is_input
            ):
                if not isinstance(info.base_idx, int):
                    raise AssertionError(
                        f"expected info.base_idx to be int, got {type(info.base_idx)}"
                    )
                epilogue_args_idx.append(info.base_idx)
        self.epilogue_args_idx = tuple(epilogue_args_idx)

        if config.unlift_effect_tokens:
            if len(self.runtime_metadata.tokens) != 0:
                raise AssertionError(
                    "expected no tokens when unlift_effect_tokens is True, "
                    f"got {len(self.runtime_metadata.tokens)}"
                )

        if self.runtime_metadata.num_outputs_aliased > 0:
            self.output_handlers = tuple(
                make_output_handler(info, self.runtime_metadata, self.trace_joint)
                for info in self.runtime_metadata.output_info
            )
        else:
            self.output_handlers = ()

    def capture_orig_inputs(self, args: list[Any]) -> dict[int, Any]:
        return {i: args[i] for i in self.epilogue_args_idx}

    def increment_mutation_versions(self, args: list[Any]) -> None:
        if self.keep_input_mutations:
            mutated_args = (
                args[i]
                for i in self.runtime_metadata.mutated_graph_handled_indices_seen_by_autograd
            )
            torch.autograd.graph.increment_version(mutated_args)

    def finalize(self, orig_inputs: dict[int, Any], all_outs: list[Any]) -> Any:
        self._validate_compiled_output_arity(all_outs)
        updated_inputs, fw_outs = self._split_mutated_inputs(all_outs)
        if updated_inputs is not None:
            self._apply_input_mutations(orig_inputs, updated_inputs)

        ret_outs = self._replay_output_aliases(orig_inputs, fw_outs)
        if self.runtime_metadata.dynamic_outputs:
            for t, o in zip(ret_outs, self.runtime_metadata.output_info):
                if o.dynamic_dims is None:
                    continue
                maybe_mark_dynamic_helper(t, o.dynamic_dims)
        if self.runtime_metadata.grad_enabled_mutation is not None:
            torch._C._set_grad_enabled(self.runtime_metadata.grad_enabled_mutation)
        return ret_outs

    def _validate_compiled_output_arity(self, all_outs: list[Any]) -> None:
        expected_outs = (
            self.runtime_metadata.num_mutated_inp_runtime_indices
            + self.runtime_metadata.num_outputs
            + self.runtime_metadata.num_intermediate_bases
        )
        if len(all_outs) != expected_outs:
            raise AssertionError(
                f"expected {expected_outs} outputs, got {len(all_outs)}"
            )

    def _split_mutated_inputs(
        self, all_outs: list[Any]
    ) -> tuple[list[Any] | None, list[Any]]:
        num_mutated_runtime_inps = self.runtime_metadata.num_mutated_inp_runtime_indices
        if num_mutated_runtime_inps == 0:
            return None, all_outs
        return (
            all_outs[:num_mutated_runtime_inps],
            all_outs[num_mutated_runtime_inps:],
        )

    def _apply_input_mutations(
        self, orig_inputs: dict[int, Any], updated_inputs: list[Any]
    ) -> None:
        for i, inpt_idx in enumerate(self.runtime_metadata.mutated_inp_runtime_indices):
            meta = self.runtime_metadata.input_info[inpt_idx]
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
                if self.trace_joint:
                    if not isinstance(updated_inpt, TensorAlias):
                        raise AssertionError(
                            f"expected TensorAlias for updated_inpt, got {type(updated_inpt)}"
                        )
                    updated_inpt = updated_inpt.alias
                with torch.no_grad():
                    original_inpt.set_(updated_inpt)
                continue
            if meta.mutates_metadata and not meta.mutates_data:
                if self.trace_joint:
                    if not isinstance(updated_inpt, TensorAlias):
                        raise AssertionError(
                            f"expected TensorAlias for updated_inpt, got {type(updated_inpt)}"
                        )
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
                    if not meta.mutates_data:
                        raise AssertionError("expected meta.mutates_data to be True")
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
                        self.runtime_metadata.mutated_inp_stream_indices is not None
                        and i < len(self.runtime_metadata.mutated_inp_stream_indices)
                        and self.runtime_metadata.mutated_inp_stream_indices[i]
                        is not None
                    ):
                        raise RuntimeError(
                            "Mutations on inputs with user-specified streams are not yet supported. "
                            "See: https://github.com/pytorch/pytorch/issues/172522"
                        )
                    original_inpt.copy_(updated_inpt)

    def _replay_output_aliases(
        self, orig_inputs: dict[int, Any], fw_outs: list[Any]
    ) -> Any:
        if self.runtime_metadata.num_outputs_aliased == 0:
            return fw_outs

        # The compiled forward also returned intermediate bases. We don't want to return them to the user.
        expect_num_outputs = (
            len(self.output_handlers) + self.runtime_metadata.num_intermediate_bases
        )
        if len(fw_outs) != expect_num_outputs:
            raise AssertionError(
                f"expected {expect_num_outputs} fw_outs, got {len(fw_outs)}"
            )
        return [
            handler(orig_inputs, fw_outs, out)
            for out, handler in builtins.zip(fw_outs, self.output_handlers)
        ]


def _create_runtime_wrapper(
    compiled_fn: Callable[..., Any],
    *,
    runtime_metadata: ViewAndMutationMeta,
    indices_of_inps_to_detach: list[int],
    trace_joint: bool,
    keep_input_mutations: bool,
    disable_amp: bool,
) -> Callable[..., Any]:
    compiled_invoker = _RuntimeCompiledFnInvoker(
        compiled_fn=compiled_fn,
        indices_of_inps_to_detach=indices_of_inps_to_detach,
        trace_joint=trace_joint,
        disable_amp=disable_amp,
    )
    runtime_epilogue = _RuntimeForwardEpilogue(
        runtime_metadata=runtime_metadata,
        trace_joint=trace_joint,
        keep_input_mutations=keep_input_mutations,
    )

    def record_runtime_wrapper_prologue_enter() -> AbstractContextManager[None] | None:
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
        cm: AbstractContextManager[None] | None,
    ) -> None:
        if cm is not None:
            cm.__exit__(None, None, None)

    # Codegen mutation epilogue: emit straight-line code per mutated input
    # with all branches resolved at compile time.
    if runtime_metadata.num_mutated_inp_runtime_indices > 0:
        mut_lines = ["def _apply_mutations(orig_inputs, updated_inputs):"]
        mut_globals: dict[str, object] = {
            "torch": torch,
            "_unwrap_tensoralias": _unwrap_tensoralias,
        }
        for i, inpt_idx in enumerate(runtime_metadata.mutated_inp_runtime_indices):
            meta = runtime_metadata.input_info[inpt_idx]
            if not meta.mutates_data and not meta.mutates_metadata:
                continue
            oi = f"orig_inputs[{inpt_idx}]"
            ui = f"updated_inputs[{i}]"
            if meta.mutates_storage_metadata:
                if trace_joint:
                    mut_lines.append(f"    _u{i} = _unwrap_tensoralias({ui})")
                else:
                    mut_lines.append(f"    _u{i} = {ui}")
                mut_lines.append(f"    with torch.no_grad(): {oi}.set_(_u{i})")
            elif meta.mutates_metadata and not meta.mutates_data:
                if trace_joint:
                    mut_lines.append(f"    _u{i} = _unwrap_tensoralias({ui})")
                else:
                    mut_lines.append(f"    _u{i} = {ui}")
                mut_lines.append(
                    f"    {oi}.as_strided_(_u{i}.size(), _u{i}.stride(), _u{i}.storage_offset())"
                )
            else:
                if meta.mutates_data and meta.mutates_metadata:
                    mut_lines.append(
                        f"    {oi}.as_strided_({ui}.size(), {ui}.stride(), {ui}.storage_offset())"
                    )
                else:
                    assert meta.mutates_data, (  # noqa: S101
                        f"expected mutates_data for input {inpt_idx}"
                    )
                if meta.is_leaf:
                    mut_lines.append(
                        f"    if {oi}.requires_grad: {oi}.detach().copy_({ui})"
                    )
                    mut_lines.append(f"    else: {oi}.copy_({ui})")
                else:
                    has_stream = (
                        runtime_metadata.mutated_inp_stream_indices is not None
                        and i < len(runtime_metadata.mutated_inp_stream_indices)
                        and runtime_metadata.mutated_inp_stream_indices[i] is not None
                    )
                    if has_stream:
                        msg_name = f"_stream_err_{i}"
                        mut_globals[msg_name] = (
                            "Mutations on inputs with user-specified streams are not yet supported. "
                            "See: https://github.com/pytorch/pytorch/issues/172522"
                        )
                        mut_lines.append(f"    raise RuntimeError({msg_name})")
                    else:
                        mut_lines.append(f"    {oi}.copy_({ui})")
        if len(mut_lines) == 1:
            mut_lines.append("    pass")
        mut_source = "\n".join(mut_lines)

        from .subclass_codegen import _compile_and_exec_source

        codegen_apply_mutations = _compile_and_exec_source(
            mut_source, mut_globals, "_apply_mutations", "mutation_epilogue"
        )
        import types

        runtime_epilogue._apply_input_mutations = types.MethodType(  # type: ignore[attr-defined]
            lambda self, orig_inputs, updated_inputs: codegen_apply_mutations(
                orig_inputs, updated_inputs
            ),
            runtime_epilogue,
        )

    @simple_wraps(compiled_invoker.compiled_fn)
    def runtime_wrapper(args: list[Any]) -> Any:
        # Create context manager for profiler
        cm = record_runtime_wrapper_prologue_enter()
        prologue_exited = False

        def exit_prologue() -> None:
            nonlocal prologue_exited
            if not prologue_exited:
                record_runtime_wrapper_prologue_exit(cm)
                prologue_exited = True

        try:
            # stash a ref to each input tensor we plan to use after the compiled function
            orig_inputs = runtime_epilogue.capture_orig_inputs(args)
            runtime_epilogue.increment_mutation_versions(args)
            all_outs = compiled_invoker.run(args, on_before_call=exit_prologue)
        finally:
            exit_prologue()

        del args
        return runtime_epilogue.finalize(orig_inputs, all_outs)

    if not (trace_joint and _should_disable_saved_tensors_hooks()):
        return runtime_wrapper

    # Disabling saved tensors hooks
    @simple_wraps(runtime_wrapper)
    def _runtime_wrapper(*args: Any, **kwargs: Any) -> Any:
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
        fw_module: torch.fx.GraphModule,
        flat_args: list[Any],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ) -> None:
        if config.functionalize_rng_ops:
            # Update example inputs for the fw_compiler
            fake_mode = detect_fake_mode()
            if fake_mode is None:
                raise AssertionError(
                    "fake_mode must not be None when functionalize_rng_ops is True"
                )
            seed, offset = CUDARngStateHelper.get_torch_state_as_tuple(fake_mode)
            flat_args.extend([seed, offset])
            # We are not clearing flat_args here because
            # 1) There is a check in the debug compiler at the end
            # 2) It does not matter as these are fake tensors

    def post_compile(
        self,
        compiled_fn: Callable[..., Any],
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ) -> Callable[..., Any]:
        @wraps(compiled_fn)
        def wrapper(runtime_args: list[Any]) -> Any:
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
        outs: Any,
        offset_index: int,
    ) -> Any:
        if metadata.is_rng_op_functionalized:
            if metadata.num_outputs_rng_offset != 1:
                raise AssertionError(
                    f"expected num_outputs_rng_offset == 1, got {metadata.num_outputs_rng_offset}"
                )
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
    fwd_output_strides: list[list[int] | None] | None = None
    needs_post_compile: bool = True

    def pre_compile(
        self,
        fw_module: fx.GraphModule,  # Must be fw_module from aot_dispatch_*_graph
        flat_args: list[Any],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ) -> None:
        tracing_context = torch._guards.TracingContext.try_get()
        if tracing_context and tracing_context.fakify_first_call:
            self.out_metas = [
                n.meta["val"] for n in (list(fw_module.graph.nodes)[-1].args[0])
            ]
        else:
            self.needs_post_compile = False

    def _compute_output_meta_with_inductor_strides(self) -> list[torch.Tensor]:
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
    def set_fwd_output_strides(
        self, fwd_output_strides: list[list[int] | None]
    ) -> None:
        self.fwd_output_strides = fwd_output_strides

    def post_compile(
        self,
        compiled_fn: Callable[..., Any],
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ) -> Callable[..., Any]:
        if self.needs_post_compile:
            if self.fwd_output_strides is None:
                raise AssertionError(
                    "fwd_output_strides must not be None when needs_post_compile is True"
                )
            fakified_out = self._compute_output_meta_with_inductor_strides()

            @wraps(compiled_fn)
            def wrapper(runtime_args: list[Any]) -> Any:
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
    fw_only: Callable[..., Any] | None  # Not cached, only used in pre_compile
    maybe_subclass_meta: SubclassMeta | None
    num_fw_outs_saved_for_bw: int | None

    def pre_compile(
        self,
        flat_fn: TraceFn,
        flat_args: list[FxValue],
        flat_args_descs: list[AOTInput],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ) -> tuple[TraceFn, list[FxValue], list[AOTInput], ViewAndMutationMeta]:
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

    @staticmethod
    def _get_frozen_inp_indices() -> frozenset[int]:
        # fw_compiler_freezing (compile_fx.py) bakes frozen params into the
        # graph and sets their TracingContext.params_flat entries to None
        # before post_compile runs.  We pass these indices to codegen so it
        # can emit straight-line code instead of a runtime None check.
        tc = TracingContext.try_get()
        if tc is None or tc.params_flat is None:
            return frozenset()
        return frozenset(i for i, p in enumerate(tc.params_flat) if p is None)

    def post_compile(
        self,
        compiled_fn: Callable[..., Any],
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ) -> Callable[..., Any]:
        if self.maybe_subclass_meta is None and not runtime_metadata.act_input_indices:
            return compiled_fn

        from .subclass_codegen import codegen_subclass_wrapper

        inner_fn = codegen_subclass_wrapper(
            compiled_fn=compiled_fn,
            inp_metas=runtime_metadata.subclass_inp_meta,
            out_metas=runtime_metadata.subclass_fw_graph_out_meta,
            num_fw_outs_saved_for_bw=self.num_fw_outs_saved_for_bw,
            frozen_inp_indices=self._get_frozen_inp_indices(),
            act_input_indices=runtime_metadata.act_input_indices,
        )
        inner_fn._boxed_call = True  # type: ignore[attr-defined]
        return inner_fn


@dataclass
class EffectTokensWrapper(CompilerWrapper):
    def post_compile(
        self,
        compiled_fn: Callable[..., Any],
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ) -> Callable[..., Any]:
        num_tokens = len(runtime_metadata.tokens)

        @wraps(compiled_fn)
        def inner_fn(args: list[Any]) -> Any:
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
    def remove_dupe_args(self, args: list[Any]) -> list[Any]:
        return [t for t, keep in zip(args, self.keep_arg_mask) if keep]

    def add_dupe_args(self, args: list[Any]) -> list[Any]:
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

        if requires_subclass_dispatch(leaf_flat_args, fw_metadata):  # type: ignore[arg-type]
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
        if len(add_dupe_map) != duped_arg_len:
            raise AssertionError(
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
                flat_fn,
                self.add_dupe_args(args),  # type: ignore[arg-type]
            )
            return outs, out_descs

        if config.debug_assert:
            ref_fw_metadata = run_functionalized_fw_and_collect_metadata(
                without_output_descs(wrapped_flat_fn),
                flat_args_descs=deduped_flat_args_descs,
                static_input_indices=aot_config.static_input_indices,
                keep_input_mutations=fw_metadata.keep_input_mutations,
            )(*deduped_flat_args)
            if ref_fw_metadata != updated_fw_metadata:
                raise AssertionError(
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
        compiled_fn: Callable[..., Any],
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ) -> Callable[..., Any]:
        if not self.needs_post_compile:
            return compiled_fn

        keep_indices = [i for i, keep in enumerate(self.keep_arg_mask) if keep]
        idx_list = ", ".join(f"args[{i}]" for i in keep_indices)
        source = (
            f"def inner_fn(args):\n"
            f"    deduped_args = [{idx_list}]\n"
            f"    args.clear()\n"
            f"    return compiled_fn(deduped_args)\n"
        )
        from .subclass_codegen import _compile_and_exec_source

        wrapped_compiled_fn: Callable[..., Any] = _compile_and_exec_source(  # type: ignore[assignment]
            source,
            {"compiled_fn": compiled_fn},
            "inner_fn",
            "dedup_wrapper",
            wrapped_fn=compiled_fn,
        )

        wrapped_compiled_fn._boxed_call = True  # type: ignore[attr-defined]

        # This can be uncommented when we properly guard for duplicates,
        # but right now we must not do it.
        # if not config.debug_assert:
        #     return wrapped_compiled_fn

        @wraps(wrapped_compiled_fn)
        def debugged_compiled_fn(args: list[Any]) -> Any:
            # Test that the computed remove/add arg functions are an inverse
            new_args = self.add_dupe_args(self.remove_dupe_args(args))
            seen: dict[Any, None] = {}
            for i, (x, y) in enumerate(zip(new_args, args)):
                seen[y] = None
                if x is not y:
                    raise AssertionError(
                        format_guard_bug_msg(
                            aot_config,
                            f"{describe_input(i, aot_config)} would be a duplicate of "
                            f"{describe_input(self.add_dupe_map[i], aot_config)}",
                        )
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
    ) -> tuple[Callable[..., Any], list[FxValue], list[AOTInput], ViewAndMutationMeta]:
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
        if requires_subclass_dispatch(flat_args, fw_metadata):  # type: ignore[arg-type]
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

        if len(fw_metadata.input_info) != len(synthetic_base_info):
            raise AssertionError(
                f"expected len(fw_metadata.input_info) == len(synthetic_base_info), "
                f"got {len(fw_metadata.input_info)} != {len(synthetic_base_info)}"
            )

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
        def wrapped_flat_fn(*args: Any) -> Any:
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
            out, out_descs = call_and_expect_output_descs(flat_fn, unpacked_args)  # type: ignore[arg-type]
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
            )(*flat_args_with_synthetic_bases)
            if ref_fw_metadata != fw_metadata_updated:
                raise AssertionError(
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
        compiled_fn: Callable[..., Any],
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ) -> Callable[..., Any]:
        if not self.needs_post_compile:
            return compiled_fn

        is_inference = not self.trace_joint

        @wraps(compiled_fn)
        def wrapped_compiled_fn(args: list[Any]) -> Any:
            # TODO: this sure seems expensive to run at runtime (which
            # post_compile seems to imply it does?!)
            args_with_synthetic_bases, _, synthetic_base_info = merge_view_inputs(
                aot_config, args, None, self.old_input_info, is_inference=is_inference
            )
            if synthetic_base_info is None:
                raise AssertionError("synthetic_base_info must not be None")
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
    fwd_inputs_descs: list[AOTInput] | None,
    mutated_input_info: list[InputAliasInfo],
    *,
    # The autograd case currently has more restrictions than the inference case.
    is_inference: bool,
) -> tuple[list[Any], list[AOTInput], list[int | tuple[int, torch.Tensor]] | None]:
    if fwd_inputs_descs is None:
        fwd_inputs_descs = [DummyAOTInput(i) for i in range(len(fwd_inputs))]

    def _are_differentiable_views(view1: torch.Tensor, view2: torch.Tensor) -> bool:
        if view1 is view2:
            return True
        if view1._base is None and view2._base is None:
            return False
        if view1._base is view2._base or view1._base is view2 or view1 is view2._base:
            return True
        return False

    def _same_dtype_views(view1: torch.Tensor, view2: torch.Tensor) -> bool:
        if view1.dtype != view2.dtype:
            return False
        if view1._base is not None and view1.dtype != view1._base.dtype:
            return False
        if view2._base is not None and view2.dtype != view2._base.dtype:
            return False
        return True

    if len(fwd_inputs) != len(mutated_input_info):
        raise AssertionError(
            f"expected len(fwd_inputs) == len(mutated_input_info), "
            f"got {len(fwd_inputs)} != {len(mutated_input_info)}"
        )
    if not [info for info in mutated_input_info if info.mutates_data]:
        # Return early when there are no mutations.
        return fwd_inputs, fwd_inputs_descs, None

    storage_ref_to_idx: dict[StorageWeakRef, list[int]] = collections.defaultdict(list)
    # pyrefly: ignore [implicit-any]
    base_args = []
    # pyrefly: ignore [implicit-any]
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
    inner_calling_convention_meta: dict[int, int | tuple[int, torch.Tensor]] = {}
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
                if not _are_differentiable_views(view1, view2):
                    raise AssertionError(
                        "aot_autograd() does not yet handle non-differentiable view input mutations."
                    )
            # Regenerating views when reinterpreting complex / real tensors seems non-trivial,
            # not handling for now
            if not _same_dtype_views(view1, view2):
                raise AssertionError(
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
                if other_base is not synthetic_base:
                    raise AssertionError(
                        "aot_autograd() does not yet handle non-differentiable view input mutations."
                    )
            for alias in aliases_with_none_bases:
                if alias is not synthetic_base:
                    raise AssertionError(
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
        if len(other_args) != len(fwd_inputs):
            raise AssertionError(
                f"expected len(other_args) == len(fwd_inputs), "
                f"got {len(other_args)} != {len(fwd_inputs)}"
            )
        # If no synthetic bases are necessary, just return the original inputs.
        return fwd_inputs, fwd_inputs_descs, None
    else:
        from torch.fx.experimental.symbolic_shapes import SymIntEqByExpr

        def make_hashable(arg: Any) -> Any:
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
        post_processed_calling_convention_meta: list[int | tuple[int, torch.Tensor]] = [
            -1 for _ in range(len(inner_calling_convention_meta))
        ]
        for k, v in inner_calling_convention_meta.items():
            post_processed_calling_convention_meta[k] = v
        # Quick assert: every argument in the inner calling convention should be accounted for.
        for x in post_processed_calling_convention_meta:
            if x == -1:
                raise AssertionError(
                    "every argument in the inner calling convention should be accounted for"
                )
        # pyrefly: ignore [bad-return]
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
    bw_module: Callable[..., Any]
    placeholder_list: list[Any]
    saved_context: TracingContext | None
    saved_compile_context: CompileContext | None


# On an AOT Autograd cache hit, we already have a lowered backward, so there is usually
# no need to keep information around for a new lazy compilation. Except for compiled autograd,
# which wants to retrace this backward into a larger graph, and it needs the graph module to do so.
@dataclass
class CachedAutogradLazyBackwardCompileInfo:
    bw_module_fn: Callable[..., Any]


def _raise_if_functorch_active() -> None:
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
    ctx_saved_tensors: Sequence[torch.Tensor],
    ctx_symints: Sequence[IntLikeType],
    ctx_opaque_objects: Sequence[Any],
    metadata: ViewAndMutationMeta,
    maybe_subclass_metadata: SubclassMeta | None,
    flat_args: Sequence[Any],
    codegen_unwrap_fn: Callable[..., Any] | None = None,
) -> list[Any]:
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

    if len(flat_args) != expected_grad_outs:
        raise AssertionError(
            f"expected {expected_grad_outs} grad_outs, got {len(flat_args)}"
        )
    out_info = metadata.output_info

    inp_tangents, out_tangents, intermediate_base_tangents = (
        flat_args[:num_mutated_runtime_inps],
        flat_args[
            num_mutated_runtime_inps : num_mutated_runtime_inps + metadata.num_outputs
        ],
        flat_args[num_mutated_runtime_inps + metadata.num_outputs :],
    )
    # Release grad refs from the caller's list (boxed calling convention).
    # Slicing already copied refs into sub-lists above, so clearing the
    # original list only drops redundant refs. The isinstance guard skips
    # this when flat_args is a tuple (non-boxed path from compiled_autograd).
    if isinstance(flat_args, list):
        flat_args.clear()
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
        and info.requires_grad_for_backward
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
    # pyrefly: ignore [implicit-any]
    rng_args = []
    if metadata.is_rng_op_functionalized:
        # Add the seed and offset to args
        rng_args = CUDARngStateHelper.get_torch_state_as_tuple()

    bw_tokens = [None] * metadata.num_backward_tokens

    # - note: donated buffer logic requires (*ctx.symints, *ctx.saved_tensors, *ctx.opaques) showing up first
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
    expected_tangents_start = (
        len(ctx_symints) + num_ctx_saved_tensors + len(ctx_opaque_objects)
    )
    if tangents_start_idx != expected_tangents_start:
        raise AssertionError(
            f"expected tangents_start_idx == {expected_tangents_start}, got {tangents_start_idx}"
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

        stack_traces = metadata.tangent_source_stack_traces or ()

        flat_processed_tangents = list(
            itertools.chain.from_iterable(
                (
                    AOTDispatchAutograd.process_runtime_tangent(
                        t,
                        m,
                        tangent_idx=idx,
                        tangent_desc=desc,
                        compile_id_str=metadata.compile_id_str,
                        tangent_stack_trace=stack_traces[idx] if stack_traces else None,
                    )[1]
                )
                for idx, (t, m, desc) in enumerate(
                    zip(
                        tangents,
                        metadata.subclass_tangent_meta,
                        metadata.traced_tangents_descs,
                    )
                )
            )
        )

        if codegen_unwrap_fn is not None:
            unwrap = codegen_unwrap_fn
        else:
            unwrap = _unwrap_no_symints
        all_args = (
            unwrap(all_args[:tangents_start_idx])
            + flat_processed_tangents
            + unwrap(all_args[tangents_end_idx:])
        )
    else:
        stack_traces = metadata.tangent_source_stack_traces or ()

        all_args = [
            (
                AOTDispatchAutograd.process_runtime_tangent(
                    t,
                    metadata.subclass_tangent_meta[i - tangents_start_idx],
                    tangent_idx=i - tangents_start_idx,
                    tangent_desc=metadata.traced_tangents_descs[i - tangents_start_idx],
                    compile_id_str=metadata.compile_id_str,
                    tangent_stack_trace=(
                        stack_traces[i - tangents_start_idx] if stack_traces else None
                    ),
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
) -> None:
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
    metadata: ViewAndMutationMeta,
    maybe_subclass_metadata: SubclassMeta | None,
    out: Any,
    *,
    ctx_opaque_objects: Sequence[Any] = (),
    make_subclass_override: Callable[..., Any] | None = None,
    codegen_wrap_fn: Callable[..., Any] | None = None,
) -> tuple[Any, ...]:
    # Toss out the backward output tokens
    num_bw_tokens = metadata.num_backward_tokens
    if num_bw_tokens > 0:
        out = out[:-num_bw_tokens]

    # TODO: replace this with FunctionalizedRngRuntimeWrapper.post_compile
    out = FunctionalizedRngRuntimeWrapper()._functionalized_rng_runtime_epilogue(
        metadata, out, offset_index=len(out) - 1
    )
    out = tuple(out)

    # Replace compile-time opaque constants in the backward output with the
    # real runtime opaques saved from the forward pass. During joint graph
    # tracing, backward output opaques come from tangent constants (baked at
    # compile time). At runtime we need the actual opaque objects that were
    # saved for backward from the forward pass.
    if ctx_opaque_objects:
        opaque_iter = iter(ctx_opaque_objects)
        out = tuple(
            next(opaque_iter) if isinstance(v, FakeScriptObject) else v for v in out
        )
        remaining = list(opaque_iter)
        if remaining:
            raise AssertionError(
                f"ctx_opaque_objects had {len(remaining)} leftover entries "
                "(expected all to be consumed by FakeScriptObject slots in backward output)"
            )

    # TODO: figure out how to refactor the backward properly so I can use aot_dispatch_subclass_wrapper() here.
    if maybe_subclass_metadata is not None:
        if maybe_subclass_metadata.grad_input_metas is None:
            raise AssertionError("grad_input_metas must not be None")
        if codegen_wrap_fn is not None and make_subclass_override is None:
            return codegen_wrap_fn(out)
        outs_wrapped = wrap_tensor_subclasses(
            out,
            subclass_metas=maybe_subclass_metadata.grad_input_metas,
            included_subclass_symints=True,
            is_runtime=True,
            make_subclass_override=make_subclass_override,
        )
        return outs_wrapped
    return out


def coerce_to_expected_memory_format(
    x: torch.Tensor, memory_format: MemoryFormatMeta
) -> torch.Tensor:
    if memory_format.memory_format is not None:
        # Coerce to torch.memory_format
        if not x.is_contiguous(memory_format=memory_format.memory_format):
            x = x.contiguous(memory_format=memory_format.memory_format)
        return x

    expected_size = memory_format.size
    if expected_size is None:
        raise AssertionError("memory_format.size must not be None")
    expected_stride = memory_format.stride
    if expected_stride is None:
        raise AssertionError("memory_format.stride must not be None")
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
def _disable_saved_tensors_hooks() -> Generator[None, None, None]:
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

    compiled_fn: Callable[..., Any]
    serialize_fn: Callable[[], Any]

    def __init__(
        self, compiled_fn: Callable[..., Any], serialize_fn: Callable[[], Any]
    ) -> None:
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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.compiled_fn(*args, **kwargs)


@dataclass
class AOTDispatchAutogradCompileSpec:
    compiled_fw_func: Callable[..., Any]
    compiled_bw_func: Callable[..., Any] | None
    maybe_subclass_meta: SubclassMeta | None
    num_symints_saved_for_bw: int
    backward_state_indices: list[int]
    disable_amp: bool
    indices_of_inps_to_detach: list[int]
    lazy_backward_info: (
        AutogradLazyBackwardCompileInfo | CachedAutogradLazyBackwardCompileInfo | None
    )
    aot_config: AOTConfig
    fw_metadata: ViewAndMutationMeta
    try_save_cache_entry: Callable[..., Any] | None


@dataclass
class _AutogradSavedState:
    metadata: ViewAndMutationMeta

    def save_from_forward(self, ctx: Any, fw_outs: Sequence[Any]) -> None:
        tensors_saved_with_vc_check = fw_outs[
            self.metadata.tensors_saved_for_backwards_with_vc_check_slice
        ]
        tensors_saved_no_vc_check = fw_outs[
            self.metadata.tensors_saved_for_backwards_no_vc_check_slice
        ]
        if not all(isinstance(x, torch.Tensor) for x in tensors_saved_with_vc_check):
            raise AssertionError(
                "expected all tensors_saved_with_vc_check to be Tensors, "
                f"got types: {[type(x) for x in tensors_saved_with_vc_check]}"
            )
        if not all(isinstance(x, torch.Tensor) for x in tensors_saved_no_vc_check):
            raise AssertionError(
                "expected all tensors_saved_no_vc_check to be Tensors, "
                f"got types: {[type(x) for x in tensors_saved_no_vc_check]}"
            )

        # See Note [Detaching saved tensors in AOTAutograd]
        num_vc_check = len(tensors_saved_with_vc_check)
        tensors_to_save = [
            x.detach() if x._is_view() else x for x in tensors_saved_with_vc_check
        ]
        tensors_no_vc_check = [
            x.detach() if x._is_view() else x for x in tensors_saved_no_vc_check
        ]

        # dynamic_saved_tensors_idxs has indices relative to all saved tensors
        # (vc_check + no_vc_check combined). Mark dynamics on the detached tensors.
        for idx, dims in self.metadata.dynamic_saved_tensors_idxs.items():
            if idx < num_vc_check:
                maybe_mark_dynamic_helper(tensors_to_save[idx], dims)
            else:
                maybe_mark_dynamic_helper(tensors_no_vc_check[idx - num_vc_check], dims)

        ctx.save_for_backward(*tensors_to_save)
        ctx._tensors_no_vc_check = tensors_no_vc_check

        symint_outs = fw_outs[self.metadata.symints_saved_for_backwards_slice]
        if not all(
            isinstance(x, (int, float, torch.SymInt, torch.SymFloat))
            for x in symint_outs
        ):
            raise AssertionError(
                "expected all symint_outs to be int/float/SymInt/SymFloat, "
                f"got types: {[type(x) for x in symint_outs]}"
            )
        ctx.symints = symint_outs

        opaque_object_outs = fw_outs[
            self.metadata.opaque_objects_saved_for_backwards_slice
        ]
        if not all(
            is_opaque_type(type(obj)) or isinstance(obj, OpaqueBase)
            for obj in opaque_object_outs
        ):
            raise AssertionError(
                "expected all opaque_object_outs to be opaque types, "
                f"got types: {[type(obj) for obj in opaque_object_outs]}"
            )
        ctx.opaque_objects = opaque_object_outs

    def load_tensors(self, ctx: Any) -> Sequence[torch.Tensor]:
        if len(ctx._tensors_no_vc_check) > 0:
            return list(ctx.saved_tensors) + ctx._tensors_no_vc_check
        return ctx.saved_tensors


@dataclass
class _AutogradForwardEpilogue:
    metadata: ViewAndMutationMeta

    def finalize(self, ctx: Any, fw_outs: Sequence[Any]) -> tuple[Any, ...]:
        num_outputs = self.metadata.num_outputs
        num_outputs_aliased = self.metadata.num_outputs_aliased
        num_mutated_runtime_inps = self.metadata.num_mutated_inp_runtime_indices
        num_forward_returns = self.metadata.num_forward_returns

        raw_returns = list(fw_outs[:num_forward_returns])

        # Wrap all autograd.Function.forward() outputs that are aliases
        # so that autograd.Function doesn't treat them as tensors
        if num_mutated_runtime_inps > 0:
            for i, idx in enumerate(self.metadata.mutated_inp_runtime_indices):
                # We could make this faster by only looping over inputs with metadata-only mutations
                # (instead of looping over inputs with either data or metadata mutations), but there shouldn't be many.
                info = self.metadata.input_info[idx]
                if info.mutates_metadata and not info.mutates_data:
                    raw_returns[i] = TensorAlias(raw_returns[i])

            if config.debug_assert:
                user_mutated_inputs_raw = raw_returns[0:num_mutated_runtime_inps]
                mut_inp_infos = [
                    x
                    for x in self.metadata.input_info
                    if x.mutates_data or x.mutates_metadata
                ]
                if len(user_mutated_inputs_raw) != len(mut_inp_infos):
                    raise AssertionError(
                        "expected len(user_mutated_inputs_raw) == len(mut_inp_infos), "
                        f"got {len(user_mutated_inputs_raw)} != {len(mut_inp_infos)}"
                    )

        if self.metadata.num_unsafe_view_outputs > 0:
            for idx in self.metadata.unsafe_view_out_indices:
                raw_return_idx = num_mutated_runtime_inps + idx
                o = raw_returns[raw_return_idx]
                raw_returns[raw_return_idx] = torch.ops.aten._unsafe_view(o, o.shape)

        if num_outputs_aliased > 0:
            for idx in self.metadata.aliased_out_indices:
                raw_return_idx = num_mutated_runtime_inps + idx
                raw_returns[raw_return_idx] = TensorAlias(raw_returns[raw_return_idx])

            if config.debug_assert:
                intermediates_raw = raw_returns[
                    num_mutated_runtime_inps + num_outputs :
                ]
                if any(isinstance(x, TensorAlias) for x in intermediates_raw):
                    raise AssertionError("expected no TensorAlias in intermediates_raw")

        # invariant: intermediate bases always require gradients, so we don't have to
        # consider marking them as non-differentiable.
        raw_returns_not_including_intermediate_bases = raw_returns[
            : num_mutated_runtime_inps + num_outputs
        ]
        raw_returns_meta = [
            x
            for x in self.metadata.input_info
            if x.mutation_type == MutationType.MUTATED_OUT_GRAPH
        ] + self.metadata.output_info

        fw_outs_not_requiring_grad = [
            x
            for (i, x) in enumerate(raw_returns_not_including_intermediate_bases)
            if isinstance(x, torch.Tensor) and not raw_returns_meta[i].requires_grad
        ]
        ctx.mark_non_differentiable(*fw_outs_not_requiring_grad)
        ctx._materialize_non_diff_grads = False
        return tuple(raw_returns)


@dataclass
class _AutogradRngStateTracker:
    num_rng: int
    graphsafe_idx: int | None
    fwd_rng_states: list[torch.Generator] = field(default_factory=list)
    bwd_rng_states: list[torch.Generator] = field(default_factory=list)
    curr_fwd_iter: Any = field(default_factory=lambda: itertools.count(0))
    backward_state_position: int = 0
    pending_forwards: set[int] = field(default_factory=set)
    saved_backward_tensor_states: dict[int, list[torch.Tensor]] = field(
        default_factory=dict
    )

    def add_forward_args(self, ctx: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if self.num_rng == 0:
            return args

        if len(self.fwd_rng_states) == 0:
            if self.graphsafe_idx is None:
                raise AssertionError("graphsafe_idx must not be None when num_rng > 0")
            initialize_rng_states(
                self.num_rng,
                self.graphsafe_idx,
                self.fwd_rng_states,
                self.bwd_rng_states,
            )

        curr_iter = next(self.curr_fwd_iter)
        ctx._curr_iter = curr_iter

        # if this state is not contained in the backward,
        # we need to save it for when its backward pass happens
        if curr_iter != self.backward_state_position:
            self.saved_backward_tensor_states[curr_iter] = [
                rng_state.get_state() for rng_state in self.fwd_rng_states
            ]

        self.pending_forwards.add(curr_iter)
        return (*args, *self.fwd_rng_states)

    def add_backward_args(self, ctx: Any, all_args: list[Any]) -> None:
        if self.num_rng == 0:
            return

        curr_backward_iter = ctx._curr_iter
        retain_graph = torch._C._autograd._get_current_graph_task_keep_graph()

        # Save current state if we have a pending forward that needs this state
        # or this state may be needed again because of retain graph
        if (
            self.backward_state_position in self.pending_forwards
            and self.backward_state_position not in self.saved_backward_tensor_states
            and (self.backward_state_position != curr_backward_iter or retain_graph)
        ):
            self.saved_backward_tensor_states[self.backward_state_position] = [
                rng_state.get_state() for rng_state in self.bwd_rng_states
            ]

        # Restore saved states if needed
        if curr_backward_iter in self.saved_backward_tensor_states:
            if self.backward_state_position != curr_backward_iter:
                for bwd_state, saved_state in zip(
                    self.bwd_rng_states,
                    self.saved_backward_tensor_states[curr_backward_iter],
                ):
                    bwd_state.set_state(saved_state)
            if not retain_graph:
                del self.saved_backward_tensor_states[curr_backward_iter]
        else:
            if self.backward_state_position != curr_backward_iter:
                raise AssertionError(
                    "expected backward_state_position == curr_backward_iter, "
                    f"got {self.backward_state_position} != {curr_backward_iter}"
                )

        self.backward_state_position = curr_backward_iter + 1
        if not retain_graph:
            self.pending_forwards.remove(curr_backward_iter)
        all_args.extend(self.bwd_rng_states)


@dataclass
class _AutogradBackwardCompiler:
    compiled_bw: Callable[..., Any] | None
    lazy_backward_info: (
        AutogradLazyBackwardCompileInfo | CachedAutogradLazyBackwardCompileInfo | None
    )
    disable_amp: bool
    aot_config: AOTConfig
    fw_metadata: ViewAndMutationMeta
    try_save_cache_entry: Callable[..., Any] | None

    def get_or_compile(self, *, saved_tensors_use_once: bool) -> Callable[..., Any]:
        if self.compiled_bw is not None:
            return self.compiled_bw

        if self.lazy_backward_info is None:
            raise AssertionError("lazy_backward_info must not be None")
        if not isinstance(self.lazy_backward_info, AutogradLazyBackwardCompileInfo):
            raise AssertionError(
                "expected AutogradLazyBackwardCompileInfo, "
                f"got {type(self.lazy_backward_info)}"
            )

        self._prepare_lazy_backward_context(saved_tensors_use_once)

        bw_module = self.lazy_backward_info.bw_module
        placeholder_list = self.lazy_backward_info.placeholder_list
        saved_context = self.lazy_backward_info.saved_context
        saved_compile_context = self.lazy_backward_info.saved_compile_context

        context = torch._C._DisableAutocast if self.disable_amp else nullcontext
        metrics_context = get_metrics_context()
        with (
            tracing(saved_context),
            compile_context(saved_compile_context),
            context(),
            track_graph_compiling(self.aot_config, "backward"),
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
            if self.aot_config.bw_compiler is None:
                raise AssertionError("aot_config.bw_compiler must not be None")
            self.compiled_bw = self.aot_config.bw_compiler(
                copy.deepcopy(bw_module), placeholder_list
            )
            # Maybe save cache entry
            if self.try_save_cache_entry is not None:
                self.try_save_cache_entry(
                    self.compiled_bw,
                    bw_module,
                    self.fw_metadata,
                    self.aot_config,
                )

        return self.compiled_bw

    def _prepare_lazy_backward_context(self, saved_tensors_use_once: bool) -> None:
        if self.lazy_backward_info is None:
            raise AssertionError("lazy_backward_info must not be None")
        if not isinstance(self.lazy_backward_info, AutogradLazyBackwardCompileInfo):
            raise AssertionError(
                "expected AutogradLazyBackwardCompileInfo, "
                f"got {type(self.lazy_backward_info)}"
            )

        if (
            hasattr(self.lazy_backward_info, "saved_context")
            and self.lazy_backward_info.saved_context is not None
        ):
            if not isinstance(self.lazy_backward_info.saved_context, TracingContext):
                raise AssertionError(
                    f"expected TracingContext, got {type(self.lazy_backward_info.saved_context)}"
                )
            ddp_ctx = self.lazy_backward_info.saved_context.ddp_optimizer_ctx
            if ddp_ctx is not None:
                if ddp_ctx.curr_bucket < 0:
                    raise AssertionError(
                        "expected same # of fw and bw compiles, "
                        f"but found bucket {ddp_ctx.curr_bucket}"
                    )
                curr_fw_meta = ddp_ctx.metadata_per_bucket[ddp_ctx.curr_bucket]
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
                self.lazy_backward_info.saved_context.fw_metadata = curr_fw_meta

        if not saved_tensors_use_once:
            self.fw_metadata.bw_donated_idxs = []
            # Update bw_donated_idxs if using lazy_backward_info from `aot_dispatch_autograd`
            if (
                hasattr(self.lazy_backward_info, "saved_context")
                and hasattr(self.lazy_backward_info.saved_context, "fw_metadata")
                and hasattr(
                    self.lazy_backward_info.saved_context.fw_metadata,  # type: ignore[union-attr]
                    "bw_donated_idxs",
                )
            ):
                self.lazy_backward_info.saved_context.fw_metadata.bw_donated_idxs = (  # type: ignore[union-attr]
                    # pyrefly: ignore [implicit-any]
                    []
                )


@dataclass
class _AOTDispatchAutogradFunctionFactory:
    spec: AOTDispatchAutogradCompileSpec

    def build(self) -> type[torch.autograd.Function]:
        compile_id = CompileContext.current_compile_id()
        compile_id_str = str(compile_id) if compile_id is not None else None
        self.spec.fw_metadata.compile_id_str = compile_id_str

        saved_state = _AutogradSavedState(self.spec.fw_metadata)
        forward_epilogue = _AutogradForwardEpilogue(self.spec.fw_metadata)
        rng_state = _AutogradRngStateTracker(
            num_rng=self.spec.fw_metadata.num_graphsafe_rng_states,
            graphsafe_idx=self.spec.fw_metadata.graphsafe_rng_state_index,
        )
        backward_compiler = _AutogradBackwardCompiler(
            compiled_bw=self.spec.compiled_bw_func,
            lazy_backward_info=self.spec.lazy_backward_info,
            disable_amp=self.spec.disable_amp,
            aot_config=self.spec.aot_config,
            fw_metadata=self.spec.fw_metadata,
            try_save_cache_entry=self.spec.try_save_cache_entry,
        )

        compiled_fw_func = self.spec.compiled_fw_func
        compiled_bw_func = self.spec.compiled_bw_func
        maybe_subclass_meta = self.spec.maybe_subclass_meta
        num_symints_saved_for_bw_ = self.spec.num_symints_saved_for_bw
        backward_state_indices = self.spec.backward_state_indices
        disable_amp = self.spec.disable_amp
        lazy_backward_info = self.spec.lazy_backward_info
        aot_config = self.spec.aot_config
        fw_metadata = self.spec.fw_metadata

        _codegen_bw_unwrap_fn = None
        _codegen_bw_wrap_fn = None
        if maybe_subclass_meta is not None:
            from .subclass_codegen import codegen_backward_subclass_fns

            _codegen_bw_unwrap_fn, _codegen_bw_wrap_fn = codegen_backward_subclass_fns(
                grad_input_metas=maybe_subclass_meta.grad_input_metas,
            )

        class CompiledFunction(torch.autograd.Function):
            compiled_fw = compiled_fw_func
            compiled_bw = compiled_bw_func
            metadata: ViewAndMutationMeta = fw_metadata  # type: ignore[assignment]
            maybe_subclass_metadata: SubclassMeta | None = maybe_subclass_meta
            num_symints_saved_for_bw = num_symints_saved_for_bw_
            _aot_id = aot_config.aot_id
            _lazy_backward_info = lazy_backward_info
            _bw_epilogue_wrap_fn = _codegen_bw_wrap_fn
            _bw_prologue_unwrap_fn = _codegen_bw_unwrap_fn
            boxed_grads_call = True

            @staticmethod
            def _compiled_autograd_key(ctx: Any) -> tuple[Any, ...]:
                return (ctx._autograd_function_id, *ctx.symints)

            @staticmethod
            # pyrefly: ignore [bad-override]
            def forward(ctx: Any, *deduped_flat_tensor_args: Any) -> Any:
                args = deduped_flat_tensor_args
                if backward_state_indices:
                    bw_state = args[backward_state_indices[0]]
                    if not isinstance(bw_state, BackwardState):
                        raise AssertionError(
                            f"expected BackwardState, got {type(bw_state)}"
                        )
                    ctx._compiled_autograd_backward_state = bw_state

                args = rng_state.add_forward_args(ctx, args)

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

                saved_state.save_from_forward(ctx, fw_outs)
                return forward_epilogue.finalize(ctx, fw_outs)

            @staticmethod
            def backward(ctx: Any, *flat_args: Any) -> tuple[Any, ...]:
                # With boxed_grads_call, grads arrive as a single mutable
                # list (not *args) so backward can free them individually
                # to reduce peak memory.
                if CompiledFunction.boxed_grads_call:
                    if len(flat_args) != 1 or not isinstance(flat_args[0], list):
                        raise AssertionError(
                            "boxed_grads_call is set but backward received "
                            f"{len(flat_args)} args instead of a single mutable "
                            "list. When boxed_grads_call=True, grads must be "
                            "passed as a single list argument [grad0, grad1, ...] "
                            "to allow freeing individual grads mid-backward."
                        )
                    grad_args = flat_args[0]
                else:
                    # Non-boxed path: used by subclasses of CompiledFunction
                    # that override boxed_grads_call to False.
                    grad_args = list(flat_args)
                del flat_args
                all_args = _backward_prologue_functional(
                    saved_state.load_tensors(ctx),
                    ctx.symints,
                    ctx.opaque_objects,
                    CompiledFunction.metadata,
                    CompiledFunction.maybe_subclass_metadata,
                    grad_args,
                    codegen_unwrap_fn=CompiledFunction._bw_prologue_unwrap_fn,
                )
                rng_state.add_backward_args(ctx, all_args)

                def impl_fn(double_ctx: Any = None) -> Any:
                    out = CompiledFunction._backward_impl(ctx, all_args)
                    return _backward_epilogue_functional(
                        CompiledFunction.metadata,
                        CompiledFunction.maybe_subclass_metadata,
                        out,
                        codegen_wrap_fn=CompiledFunction._bw_epilogue_wrap_fn,
                    )

                if (
                    torch._C._is_key_in_tls("context")
                    and (config_ctx := torch._C._get_obj_in_tls("context")) is not None
                ):
                    impl_fn = functools.partial(config_ctx.run, impl_fn)

                needs_grad = torch.is_grad_enabled() and any(
                    t.requires_grad for t in all_args if isinstance(t, torch.Tensor)
                )
                if needs_grad:
                    # double backward
                    return CompiledFunction._double_backward(ctx, impl_fn, all_args)
                return impl_fn()

            @staticmethod
            def _double_backward(
                ctx: Any, impl_fn: Callable[..., Any], all_args: list[Any]
            ) -> Any:
                # Ensure that the graph is connected, and error if double backward is performed.
                # See comment for why once_differentiable is not sufficient:
                # https://github.com/pytorch/pytorch/pull/92348/files#r1072962107
                class CompiledFunctionBackward(torch.autograd.Function):
                    # CompiledFunctionBackward is not yet supported in dynamo skipfiles
                    _aot_id = aot_config.aot_id

                    @staticmethod
                    # pyrefly: ignore [bad-override]
                    def forward(double_ctx: Any, *unused_args: Any) -> Any:
                        return impl_fn(double_ctx)

                    @staticmethod
                    def backward(ctx: Any, *args: Any) -> None:
                        raise RuntimeError(
                            "torch.compile with aot_autograd does not currently support double backward"
                        )

                CompiledFunctionBackward._compiled_autograd_key = (  # type: ignore[method-assign]
                    CompiledFunction._compiled_autograd_key
                )

                return CompiledFunctionBackward.apply(*all_args)

            @staticmethod
            def _backward_impl(ctx: Any, all_args: list[Any]) -> Any:
                # compiled autograd reimplements this function at proxy_call_aot_backward
                if backward_state_indices:
                    raise AssertionError("BackwardState requires CompiledAutograd")
                ctx.maybe_clear_saved_tensors()

                saved_tensors_use_once = (
                    not torch._C._autograd._get_current_graph_task_keep_graph()
                )
                compiled_bw = backward_compiler.get_or_compile(
                    saved_tensors_use_once=saved_tensors_use_once
                )
                CompiledFunction.compiled_bw = compiled_bw

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

                return call_func_at_runtime_with_args(
                    compiled_bw,
                    all_args,
                    steal_args=True,
                    disable_amp=disable_amp,
                )

        return CompiledFunction


# This is wrapped in a class just for namespacing purposes
# No need to make it into an actual CompilerWrapper because it doesn't fit the abstract as cleanly
class AOTDispatchAutograd:
    @staticmethod
    def _raise_tangent_metadata_error(
        expected_type: type | None,
        expected_meta: Any,
        runtime_type: type,
        runtime_meta: Any,
        orig_x: torch.Tensor,
        tangent_idx: int | None,
        tangent_desc: Any | None,
        compile_id_str: str | None,
        tangent_stack_trace: str | None,
    ) -> RuntimeError:
        expected_subclass_got_plain_tensor = (
            expected_type is not None
            and expected_type is not torch.Tensor
            and runtime_type is torch.Tensor
        )
        if expected_subclass_got_plain_tensor:
            tangent_msg = ""
            if tangent_idx is not None:
                tangent_msg = f" (tangent index: {tangent_idx})"

            output_hint = ""
            if tangent_desc is not None:
                from .descriptors import PlainAOTOutput, TangentAOTInput

                if isinstance(tangent_desc, TangentAOTInput) and isinstance(
                    tangent_desc.output, PlainAOTOutput
                ):
                    idx = tangent_desc.output.idx
                    output_hint = f"\n\nThe problematic output is: forward output at index {idx} (0-indexed)"
                else:
                    output_hint = (
                        f"\n\nThe problematic output is: {tangent_desc.expr()}"
                    )

            graph_hint = ""
            if compile_id_str is not None:
                graph_hint = (
                    f"\n\nThis error occurred in compiled graph [{compile_id_str}]."
                )

            stack_trace_hint = ""
            if tangent_stack_trace is not None:
                stack_trace_hint = (
                    f"\n\nThe forward output was created here:\n{tangent_stack_trace}"
                )

            return RuntimeError(
                f"""
During the backward, we encountered a tensor subclass where we guessed its
metadata incorrectly.
Expected a {expected_type.__name__} tangent but got a plain Tensor{tangent_msg}.
This happens when a compiled function returns multiple outputs that
require gradients, but .backward() is only called on some of them.
To fix: call .detach() on forward outputs you don't need gradients for.{output_hint}{graph_hint}{stack_trace_hint}

This error is also more likely to occur if your compiled model is suffering
from a large number of graph breaks. For more advice on finding and fixing
graph breaks, see:
https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.graph_breaks_index.html

For more info about this error, see:
https://github.com/pytorch/pytorch/issues/172556"""
            )
        else:
            return RuntimeError(
                f"""
During the backward, we encountered a tensor subclass where we guessed its
metadata incorrectly.
Expected: {expected_meta} (type {expected_type}),
got: {runtime_meta} (type {runtime_type}), shape: {orig_x.shape}.
Your tensor subclass must implement __coerce_same_metadata_as_tangent__."""
            )

    @staticmethod
    def process_runtime_tangent(
        x: Any,
        meta: PlainTensorMeta | SubclassCreationMeta,
        tangent_idx: int | None = None,
        tangent_desc: Any | None = None,
        compile_id_str: str | None = None,
        tangent_stack_trace: str | None = None,
    ) -> tuple[Any, list[Any]]:
        if not isinstance(x, torch.Tensor):
            return x, [x]

        if isinstance(x, FakeTensor):
            if not meta.memory_format:
                raise AssertionError(
                    "meta.memory_format must not be None for FakeTensor"
                )
            x = coerce_to_expected_memory_format(x, meta.memory_format)
            return x, [x]

        expected_type: type | None = torch.Tensor
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

        def maybe_coerce(x: torch.Tensor) -> torch.Tensor | None:
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
            raise AOTDispatchAutograd._raise_tangent_metadata_error(
                expected_type,
                expected_meta,
                runtime_type,
                runtime_meta,
                orig_x,
                tangent_idx,
                tangent_desc,
                compile_id_str,
                tangent_stack_trace,
            )

        # Coerce to expected memory format
        if not meta.memory_format:
            raise AssertionError("meta.memory_format must not be None")
        x = coerce_to_expected_memory_format(x, meta.memory_format)

        if not is_traceable_wrapper_subclass(x):
            return x, [x]

        if not isinstance(meta, SubclassCreationMeta):
            raise AssertionError(f"expected SubclassCreationMeta, got {type(meta)}")
        if orig_x is not x:
            runtime_subclass_keys = x.__tensor_flatten__()[0]

        if len(meta.attrs) != len(runtime_subclass_keys):
            raise AssertionError(
                f"expected len(meta.attrs) == len(runtime_subclass_keys), "
                f"got {len(meta.attrs)} != {len(runtime_subclass_keys)}"
            )
        leaves = []
        for attr, attr_meta in meta.attrs.items():
            if isinstance(attr_meta, OpaqueMeta):
                # Opaques aren't differentiable but occupy a flat arg slot.
                leaves.append(getattr(x, attr))
                continue
            elem = getattr(x, attr)
            new_elem, elem_leaves = AOTDispatchAutograd.process_runtime_tangent(
                elem, attr_meta
            )
            if new_elem is not elem:
                setattr(x, attr, new_elem)
            leaves.extend(elem_leaves)

        return x, leaves

    @staticmethod
    def post_compile(spec: AOTDispatchAutogradCompileSpec) -> Callable[..., Any]:
        compiled_function_cls = _AOTDispatchAutogradFunctionFactory(spec).build()
        return RuntimeWrapper(
            indices_of_inps_to_detach=spec.indices_of_inps_to_detach,
            trace_joint=True,
            disable_amp=spec.disable_amp,
        ).post_compile(
            compiled_function_cls.apply,
            spec.aot_config,
            runtime_metadata=spec.fw_metadata,
        )


@dataclass
class DebugAssertWrapper(CompilerWrapper):
    flat_requires_grad: list[bool | None] = field(default_factory=list)

    def post_compile(
        self,
        compiled_fn: Callable[..., Any],
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ) -> Callable[..., Any]:
        lines = ["def inner_fn(args):"]
        globals_dict: dict[str, object] = {"compiled_fn": compiled_fn}
        for i, can_require_grad in enumerate(self.flat_requires_grad):
            if can_require_grad is None:
                lines.append(
                    f"    if isinstance(args[{i}], Tensor):"
                    f" raise AssertionError("
                    f"'expected non-Tensor for arg {i}, got Tensor')"
                )
            elif not can_require_grad:
                msg_name = f"_msg_{i}"
                globals_dict[msg_name] = format_guard_bug_msg(
                    aot_config,
                    f"{describe_input(i, aot_config)} would not require grad",
                )
                lines.append(
                    f"    if args[{i}].requires_grad: raise AssertionError({msg_name})"
                )
        lines.append("    return compiled_fn(args)")

        source = "\n".join(lines)
        globals_dict["Tensor"] = Tensor

        from .subclass_codegen import _compile_and_exec_source

        return _compile_and_exec_source(
            source,
            globals_dict,
            "inner_fn",
            "debug_assert_wrapper",
            wrapped_fn=compiled_fn,
        )


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
    compiled_fn: Callable[..., Any],
    aot_config: AOTConfig,
    *,
    runtime_metadata: ViewAndMutationMeta,
) -> tuple[Callable[..., Any], ViewAndMutationMeta]:
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
    maybe_subclass_meta: SubclassMeta | None,
) -> None:
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
