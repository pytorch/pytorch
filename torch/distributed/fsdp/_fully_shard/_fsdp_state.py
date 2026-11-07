# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import functools
import logging
from dataclasses import replace
from typing import Any, Generic, TYPE_CHECKING, TypeVar

import torch
import torch.nn as nn
from torch.autograd.graph import _MultiHandle
from torch.distributed import _spmd_no_typecheck
from torch.distributed._composable_state import (
    _get_module_state,
    _insert_module_state,
    _State,
)
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.utils import _apply_to_tensors, _to_kwargs

from ._fsdp_api import MixedPrecisionPolicy
from ._fsdp_common import _cast_fp_tensor, _dynamo_disable, TrainingState


if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle

    from .._flex_shard.bucket_runtime import BucketCommContext
    from ._fsdp_engine import FullyShardBucket
    from ._fsdp_param import FSDPParam

logger = logging.getLogger("torch.distributed.fsdp.fully_shard")

_StateType = TypeVar("_StateType", bound="FSDPState")


class FSDPStateContext(Generic[_StateType]):
    """This has state shared across FSDP states."""

    def __init__(self) -> None:
        # All FSDP states in the root state's module tree, in
        # ``named_modules()`` pre-order. The root post-forward completes
        # incomplete group forwards in ``reversed`` order so that within a
        # nesting chain the deepest group finishes first, matching the order
        # ``_post_forward`` would unwind in the non-partial case.
        self.all_states: list[_StateType] = []
        # Iteration's forward root runs the once-per-forward logic; this root
        # may not be the overall root set by lazy initialization in cases where
        # only a submodule runs forward (e.g. encoder-only for eval)
        self.iter_forward_root: _StateType | None = None
        # Whether to finalize backward in this backward's final callback
        self.is_last_backward: bool = True
        # Whether backward finalization is owned by the caller
        self.manual_backward_finalization: bool = False
        # Manual finalization mode selected when backward starts
        self.manual_backward_finalization_active: bool | None = None
        # An asynchronous explicit finalization must be waited on before this
        # FSDP tree starts more work.
        self.gradient_reduction_pending: bool = False
        # Optional user-provided event recorded after optimizer for the
        # all-gather streams to wait on in the root pre-forward
        self.post_optim_event: torch.Event | None = None


class FSDPState(_State):
    # Name used in error messages; subclasses can override
    _state_name: str = "FSDP"

    def __init__(self) -> None:
        super().__init__()
        # The engine buckets (``FullyShardBucket``) of this state's
        # ``fully_shard`` call, one per mesh (per-param mesh support), run by
        # the root's FlexShard engine context
        self._fsdp_param_groups: list[FullyShardBucket] = []
        self._engine_context: BucketCommContext | None = None
        # The context share_comm_ctx gave this root, for its lazy init
        self._shared_engine_context: BucketCommContext | None = None
        self._is_root: bool | None = None  # root set during lazy init
        self._state_ctx = FSDPStateContext()
        self._training_state: TrainingState = TrainingState.IDLE
        # ``False`` when user set reshard_after_forward
        # through ``fully_shard`` or ``set_reshard_after_forward``
        self._auto_reshard_after_forward: bool | None = True

    @property
    def _engine_buckets(self) -> "list[FullyShardBucket]":
        return self._fsdp_param_groups

    @property
    def _modules_to_run_forward(self) -> "_PendingGroupModules":
        """The modules of this state's open grouped forward, as FSDP2's
        attribute, for callers that finish such forwards themselves (e.g.
        torchtitan's evaluation cleanup)."""
        return _PendingGroupModules(self)

    def _get_state_for_module(self, module: nn.Module) -> "FSDPState | None":
        """Get the state for a module. Subclasses can override to use different state getters."""
        return _get_module_fsdp_state(module)

    @property
    def _fsdp_param_group(self) -> "FullyShardBucket | None":
        """
        Returns the param group for backward compatibility.
        This property is only valid when there is at most one param group.
        For per-param mesh support with multiple param groups, use
        ``_fsdp_param_groups`` instead.
        """
        if len(self._fsdp_param_groups) > 1:
            group_fqns = [g._module_fqn for g in self._fsdp_param_groups]
            raise AssertionError(
                f"Expected at most 1 param group for backward compatibility, "
                f"but got {len(self._fsdp_param_groups)} (fqns: {group_fqns}). "
                f"Use `_fsdp_param_groups` (plural) to access all param groups "
                f"when using per-param mesh via shard_placement_fn returning "
                f"ShardPlacementResult."
            )
        if self._fsdp_param_groups:
            return self._fsdp_param_groups[0]
        return None

    # Define a separate init since `__init__` is called in the contract
    def init(
        self,
        modules: tuple[nn.Module, ...],
        device: torch.device,
        mp_policy: MixedPrecisionPolicy,
        auto_reshard_after_forward: bool,
    ) -> None:
        for module in modules:
            _insert_module_state(module, self)
        self._modules = modules
        self._device = device
        self._device_handle = _get_device_handle(device.type)
        self._mp_policy = replace(mp_policy, param_dtype_override_fn=None)
        self._auto_reshard_after_forward = auto_reshard_after_forward
        # The engine's group hooks run per module and track the group's
        # forward pass themselves
        pre_handles = [
            module.register_forward_pre_hook(
                self._pre_forward, prepend=True, with_kwargs=True
            )
            for module in modules
        ]
        post_handles = [
            module.register_forward_hook(
                self._post_forward, prepend=False, always_call=len(modules) > 1
            )
            for module in modules
        ]
        handle = _MultiHandle(tuple(pre_handles + post_handles))
        self._pre_forward_hook_handle: RemovableHandle | _MultiHandle = handle
        self._post_forward_hook_handle: RemovableHandle | _MultiHandle = handle

    def _root_pre_forward(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        self._lazy_init()
        if self._state_ctx.iter_forward_root is not None:
            return args, kwargs
        logger.debug("FSDP::root_pre_forward")
        self._state_ctx.iter_forward_root = self
        with torch.profiler.record_function("FSDP::root_pre_forward"):
            # The engine's all-gathers wait for the current stream when issued,
            # so for the optimizer; a post-optimizer event can make them wait
            # for that point instead.
            context = self._engine_context
            if context is not None:
                context.wait_reductions_at_backward_end = (
                    self._state_ctx.is_last_backward
                )
                context.manual_backward_finalization = (
                    self._state_ctx.manual_backward_finalization
                )
                if (event := self._state_ctx.post_optim_event) is not None:
                    context.unshard_stream.wait_event(event)
                    self._state_ctx.post_optim_event = None
            if self._device.type in [
                "cuda",
                "hpu",
                "xpu",
                "mtia",
                torch._C._get_privateuse1_backend_name(),
            ]:
                if args or kwargs:
                    with torch.profiler.record_function("FSDP::inputs_to_device"):
                        args_tuple, kwargs_tuple = _to_kwargs(
                            args, kwargs, self._device, False
                        )  # same as DDP
                    args = args_tuple[0]
                    kwargs = kwargs_tuple[0]
        return args, kwargs

    def _lazy_init(self) -> None:
        """
        Lazy initialization represents when all modules' parallelisms have
        finalized (e.g. FSDP has been applied to all desired modules). This
        means that we can determine which state is the root, and we do so by
        the 1st state to run forward.
        """
        if self._is_root is not None:
            return  # no-op: already initialized
        if len(self._modules) > 1:
            raise RuntimeError(
                f"{self._state_name} requires a single root module but got {self._modules}"
            )
        root_module = self._modules[0]
        for module in root_module.modules():
            state = self._get_state_for_module(module)
            if (
                state is not None
                and state is not self
                and state._state_ctx.manual_backward_finalization
            ):
                raise RuntimeError(
                    "set_manual_backward_finalization must be called on the "
                    f"root {self._state_name} module"
                )
        self._is_root = True
        visited_states: set[FSDPState] = set()
        for module_name, module in root_module.named_modules():
            if (state := self._get_state_for_module(module)) is None:
                continue
            if module is not root_module:
                if state not in visited_states and state._is_root is not None:
                    raise RuntimeError(
                        f"{self._state_name} state has already been lazily initialized for "
                        f"{module_name}\n{self._state_name} requires running forward through "
                        "the root module first"
                    )
                state._is_root = False
            # A single state can map to multiple modules (e.g.
            # fully_shard([mod_a, mod_b, mod_c])), so dedup here.
            if state not in visited_states:
                self._state_ctx.all_states.append(state)
            visited_states.add(state)
        self._init_fqns()
        self._validate_no_duplicate_params()
        self._lazy_init_engine()

    def _lazy_init_engine(self) -> None:
        """Join every state's engine buckets to one engine context, as FSDP2
        shares the root's comm context."""
        from ._fsdp_engine import new_engine_context

        context = self._shared_engine_context
        if context is None:
            context = new_engine_context(self._device)
        context.backward_start_callbacks.append(self._engine_backward_start)
        context.backward_end_callbacks.append(self._engine_backward_end)
        for state in self._state_ctx.all_states:
            state._state_ctx = self._state_ctx
            state._engine_context = context
            buckets = state._engine_buckets
            for bucket in buckets:
                # For the root, do not reshard after forward since for
                # training, the parameters would be all-gathered right away
                reshard = bucket.storage._reshard_after_forward and not (
                    state is self and self._auto_reshard_after_forward
                )
                bucket.join(context, reshard_after_forward=reshard)
                # The last bucket's post-backward trigger wraps the others', so
                # it reduces first
                bucket.bucket.retires_pending_reductions = bucket is buckets[-1]

    def _engine_backward_start(self) -> None:
        """Fix the manual backward finalization mode until the backward
        iteration ends, as FSDP2's pre-backward does."""
        state_ctx = self._state_ctx
        if state_ctx.manual_backward_finalization_active is None:
            state_ctx.manual_backward_finalization_active = (
                state_ctx.manual_backward_finalization
            )

    def _engine_backward_end(self) -> None:
        """End of a backward on the engine: reset the frontend's per-iteration
        state, as FSDP2's root post-backward callback does."""
        logger.debug("FSDP::root_post_backward")
        self._state_ctx.iter_forward_root = None
        # With manual finalization, finalize_backward ends the iteration
        if (
            self._state_ctx.is_last_backward
            and not self._state_ctx.manual_backward_finalization_active
        ):
            self._state_ctx.manual_backward_finalization_active = None
        for state in self._state_ctx.all_states:
            state._training_state = TrainingState.IDLE
            for bucket in state._engine_buckets:
                bucket.wait_for_grad_offload()

    def _validate_no_duplicate_params(self) -> None:
        seen: set[int] = set()
        for state in self._state_ctx.all_states:
            for fsdp_param_group in state._fsdp_param_groups:
                for fsdp_param in fsdp_param_group.fsdp_params:
                    if fsdp_param._orig_param_uid in seen:
                        raise ValueError(
                            f"Parameter '{fsdp_param._param_fqn}' is shared with a "
                            f"parameter already managed by another FSDP group. "
                            f"For shared/tied parameters, use "
                            f"fully_shard([module_a, module_b]) to place them in "
                            f"the same FSDP group."
                        )
                    seen.add(fsdp_param._orig_param_uid)

    def _init_fqns(self) -> None:
        """Sets module and parameter FQN attributes for debugging."""
        if not self._is_root:
            raise AssertionError("Expected _is_root to be True")
        root_module = self._modules[0]
        param_to_fsdp_param: dict[nn.Parameter, FSDPParam] = {}
        # Build a mapping from module to all its buckets (not just one)
        module_to_fsdp_param_groups: dict[nn.Module, list[FullyShardBucket]] = {}
        for state in self._state_ctx.all_states:
            for fsdp_param_group in state._fsdp_param_groups:
                for fsdp_param in fsdp_param_group.fsdp_params:
                    param_to_fsdp_param[fsdp_param.sharded_param] = fsdp_param
                for module in fsdp_param_group.modules:
                    if module not in module_to_fsdp_param_groups:
                        module_to_fsdp_param_groups[module] = []
                    module_to_fsdp_param_groups[module].append(fsdp_param_group)
        for param_name, param in root_module.named_parameters():
            if param in param_to_fsdp_param:
                param_to_fsdp_param[param]._param_fqn = param_name
        for module_name, module in root_module.named_modules():
            if module in module_to_fsdp_param_groups:
                # Set FQN for all param groups associated with this module
                for bucket in module_to_fsdp_param_groups[module]:
                    if bucket.module_fqn is None:
                        bucket.module_fqn = module_name
                    else:
                        bucket.module_fqn += f", {module_name}"

    @_dynamo_disable
    def _pre_forward(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        from .._flex_shard.bucket_runtime import _in_backward

        context = self._engine_context
        if self._state_ctx.gradient_reduction_pending or (
            context is not None and context.pending_finalization is not None
        ):
            raise RuntimeError(
                "The previous gradient reduction must be waited on before forward"
            )
        # In an activation checkpointing recompute, the buckets' hooks unshard
        if not _in_backward():
            with _spmd_no_typecheck():
                # With grouped ``fully_shard([a, b, ...])`` the pre-hook fires
                # per module; root setup runs on the first module's entry
                state_first_in_pass = self._training_state != TrainingState.FORWARD
                self._training_state = TrainingState.FORWARD
                if state_first_in_pass:
                    args, kwargs = self._root_pre_forward(module, args, kwargs)
        args, kwargs = self._cast_forward_inputs(args, kwargs)
        # The buckets' hooks unshard outside spmd_types' type checker
        # themselves; the post-backward trigger they wrap the typed inputs in
        # stays checked
        for bucket in self._engine_buckets:
            runtime = bucket.bucket
            hook = (
                runtime.group_pre_forward_hook
                if runtime.group_modules
                else runtime.pre_forward_hook
            )
            if (result := hook(module, args, kwargs)) is not None:
                args, kwargs = result
            if bucket.has_spmd_types:
                for fsdp_param in bucket.fsdp_params:
                    fsdp_param._restore_spmd_types(fsdp_param.unsharded_param)
        return args, kwargs

    @_dynamo_disable
    def _post_forward(self, module: nn.Module, input: Any, output: Any) -> Any:
        from .._flex_shard.bucket_runtime import _in_backward

        with _spmd_no_typecheck():
            for bucket in self._engine_buckets:
                runtime = bucket.bucket
                if runtime.group_modules:
                    runtime.group_post_forward_hook(module, input, output)
                else:
                    runtime.post_forward_hook(module, input, output)
            if _in_backward():
                return self._cast_output_dtype(output)
            if not any(b.bucket.group_pass is not None for b in self._engine_buckets):
                self._training_state = TrainingState.IDLE
            if self._state_ctx.iter_forward_root is self:
                # Complete the forward of groups whose modules did not all run,
                # e.g. a decoder that leaves its output projection to the loss.
                # See ``all_states`` for why this iterates ``reversed``.
                for state in reversed(self._state_ctx.all_states):
                    if state is self:
                        continue
                    for bucket in state._engine_buckets:
                        if bucket.bucket.group_pass is not None:
                            bucket.bucket.complete_group_pass(output)
                            state._training_state = TrainingState.IDLE
                self._state_ctx.iter_forward_root = None
            if torch._C._are_functorch_transforms_active():
                # Under functorch, outputs that outer autograd differentiates can
                # report requires_grad=False, so the buckets hooked none of them
                output = _apply_to_tensors(
                    lambda t: _FunctorchPreBackward.apply(self, t)
                    if not t.requires_grad and (t.is_floating_point() or t.is_complex())
                    else t,
                    output,
                )
            return self._cast_output_dtype(output)

    def _functorch_pre_backward(self, grad: torch.Tensor) -> None:
        for bucket in self._engine_buckets:
            bucket.bucket.pre_backward_hook((grad,))

    def _cast_forward_inputs(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if not (self._mp_policy.cast_forward_inputs and self._mp_policy.param_dtype):
            return args, kwargs
        with torch.profiler.record_function("FSDP::cast_forward_inputs"):
            cast_fn = functools.partial(_cast_fp_tensor, self._mp_policy.param_dtype)
            return _apply_to_tensors(cast_fn, args), _apply_to_tensors(cast_fn, kwargs)

    def _cast_output_dtype(self, output: Any) -> Any:
        if self._mp_policy.output_dtype is None:
            return output
        with torch.profiler.record_function("FSDP::cast_forward_outputs"):
            return _apply_to_tensors(
                functools.partial(_cast_fp_tensor, self._mp_policy.output_dtype),
                output,
            )


class _PendingGroupModules:
    """A view of the modules an open grouped forward has yet to run; clearing
    it completes that forward."""

    def __init__(self, state: FSDPState) -> None:
        self._state = state

    def _modules(self) -> set[nn.Module]:
        return {
            module
            for bucket in self._state._engine_buckets
            for module in bucket.bucket.modules_to_run
        }

    def __bool__(self) -> bool:
        return bool(self._modules())

    def __len__(self) -> int:
        return len(self._modules())

    def __iter__(self):
        return iter(self._modules())

    def __contains__(self, module: object) -> bool:
        return module in self._modules()

    def clear(self) -> None:
        with _spmd_no_typecheck():
            for bucket in self._state._engine_buckets:
                if bucket.bucket.group_pass is not None:
                    bucket.bucket.complete_group_pass(None)
        self._state._training_state = TrainingState.IDLE


class _FunctorchPreBackward(torch.autograd.Function):
    """Runs the buckets' pre-backward hooks for an output under functorch
    transforms (e.g. vmap, jvp)."""

    generate_vmap_rule = True

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(state: FSDPState, output: torch.Tensor) -> torch.Tensor:
        return output

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        ctx.state = inputs[0]

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[None, torch.Tensor]:
        ctx.state._functorch_pre_backward(grad)
        return None, grad

    @staticmethod
    # pyrefly: ignore [bad-override]
    def jvp(ctx: Any, state_tangent: Any, tangent: torch.Tensor | None) -> Any:
        # Keep the tangent's backward on the pre-backward path
        if tangent is None:
            return None
        return _FunctorchPreBackward.apply(ctx.state, tangent)


def _get_module_fsdp_state(module: nn.Module) -> FSDPState | None:
    state = _get_module_state(module)
    if isinstance(state, FSDPState):
        return state
    return None
