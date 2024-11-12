import functools
from contextlib import contextmanager
from typing import Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp._fsdp_api import MixedPrecisionPolicy
from torch.distributed._composable.fsdp._fsdp_common import _cast_fp_tensor
from torch.distributed.tensor import (
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.nn.modules.lazy import LazyModuleMixin
from torch.utils._pytree import tree_map
from torch.utils.checkpoint import (
    checkpoint,
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)


_active_parametrization = False


@contextmanager
def disable_active_parametrization():  # type: ignore[no-untyped-def]
    global _active_parametrization
    try:
        _active_parametrization = False
        yield
    finally:
        _active_parametrization = True


@contextmanager
def enable_active_parametrization():  # type: ignore[no-untyped-def]
    global _active_parametrization
    try:
        _active_parametrization = True
        yield
    finally:
        _active_parametrization = False


def active_parametrization():  # type: ignore[no-untyped-def]
    global _active_parametrization
    return _active_parametrization


def fsdp_must_recompute_policy():  # type: ignore[no-untyped-def]
    def _fsdp_recomp_policy():  # type: ignore[no-untyped-def]
        def _custom_policy(ctx, func, *args, **kwargs):  # type: ignore[no-untyped-def]
            to_recompute = func in {
                torch.ops._c10d_functional.all_gather_into_tensor.default,
                torch.ops._c10d_functional.wait_tensor.default,
                torch.ops.aten._to_copy.default,  # for dtype cast in FSDP
            }
            return (
                CheckpointPolicy.MUST_RECOMPUTE
                if to_recompute
                else CheckpointPolicy.MUST_SAVE
            )

        return _custom_policy

    return create_selective_checkpoint_contexts(_fsdp_recomp_policy())


def fsdp_must_save_policy():  # type: ignore[no-untyped-def]
    def _fsdp_save_policy():  # type: ignore[no-untyped-def]
        def _custom_policy(ctx, func, *args, **kwargs):  # type: ignore[no-untyped-def]
            to_save = func in {
                torch.ops._c10d_functional.wait_tensor.default,
            }
            return (
                CheckpointPolicy.MUST_SAVE
                if to_save
                # NOTE(yf225): since wait_tensor output is saved, all-gather will never actually be recomputed.
                # So MUST_RECOMPUTE here for other ops is actually okay.
                else CheckpointPolicy.MUST_RECOMPUTE
            )

        return _custom_policy

    return create_selective_checkpoint_contexts(_fsdp_save_policy())


class ReplicateComputation(torch.nn.Module):
    def __init__(  # type: ignore[no-untyped-def]
        self, device_mesh, param_sharding, mode, reshard_after_forward, mp_policy
    ):
        super().__init__()
        self.device_mesh = device_mesh
        self.param_sharding = param_sharding
        self.mode = mode
        self.compute_placements = [Replicate()] * self.device_mesh.ndim
        self.grad_placements = [Partial(reduce_op="avg")] * self.device_mesh.ndim
        self.reshard_after_forward = reshard_after_forward
        self.mp_policy = mp_policy or MixedPrecisionPolicy()

    def replicate_compute(self, x):
        # data parallel runtime replicate parameters and do local compute
        # the gradients are partial tensors that needs to perform reduction
        # (i.e. DDP: allreduce, FSDP: reduce_scatter, HSDP: mix of both)

        # NOTE: available in pytorch main
        # if self.mp_policy.param_dtype is not None:
        #     x = x.to(dtype=self.mp_policy.param_dtype)
        # output = x.redistribute(placements=self.compute_placements).to_local(grad_placements=self.grad_placements)

        # NOTE: specifying mixed precision is only available in pytorch_intern24
        #       https://github.com/tianyu-l/pytorch_intern24/pull/20
        # support for FSDP + TP (assuming TP shards the inner-most dim)
        if self.mode == "fully_shard" and x._spec.mesh.ndim == 2:
            dp_placement, tp_placement = x._spec.placements
            dp_mesh, tp_mesh = self.device_mesh, x._spec.mesh["tp"]

            # re-wrap 2D DTensor to 1D DTensor on dp_mesh for efficient FSDP all-gather
            # TODO: we should consider merging this logic into DTensor redistribute API
            sharded_local_tensor = x.to_local()
            sharded_dtensor = DTensor.from_local(
                sharded_local_tensor, dp_mesh, self.param_sharding
            )

            # the actual FSDP all-gather on dp_mesh
            replicated_dtensor = sharded_dtensor.redistribute(
                placements=self.compute_placements,
                forward_dtype=self.mp_policy.param_dtype,
                backward_dtype=self.mp_policy.reduce_dtype,
            )

            # re-wrap 1D all-gathered DTensor on dp_mesh to 1D DTensor on tp_mesh
            # TODO: DTensor should support this mesh collasping operation
            replicated_local_tensor = replicated_dtensor.to_local(
                grad_placements=self.grad_placements
            )
            output = DTensor.from_local(
                replicated_local_tensor, tp_mesh, (tp_placement,)
            )
        else:
            output = x.redistribute(
                placements=self.compute_placements,
                forward_dtype=self.mp_policy.param_dtype,
                backward_dtype=self.mp_policy.reduce_dtype,
            ).to_local(grad_placements=self.grad_placements)

        return output

    def forward(self, x):
        if torch.compiler.is_compiling():
            assert active_parametrization()

        # This should never be set to true during forward, only outside for model
        # inspection / debugging / initialization
        # model initialization can be done now through
        # with disable_data_parallel():
        #     model.init_weights()
        # NOTE(yf225): note: also useful for Ads optimizer init (which needs sharded tensor)
        if not active_parametrization():
            return x

        if self.mode in ("fully_shard", "hybrid_shard"):
            if self.reshard_after_forward:
                # apply checkpointing to implement reshard_after_forward=True
                output = checkpoint(
                    self.replicate_compute,
                    x,
                    use_reentrant=False,
                    context_fn=fsdp_must_recompute_policy,
                )
            else:
                # apply must-save policy to implement reshard_after_forward=False
                # NOTE(yf225): this is important for avoiding different ranks making different recompute vs. save decision for all-gather output,
                # leading to NCCL stuckness in AOT backward graph.
                output = checkpoint(
                    self.replicate_compute,
                    x,
                    use_reentrant=False,
                    context_fn=fsdp_must_save_policy,
                )
        else:
            output = self.replicate_compute(x)
        return output


cls_key_to_SimpleFSDP_cls: Dict[Tuple[Type, str], Type] = {}


# NOTE(yf225): Ads model doesn't do explicit dtype casting for inputs
# and rely on FSDP2 `MixedPrecisionPolicy.cast_forward_inputs=True` to do it.
# Here we need to do the same dtype casting in SimpleFSDP.
def _pre_forward(mp_policy, module: nn.Module, args, kwargs):
    if mp_policy.cast_forward_inputs and mp_policy.param_dtype:
        cast_fn = functools.partial(_cast_fp_tensor, mp_policy.param_dtype)
        args, kwargs = tree_map(cast_fn, args), tree_map(cast_fn, kwargs)
    return args, kwargs


# NOTE(yf225): to match FSDP2 behavior (output_dtype casting)
def _post_forward(mp_policy, module: nn.Module, input, output):
    if mp_policy.output_dtype is not None:
        output = tree_map(
            functools.partial(_cast_fp_tensor, mp_policy.output_dtype),
            output,
        )
    return output


def create_fsdp_managed_attr(p_name):
    def getter(self):
        # TODO(yf225): if this function throws exception, how does it behave? add a unit test for it.
        return self._name_to_fsdp_managed_attr_getter[p_name]()

    def setter(self, value):
        raise RuntimeError("Setting FSDP-managed attribute is not supported")

    return property(getter, setter)


def unimplemented_deepcopy(*args, **kwargs):
    raise AssertionError(
        "FSDP does not support deepcopy. Please use state dict for serialization."
    )


def SimpleFSDP_data_parallel(
    model,
    device_mesh,
    mode="replicate",
    reshard_after_forward: bool = True,
    mp_policy: Optional[MixedPrecisionPolicy] = None,
):
    if isinstance(model, SimpleFSDPModule):
        return model

    if mode == "replicate":
        param_sharding = (Replicate(),)
    elif mode == "fully_shard":
        param_sharding = (Shard(0),)
    elif mode == "hybrid_shard":
        # replicate inter-host, fully shard intra-host
        param_sharding = (Replicate(), Shard(0))
        assert (
            device_mesh.ndim == 2
        ), "hybrid sharded data parallel requires 2D DeviceMesh"
    else:
        raise ValueError(f"Unsupported mode {mode}")

    for mod_name, mod in sorted(list(model.named_modules())):
        if isinstance(mod, SimpleFSDPModule):
            continue

        # NOTE(yf225): we need to be careful with LazyModuleMixin, as those modules don't have their params created until after the first iteration.
        # We need to detect whether module is LazyModuleMixin and if so, assert they are initialized before doing register_parameter here.
        assert not (
            isinstance(mod, LazyModuleMixin) and mod.has_uninitialized_params()
        ), "Lazy modules (inherited from LazyModuleMixin) must be initialized before applying SimpleFSDP. Please run your model at least once before applying SimpleFSDP"

        params_dict = dict(mod.named_parameters(recurse=False))

        # Create new class for this module with all parametrized parameters
        param_properties = {}
        for p_name, p in params_dict.items():
            assert (
                p is not None
            ), "Expected None parameter to never appear in module.named_parameters()"
            if p.numel() > 0:
                param_local = nn.Parameter(
                    distribute_tensor(p, device_mesh, param_sharding)
                )
                # TODO(yf225): verify whether new param registration for ParameterDict and ParameterList need special handling
                if isinstance(mod, torch.nn.ParameterDict):
                    mod[p_name] = param_local
                elif isinstance(mod, torch.nn.ParameterList):
                    mod[int(p_name)] = param_local
                else:
                    mod.register_parameter(
                        p_name,
                        # NOTE: for 2D we need to distribute_tensor a DTensor
                        #       which requires latest change in pytorch_intern24
                        #       https://github.com/tianyu-l/pytorch_intern24/pull/25
                        param_local,
                    )
                parametrization = ReplicateComputation(
                    device_mesh,
                    param_sharding,
                    mode,
                    reshard_after_forward,
                    mp_policy=mp_policy,
                )

                # NOTE(yf225): adapted from Francisco's prototype (P1668496721) for fixing param FQN
                def getter(
                    self_mod=mod,
                    _param_name=p_name,
                    _pn=parametrization,
                    _param_local=param_local,
                ):
                    def _inner():
                        _param = self_mod._parameters[_param_name]
                        assert _param is not None, "Unexpected None parameter!"
                        ret = _pn(_param)
                        return ret

                    if torch.compiler.is_compiling():
                        return _inner()
                    else:
                        try:
                            return _inner()
                        except Exception as e:
                            e_str = str(e)
                            # NOTE(yf225): important for not letting the exception message be swallowed in the log.
                            # This prints false alarms with empty exception message at end of training process,
                            # so need to check exception message length first.
                            if len(e_str) > 0:
                                print(f"SimpleFSDP getter exception: {e_str}")
                            raise

                param_properties[p_name] = getter
            else:
                param_properties[p_name] = lambda: p

        # NOTE(yf225): This is important for Dynamo to route the module to FSDPManagedNNModuleVariable
        mod._is_fsdp_managed_module = True  # type: ignore[assignment]
        mod._fsdp_use_orig_params = True  # type: ignore[assignment]

        cls = mod.__class__
        # NOTE(yf225): nn.Linear(bias=True) and nn.Linear(bias=False) has different set of parameter names in named_parameters() (i.e. one has "bias" and one doesn't).
        # In general, we cannot expect different configurations of the same nn module class to have the same set of parameters, making it difficult to use one single custom class
        # (with the same `namespace` dict) to represent all of those variations.
        # Easiest way to manage this is to treat different configurations of the same nn module class as separate module classes in the class cache.
        # If we are not careful here, there can be weird and hard-to-debug "accessing None parameter" error within getter.
        # TODO(yf225): We need to add unit tests for this block of code: 1. model has two linears (first bias=True then bias=False), 2. model has two linears of different shapes.
        param_properties_key = "#".join(sorted(param_properties.keys()))
        # NOTE(yf225): Q: why do we need class cache? A: we intentionally don't want to create one new class per module instance (to avoid recompilation per new class).
        new_cls = cls_key_to_SimpleFSDP_cls.get((cls, param_properties_key), None)
        if not new_cls:
            namespace = {"__deepcopy__": unimplemented_deepcopy}
            for p_name in param_properties:
                # NOTE(yf225): it's important to have this indirection, to make sure that:
                # Different instances of the same class can resolve their parameter access to instance-specific getters
                # (which contains unique objects used in that instance-specific parameter's unshard operation).
                namespace[p_name] = create_fsdp_managed_attr(p_name)
            new_cls = type(
                f"SimpleFSDP{cls.__name__}", (SimpleFSDPModule, cls), namespace
            )
            cls_key_to_SimpleFSDP_cls[(cls, param_properties_key)] = new_cls
        mod.__class__ = new_cls
        mod._name_to_fsdp_managed_attr_getter = param_properties

    model.register_forward_pre_hook(
        functools.partial(_pre_forward, mp_policy), prepend=True, with_kwargs=True
    )
    model.register_forward_hook(
        functools.partial(_post_forward, mp_policy), prepend=False
    )
    return model


# NOTE(yf225): We have this wrapping module class, because existing Ads FSDP2 integration
# wants to dispatch to FSDP2 vs. FSDP1 path based on module class type.
class SimpleFSDPModule:
    def __new__(cls, *args, **kwargs):
        """
        Override ``__new__`` to remove the FSDP class and directly construct
        the original class for cases like indexing into a container module.
        """
        # Use index 2 since 0 is the dynamically constructed `FSDP<...>` class
        # and index 1 is the `SimpleFSDPModule` class itself
        assert len(cls.__mro__) > 2
        orig_cls = cls.__mro__[2]
        self = orig_cls.__new__(orig_cls, *args, **kwargs)
        self.__init__(*args, **kwargs)
        return self

    # NOTE(yf225): per Andrew: this is only meant for implicit prefetching, which SimpleFSDP will not use.
    def set_post_optim_event(self, event: torch.Event) -> None:
        pass

    # TODO(yf225): do we need to implement these?
    # def reshard(self) -> None:
    # def unshard(self, async_op: bool = False) -> Optional["UnshardHandle"]:
    # def set_is_last_backward(self, is_last_backward: bool) -> None:
    # def set_requires_gradient_sync(
    #     self, requires_gradient_sync: bool, *, recurse: bool = True
    # ) -> None:
    # def set_requires_all_reduce(
    #     self, requires_all_reduce: bool, *, recurse: bool = True
    # ) -> None:
    # def set_reshard_after_backward(
    #     self, reshard_after_backward: bool, *, recurse: bool = True
    # ) -> None:
    # def set_modules_to_forward_prefetch(self, modules: List["FSDPModule"]) -> None:
    # def set_modules_to_backward_prefetch(self, modules: List["FSDPModule"]) -> None:
    # def set_reduce_scatter_divide_factor(self, factor: float) -> None:
    # def set_unshard_in_backward(self, unshard_in_backward: bool) -> None:
    # def _set_unshard_async_op(self, async_op: bool):
    # def _get_fsdp_state(self) -> FSDPState:
    # def _apply(self, *args: Any, **kwargs: Any) -> Any:
