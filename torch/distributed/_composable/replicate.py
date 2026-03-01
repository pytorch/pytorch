# mypy: allow-untyped-defs
import weakref
from collections.abc import Iterable
from typing import Any, NoReturn

import torch
import torch.nn as nn
from torch.distributed._composable_state import _State
from torch.nn.parallel import DistributedDataParallel

from .contract import _get_registry, contract


_ROOT_MODULE_PREFIX = ""


class _ReplicateState(_State):
    _ddp_weakref: weakref.ref

    def __init__(self) -> None:
        super().__init__()
        self.module: nn.Module = nn.ParameterList()
        self.has_initialized: bool = False
        self._param_list: nn.ParameterList = nn.ParameterList()
        # TODO(@fegin): this variable is originally create for testing, we
        # should remove this if possible.
        self._orig_module = self.module
        self._param_names: list[str] = []
        self._no_sync: bool = False
        self._init_args: tuple[Any, ...] | None = None
        self._init_kwargs: dict[str, Any] = {}
        self._comm_hook_args: list[Any] = []

    def _collect_params(
        self,
        module: nn.Module,
        ignored_modules: set[nn.Module],
        ignored_params: set[nn.Parameter],
        prefix: str = _ROOT_MODULE_PREFIX,
    ) -> None:
        # skip if managed by fully_sharded API
        if _is_fully_sharded(module):
            return

        # if a module is ignored, all descendants of the module are ignored.
        if module in ignored_modules:
            return

        recurse_prefix = (
            f"{prefix}." if prefix != _ROOT_MODULE_PREFIX else _ROOT_MODULE_PREFIX
        )

        for n, p in module.named_parameters(recurse=False):
            if p not in ignored_params:
                self._param_list.append(p)
                self._param_names.append(f"{recurse_prefix}{n}")

        for name, child_module in module.named_children():
            self._collect_params(
                child_module,
                ignored_modules,
                ignored_params,
                prefix=f"{recurse_prefix}{name}",
            )

    def lazy_init(self) -> None:
        @torch._disable_dynamo(recursive=True)
        def _lazy_init():
            if self._init_args is None:
                raise AssertionError
            self.init(*self._init_args, **self._init_kwargs)
            self.register_comm_hook()
            self._init_args = ()
            self._init_kwargs = {}

        _lazy_init()

    def init(
        self,
        module: nn.Module,
        ignored_modules: set[nn.Module],
        **kwargs,
    ) -> None:
        if self.has_initialized:
            return

        self.has_initialized = True
        self.module = module
        ignored_params = {p for m in ignored_modules for p in m.parameters()}
        for submodule in module.modules():
            if _is_fully_sharded(submodule):
                ignored_params.update(submodule.parameters())
        from torch.distributed.tensor.parallel.ddp import _localize_dtensor

        _localize_dtensor(module, ignored_params=ignored_params)
        self._collect_params(module, ignored_modules, ignored_params)

        if "device_id" in kwargs:
            # replicate() supports a small usability enhancement where
            # user can pass in device_id as a Union[int, torch.device] even for
            # CPU devices so users don't have to change code for CPU/GPU runs.
            # We derive the right device_ids to feed into DDP to support this.
            if kwargs["device_id"] is not None:
                device_id = kwargs["device_id"]
                # Convert to device_ids that DDP expects.
                if isinstance(device_id, torch.device) and device_id.type == "cpu":
                    # CPU modules receive device_ids None
                    kwargs["device_ids"] = None
                else:
                    # GPU modules expect device_ids=[cuda_device]
                    kwargs["device_ids"] = [device_id]
            else:
                kwargs["device_ids"] = None
            kwargs.pop("device_id")

        self._ddp = DistributedDataParallel(self._param_list, **kwargs)
        # Weakref to the DDP instance is currently only used for testing.
        replicate.state(self.module)._ddp_weakref = weakref.ref(self._ddp)

    def register_comm_hook(self) -> None:
        for comm_args, comm_kwargs in self._comm_hook_args:
            self._ddp.register_comm_hook(*comm_args, **comm_kwargs)
        self._comm_hook_args.clear()

    def record_init_args(self, *args, **kwargs) -> None:
        self._init_args = args
        self._init_kwargs = kwargs

    def forward_pre_hook(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        if self._init_args or self._init_kwargs:
            self.lazy_init()
        self._ddp.require_backward_grad_sync = not self._no_sync
        DistributedDataParallel._active_ddp_module = self._ddp
        return self._ddp._pre_forward(*args, **kwargs)

    def forward_post_hook(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        DistributedDataParallel._active_ddp_module = None
        return self._ddp._post_forward(output)


def unimplemented_deepcopy(*args: Any, **kwargs: Any) -> NoReturn:
    raise AssertionError(
        "DDP does not support deepcopy. Please use state dict for serialization."
    )


# Follow the same pattern as FSDP/fully_shard
class DDP:
    def __new__(cls, *args, **kwargs):
        """
        Override ``__new__`` to remove the DDP class and directly construct
        the original class for cases like indexing into a container module.
        """
        # Use index 2 since 0 is the dynamically constructed `DDP<...>` class
        # and index 1 is the `DDP` class itself
        orig_cls = cls.__mro__[2]
        return orig_cls.__new__(orig_cls, *args, **kwargs)

    def set_requires_gradient_sync(self, requires_gradient_sync: bool) -> None:
        """
        Sets if the module should sync gradients. This can be used to implement
        gradient accumulation without communication.

        Args:
            requires_gradient_sync (bool): Whether to reduce gradients for the
                module's parameters.
        """
        replicate.state(self)._no_sync = not requires_gradient_sync  # type: ignore[arg-type]

    def register_comm_hook(self, *args, **kwargs) -> None:
        replicate.state(self)._comm_hook_args.append((args, kwargs))  # type: ignore[arg-type]


@contract(state_cls=_ReplicateState)
def replicate(
    module: nn.Module,
    ignored_modules: Iterable[torch.nn.Module] | None = None,
    **kwargs,
) -> nn.Module:
    r"""Replicates a module

    Args:
        module (torch.nn.Module): module to replicate

    Example::
        >>> # xdoctest: +REQUIRES(module:torch._C._distributed_c10d)
        >>> module = nn.Linear(3, 3)
        >>> replicate(module)
    """
    torch._C._log_api_usage_once("torch.distributed.replicate")

    # TODO(fegin): using kwargs is not a good idea if we would like to make
    # replicate a formal API to replace DDP.
    if "device_id" in kwargs:
        if not isinstance(kwargs["device_id"], (int, torch.device)):
            raise RuntimeError(
                "Expected device_id to be int or torch.device, "
                f"but got {type(kwargs['device_id'])}"
            )

    if _is_fully_sharded(module):
        raise RuntimeError(
            "Cannot apply `replicate()` on a Module already managed by `fully_shard`"
        )

    if ignored_modules is None:
        ignored_modules = {}
    else:
        ignored_modules = set(ignored_modules)

    state = replicate.state(module)
    module.register_forward_pre_hook(state.forward_pre_hook, with_kwargs=True)
    device_mesh = kwargs.get("device_mesh")
    if device_mesh is not None:
        root_mesh = device_mesh._get_root_mesh()
        # if a root mesh is not the same as device_mesh,
        # meaning the device_mesh is sliced out from the root mesh.
        if root_mesh != device_mesh:
            # TODO: This is a temporary work around to enable DDP + TP.
            # We should do the logic in DDP so that the 2D implementation is
            # sound and the state_dict works out of the box.
            #
            # This won't conflict with what is done in DDP class as the module
            # replicate is going to pass is NOT the original module.
            from torch.distributed.tensor.parallel.ddp import (
                _localize_dtensor,
                _reconstruct_dtensor,
            )

            module.register_forward_pre_hook(_reconstruct_dtensor)
            module.register_forward_hook(_localize_dtensor)

    module.register_forward_hook(state.forward_post_hook)  # type: ignore[arg-type]

    state.record_init_args(module, ignored_modules, **kwargs)

    # Place DDP leftmost for highest priority in the method resolution order
    cls = module.__class__
    dct = {"__deepcopy__": unimplemented_deepcopy}
    new_cls = type(f"DDP{cls.__name__}", (DDP, cls), dct)
    module.__class__ = new_cls
    return module


def _is_fully_sharded(module: nn.Module) -> bool:
    r"""Check if module is marked with fully_shard."""
    registry = _get_registry(module)
    if registry is None:
        return False
    return "fully_shard" in registry
