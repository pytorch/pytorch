import weakref
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch.distributed._composable_state import _State
from torch.nn.parallel import DistributedDataParallel

from .contract import _get_registry, contract

_ROOT_MODULE_PREFIX = ""


class _ReplicateState(_State):
    def __init__(self) -> None:
        super().__init__()
        self.module: nn.Module = nn.ParameterList()
        self.has_initialized: bool = False
        self._param_list: nn.ParameterList = nn.ParameterList()
        # TODO(@fegin): this variable is originally create for testing, we
        # should remove this if possible.
        self._param_names: List[str] = []

    def _collect_params(
        self,
        module: nn.Module,
        ignored_modules: Set[nn.Module],
        ignored_params: Set[nn.Parameter],
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

    def init(
        self,
        module: nn.Module,
        ignored_modules: Set[nn.Module],
        **kwargs,
    ) -> None:
        if _is_fully_sharded(module):
            raise RuntimeError(
                "Cannot apply `replicate()` on a Module already managed by `fully_shard`"
            )

        if self.has_initialized:
            return

        self.has_initialized = True
        self.module = module
        ignored_params = {p for m in ignored_modules for p in m.parameters()}
        self._collect_params(module, ignored_modules, ignored_params)
        module.register_forward_pre_hook(self.forward_pre_hook, with_kwargs=True)
        module.register_forward_hook(self.forward_post_hook)  # type: ignore[arg-type]

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

    def forward_pre_hook(
        self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        return self._ddp._pre_forward(*args, **kwargs)

    def forward_post_hook(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        return self._ddp._post_forward(output)


@contract(state_cls=_ReplicateState)
def replicate(
    module: nn.Module,
    ignored_modules: Optional[Iterable[torch.nn.Module]] = None,
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

    if ignored_modules is None:
        ignored_modules = {}
    else:
        ignored_modules = set(ignored_modules)
    replicate.state(module).init(module, ignored_modules, **kwargs)

    return module


def _is_fully_sharded(module: nn.Module) -> bool:
    r"""Check if module is marked with fully_shard."""
    registry = _get_registry(module)
    if registry is None:
        return False
    return "fully_shard" in registry
