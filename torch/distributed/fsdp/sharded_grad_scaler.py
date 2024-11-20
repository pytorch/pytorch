# mypy: allow-untyped-defs
import logging
from collections import abc, defaultdict
from typing import Any, Dict, Iterable, List, Optional, overload, Tuple, Union

import torch
import torch.distributed as dist
from torch.amp.grad_scaler import _MultiDeviceReplicator, GradScaler, OptState
from torch.distributed.distributed_c10d import ProcessGroup


logger = logging.getLogger(__name__)


def _refresh_per_optimizer_state() -> Dict[str, Any]:
    return {"stage": OptState.READY, "found_inf_per_device": {}}


def _is_supported_device(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda or tensor.device.type in (
        "xla",
        "cpu",
        "hpu",
        "mtia",
        "xpu",
        torch._C._get_privateuse1_backend_name(),
    )


class _GeneralMultiDeviceReplicator(_MultiDeviceReplicator):
    """
    Lazily serves tensor to request device. This class extends
    _MultiDeviceReplicator to allow support for "cpu" as a device.
    """

    def __init__(self, master_tensor: torch.Tensor) -> None:
        assert _is_supported_device(master_tensor)
        self.master = master_tensor
        self._per_device_tensors: Dict[torch.device, torch.Tensor] = {}


class ShardedGradScaler(GradScaler):
    """
    ShardedGradScaler helps perform gradient scaling in a shard aware manner. It extends
    functionality from GradScaler:
    * Supports Pytorch DDP and FSDP implementations
    * Support CPU offloaded tensors (as used in fully sharded data parallel[FSDP])
    * Supports the custom Mixed Precision loss dtype (fp16, bf16) that FSDP returns
    * Sync inf/nan for scaled gradient tensors on any torch.device (where tensors are placed) across
    nodes

    Example::

        # Creates a ShardedGradScaler once at the beginning of training.
        scaler = ShardedGradScaler()

        for epoch in epochs:
            for input, target in data:
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)

                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                scaler.scale(loss).backward()

                # scaler.step() first unscales gradients of the optimizer's params.
                # If gradients don't contain infs/NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()

    See :class:`GradScaler` for explanation of scaling/unscaling and more use cases.

    Args:
        init_scale (float, optional, default=2.**16):  Initial scale factor.
        growth_factor (float, optional, default=2.0):  Factor by which the scale is multiplied during
            :meth:`update` if no inf/NaN gradients occur for ``growth_interval`` consecutive iterations.
        backoff_factor (float, optional, default=0.5):  Factor by which the scale is multiplied during
            :meth:`update` if inf/NaN gradients occur in an iteration.
        growth_interval (int, optional, default=2000):  Number of consecutive iterations without inf/NaN gradients
            that must occur for the scale to be multiplied by ``growth_factor``.
        enabled (bool, optional):  If ``False``, disables gradient scaling. :meth:`step` simply
            invokes the underlying ``optimizer.step()``, and other methods become no-ops.
            Default: ``True``
        process_group (ProcessGroup, optional, default=torch.distributed.group.WORLD):
            process group for sharding
    """

    def __init__(
        self,
        device: str = "cuda",
        init_scale: float = 2.0**16,
        backoff_factor: float = 0.5,
        growth_factor: float = 2.0,
        growth_interval: int = 2000,
        enabled: bool = True,
        process_group: Optional[ProcessGroup] = dist.group.WORLD,
    ) -> None:
        super().__init__(
            device,
            init_scale=init_scale,
            backoff_factor=backoff_factor,
            growth_factor=growth_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )
        if self._enabled:
            self.process_group = process_group
            self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)

    @overload
    def scale(self, outputs: torch.Tensor) -> torch.Tensor:
        ...

    @overload
    def scale(self, outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        ...

    @overload
    def scale(self, outputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        ...

    @overload
    def scale(self, outputs: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]:
        ...

    def scale(
        self, outputs: Union[torch.Tensor, Iterable[torch.Tensor]]
    ) -> Union[torch.Tensor, Iterable[torch.Tensor]]:
        if not self._enabled:
            return outputs

        if isinstance(outputs, torch.Tensor):
            assert _is_supported_device(outputs)
            if self._scale is None:
                self._lazy_init_scale_growth_tracker(outputs.device)
            assert self._scale is not None
            scaled_output = outputs * self._scale.to(
                device=outputs.device, non_blocking=True
            )
            # Here we ensure the return dtype is the same as the outputs dtype.
            # For the FSDP + Mixed Precision use case, the loss output is in the Mixed Precision
            # format (fp16, bf16) and so the scaled loss should be of the same dtype.
            return scaled_output.type(outputs.dtype)

        stash: List[_GeneralMultiDeviceReplicator] = []

        def apply_scale(val: Union[torch.Tensor, Iterable[torch.Tensor]]):
            if isinstance(val, torch.Tensor):
                assert _is_supported_device(val)
                if len(stash) == 0:
                    if self._scale is None:
                        self._lazy_init_scale_growth_tracker(val.device)
                    assert self._scale is not None
                    stash.append(_GeneralMultiDeviceReplicator(self._scale))
                scaled_val = val * stash[0].get(val.device)
                # Here we ensure the return dtype is the same as the outputs dtype.
                # For the FSDP + Mixed Precision use case, the loss output is in the Mixed Precision
                # format (fp16, bf16) and so the scaled loss should be of the same dtype.
                return scaled_val.type(val.dtype)
            if isinstance(val, abc.Iterable):
                iterator = map(apply_scale, val)
                if isinstance(val, (list, tuple)):
                    return type(val)(iterator)
                return iterator
            raise ValueError("outputs must be a Tensor or an iterable of Tensors")

        return apply_scale(outputs)

    def _unscale_grads_(
        self,
        optimizer: torch.optim.Optimizer,
        inv_scale: torch.Tensor,
        found_inf: torch.Tensor,
        allow_fp16: bool = True,
    ) -> Dict[torch.device, torch.Tensor]:
        per_device_inv_scale = _GeneralMultiDeviceReplicator(inv_scale)
        per_device_found_inf = _GeneralMultiDeviceReplicator(found_inf)

        # To set up _amp_foreach_non_finite_check_and_unscale_, split grads by device and dtype.
        # There could be thousands of grads, so we'd like to iterate through them just once.
        # However, we don't know their devices or dtypes in advance.

        # https://stackoverflow.com/questions/5029934/defaultdict-of-defaultdict
        # Google says mypy struggles with defaultdicts type annotations.
        per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))  # type: ignore[var-annotated]
        with torch.no_grad():
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    if (not allow_fp16) and param.grad.dtype == torch.float16:
                        raise ValueError("Attempting to unscale FP16 gradients.")
                    if param.grad.is_sparse:
                        # is_coalesced() == False means the sparse grad has values with duplicate indices.
                        # coalesce() deduplicates indices and adds all values that have the same index.
                        # For scaled fp16 values, there's a good chance coalescing will cause overflow,
                        # so we should check the coalesced _values().
                        if param.grad.dtype is torch.float16:
                            # coalesce is not supported in torch.float16
                            param_grad_fp32 = param.grad.type(torch.float32).coalesce()
                            param.grad = param_grad_fp32.type(torch.float16)
                        to_unscale = param.grad._values()
                    else:
                        to_unscale = param.grad

                    per_device_and_dtype_grads[to_unscale.device][
                        to_unscale.dtype
                    ].append(to_unscale)

            for device, per_dtype_grads in per_device_and_dtype_grads.items():
                for grads in per_dtype_grads.values():
                    torch._amp_foreach_non_finite_check_and_unscale_(
                        grads,
                        per_device_found_inf.get(device),
                        per_device_inv_scale.get(device),
                    )
        # There exist contexts (e.g. w/ `use_orig_params=True`) wherein some
        # ranks may have no (non-zero sized) parameter shards, necessitating the
        # initialization of `per_device_found_inf._per_device_tensors` here
        if not per_device_found_inf._per_device_tensors:
            assert self._scale is not None
            per_device_found_inf.get(self._scale.device)
        return per_device_found_inf._per_device_tensors

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        if not self._enabled:
            return

        self._check_scale_growth_tracker("unscale_")

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state["stage"] is OptState.UNSCALED:
            raise RuntimeError(
                "unscale_() has already been called on this optimizer since the last update()."
            )
        elif optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("unscale_() is being called after step().")

        # FP32 division can be imprecise for certain compile options, so we carry out the reciprocal in FP64.
        assert self._scale is not None
        inv_scale = self._scale.double().reciprocal().float()
        found_inf = torch.full(
            (1,), 0.0, dtype=torch.float32, device=self._scale.device
        )

        optimizer_state["found_inf_per_device"] = self._unscale_grads_(
            optimizer, inv_scale, found_inf, True
        )
        optimizer_state["stage"] = OptState.UNSCALED

        # Synchronize the detected inf across the ranks
        optimizer_state = self._per_optimizer_states[id(optimizer)]
        works = []
        found_inf_on_cpus = []
        found_inf_on_devices = []

        for found_inf in optimizer_state["found_inf_per_device"].values():
            if self._device != "cpu" and found_inf.device.type == "cpu":
                found_inf_on_cpus.append(found_inf)
                found_inf_on_device = found_inf.to(self._device)
                found_inf_on_devices.append(found_inf_on_device)
                works.append(
                    dist.all_reduce(
                        found_inf_on_device, async_op=True, group=self.process_group
                    )
                )
            else:
                works.append(
                    dist.all_reduce(found_inf, async_op=True, group=self.process_group)
                )
        for work in works:
            work.wait()
        if found_inf_on_cpus:
            torch._foreach_copy_(found_inf_on_cpus, found_inf_on_devices)

    def _amp_update_scale_cpu_(self, found_inf: torch.Tensor) -> None:
        """
        If found_inf is 1.0 (True), then scale is multiplied by backoff_factor and growth_tracker is set to zero.
        Otherwise, scale is multiplied by the growth factor when the growth interval is reached.
        """
        assert self._scale is not None and self._growth_tracker is not None

        if found_inf.item() >= 1.0:
            self._scale *= self._backoff_factor
            self._growth_tracker.fill_(0)
        else:
            successful = self._growth_tracker + 1
            if successful == self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker.fill_(0)
            else:
                self._growth_tracker = successful

    def update(self, new_scale: Optional[Union[float, torch.Tensor]] = None) -> None:
        """
        Updates the scale factor.
        If any optimizer steps were skipped the scale is multiplied by ``backoff_factor``
        to reduce it. If ``growth_interval`` unskipped iterations occurred consecutively,
        the scale is multiplied by ``growth_factor`` to increase it.
        Passing ``new_scale`` sets the new scale value manually. (``new_scale`` is not
        used directly, it's used to fill GradScaler's internal scale tensor. So if
        ``new_scale`` was a tensor, later in-place changes to that tensor will not further
        affect the scale GradScaler uses internally.)
        Args:
            new_scale (float or :class:`torch.Tensor`, optional, default=None):  New scale factor.
        .. warning::
            :meth:`update` should only be called at the end of the iteration, after ``scaler.step(optimizer)`` has
            been invoked for all optimizers used this iteration.
        """

        if not self._enabled:
            return

        _scale, _growth_tracker = self._check_scale_growth_tracker("update")  # type: ignore[var-annotated]

        if new_scale is not None:
            # Accept a new user-defined scale.
            if isinstance(new_scale, float):
                self._scale.fill_(new_scale)  # type: ignore[union-attr]
            else:
                reason = "new_scale should be a float or a 1-element torch.cuda.FloatTensor or \
                    torch.FloatTensor with requires_grad=False."
                assert new_scale.device.type == self._device, reason
                assert new_scale.numel() == 1, reason
                assert new_scale.requires_grad is False, reason
                self._scale.copy_(new_scale)  # type: ignore[union-attr]
        else:
            # Consume shared inf/nan data collected from optimizers to update the scale.
            # If all found_inf tensors are on the same device as self._scale, this operation is asynchronous.
            found_infs = [
                found_inf.to(device=_scale.device, non_blocking=True)
                for state in self._per_optimizer_states.values()
                for found_inf in state["found_inf_per_device"].values()
            ]

            assert len(found_infs) > 0, "No inf checks were recorded prior to update."

            found_inf_combined = found_infs[0]
            if len(found_infs) > 1:
                for i in range(1, len(found_infs)):
                    found_inf_combined += found_infs[i]

            if _scale.device.type == "cpu":
                self._amp_update_scale_cpu_(found_inf_combined)
            else:
                torch._amp_update_scale_(
                    self._scale,  # type: ignore[arg-type]
                    self._growth_tracker,  # type: ignore[arg-type]
                    found_inf_combined,
                    self._growth_factor,  # type: ignore[arg-type]
                    self._backoff_factor,  # type: ignore[arg-type]
                    self._growth_interval,  # type: ignore[arg-type]
                )

        # To prepare for next iteration, clear the data collected from optimizers this iteration.
        self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)
