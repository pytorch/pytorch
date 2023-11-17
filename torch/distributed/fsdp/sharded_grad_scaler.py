import logging
from typing import Dict, Optional, Sequence

import torch
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.distributed.distributed_c10d import ProcessGroup

log = logging.getLogger(__name__)


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
        init_scale: float = 2.0**16,
        backoff_factor: float = 0.5,
        growth_factor: float = 2.0,
        growth_interval: int = 2000,
        enabled: bool = True,
        process_group: Optional[ProcessGroup] = dist.group.WORLD,
    ) -> None:
        super().__init__(
            init_scale=init_scale,
            backoff_factor=backoff_factor,
            growth_factor=growth_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )
        if self._enabled:
            self.process_group = process_group

    def _is_tensor_on_supported_device(self, tensor: torch.Tensor):
        return tensor.is_cuda or tensor.device.type in ("xla", "cpu")

    def _maybe_convert_dtype(
        self, tensor: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        return tensor.type(dtype)

    def _foreach_non_finite_check_and_unscale_cpu_(
        self,
        grads: Sequence[torch.Tensor],
        found_inf: torch.Tensor,
        inv_scale: torch.Tensor,
    ) -> None:
        if len(grads) == 0:
            return
        assert inv_scale.numel() == 1, "inv_scale must be a 1-element tensor."
        assert found_inf.numel() == 1, "found_inf must be a 1-element tensor."

        for grad in grads:
            if grad.device.type != "cpu":
                log.error(
                    "tensor device is %s but was expected to be ``cpu``",
                    grad.device,
                )
                raise ValueError(
                    "Gradients were found on a non-CPU device when"
                    " expected to be on CPU."
                )
            if (
                torch.isinf(grad).any().item() is True
                or torch.isnan(grad).any().item() is True
            ):
                found_inf.data = torch.tensor([1.0])
                break
            else:
                grad.data *= inv_scale.item()

    def _sparse_coalesce(self, tensor: torch.Tensor):
        return tensor.type(torch.float32).coalesce().type(torch.float16)

    def _foreach_non_finite_check_and_unscale_by_device(
        self,
        grads,
        per_device_found_inf,
        per_device_inv_scale,
        device,
        device_type="gpu",
    ):
        if device_type == "cpu":
            self._foreach_non_finite_check_and_unscale_cpu_(
                grads,
                per_device_found_inf.get(device),
                per_device_inv_scale.get(device),
            )
        else:
            torch._amp_foreach_non_finite_check_and_unscale_(
                grads,
                per_device_found_inf.get(device),
                per_device_inv_scale.get(device),
            )

    def _maybe_init_per_device_tensors(self, per_device_found_inf):
        if not per_device_found_inf._per_device_tensors:
            assert self._scale is not None
            per_device_found_inf.get(self._scale.device)
        return per_device_found_inf._per_device_tensors

    def _init_found_inf(self, device):
        return torch.full((1,), 0.0, dtype=torch.float32, device=device)

    def _update_scale_by_device(
        self, _scale, _growth_tracker, found_inf_combined, device="gpu"
    ):
        if device == "cpu":
            self._amp_update_scale_cpu_(found_inf_combined)
        else:
            torch._amp_update_scale_(
                _scale,  # type: ignore[arg-type]
                _growth_tracker,  # type: ignore[arg-type]
                found_inf_combined,
                self._growth_factor,  # type: ignore[arg-type]
                self._backoff_factor,  # type: ignore[arg-type]
                self._growth_interval,  # type: ignore[arg-type]
            )

    def _unscale_grads_(
        self,
        optimizer: torch.optim.Optimizer,
        inv_scale: torch.Tensor,
        found_inf: torch.Tensor,
        allow_fp16: bool = True,
    ) -> Dict[torch.device, torch.Tensor]:
        return super()._unscale_grads_(optimizer, inv_scale, found_inf, allow_fp16)

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        super().unscale_(optimizer)

        # Synchronize the detected inf across the ranks
        optimizer_state = self._per_optimizer_states[id(optimizer)]
        future_handles = []

        for v in optimizer_state["found_inf_per_device"].values():
            if v.device.type == "cpu":
                v_on_cuda = v.cuda()
                future_handles.append(
                    dist.all_reduce(
                        v_on_cuda, async_op=True, group=self.process_group
                    ).get_future()
                )
                v.copy_(v_on_cuda.cpu())
            else:
                future_handles.append(
                    dist.all_reduce(
                        v, async_op=True, group=self.process_group
                    ).get_future()
                )

        # Make sure that the calls are done before moving out.
        if future_handles:
            torch.futures.wait_all(future_handles)

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
