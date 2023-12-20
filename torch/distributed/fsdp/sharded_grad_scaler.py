import logging
from typing import Dict, Iterable, List, Optional, overload, Sequence, Tuple, Union

import torch
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import _MultiDeviceReplicator, GradScaler
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
            self._allow_fp16_grad = True

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
        scaled_outputs = super().scale(outputs)
        # Here we ensure the return dtype is the same as the outputs dtype.
        # For the FSDP + Mixed Precision use case, the loss output is in the Mixed Precision
        # format (fp16, bf16) and so the scaled loss should be of the same dtype.
        if isinstance(scaled_outputs, torch.Tensor):
            return scaled_outputs.type(outputs.dtype)  # type: ignore[union-attr]
        iterable = map(lambda x, y: x.type(y.dtype), scaled_outputs, outputs)
        if isinstance(scaled_outputs, (list, tuple)):
            return type(scaled_outputs)(iterable)
        return iterable

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
                found_inf.fill_(1.0)
                break
            else:
                grad.data *= inv_scale.item()

    def _update_scale_(
        self,
        _scale: torch.Tensor,
        _growth_tracker: torch.Tensor,
        found_inf_combined: torch.Tensor,
    ):
        if _scale.device.type == "cpu":
            self._amp_update_scale_cpu_(found_inf_combined)
        else:
            super()._update_scale_(
                _scale,
                _growth_tracker,
                found_inf_combined,
            )

    def _unscale_grads_(
        self,
        optimizer: torch.optim.Optimizer,
        inv_scale: torch.Tensor,
        found_inf: torch.Tensor,
        allow_fp16: bool,
    ) -> Dict[torch.device, torch.Tensor]:
        per_device_found_inf_dict = super()._unscale_grads_(
            optimizer, inv_scale, found_inf, allow_fp16
        )
        # There exist contexts (e.g. w/ `use_orig_params=True`) wherein some
        # ranks may have no (non-zero sized) parameter shards, necessitating the
        # initialization of `per_device_found_inf._per_device_tensors` here
        if not per_device_found_inf_dict:
            assert self._scale is not None
            per_device_found_inf = _MultiDeviceReplicator(found_inf)
            per_device_found_inf.get(self._scale.device)
            return per_device_found_inf._per_device_tensors
        else:
            return per_device_found_inf_dict

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        super().unscale_(optimizer)

        # Synchronize the detected inf across the ranks
        optimizer_state = self._per_optimizer_states[id(optimizer)]
        future_handles = []

        for v in optimizer_state["found_inf_per_device"].values():
            if v.device.type == "cpu":
                v_on_cuda = v.cuda()
                dist.all_reduce(v_on_cuda, async_op=False, group=self.process_group)
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

    def update(self, new_scale: Optional[Union[float, torch.Tensor]] = None) -> None:
        super().update(new_scale)

    def _is_tensor_on_supported_device(self, tensor: torch.Tensor):
        return tensor.is_cuda or tensor.device.type in ("xla", "cpu")

    def _sparse_coalesce(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.type(torch.float32).coalesce().type(torch.float16)

    def _foreach_non_finite_check_and_unscale_(
        self,
        grads: List[torch.Tensor],
        found_inf: torch.Tensor,
        inv_scale: torch.Tensor,
    ):
        if found_inf.device.type == "cpu":
            self._foreach_non_finite_check_and_unscale_cpu_(
                grads,
                found_inf,
                inv_scale,
            )
        else:
            super()._foreach_non_finite_check_and_unscale_(
                grads,
                found_inf,
                inv_scale,
            )
