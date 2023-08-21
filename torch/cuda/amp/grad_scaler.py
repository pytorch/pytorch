import inspect
import warnings
from collections import abc, defaultdict
from enum import Enum
from typing import Any, cast, Dict, List, Optional, Tuple

import torch
from .common import amp_definitely_not_available


__all__ = ["OptState", "GradScaler"]


class _MultiDeviceReplicator:
    """
    Lazily serves copies of a tensor to requested devices.  Copies are cached per-device.
    """

    def __init__(self, master_tensor: torch.Tensor) -> None:
        assert master_tensor.is_cuda or master_tensor.device.type == "xla"
        self.master = master_tensor
        self._per_device_tensors: Dict[torch.device, torch.Tensor] = {}

    def get(self, device) -> torch.Tensor:
        retval = self._per_device_tensors.get(device, None)
        if retval is None:
            retval = self.master.to(device=device, non_blocking=True, copy=True)
            self._per_device_tensors[device] = retval
        return retval


# Defines default_factory for GradScaler's _per_optimizer_states defaultdict,
# as well as associated "enum" values.  Prefers defining these at top level because
# - Lambdas can't be pickled, so we don't want to supply a lambda as the factory.
# - Defining READY, UNSCALED, STEPPED and _refresh_per_optimizer_state within GradScaler
#   causes a circular reference, which we'd rather avoid.
class OptState(Enum):
    READY = 0
    UNSCALED = 1
    STEPPED = 2


def _refresh_per_optimizer_state():
    return {"stage": OptState.READY, "found_inf_per_device": {}}


class GradScaler:
    _scale: Optional[torch.Tensor]
    _grows_tracker: Optional[torch.Tensor]
    _per_optimizer_states: Dict[int, Dict[str, Any]]
    """
    An instance ``scaler`` of :class:`GradScaler` helps perform the steps of gradient scaling
    conveniently.

    * ``scaler.scale(loss)`` multiplies a given loss by ``scaler``'s current scale factor.
    * ``scaler.step(optimizer)`` safely unscales gradients and calls ``optimizer.step()``.
    * ``scaler.update()`` updates ``scaler``'s scale factor.

    Example::

        # Creates a GradScaler once at the beginning of training.
        scaler = GradScaler()

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

    See the :ref:`Automatic Mixed Precision examples<amp-examples>` for usage
    (along with autocasting) in more complex cases like gradient clipping, gradient accumulation, gradient penalty,
    and multiple losses/optimizers.

    ``scaler`` dynamically estimates the scale factor each iteration.  To minimize gradient underflow,
    a large scale factor should be used.  However, ``float16`` values can "overflow" (become inf or NaN) if
    the scale factor is too large.  Therefore, the optimal scale factor is the largest factor that can be used
    without incurring inf or NaN gradient values.
    ``scaler`` approximates the optimal scale factor over time by checking the gradients for infs and NaNs during every
    ``scaler.step(optimizer)`` (or optional separate ``scaler.unscale_(optimizer)``, see :meth:`unscale_`).

    * If infs/NaNs are found, ``scaler.step(optimizer)`` skips the underlying ``optimizer.step()`` (so the params
      themselves remain uncorrupted) and ``update()`` multiplies the scale by ``backoff_factor``.

    * If no infs/NaNs are found, ``scaler.step(optimizer)`` runs the underlying ``optimizer.step()`` as usual.
      If ``growth_interval`` unskipped iterations occur consecutively, ``update()`` multiplies the scale by
      ``growth_factor``.

    The scale factor often causes infs/NaNs to appear in gradients for the first few iterations as its
    value calibrates.  ``scaler.step`` will skip the underlying ``optimizer.step()`` for these
    iterations.  After that, step skipping should occur rarely (once every few hundred or thousand iterations).

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
    """

    def __init__(
        self,
        init_scale=2.0**16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=True,
    ):
        if enabled and amp_definitely_not_available():
            warnings.warn(
                "torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling.",
                stacklevel=TO_BE_DETERMINED,
            )
            self._enabled = False
        else:
            self._enabled = enabled

        if self._enabled:
            assert growth_factor > 1.0, "The growth factor must be > 1.0."
            assert backoff_factor < 1.0, "The backoff factor must be < 1.0."

            self._init_scale = init_scale
            # self._scale will be lazily initialized during the first call to scale()
            self._scale = None
            self._growth_factor = growth_factor
            self._backoff_factor = backoff_factor
            self._growth_interval = growth_interval
            self._init_growth_tracker = 0
            # self._growth_tracker will be lazily initialized during the first call to scale()
            self._growth_tracker = None
            self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)

    def _check_scale_growth_tracker(
        self, funcname
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fix = "This may indicate your script did not use scaler.scale(loss or outputs) earlier in the iteration."
        assert self._scale is not None, (
            f"Attempted {funcname} but _scale is None.  " + fix
        )
        assert self._growth_tracker is not None, (
            f"Attempted {funcname} but _growth_tracker is None.  " + fix
        )
        return (self._scale, self._growth_tracker)

    def _lazy_init_scale_growth_tracker(self, dev):
        assert self._growth_tracker is None, "_growth_tracker initialized before _scale"
        self._scale = torch.full((), self._init_scale, dtype=torch.float32, device=dev)
        self._growth_tracker = torch.full(
            (), self._init_growth_tracker, dtype=torch.int32, device=dev
        )

    def scale(self, outputs):
        """
        Multiplies ('scales') a tensor or list of tensors by the scale factor.

        Returns scaled outputs.  If this instance of :class:`GradScaler` is not enabled, outputs are returned
        unmodified.

        Args:
            outputs (Tensor or iterable of Tensors):  Outputs to scale.
        """
        if not self._enabled:
            return outputs

        # Short-circuit for the common case.
        if isinstance(outputs, torch.Tensor):
            assert outputs.is_cuda or outputs.device.type == "xla"
            if self._scale is None:
                self._lazy_init_scale_growth_tracker(outputs.device)
            assert self._scale is not None
            return outputs * self._scale.to(device=outputs.device, non_blocking=True)

        # Invoke the more complex machinery only if we're treating multiple outputs.
        stash: List[
            _MultiDeviceReplicator
        ] = []  # holds a reference that can be overwritten by apply_scale

        def apply_scale(val):
            if isinstance(val, torch.Tensor):
                assert val.is_cuda or val.device.type == "xla"
                if len(stash) == 0:
                    if self._scale is None:
                        self._lazy_init_scale_growth_tracker(val.device)
                    assert self._scale is not None
                    stash.append(_MultiDeviceReplicator(self._scale))
                return val * stash[0].get(val.device)
            elif isinstance(val, abc.Iterable):
                iterable = map(apply_scale, val)
                if isinstance(val, (list, tuple)):
                    return type(val)(iterable)
                else:
                    return iterable
            else:
                raise ValueError("outputs must be a Tensor or an iterable of Tensors")

        return apply_scale(outputs)

    def _unscale_grads_(self, optimizer, inv_scale, found_inf, allow_fp16):
        per_device_inv_scale = _MultiDeviceReplicator(inv_scale)
        per_device_found_inf = _MultiDeviceReplicator(found_inf)

        # To set up _amp_foreach_non_finite_check_and_unscale_, split grads by device and dtype.
        # There could be hundreds of grads, so we'd like to iterate through them just once.
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
                            param.grad = param.grad.coalesce()
                        to_unscale = param.grad._values()
                    else:
                        to_unscale = param.grad

                    # TODO: is there a way to split by device and dtype without appending in the inner loop?
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

        return per_device_found_inf._per_device_tensors

    def unscale_(self, optimizer):
        """
        Divides ("unscales") the optimizer's gradient tensors by the scale factor.

        :meth:`unscale_` is optional, serving cases where you need to
        :ref:`modify or inspect gradients<working-with-unscaled-gradients>`
        between the backward pass(es) and :meth:`step`.
        If :meth:`unscale_` is not called explicitly,  gradients will be unscaled  automatically during :meth:`step`.

        Simple example, using :meth:`unscale_` to enable clipping of unscaled gradients::

            ...
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()

        Args:
            optimizer (torch.optim.Optimizer):  Optimizer that owns the gradients to be unscaled.

        .. note::
            :meth:`unscale_` does not incur a CPU-GPU sync.

        .. warning::
            :meth:`unscale_` should only be called once per optimizer per :meth:`step` call,
            and only after all gradients for that optimizer's assigned parameters have been accumulated.
            Calling :meth:`unscale_` twice for a given optimizer between each :meth:`step` triggers a RuntimeError.

        .. warning::
            :meth:`unscale_` may unscale sparse gradients out of place, replacing the ``.grad`` attribute.
        """
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
        found_inf = torch.full((), 0.0, dtype=torch.float32, device=self._scale.device)

        optimizer_state["found_inf_per_device"] = self._unscale_grads_(
            optimizer, inv_scale, found_inf, False
        )
        optimizer_state["stage"] = OptState.UNSCALED

    def _maybe_opt_step(self, optimizer, optimizer_state, *args, **kwargs):
        retval = None
        if not sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
            retval = optimizer.step(*args, **kwargs)
        return retval

    def step(self, optimizer, *args, **kwargs):
        """
        :meth:`step` carries out the following two operations:

        1.  Internally invokes ``unscale_(optimizer)`` (unless :meth:`unscale_` was explicitly called for ``optimizer``
            earlier in the iteration).  As part of the :meth:`unscale_`, gradients are checked for infs/NaNs.
        2.  If no inf/NaN gradients are found, invokes ``optimizer.step()`` using the unscaled
            gradients.  Otherwise, ``optimizer.step()`` is skipped to avoid corrupting the params.

        ``*args`` and ``**kwargs`` are forwarded to ``optimizer.step()``.

        Returns the return value of ``optimizer.step(*args, **kwargs)``.

        Args:
            optimizer (torch.optim.Optimizer):  Optimizer that applies the gradients.
            args:  Any arguments.
            kwargs:  Any keyword arguments.

        .. warning::
            Closure use is not currently supported.
        """
        if not self._enabled:
            return optimizer.step(*args, **kwargs)

        if "closure" in kwargs:
            raise RuntimeError(
                "Closure use is not currently supported if GradScaler is enabled."
            )

        self._check_scale_growth_tracker("step")

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError(
                "step() has already been called since the last update()."
            )

        retval = None

        if (
            hasattr(optimizer, "_step_supports_amp_scaling")
            and optimizer._step_supports_amp_scaling
        ):
            # This optimizer has customized scale-handling logic, so we can call optimizer.step() directly.
            # The contract with custom optimizers is that their step() should accept an additional,
            # optional grad_scaler kwarg.  We append self to the kwargs so the custom optimizer has full information:
            # it can query its own state, invoke unscale_ on itself, etc
            # The contract above is being deprecated to avoid introducing `grad_scaler: GradScaler` argument
            # to `Optimizer.step`. The new behavior is going to add two Tensor attributes of `grad_scale`
            # and `found_inf` to the passed optimizer so that the optimizer can utilize those
            # to skip the parameter updates or unscale gradients before updating parameters in
            # the fused kernel, e.g. `FusedAdamMathFunctor`.
            # In this behavior, `GradScaler._check_inf_per_device` is called if `OptState.READY`,
            # while the method is expected to be called by users side, i.e. their optimizers.
            kwargs_ = kwargs
            has_grad_scaler_kwarg = (
                "grad_scaler" in inspect.signature(optimizer.step).parameters
            )
            if has_grad_scaler_kwarg:
                warnings.warn(
                    "GradScaler is going to stop passing itself as a keyword argument to the passed "
                    "optimizer. In the near future GradScaler registers `grad_scale: Tensor` and "
                    "`found_inf: Tensor` to the passed optimizer and let the optimizer use them directly.",
                    FutureWarning,
                    stacklevel=TO_BE_DETERMINED,
                )
                kwargs_.update({"grad_scaler": self})
            else:
                if optimizer_state["stage"] is OptState.READY:
                    self._check_inf_per_device(optimizer)
                scaler = self._get_scale_async()
                found_inf = cast(
                    torch.Tensor,
                    sum(
                        [
                            t.to(scaler.device, non_blocking=True)
                            for t in optimizer_state["found_inf_per_device"].values()
                        ]
                    ),
                )
                optimizer.grad_scale = (
                    None if optimizer_state["stage"] == OptState.UNSCALED else scaler
                )
                optimizer.found_inf = found_inf
            retval = optimizer.step(*args, **kwargs_)
            optimizer_state["stage"] = OptState.STEPPED
            if not has_grad_scaler_kwarg:
                del optimizer.grad_scale
                del optimizer.found_inf
            return retval

        if optimizer_state["stage"] is OptState.READY:
            self.unscale_(optimizer)

        assert (
            len(optimizer_state["found_inf_per_device"]) > 0
        ), "No inf checks were recorded for this optimizer."

        retval = self._maybe_opt_step(optimizer, optimizer_state, *args, **kwargs)

        optimizer_state["stage"] = OptState.STEPPED

        return retval

    def update(self, new_scale=None):
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
            new_scale (float or :class:`torch.cuda.FloatTensor`, optional, default=None):  New scale factor.

        .. warning::
            :meth:`update` should only be called at the end of the iteration, after ``scaler.step(optimizer)`` has
            been invoked for all optimizers used this iteration.

        .. warning::
            For performance reasons, we do not check the scale factor value to avoid synchronizations,
            so the scale factor is not guaranteed to be above 1. If the scale falls below 1 and/or
            you are seeing NaNs in your gradients or loss, something is likely wrong. For example,
            bf16-pretrained models are often incompatible with AMP/fp16 due to differing dynamic ranges.
        """
        if not self._enabled:
            return

        _scale, _growth_tracker = self._check_scale_growth_tracker("update")

        if new_scale is not None:
            # Accept a new user-defined scale.
            if isinstance(new_scale, float):
                self._scale.fill_(new_scale)  # type: ignore[union-attr]
            else:
                reason = "new_scale should be a float or a 1-element torch.cuda.FloatTensor with requires_grad=False."
                assert isinstance(new_scale, torch.cuda.FloatTensor), reason  # type: ignore[attr-defined]
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

            torch._amp_update_scale_(
                _scale,
                _growth_tracker,
                found_inf_combined,
                self._growth_factor,
                self._backoff_factor,
                self._growth_interval,
            )

        # To prepare for next iteration, clear the data collected from optimizers this iteration.
        self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)

    def _get_scale_async(self):
        return self._scale

    def get_scale(self):
        """
        Returns a Python float containing the current scale, or 1.0 if scaling is disabled.

        .. warning::
            :meth:`get_scale` incurs a CPU-GPU sync.
        """
        if self._enabled:
            return (
                self._init_scale
                if self._scale is None
                else self._get_scale_async().item()
            )
        else:
            return 1.0

    def get_growth_factor(self):
        r"""
        Returns a Python float containing the scale growth factor.
        """
        return self._growth_factor

    def set_growth_factor(self, new_factor):
        r"""
        Args:
            new_scale (float):  Value to use as the new scale growth factor.
        """
        self._growth_factor = new_factor

    def get_backoff_factor(self):
        r"""
        Returns a Python float containing the scale backoff factor.
        """
        return self._backoff_factor

    def set_backoff_factor(self, new_factor):
        r"""
        Args:
            new_scale (float):  Value to use as the new scale backoff factor.
        """
        self._backoff_factor = new_factor

    def get_growth_interval(self):
        r"""
        Returns a Python int containing the growth interval.
        """
        return self._growth_interval

    def set_growth_interval(self, new_interval):
        r"""
        Args:
            new_interval (int):  Value to use as the new growth interval.
        """
        self._growth_interval = new_interval

    def _get_growth_tracker(self):
        if self._enabled:
            return (
                self._init_growth_tracker
                if self._growth_tracker is None
                else self._growth_tracker.item()
            )
        else:
            return 0

    def is_enabled(self):
        r"""
        Returns a bool indicating whether this instance is enabled.
        """
        return self._enabled

    def state_dict(self):
        r"""
        Returns the state of the scaler as a :class:`dict`.  It contains five entries:

        * ``"scale"`` - a Python float containing the current scale
        * ``"growth_factor"`` - a Python float containing the current growth factor
        * ``"backoff_factor"`` - a Python float containing the current backoff factor
        * ``"growth_interval"`` - a Python int containing the current growth interval
        * ``"_growth_tracker"`` - a Python int containing the number of recent consecutive unskipped steps.

        If this instance is not enabled, returns an empty dict.

        .. note::
           If you wish to checkpoint the scaler's state after a particular iteration, :meth:`state_dict`
           should be called after :meth:`update`.
        """
        return (
            {
                "scale": self.get_scale(),
                "growth_factor": self._growth_factor,
                "backoff_factor": self._backoff_factor,
                "growth_interval": self._growth_interval,
                "_growth_tracker": self._get_growth_tracker(),
            }
            if self._enabled
            else {}
        )

    def load_state_dict(self, state_dict):
        r"""
        Loads the scaler state.  If this instance is disabled, :meth:`load_state_dict` is a no-op.

        Args:
           state_dict(dict): scaler state.  Should be an object returned from a call to :meth:`state_dict`.
        """
        if not self._enabled:
            return

        if len(state_dict) == 0:
            raise RuntimeError(
                "The source state dict is empty, possibly because it was saved "
                "from a disabled instance of GradScaler."
            )

        self._init_scale = state_dict["scale"]
        if self._scale is not None:
            self._scale.fill_(state_dict["scale"])
        self._growth_factor = state_dict["growth_factor"]
        self._backoff_factor = state_dict["backoff_factor"]
        self._growth_interval = state_dict["growth_interval"]
        self._init_growth_tracker = state_dict["_growth_tracker"]
        if self._growth_tracker is not None:
            self._growth_tracker.fill_(state_dict["_growth_tracker"])

    def __getstate__(self):
        state = self.__dict__.copy()
        if self._enabled:
            assert len(self._per_optimizer_states) == 0, (
                "A GradScaler instance may only be pickled at the beginning "
                "of an iteration, or at the end after scaler.update()."
            )
            # Pickling _scale and _growth_tracker Tensors directly triggers
            # "warnings.warn("pickle support for Storage will be removed in 1.5..."
            # so instead, we set the unpickled instance up to reinitialize them lazily.
            state["_init_scale"] = self.get_scale()
            state["_init_growth_tracker"] = self._get_growth_tracker()
            state["_scale"] = None
            state["_growth_tracker"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _check_inf_per_device(self, optimizer):
        _scale, _ = self._check_scale_growth_tracker("_check_inf_per_device")

        dummy_inv_scale = torch.full((), 1.0, dtype=torch.float32, device=_scale.device)
        found_inf = torch.full((), 0.0, dtype=torch.float32, device=_scale.device)

        self._per_optimizer_states[id(optimizer)][
            "found_inf_per_device"
        ] = self._unscale_grads_(optimizer, dummy_inv_scale, found_inf, True)

        return self._per_optimizer_states[id(optimizer)]["found_inf_per_device"]

    def _found_inf_per_device(self, optimizer):
        return self._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
