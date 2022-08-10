import warnings
from collections import abc
from typing import List, Tuple

import torch
from .grad_scaler import _MultiDeviceReplicator
from .common import amp_definitely_not_available


class OverlappedGradScaler:
    """
    An instance ``scaler`` of :class:`OverlappedGradScaler` helps perform the steps of gradient scaling
    conveniently with :class:`OverlappedOverlappedOptimizer`.

    * ``scaler.scale(loss)`` multiplies a given loss by ``scaler``'s current scale factor.
    * ``scaler.update()`` updates ``scaler``'s scale factor.

    Example::

        # Creates a OverlappedGradScaler and OverlappedOptimizer once at the beginning of training.
        scaler = OverlappedGradScaler()
        optimizer = OverlappedSGD(grad_scaler=scaler)

        # Creats DDP model and register the OverlappedOptimizer
        model = torch.nn.parallel.DistributedDataParallel(model)
        model._register_overlapped_optim(optimizer)

        for epoch in epochs:
            for input, target in data:
                output = model(input)
                loss = loss_fn(output, target)

                # Scales loss. Calls backward() on scaled loss to create scaled gradients.
                scaler.scale(loss).backward()

                # If gradients don't contain infs/NaNs, optimizer.step_param() is then called 
                # once gradients in a DDP bucket has been reduced,
                # otherwise, optimizer.step_param() is skipped.

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
    iteration.

    * If infs/NaNs are found,  the underlying ``optimizer.step_param()`` will be skipped for the current bucket 
      (so the params in this bucket will not be updated but the reset buckets may be updated if no infs/NaNs are found) 
      and ``update()`` multiplies the scale by ``backoff_factor``.

    * If no infs/NaNs are found, the underlying ``optimizer.step_param()`` will be applied as usual.
      If ``growth_interval`` unskipped iterations occur consecutively, ``update()`` multiplies the scale by
      ``growth_factor``.

    The scale factor often causes infs/NaNs to appear in gradients for the first few iterations as its
    value calibrates. The underlying ``optimizer.step_param()`` will be skipped for these iterations.
    After that, step skipping should occur rarely (once every few hundred or thousand iterations).

    Args:
        init_scale (float, optional, default=2.**16):  Initial scale factor.
        growth_factor (float, optional, default=2.0):  Factor by which the scale is multiplied during
            :meth:`update` if no inf/NaN gradients occur for ``growth_interval`` consecutive iterations.
        backoff_factor (float, optional, default=0.5):  Factor by which the scale is multiplied during
            :meth:`update` if inf/NaN gradients occur in an iteration.
        growth_interval (int, optional, default=2000):  Number of consecutive iterations without inf/NaN gradients
            that must occur for the scale to be multiplied by ``growth_factor``.
        enabled (bool, optional, default=True):  If ``False``, disables gradient scaling. :meth:`step` simply
            invokes the underlying ``optimizer.step()``, and other methods become no-ops.
    """
    # TODO: Find a way to register update() to torch.distributed.Reducer's finalize_backward()
    def __init__(self, 
                 init_scale=2 ** 16, 
                 growth_factor=2, 
                 backoff_factor=0.5, 
                 growth_interval=2000, 
                 enabled=True):
        if enabled and amp_definitely_not_available():
            warnings.warn("torch.cuda.amp.OverlappedGradScaler is enabled, but CUDA is not available.  Disabling.")
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

            # Records whether found infs/NaNs or not within a step
            self._found_inf_within_step = False

    def _check_scale_growth_tracker(self, funcname) -> Tuple[torch.Tensor, torch.Tensor]:
        fix = "This may indicate your script did not use scaler.scale(loss or outputs) earlier in the iteration."
        assert self._scale is not None, "Attempted {} but _scale is None.  ".format(funcname) + fix
        assert self._growth_tracker is not None, "Attempted {} but _growth_tracker is None.  ".format(funcname) + fix
        return (self._scale, self._growth_tracker)

    def _lazy_init_scale_growth_tracker(self, dev):
        assert self._growth_tracker is None, "_growth_tracker initialized before _scale"
        self._scale = torch.full((1,), self._init_scale, dtype=torch.float32, device=dev)
        self._growth_tracker = torch.full((1,), self._init_growth_tracker, dtype=torch.int32, device=dev)

    def scale(self, outputs):
        """
        Multiplies ('scales') a tensor or list of tensors by the scale factor.

        Returns scaled outputs.  If this instance of :class:`OverlappedGradScaler` is not enabled, outputs are returned
        unmodified.

        Args:
            outputs (Tensor or iterable of Tensors):  Outputs to scale.
        """
        if not self._enabled:
            return outputs

        # Short-circuit for the common case.
        if isinstance(outputs, torch.Tensor):
            assert outputs.is_cuda or outputs.device.type == 'xla'
            if self._scale is None:
                self._lazy_init_scale_growth_tracker(outputs.device)
            assert self._scale is not None
            return outputs * self._scale.to(device=outputs.device, non_blocking=True)

        # Invoke the more complex machinery only if we're treating multiple outputs.
        stash: List[_MultiDeviceReplicator] = []  # holds a reference that can be overwritten by apply_scale

        def apply_scale(val):
            if isinstance(val, torch.Tensor):
                assert val.is_cuda or val.device.type == 'xla'
                if len(stash) == 0:
                    if self._scale is None:
                        self._lazy_init_scale_growth_tracker(val.device)
                    assert self._scale is not None
                    stash.append(_MultiDeviceReplicator(self._scale))
                return val * stash[0].get(val.device)
            elif isinstance(val, abc.Iterable):
                iterable = map(apply_scale, val)
                if isinstance(val, list) or isinstance(val, tuple):
                    return type(val)(iterable)
                else:
                    return iterable
            else:
                raise ValueError("outputs must be a Tensor or an iterable of Tensors")

        return apply_scale(outputs)
        
    def unscale_grad(self, grad) -> bool:
        """
        Unscale the grad.
        Return True if no infs/NaNs founded.
        """
        if not self._enabled:
            return True

        fix = "This may indicate your script did not use scaler.scale(loss or outputs) earlier in the iteration."
        assert self._scale is not None, "Attempted unscale_grad but _scale is None. " + fix
        inv_scale = self._scale.double().reciprocal().float()
        found_inf = torch.full((1,), 0.0, dtype=torch.float32, device=grad.device)
        with torch.no_grad():
            if grad is None:
                return
            if grad.dtype == torch.float16:
                raise ValueError("Attempting to unscale FP16 gradients.")
            if grad.is_sparse:
                # is_coalesced() == False means the sparse grad has values with duplicate indices.
                # coalesce() deduplicates indices and adds all values that have the same index.
                # For scaled fp16 values, there's a good chance coalescing will cause overflow,
                # so we should check the coalesced _values().
                if grad.dtype is torch.float16:
                    grad = grad.coalesce()
                to_unscale = grad._values()
            else:
                # Compatible with _amp_foreach_non_finite_check_and_unscale_
                to_unscale = [grad]
            torch._amp_foreach_non_finite_check_and_unscale_(to_unscale,
                                                             found_inf,
                                                             inv_scale)
                                                            
            found_inf = (found_inf.item() != 0.0)                                   
            self._found_inf_within_step = (found_inf or self._found_inf_within_step)

        return not found_inf

    def update(self, new_scale=None):
        """
        Updates the scale factor.

        If any optimizer steps were skipped the scale is multiplied by ``backoff_factor``
        to reduce it. If ``growth_interval`` unskipped iterations occurred consecutively,
        the scale is multiplied by ``growth_factor`` to increase it.

        Passing ``new_scale`` sets the new scale value manually. (``new_scale`` is not
        used directly, it's used to fill OverlappedGradScaler's internal scale tensor. So if
        ``new_scale`` was a tensor, later in-place changes to that tensor will not further
        affect the scale OverlappedGradScaler uses internally.)

        Args:
            new_scale (float or :class:`torch.cuda.FloatTensor`, optional, default=None):  New scale factor.

        .. warning::
            :meth:`update` should only be called at the end of the iteration, after ``scaler.step(optimizer)`` has
            been invoked for all optimizers used this iteration.
        """
        if not self._enabled:
            return

        # Reset _found_inf_within_step for a new step
        self._found_inf_within_step = False

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
            # Compatible with _amp_update_scale_
            if self._found_inf_within_step:
                found_inf = torch.full((1,), 1.0, dtype=torch.float32, device=self._scale.device)
            else:
                found_inf = torch.full((1,), 0.0, dtype=torch.float32, device=self._scale.device)

            torch._amp_update_scale_(_scale,
                                     _growth_tracker,
                                     found_inf,
                                     self._growth_factor,
                                     self._backoff_factor,
                                     self._growth_interval)

    def _get_scale_async(self):
        return self._scale

    def get_scale(self):
        """
        Returns a Python float containing the current scale, or 1.0 if scaling is disabled.

        .. warning::
            :meth:`get_scale` incurs a CPU-GPU sync.
        """
        if self._enabled:
            return self._init_scale if self._scale is None else self._get_scale_async().item()
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
            return self._init_growth_tracker if self._growth_tracker is None else self._growth_tracker.item()
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
        * ``"_found_inf_within_step"`` - a Python bool containing wheather found inf within current step.

        If this instance is not enabled, returns an empty dict.

        .. note::
           If you wish to checkpoint the scaler's state after a particular iteration, :meth:`state_dict`
           should be called after :meth:`update`.
        """
        return {"scale": self.get_scale(),
                "growth_factor": self._growth_factor,
                "backoff_factor": self._backoff_factor,
                "growth_interval": self._growth_interval,
                "_growth_tracker": self._get_growth_tracker(),
                "_found_inf_within_step": self._found_inf_within_step} if self._enabled else {}

    def load_state_dict(self, state_dict):
        r"""
        Loads the scaler state.  If this instance is disabled, :meth:`load_state_dict` is a no-op.

        Args:
           state_dict(dict): scaler state.  Should be an object returned from a call to :meth:`state_dict`.
        """
        if not self._enabled:
            return

        if len(state_dict) == 0:
            raise RuntimeError("The source state dict is empty, possibly because it was saved "
                               "from a disabled instance of OverlappedGradScaler.")

        self._init_scale = state_dict["scale"]
        if self._scale is not None:
            self._scale.fill_(state_dict["scale"])
        self._growth_factor = state_dict["growth_factor"]
        self._backoff_factor = state_dict["backoff_factor"]
        self._growth_interval = state_dict["growth_interval"]
        self._init_growth_tracker = state_dict["_growth_tracker"]
        if self._growth_tracker is not None:
            self._growth_tracker.fill_(state_dict["_growth_tracker"])
        self._found_inf_within_step = state_dict["_found_inf_within_step"]

    def __getstate__(self):
        state = self.__dict__.copy()
        if self._enabled:
            # Pickling _scale and _growth_tracker Tensors directly triggers
            # "warnings.warn("pickle support for Storage will be removed in 1.5..."
            # so instead, we set the unpickled instance up to reinitialize them lazily.
            state['_init_scale'] = self.get_scale()
            state['_init_growth_tracker'] = self._get_growth_tracker()
            state['_scale'] = None
            state['_growth_tracker'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
