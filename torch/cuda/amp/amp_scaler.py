import torch
from collections import defaultdict
from torch._six import container_abcs


class _MultiDeviceReplicator(object):
    """
    Lazily serves copies of a tensor to requested devices.  Copies are cached per-device.
    """
    def __init__(self, master_tensor):
        assert master_tensor.is_cuda
        self.master = master_tensor
        self._per_device_tensors = {}

    def get(self, device):
        retval = self._per_device_tensors.get(device, None)
        if retval is None:
            retval = self.master.to(device=device, non_blocking=True, copy=True)
            self._per_device_tensors[device] = retval
        return retval


class AmpScaler(object):
    """
    :class:`AmpScaler` performs dynamic gradient scaling.

    Here's how that looks in a simple example::

        # Create an AmpScaler instance.
        scaler = AmpScaler()
        ...
        for input, target in data:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)

            # Scale the loss, and call backward() on the scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            # Carry out a scaling-safe step.  scaler.step() unscales the optimizer's gradients
            # and skips optimizer.step() if the gradients contain infs or NaNs.
            scaler.step(optimizer)

            # Update the scale for next iteration.
            scaler.update()

    See the :ref:`Gradient Scaling Examples<gradient-scaling-examples>` for usage in more complex cases.

    ``scaler`` maintains the scale factor internally.  To leverage ``float16``'s full dynamic range,
    ``scaler`` attempts to use the largest scale factor it can without incurring overflow.
    It does so by checking the gradients for infs and NaNs during every :meth:`step` or separate :meth:`unscale`.
    If no infs/NaNs are found, :meth:`step` runs the underlying ``optimizer.step()`` as usual and
    :meth:`update` multiplies the scale factor by the growth factor.  If infs/NaNs are found, :meth:`step` skips the
    underlying ``optimizer.step()`` (so the params themselves remain uncorrupted) and multiplies the scale factor by
    the backoff factor.

    The scale factor often causes infs/NaNs to appear in gradients for the first few iterations as its
    value calibrates.  ``scaler.step`` will skip the underlying ``optimizer.step()`` for these
    iterations.  After that, step skipping should occur rarely (once every few hundred iterations).

    Arguments:
        init_scale (float, optional, default=2.**24):  Initial scale factor.
        growth_factor (float, optional, default=1.001):  Factor by which the scale is multiplied during
            :meth:`update` if no inf/NaN gradients were found this iteration.
        backoff_factor (float, optional, default=0.5):  Factor by which the scale is multiplied during
            :meth:`update` if inf/NaN gradients were found this iteration.
        enabled (bool,optional, default=True):  If ``False``, disables gradient scaling. :meth:`step` simply
            invokes the underlying ``optimizer.step()``, and other methods become no-ops.
    """
    def __init__(self,
                 init_scale=2.**24,
                 growth_factor=1.001,
                 backoff_factor=0.5,
                 enabled=True):
        self._enabled = enabled
        if enabled:
            self._init_scale = init_scale
            # self._scale will be lazily initialized during the first call to scaler.scale(loss or outputs)
            self._scale = None
            self._growth_factor = growth_factor
            self._backoff_factor = backoff_factor
            self._per_optimizer_states = defaultdict(dict)

    @staticmethod
    def _scale_not_initialized_error(funcname):
        return "Attempted to call {} but the scale tensor is None. This may indicate your ".format(funcname) + \
               "script did not use scaler.scale(loss or outputs) earlier in the iteration."

    def scale(self, outputs):
        """
        Multiplies ('scales') a tensor or list of tensors by the scale factor.

        These tensors are typically the outputs of a network. When ``backward()``
        is called on scaled outputs the resulting gradients will also be scaled.

        Arguments:
            outputs (Tensor or iterable of Tensors):  Outputs to scale.

        Returns:
            Scaled outputs.  If this instance of :class:`AmpScaler` is not enabled, outputs are returned unmodified.
        """
        if not self._enabled:
            return outputs

        # Short-circuit for the common case.
        if isinstance(outputs, torch.Tensor):
            assert outputs.is_cuda
            if self._scale is None:
                self._scale = torch.full((1,), self._init_scale, dtype=torch.float32, device=outputs.device)
            return outputs * self._scale.to(device=outputs.device, non_blocking=True)

        # Invoke the more complex machinery only if we're treating multiple outputs.
        stash = [None]  # trick to hold a reference that can be overwritten at any level of the recursion below.

        def apply_scale(val):
            if isinstance(val, torch.Tensor):
                assert val.is_cuda
                if self._scale is None:
                    self._scale = torch.full((1,), self._init_scale, dtype=torch.float32, device=val.device)
                if stash[0] is None:
                    stash[0] = _MultiDeviceReplicator(self._scale)
                return val * stash[0].get(val.device)
            elif isinstance(val, container_abcs.Iterable):
                return type(val)(apply_scale(v) for v in val)
            else:
                raise ValueError("outputs must be a Tensor or an iterable of Tensors")

        return apply_scale(outputs)

    def _unscale_grads(self, optimizer, inv_scale, found_inf, allow_fp16=False):
        per_device_inv_scale = _MultiDeviceReplicator(inv_scale)
        per_device_found_inf = _MultiDeviceReplicator(found_inf)

        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    if (not allow_fp16) and param.grad.dtype == torch.float16:
                        raise ValueError("Attempting to unscale FP16 gradients.  If you want to check for "
                                         "infs/nans without unscaling, use optimizer.check_infs() instead.")
                    else:
                        torch._amp_non_finite_check_and_unscale_(param.grad,
                                                                 per_device_found_inf.get(param.grad.device),
                                                                 per_device_inv_scale.get(param.grad.device))

        return per_device_found_inf._per_device_tensors

    def unscale(self, optimizer):
        """
        Divides ('unscales') the optimizer's gradient tensors by the scale factor.

        If :meth:`unscale` is not called explicitly then gradients will be
        automatically unscaled during :meth:`step`.

        :meth:`unscale` can be called explicitly to manipulate a network's
        gradients after backward and before :meth:`step`.

        Simple example, using :meth:`unscale` to enable clipping of unscaled gradients::

            ...
            scaler.scale(loss).backward()
            scaler.unscale(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()

        If this instance of :class:`AmpScaler` is not enabled, :meth:`unscale` is a no-op.

        Arguments:
            optimizer (torch.optim.Optimizer):  Optimizer that owns the gradients to be unscaled.

        .. note::
            :meth:`unscale` does not incur a CPU-GPU sync.

        .. warning::
            :meth:`unscale` should only be called once per optimizer per step,
            and only after all gradients for that optimizer's owned parameters
            have been accumulated.
        """
        if not self._enabled:
            return

        assert self._scale is not None, self._scale_not_initialized_error("unscale")

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if "unscaled" in optimizer_state:
            raise RuntimeError("unscale() has already been called on this optimizer this iteration.")

        # FP32 division can be imprecise for certain compile options, so we carry out the reciprocal in FP64.
        inv_scale = self._scale.double().reciprocal().float()
        found_inf = torch.full((1,), 0.0, dtype=torch.float32, device=self._scale.device)

        optimizer_state["found_inf_per_device"] = self._unscale_grads(optimizer, inv_scale, found_inf)
        optimizer_state["unscaled"] = True

    def step(self, optimizer, *args, **kwargs):
        """
        Carry out a scaling-safe ``optimizer.step()``.  "Scaling-safe" means two things:

        1.  , :meth:`step` invokes :meth:`unscale` for ``optimizer`` before calling ``optimizer.step()``
            (unless :meth:`unscale` was explicitly called for ``optimizer`` earlier in the iteration).
            This ensures ``optimizer.step()`` is carried out using unscaled gradients.
        2.  If inf/NaN gradients are found, :meth:`step` skips ``optimizer.step()`` to avoid polluting the params.

        ``*args`` and ``**kwargs`` are forwarded to ``optimizer.step()``.

        Arguments:
            optimizer (torch.optim.Optimizer):  Optimizer that applies the gradients.
            args:  Any arguments.
            kwargs:  Any keyword arguments.

        Returns:
            The return value of ``optimizer.step(*args, **kwargs)``.

        .. warning::
            Closure use is not currently supported.
        """
        if (not self._enabled):
            return optimizer.step(*args, **kwargs)

        if "closure" in kwargs:
            raise RuntimeError("Closure use is not currently supported if AmpScaler is enabled.")

        assert self._scale is not None, self._scale_not_initialized_error("step")

        if (hasattr(optimizer, "_step_supports_amp_scaling") and optimizer._step_supports_amp_scaling):
            # This optimizer has customized scaling-safe step logic, so we call it directly.
            # The contract with custom optimizers is that their step methods should accept an additional,
            # optional amp_scaler kwarg.  We append self to the kwargs so the custom optimizer has full information:
            # it can query its own state, invoke unscale on its own gradients for convenience, etc.
            return optimizer.step(*args, **dict(kwargs, amp_scaler=self))

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if "unscaled" not in optimizer_state:
            self.unscale(optimizer)

        assert len(optimizer_state["found_inf_per_device"]) > 0, "No inf checks were recorded for this optimizer."

        if not sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
            return optimizer.step(*args, **kwargs)

    def update(self, new_scale=None):
        """
        Updates the scale factor.

        If any optimizer steps were skipped the scale factor is multipled by
        backoff_factor to reduce it. If all optimizer steps were taken
        it is multiplied by growth_factor to increase it.

        Passing ``new_scale`` sets the scale_factor directly.

        Arguments:
            new_scale (float or :class:`torch.cuda.FloatTensor`, optional, default=None):  New shared scale factor.

        .. warning::
            :meth:`update` should only be called at the end of the iteration, after ``scaler.step(optimizer)`` has
            been invoked for all optimizers used this iteration.
        """
        if not self._enabled:
            return

        assert self._scale is not None, self._scale_not_initialized_error("update")

        if new_scale is not None:
            # Accept a new user-defined scale.
            if isinstance(new_scale, float):
                self._scale = torch.full((1,), new_scale, dtype=torch.float32, device=self._scale.device)
            else:
                reason = "new_scale should be a float or a 1-element torch.cuda.FloatTensor with requires_grad=False."
                assert isinstance(new_scale, torch.cuda.FloatTensor), reason
                assert new_scale.numel() == 1, reason
                assert new_scale.requires_grad is False, reason
                self._scale = new_scale
        else:
            # Consume shared inf/nan data collected from optimizers to update the scale.
            # If all found_inf tensors are on the same device as self._scale, this operation is asynchronous.
            found_infs = [found_inf.to(device=self._scale.device, non_blocking=True)
                          for state in self._per_optimizer_states.values()
                          for found_inf in state["found_inf_per_device"].values()]

            assert len(found_infs) > 0, "No inf checks were recorded prior to update."

            found_inf_combined = found_infs[0]
            if len(found_infs) > 1:
                for i in range(1, len(found_infs)):
                    found_inf_combined += found_infs[i]

            self._scale = torch._amp_update_scale(self._scale,
                                                  found_inf_combined,
                                                  self._growth_factor,
                                                  self._backoff_factor)

        # To prepare for next iteration, clear the data collected from optimizers this iteration.
        self._per_optimizer_states = defaultdict(dict)

    def _get_scale_async(self):
        return self._scale

    def get_scale(self):
        """
        Returns:
            A Python float containing the current scale, or 1.0 if scaling is disabled.

        .. warning::
            :meth:`get_scale` incurs a CPU-GPU sync.
        """
        if self._enabled:
            return self._init_scale if self._scale is None else self._get_scale_async().item()
        else:
            return 1.0

    def get_growth_factor(self):
        r"""
        Returns:
            A Python float containing the scale growth factor.
        """
        return self._growth_factor

    def set_growth_factor(self, new_factor):
        r"""
        Arguments:
            new_scale (float):  Value to use as the new scale growth factor.
        """
        self._growth_factor = new_factor

    def get_backoff_factor(self):
        r"""
        Returns:
            A Python float containing the scale backoff factor.
        """
        return self._backoff_factor

    def set_backoff_factor(self, new_factor):
        r"""
        Arguments:
            new_scale (float):  Value to use as the new scale backoff factor.
        """
        self._backoff_factor = new_factor

    def is_enabled(self):
        r"""
        Returns:
            A bool indicating whether this instance is enabled.
        """
        return self._enabled

    def state_dict(self):
        r"""
        Returns the state of the scaler as a :class:`dict`.  It contains three entries:

        * ``"scale"`` - a Python float containing the current scale
        * ``"growth_factor"`` - a Python float containing the current growth factor
        * ``"backoff_factor"`` - a Python float containing the current backoff factor

        If this instance is not enabled, returns an empty dict.
        """
        return {"scale": self.get_scale(),
                "growth_factor": self._growth_factor,
                "backoff_factor": self._backoff_factor} if self._enabled else {}

    def load_state_dict(self, state_dict):
        r"""
        Loads the scaler state.  If this instance is disabled, :meth:`load_state_dict` is a no-op.

        Arguments:
           state_dict(dict): scaler state.  Should be an object returned from a call to :meth:`state_dict`.
        """
        if not self._enabled:
            return

        if len(state_dict) == 0:
            raise RuntimeError("The source state dict is empty, possibly because it was saved "
                               "from a disabled instance of AmpScaler.")

        self._init_scale = state_dict["scale"]
        if self._scale is not None:
            self._scale.fill_(state_dict["scale"])
        self._growth_factor = state_dict["growth_factor"]
        self._backoff_factor = state_dict["backoff_factor"]

    def _check_inf_per_device(self, optimizer):
        assert self._scale is not None, self._scale_not_initialized_error("_check_inf_per_device")

        dummy_inv_scale = torch.full((1,), 1.0, dtype=torch.float32, device=self._scale.device)
        found_inf = torch.full((1,), 0.0, dtype=torch.float32, device=self._scale.device)

        self._per_optimizer_states[id(optimizer)]["found_inf_per_device"] = \
            self._unscale_grads(optimizer, dummy_inv_scale, found_inf, allow_fp16=True)

        return self._per_optimizer_states[id(optimizer)]["found_inf_per_device"]

    def _found_inf_per_device(self, optimizer):
        return self._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
