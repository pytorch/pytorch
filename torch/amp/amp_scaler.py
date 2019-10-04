import torch
from collections import defaultdict
from torch._six import container_abcs


class AmpScaler(object):
    """
    Performs dynamic gradient scaling, intended to be used when gradients
    have a dtype with limited dynamic range (e.g. if part of your network is ``torch.float16``).

    Here's how that looks in a simple example::

        amp_scaler = AmpScaler()

        for input, target in data:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()

    By default, ``amp_scaler.step`` internally unscales ``optimizer``'s gradients before applying them, so the
    learning rate and other hyperparameters don't need to change.  If you wish to unscale gradients manually
    prior to :meth:`step`, use :meth:`unscale`.

    ``amp_scaler`` maintains the scale factor internally.  To leverage the dtype's full dynamic range,
    ``amp_scaler`` attempts to use the largest scale factor it can without incurring overflow.
    It does so by checking the gradients for infs and NaNs during every :meth:`step` or separate :meth:`unscale`.
    If no infs/NaNs are found, :meth:`step` runs the underlying ``optimizer.step()`` as usual and
    :meth:`update` increases the scale factor slightly.  If infs/NaNs are found, :meth:`step` skips the underlying
    ``optimizer.step()`` (so the params themselves remain unpolluted) and decreases the scale factor.

    The first several ``optimizer.step()``\ s may be skipped as the scaler calibrates.  After that, step skipping
    should occur rarely (once every few hundred iterations).
    """
    def __init__(self,
                 init_scale=2.**24,
                 growth_factor=1.001,
                 backoff_factor=0.5,
                 enabled=True):
        """
        Arguments:
            init_scale (float, optional, default=2.**24):  Initial scale factor.
            growth_factor (float, optional, default=1.001):  Factor by which the scale is multiplied during
                :meth:`update` if no inf/NaN gradients were found this iteration.
            backoff_factor (float, optional, default=0.5):  Factor by which the scale is multiplied during
                :meth:`update` if inf/NaN gradients were found this iteration.
            enabled (bool,optional, default=True):  If ``False``, disables gradient scaling. :meth:`step` simply
                invokes the underlying ``optimizer.step()``, and other methods become no-ops.
        """
        self._enabled = enabled
        if enabled:
            self._scale = torch.full((1,), init_scale, dtype=torch.float32, device="cuda")
            self._growth_factor = 1.001
            self._backoff_factor = 0.5
            self._per_optimizer_states = defaultdict(dict)

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

        def apply_scale(val):
            if isinstance(val, torch.Tensor):
                assert val.is_cuda
                return val * self._scale
            elif isinstance(val, container_abcs.Iterable):
                return type(val)(apply_scale(v) for v in val)
            else:
                raise ValueError("outputs must be a Tensor or an iterable of Tensors")

        return apply_scale(outputs)

    def _unscale_grads(optimizer, rscale, found_inf, allow_fp16=False):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    if (not allow_fp16) and param.grad.dtype == torch.float16:
                        raise ValueError("Attempting to unscale FP16 gradients.  If you want to check for "
                                         "infs/nans without unscaling, use optimizer.check_infs() instead.")
                    else:
                        torch._amp_unscale_inf_check_(param.grad, rscale, found_inf)

    def unscale(self, optimizer):
        """
        Divides ('unscales') the optimizer's gradient tensors by the scale factor.

        If :meth:`unscale` is not called explicitly then gradients will be
        automatically unscaled during :meth:`step`.

        :meth:`unscale` can be called explicitly to manipulate a network's
        gradients after backward and before :meth:`step`.

        Simple example, using :meth:`unscale` to enable clipping of unscaled gradients::

            ...
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            amp_scaler.step(optimizer)
            amp_scaler.update()

        If this instance of :class:`AmpScaler` is not enabled, :meth:`unscale` is a no-op.

        Arguments:
            optimizer (torch.optim.Optimizer):  Optimizer that owns the gradients to be unscaled.

        .. note::
            :meth:`unscale` does not incur a CPU-GPU sync.

        .. warning::
            :meth:`unscale` should only be called once per optimizer per step,
            and only after _all_ gradients for that optimizer's owned parameters
            have been accumulated.
        """
        if not self._enabled:
            return

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if "unscaled" in optimizer_state:
            raise RuntimeError("unscale() has already been called on this optimizer this iteration.")

        # FP32 division can be imprecise for certain compile options, so we carry out the reciprocal in FP64.
        rscale = self.amp_state.scale.double().reciprocal().float()
        found_inf = torch.full((1,), 0.0, dtype=torch.float32, device="cuda")

        self._unscale_grads(optimizer, rscale, found_inf, allow_fp16=False)

        optimizer_state["found_inf"] = found_inf
        optimizer_state["unscaled"] = True

    def step(self, optimizer, *args, **kwargs):
        """
        Carry out a scaling-safe step using ``optimizer``.  "Scaling-safe" means two things::

            1.  If :meth:`unscale` has not yet been invoked, :meth:`step` will make sure to invoke :meth:`unscale`
                internally.
            2.  If inf/NaN gradients are found, :meth:`step` will skip ``optimizer.step()`` to avoid polluting the
                params.

        ``*args`` and ``**kwargs`` will be forwarded to ``optimizer.step()``.

        Arguments:
            optimizer (torch.optim.Optimizer):  Optimizer that applies the gradients.
            args:  Any arguments.
            kwargs:  Any keyword arguments.

        Returns:
            The return value of ``optimizer.step(*args, **kwargs)``.

        note::
            If you're writing a custom optimizer, and wish to define your own scaling-safe ``step`` method that
            ``AmpScaler.step`` may call directly without any wrapping logic, see the :ref:`Custom Optimizer Guide`.

        warning::
            Closure use is not currently supported.
        """
        if (not self._enabled):
            return optimizer.step(*args, **kwargs)

        if "closure" in kwargs:
            raise RuntimeError("Closure use is not currently supported if AmpScaler is enabled.")

        if (hasattr(optimizer, "step_supports_amp_scaling") and optimizer.step_supports_amp_scaling):
            # This optimizer has customized scaling-safe step logic, so we call it directly.
            # The contract with custom optimizers is that their step methods should accept an additional,
            # optional amp_scaler kwarg.  We append self to the kwargs so the custom optimizer has full information:
            # it can query its own state, invoke unscale on its own gradients for convenience, etc.
            return optimizer.step(*args, **dict(kwargs, amp_scaler=self))

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if "unscaled" not in optimizer_state:
            self.unscale(optimizer)

        if not optimizer_state.found_inf().item():
            return optimizer.step(*args, **kwargs)

    def update(self, new_scale=None):
        """
        Updates the scale factor.

        If any optimizer steps were skipped the scale factor is multipled by
        backoff_factor to reduce it. If all optimizer steps were taken
        it is multiplied by growth_factor to increase it.

        Passing ``new_scale`` sets the scale_factor directly.

        Arguments:
            new_scale (float or torch.cuda.FloatTensor, optional, default=None):  New shared scale factor.

        .. warning::
            ``update`` should only be called at the end of the iteration, after ``amp_scaler.step(optimizer)`` has
            been invoked for all optimizers used this iteration.
        """
        if not self._enabled:
            return

        if new_scale is not None:
            # Accept a new user-defined scale.
            if isinstance(new_scale, float):
                self._scale = torch.full((1,), new_scale, dtype=torch.float32, device="cuda")
            else:
                string = "new_scale should be a float or a 1-element torch.cuda.FloatTensor."
                assert isinstance(new_scale, torch.cuda.FloatTensor), string
                assert new_scale.numel() == 1, string
                self._scale = new_scale
        else:
            # Consume shared inf/nan data collected from optimizers to update the scale asynchronously.
            found_inf_combined = sum(v["found_inf"] for k, v in self._per_optimizer_states.items())
            self.scale = torch._amp_update_scale(self.scale,
                                                 found_inf_combined,
                                                 self._growth_factor,
                                                 self._backoff_factor)

        # To prepare for next iteration, clear the data collected from optimizers this iteration.
        self._per_optimizer_states = defaultdict(dict)

    def get_scale(self):
        """
        Returns:
            The scale factor (a single-element ``torch.cuda.FloatTensor``),
            or 1.0 if scaling is disabled.

        .. note::
            :meth:`get_scale` alone does not incur a CPU-GPU sync, but if you wish to print the scale Tensor, or
            inspect its value on the CPU by calling ``.item()``, that will incur a sync.
        """
        return self._scale if self._enabled else 1.0

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

    def _check_inf(self, optimizer):
        dummy_rscale = torch.full((1,), 1.0, dtype=torch.float32, device="cuda")
        found_inf = torch.full((1,), 0.0, dtype=torch.float32, device="cuda")

        self._unscale_grads(optimizer, dummy_rscale, found_inf, allow_fp16=True)

        self._per_optimizer_states[id(optimizer)]["found_inf"] = found_inf

        return found_inf

    def _found_inf(self, optimizer):
        return self._per_optimizer_states[id(optimizer)].get("found_inf")
