import torch
from collections import defaultdict
from torch._six import container_abcs

class AmpScaler(object):
    """ Class that manages outputs and optimizers for dynamic gradient scaling. """
    def __init__(self,
                 init_scale=2.**24,
                 growth_factor=1.001,
                 backoff_factor=0.5,
                 enabled=True):
        self._enabled = enabled
        if enabled:
            self._scale = torch.full((1,), init_scale, dtype=torch.float32, device="cuda")
            self._growth_factor = 1.001
            self._backoff_factor = 0.5
            self._per_optimizer_states = defaultdict(dict)

    def scale(self, outputs):
        r"""
        Scales a network's output(s) by the current scale, so that when ``backward()`` is called on the scaled outputs,
        the resulting gradients will also be scaled.
    
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
                return type(val)(scale(v) for v in val)
            else:
                raise ValueError("outputs must be a Tensor or an iterable of Tensors")
    
        return apply_scale(outputs)

    def unscale(self, optimizer):
        r""" :func:`unscale` divides the optimizer's owned gradients by the current scale value, and checks if
        gradients contain inf/nan.

        :func:`unscale` is exposed for convenience, to enable use cases where you'd prefer to see and manipulate
        unscaled gradients between the backward pass(es) and ``amp_scaler.step(optimizer)``.
        
        The :class:`AmpScaler` instance is aware of whether or not :func:`unscale` has been called on each optimizer
        instance.  Therefore, in the common case where you don't need to see or manipulate unscaled gradients between
        the backward pass(es) and ``amp_scaler.step``, you can (and should) simply call ``amp_scaler.step(optimizer)``
        without invoking :func:`unscale`.  ``amp_scaler.step(optimizer)`` will realize that :func:`unscale` has not yet
        been called on ``optimizer``, and will internally unscale the gradients before applying them.

        Returns:
            A flat list of unscaled gradients, ordered as they appear in the param_groups.  If any gradients are None,
            they will also appear as None at their spot in the output list.

        .. note::
            :func:`unscale` does not incur a CPU<->GPU sync.

        .. warning::
            :func:`unscale` should only be called once for each optimizer, after _all_ gradients for that optimizer's
            upcoming ``step`` have been populated.
        """
        if not self._enabled:
            return [p.grad for group in optimizer.param_groups for p in group["params"]]

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if "unscaled" in optimizer_state:
            raise RuntimeError("unscale() has already been called on this optimizer this iteration.")

        found_inf = torch.full((1,), 0.0, dtype=torch.float32, device="cuda")

        """ FP32 division can be imprecise for certain compile options, so we carry out the reciprocal in FP64. """
        rscale = self.amp_state.scale.double().reciprocal().float()

        """ Define a generator that unscales and errors on fp16 in a single O(N) traversal of the gradients. """
        def iter_params_and_check_fp16():
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        if param.grad.dtype == torch.float16:
                            raise ValueError("Attempting to unscale FP16 gradients.  If you want to check for infs/nans "
                                             "without unscaling, use optimizer.check_infs() instead.")
                        else:
                            yield torch._amp_unscale_inf_check_(param.grad, rscale, found_inf)
                    else:
                        yield param.grad

        optimizer_state["found_inf"] = found_inf
        optimizer_state["unscaled"] = True

        return list(iter_params_and_check_fp16())

    def step(self, optimizer, *args, **kwargs):
        if (not self._enabled):
            return optimizer.step(*args, **kwargs)

        if (hasattr(optimizer, "step_supports_amp_scaling") and optimizer.step_supports_amp_scaling):
            """ The contract with custom optimizers is that their step methods should accept an additional,
            optional amp_scaler kwarg.  We append self to the kwargs so the custom optimizer has full information:
            it can query its own state, invoke unscale on its own gradients for convenience, etc. """
            return optimizer.step(*args, **dict(kwargs, amp_scaler=self))

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if "unscaled" not in optimizer_state:
            self.unscale(optimizer)

        if not optimizer_state.found_inf().item():
            return optimizer.step(*args, **kwargs)

    def update(self, new_scale=None):
        r"""
        Update the scale for next iteration, based on whether any optimizers found inf/nan gradients this iteration.
        By default :func:`update` carries out the following (pseudocode):

            if gradients had inf/nan:
                new scale = current scale * scale_backoff_factor
            else:
                new scale = current scale * scale_growth_factor

        You can also set the scale for next iteration manually, by supplying ``new_scale``.

        Arguments:
            new_scale (float or torch.cuda.FloatTensor, optional, default=None):  New shared scale factor.

        Returns:
            A one-element ``torch.cuda.FloatTensor`` containing the updated scale value,
            or ``new_scale`` if scaling is not enabled.

        .. warning::
            ``update`` should only be called at the end of the iteration, after ``amp_scaler.step(optimizer)`` has
            been invoked for all optimizers used this iteration.
        """
        if not self._enabled:
            return new_scale

        if new_scale is not None:
            """ Accept a new user-defined scale. """
            if isinstance(new_scale, float):
                self._scale = torch.full((1,), new_scale, dtype=torch.float32, device="cuda")
            else:
                string = "new_scale should be a float or a 1-element torch.cuda.FloatTensor."
                assert isinstance(new_scale, torch.cuda.FloatTensor), string
                assert new_scale.numel() == 1, string
                self._scale = new_scale
        else:
            """ Consume shared inf/nan data collected from optimizers to update the scale asynchronously. """
            found_inf_combined = sum(v["found_inf"] for k, v in self._per_optimizer_states.items())
            self.scale = torch._amp_update_scale(self.scale,
                                                found_inf_combined,
                                                self._growth_factor,
                                                self._backoff_factor)

        """ To prepare for next iteration, clear the data collected from optimizers this iteration. """
        self._per_optimizer_states = defaultdict(dict)

        return self._scale

    def get_scale(self):
        r"""
        Returns:
            A one-element ``torch.cuda.FloatTensor`` containing this AmpScaler instance's current scale.
            If scaling is disabled, returns 1.0 to avoid creating a Cuda context.

        .. note::
            :func:`get_scale` alone does not incur a CPU-GPU sync, but if you wish to print the scale Tensor, or
            inspect its value on the CPU by calling ``.item()``, that will incur a sync.
        """
        return self._scale if self._enabled else 1.0

    def get_growth_factor(self):
        r"""
        Returns:
            A Python float containing the scale growth factor.
        """
        return self._growth_factor
    
    def set_growth_factor(new_factor):
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
    
    def set_backoff_factor(new_factor):
        r"""
        Arguments:
            new_scale (float):  Value to use as the new scale backoff factor.
        """
        self._backoff_factor = new_factor
    
    def _check_inf(self, optimizer):
        """ Utility function to check if gradients currently contain inf/nan.  Asynchronous. """
        found_inf = torch.full((1,), 0.0, dtype=torch.float32, device="cuda")
        dummy_rscale = torch.full((1,), 1.0, dtype=torch.float32, device="cuda")
    
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    torch._amp_unscale_inf_check_(param.grad, dummy_rscale, found_inf)

        self._per_optimizer_states[id(optimizer)]["found_inf"] = found_inf
    
        return found_inf

    def _found_inf(self, optimizer):
        """ Utility function that returns the result of the most recent inf check performed on optimizer. """
        return self._per_optimizer_states[id(optimizer)].get("found_inf")
