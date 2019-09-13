import torch

_scale_growth_rate = 1.001


def get_scale_growth_rate(new_growth_rate):
    return _scale_growth_rate


def set_scale_growth_rate(new_growth_rate):
    _scale_growth_rate = new_growth_rate


def scale_outputs(outputs, current_scale, scaling_enabled=True):
    r"""
    Scales a network's output(s) by ``current_scale``, so that when ``backward()`` is called on the scaled outputs, the resulting
    gradients will also be scaled by ``current_scale``.

    Arguments:
        outputs (Tensor or iteratble of Tensors):  Outputs to multiply by ``current_scale``.
        scaling_enabled (bool, default=True):  If False, ``scale_outputs`` is a no-op, and returns ``outputs`` directly.

    Returns:
        Outputs scaled by ``current_scale``, or ``outputs`` if ``scaling_enabled=False``.
    """
    outputs = list(outputs)
    if scaling_enabled:
        return [output*current_scale for output in outputs]
    else:
        return outputs

# Should we allow a found_inf argument to communicate the result of a previous inf check (as for step_after_unscale)
# so that unscale_and_step can bail out early?
def unscale_and_step(self,
                     closure=None,
                     current_scale=None,
                     scale_scheduler=None,
                     skip_if_inf=True,
                     found_inf=None,
                     scaling_enabled=True)
    r"""
    ``unscale_and_step`` is patched onto an optimizer instance by :func:`add_amp_attributes`.  Since ``unscale_and_step`` becomes a method, ``self`` is passed automatically.

    ``unscale_and_step`` unscales the gradients owned by this optimizer (dividing them by ``current_scale``) and checks them for
    infs/nans.  By default, if no infs/nans are found, it runs the underlying ``self.step()``.
    If infs/nans are found, it skips the step().

    Based on whether infs/nans were found, ``unscale_and_step`` then adjusts the current scale to a new recommended value
    for next iteration.  If no infs/nans were found, the adjusted value will be slightly increased:
    ``current_scale``*:func:`get_scale_growth_rate()`.  If infs/nans were found the adjusted value will be ``current_scale/2.0``.
    See Returns below.

    .. warning::
            ``unscale_and_step`` should only be called after all gradients for this optimizer's ``step`` have been populated.


    Arguments:
        current_scale (torch.cuda.FloatTensor):  The value most recently used to scale gradients (by e.g. a call to :func:`amp.scale_outputs`).
        scale_scheduler (callable, optional):  Function to customize how the recommended scale is updated.
            If supplied, ``scale_scheduler`` should take 2 arguments.  Both will be one-element ``torch.cuda.FloatTensors``.
            The first will contain the current scale value.  The second will contain ``1.0`` if gradients currently contain
            inf or nan, and ``0.0`` otherwise.  The scheduler should use these arguments to determine the next recommended scale,
            and return a one-element ``torch.cuda.FloatTensor`` containing the next recommended scale.
        skip_if_inf (bool, optional, default=True):  If False, unscale gradients but call ``step()`` blindly without checking
            if the gradients contain inf or nan.  If you're manually using the same gradient scale every iteration and you're fairly
            certain no inf/nan gradients will be encountered, setting ``skip_if_inf=False`` can provide a minor performance
            improvement, because for typical optimizers it avoids copying ``found_inf``'s value back to the CPU.
            Another case where it makes sense to supply ``skip_if_inf=False`` is batch replay, where you replay this iteration
            until inf/nan-free gradients are created.  By the time you reach the `optimizer.unscale_and_step()` call,
            you're sure the gradients do not contain inf/nan, so it's safe to elide the check.
            For other (typical) use, the default ``skip_if_inf=True`` is strongly recommended.
        found_inf (torch.cuda.FloatTensor, optional):  If you've already created a 1-element Tensor that contains 1.0 if the
            optimizer's current gradients have an inf/nan and 0.0 if they don't (by, for example, an earlier call to
            ``optimizer.check_inf()``) you may pass this as ``found_inf``.  By default, you don't need to pass ``found_inf`` manually.
            ``unscale_and_step`` will inspect the gradients for you, create a ``found_inf`` tensor, and return it.
            Leaving ``found_inf=None`` and letting ``unscale_and_step`` create it for you is typical.
        scaling_enabled (bool, optional):  If ``False``, ``unscale_and_step`` simply calls ``self.step()`` and returns
            ``opt_ret, None, None``, where opt_ret is the default return value of ``self.step()``.

    Returns:
        A tuple ``opt_ret, found_inf, recommended_scale``.  ``found_inf`` is a ``torch.cuda.FloatTensor`` that contains 1.0 if an inf/nan was found during unscaling, and 0.0 otherwise.  ``recommended_scale`` is a ``torch.cuda.FloatTensor`` containing the recommended gradient scale to use next iteration.  ``opt_ret`` is the default return value of the underlying ``self.step()`` call.  If ``scaling_enabled=False``, the return values will be ``opt_ret, None, None``.
    """
    if scaling_enabled:
        if closure is not None:
            raise ValueError("Closure use is not currently supported.  It's tricky, but not impossible, and we're trying "
                             "to decide if it's worth implementing.  If you require closure use, please comment on "
                             "https://github.com/pytorch/pytorch/issues/25081, which will help us gauge demand.")
    
        found, recommended_scale = self.unscale(current_scale,
                                                scale_scheduler=scale_scheduler)

        if found_inf is None:
            found_inf = found

        if skip_if_inf:
            if found_inf.item():
                return None, found_inf, recommended_scale
            else:
                return self.step(), found_inf, recommended_scale
        else:
            return self.step(), found_inf, recommended_scale
    else:
        return self.step(closure), None, None


def _next_recommended_scale(current_scale, found_inf):
    # Implement
    # if found_inf:
    #     current_scale = current_scale*0.5
    # else:
    #     current_scale = current_scale*_scale_growth_rate
    # using purely asynchronous dispatches.
    # These are only manipulating a single value, so they shouldn't be a bottleneck, but they could be fused into one custom op.
    return (current_scale - 0.5 * current_scale * found_inf) * (_get_scale_growth_rate() - (_get_scale_growth_rate - 1.0) *found_inf)


def unscale(self, current_scale=None, scale_scheduler=None, scaling_enabled=True)
    r"""
    ``unscale`` is patched onto an optimizer instance by :func:`add_amp_attributes`.  Since ``unscale`` becomes a method, ``self`` is passed automatically.

    ``unscale`` divides the optimizer's owned gradients by ``current_scale``, and returns two Tensors indicating whether or not
    the gradients contained infs/nans and a recommended scale to use for next iteration.

    .. warning::
        ``unscale`` should only be called after all gradients for this optimizer's upcoming ``step`` have been populated.

    All operations in unscale() are asynchronous (this is why ``unscale`` returns Tensors, as opposed to Python floats).
    If you wish to inspect whether an inf/nan was found, or the recommended next scale value, query the ``.item()`` attributes
    of the returned Tensors.  In general this should be done sparingly (not every iteration), because ``.item()`` incurs a
    CPU-GPU sync which can impair GPU utilization.

    Arguments:
        current_scale (torch.cuda.FloatTensor):  The value most recently used to scale gradients (by e.g. a call to :func:`amp.scale_outputs`).
        scale_scheduler (callable, optional):  Function to customize how the recommended scale is updated
            If supplied, ``scale_scheduler`` should take 2 arguments.  Both will be one-element ``torch.cuda.FloatTensors``.
            The first will contain the current scale value.  The second will contain ``1.0`` if gradients currently contain
            inf or nan, and ``0.0`` otherwise.  The scheduler should use these arguments to determine the next recommended scale,
            and return a one-element ``torch.cuda.FloatTensor`` containing the next recommended scale.
        scaling_enabled (bool, optional):  If ``False``, ``unscale`` becomes a no-op and returns ``None, None``.

    Returns:
        A tuple ``found_inf, recommended_scale``.  ``found_inf`` is a ``torch.cuda.FloatTensor`` that contains 1.0 if an inf/nan was found during unscaling, and 0.0 otherwise.  ``recommended_scale`` is a ``torch.cuda.FloatTensor`` containing the recommended gradient scale to use next iteration.  If ``scaling_enabled=False``, the return values will be ``None, None``.
    """
    if scaling_enabled:
        found_inf = torch.cuda.FloatTensor([0.0])

        if scale_scheduler is None:
            scale_scheduler = _next_recommended_scale

        # Eventually, we'd like this to become a multi-tensor apply call dispatched via NestedTensor.
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    if param.grad.dtype == torch.float16:
                        raise ValueError("Attempting to unscale FP16 gradients.  If you want to check for infs/nans without "
                                         "unscaling, use optimizer.check_infs() instead.")
                    else:
                        torch._amp_unscale_inf_check(param.grad, current_scale, found_inf)

        return found_inf, _next_recommended_scale(current_scale, found_inf)
    else:
        return None, None


def check_inf(self, scaling_enabled=True):
    r"""
    ``check_inf`` is patched onto an optimizer instance by :func:`add_amp_attributes`.  Since ``check_inf`` becomes a method, ``self`` is passed automatically.

    Check the gradients currently owned by this optimizer for infs/nans, without unscaling.  For typical implementations of gradient
    unscaling in your script, it's not necessary to call this function; prefer ``unscale_and_step`` or the
    ``unscale``+``step_after_unscale`` combination as shown in the Examples.

    Arguments:
        scaling_enabled (bool, optional):  If ``False``, ``check_inf`` becomes a no-op and returns ``None``.

    Returns:
        ``found_inf``, a ``torch.cuda.FloatTensor`` that contains 1.0 if an inf/nan is present in the current gradients,
        and 0.0 otherwise.  If ``scaling_enabled=False``, the return value will be ``None``.
    """
    if scaling_enabled:
        found_inf = torch.cuda.FloatTensor([0.0])

        # This duplicates code from unscale.  In principle I could just call into unscale(1.0), but that would error
        # if gradients are fp16, and I want this function to be usable on fp16 gradients.  I also don't want to add an argument
        # to "unscale" that permits unscaling in fp16, because then users might use it...
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                  torch._amp_unscale_inf_check(param.grad, 1.0, found_inf)

        return found_inf
    else:
        return None


def step_after_unscale(self,
                       closure=None,
                       found_inf=None,
                       skip_if_inf=True,
                       scaling_enabled=True)
    r"""
    ``step_after_unscale`` is patched onto an optimizer instance by :func:`add_amp_attributes`.  Since ``step_after_unscale`` becomes a method, ``self`` is passed automatically.

    ``found_inf`` should be the result of an earlier call to ``optimizer.unscale`` called on the same optimizer instance after
    all of its gradients have been populated.
    By default, ``step_after_unscale`` checks the value contained by ``found_inf``.  If ``found_inf`` contains 0.0
    (in other words if the gradients do not contain infs/nans), ``step_after_unscale`` runs the underlying ``self.step()``.
    If ``found_inf`` contains 1.0, indicating that the gradients do contain infs or nans, it skips the step().

    Arguments:
        found_inf (torch.cuda.FloatTensor):  1-element Tensor produced by an earlier call to ``optimizer.unscale()`` that should
            contain 1.0 if the optimizer's current gradients have an inf/nan and 0.0 if they don't.
        skip_if_inf (bool, optional, default=True):  If False, unscale gradients but call ``step()`` blindly without checking
            if the gradients contain inf or nan.  If you're manually using the same gradient scale every iteration and you're fairly
            certain no inf/nan gradients will be encountered, setting ``skip_if_inf=False`` can provide a minor performance
            improvement, because for typical optimizers it avoids copying ``found_inf``'s value back to the CPU.
            Another case where it makes sense to supply ``skip_if_inf=False`` is batch replay, where you replay this iteration
            until inf/nan-free gradients are created.  By the time you reach the `optimizer.step_after_unscale()` call,
            you're sure the gradients do not contain inf/nan, so it's safe to elide the check.
            For other (typical) use, the default ``skip_if_inf=True`` is strongly recommended.
        scaling_enabled (bool, optional):  If ``False``, ``step_after_unscale`` simply calls ``self.step()`` and returns
            ``opt_ret, None, None``, where opt_ret is the default return value of ``self.step()``.

    Returns:
        If ``found_inf`` contains 0.0, ``step_after_unscale`` returns the default return value of the ``self.step()`` call.
        If ``found_inf`` contains 1.0, ``step_after_unscale`` returns None.
    """
    if scaling_enabled:
        if closure is not None:
            raise ValueError("Closure use is not currently supported.  It's tricky, but not impossible, and we're trying "
                             "to decide if it's worth implementing.  If you require closure use, please comment on "
                             "https://github.com/pytorch/pytorch/issues/25081, which will help us gauge demand.")

        if skip_if_inf:
            if found_inf.item():
                return None, found_inf, recommended_scale
            else:
                return self.step(), found_inf, recommended_scale
        else:
            return self.step(), found_inf, recommended_scale
    else:
        return self.step(closure), None, None
    if scaling_enabled:
        if not found_inf.item():
            return self.step()
        else:
            return None
    else:
        return self.step()


def _add_amp_attributes(optimizer):
    # If the user defined a custom unscale_and_step, don't touch it.
    if not hasattr(optimizer, "unscale_and_step"):
        optimizer.unscale_and_step = types.MethodType(unscale_and_step, optimizer)

    if not hasattr(optimizer, "step_after_unscale"):
        optimizer.step_after_unscale = types.MethodType(step_after_unscale, optimizer)

    if not hasattr(optimizer, "unscale"):
        optimizer.unscale = types.MethodType(unscale, optimizer)

    if not hasattr(optimizer, "check_inf"):
        optimizer.unscale = types.MethodType(unscale, optimizer)


def add_amp_attributes(optimizers):
    r"""
    Adds convenience methods for gradient unscaling and inf/nan checking to an optimizer instance.

    The methods added are ``unscale``, ``check_inf``, ``unscale_and_step``, and ``step_after_unscale``.

    Arguments:
        optimizers (torch.optim.Optimizer or iterable of torch.optim.Optimizers):  Optimizer(s) for which to
            add gradient scaling methods.
    """
    optimizers = list(optimizers)
    if len(optimizers) == 0:
        raise ValueError("add_amp_attributes received an empty optimizer list.")

    for optimizer in optimizers:
        _add_amp_attributes(optimizer)
