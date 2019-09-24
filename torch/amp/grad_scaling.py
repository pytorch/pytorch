import torch
import types
from torch._six import container_abcs


_default_init_scale = 2.**24
_default_scale_growth_factor = 1.001
_default_scale_backoff_factor = 0.5


def _recommend_init_scale():
    return torch.full((1,), _default_init_scale, dtype=torch.float32, device="cuda")


def scale_outputs(outputs, current_scale, scaling_enabled=True):
    r"""
    Scales a network's output(s) by ``current_scale``, so that when ``backward()`` is called on the scaled outputs,
    the resulting gradients will also be scaled by ``current_scale``.

    Arguments:
        outputs (Tensor or iteratble of Tensors):  Outputs to multiply by ``current_scale``.
        current_scale (float or torch.cuda.FloatTensor):  Scale factor by which to multiply the outputs.
        scaling_enabled (bool, default=True):  If False, :func:`scale_outputs` is a no-op, and returns ``outputs``
        directly.

    Returns:
        Outputs scaled by ``current_scale``, or ``outputs`` if ``scaling_enabled=False``.
    """
    if not scaling_enabled:
        return outputs

    # Not necessary.
    # if isinstance(current_scale, float):
    #     current_scale = torch.full((1,), current_scale, dtype=torch.float32, device="cuda")

    def scale(val):
        if isinstance(val, torch.Tensor):
            assert val.is_cuda
            return val * current_scale
        elif isinstance(val, container_abcs.Iterable):
            return type(val)(scale(v) for v in val)
        else:
            raise ValueError("outputs must be a Tensor or an interable of Tensors")

    return scale(outputs)


def unscale_and_step(self,
                     closure=None,
                     current_scale=None,
                     scale_growth_factor=_default_scale_growth_factor,
                     scale_backoff_factor=_default_scale_backoff_factor,
                     scale_scheduler=None,
                     skip_if_inf=True,
                     found_inf=None,
                     scaling_enabled=True):
    r"""
    :func:`unscale_and_step` is patched onto an optimizer instance by :func:`add_amp_attributes`.  Since
    :func:`unscale_and_step` becomes a method, ``self`` is passed automatically. :func:`unscale_and_step` should only
    be called as an attribute of its optimizer instance (e.g. ``opt.unscale_and_step()``), not as a free function.

    :func:`unscale_and_step` unscales the gradients owned by this optimizer (dividing them by ``current_scale``) and
    checks them for infs/nans.  By default, if no infs/nans are found, it runs the underlying ``self.step()``.
    If infs/nans are found, it skips the step().

    Based on whether infs/nans were found, :func:`unscale_and_step` then uses the current scale to compute a new
    recommended scale for next iteration.  By default, the scale adjustment is the following::

       if gradients have inf/nan:
           recommended scale  = current_scale*scale_backoff_factor
       else:
           recommended_scale = current_scale*scale_growth_factor

    Default values for ``scale_backoff_factor`` and ``scale_growth_factor`` are 0.5 and 1.001 respectively.
    This means that if no infs/nans were found, the new recommended scale is slightly increased from the current scale,
    and if infs/nans were found, the new recommended scale is half the current scale.

    To minimize CPU-GPU syncs, which can impair GPU utilization, :func:`unscale` returns ``found_inf`` and
    ``recommended_scale`` as device Tensors rather than Python floats.  If you wish to inspect whether an
    inf/nan was found, or the recommended next scale value, query the ``.item()`` attributes of the returned Tensors.
    In general this should be done sparingly (not every iteration), because ``.item()`` incurs a CPU-GPU sync.

    .. warning::
            :func:`unscale_and_step` should only be called after all gradients for this optimizer's ``step`` have been
            populated.

    Arguments:
        current_scale (float or torch.cuda.FloatTensor):  The value most recently used to scale gradients
            (by e.g. a call to :func:`amp.scale_outputs`).
        scale_growth_factor (float, optional, default=1.001):  A Python float to be used as the scale growth factor.
            If ``scale_scheduler`` is supplied, this argument will be ignored.
        scale_backoff_factor (float, optional, default=0.5):  A Python float to be used as the scale backoff factor.
            If ``scale_scheduler`` is supplied, this argument will be ignored.
        scale_scheduler (callable, optional):  Function to customize how the recommended scale is updated.
            If supplied, ``scale_scheduler`` should take 2 arguments.  Both will be one-element
            ``torch.cuda.FloatTensors``.  The first will contain the current scale value.  The second will contain > 0
            if gradients currently contain inf or nan, and ``0.0`` otherwise.  The scheduler should use these arguments
            to determine the next recommended scale, and return a one-element ``torch.cuda.FloatTensor`` containing the
            next recommended scale.
        skip_if_inf (bool, optional, default=True):  If False, unscale gradients but call ``step()`` blindly without
            checking if the gradients contain inf or nan.  If you're manually using the same gradient scale every
            iteration ("static gradient scaling", as opposed to dynamic gradient scaling) and you're fairly
            certain no inf/nan gradients will be encountered, setting ``skip_if_inf=False`` can provide a minor
            performance improvement, because for typical optimizers it avoids copying ``found_inf``'s value back to
            the CPU.
            Another case where it makes sense to supply ``skip_if_inf=False`` is batch replay, where you replay this
            iteration until inf/nan-free gradients are created.  By the time you reach the
            :func:`optimizer.unscale_and_step()` call, you're sure the gradients do not contain inf/nan, so it's safe
            to elide the check.  For other (typical) use, the default ``skip_if_inf=True`` is strongly recommended.
        found_inf (torch.cuda.FloatTensor, optional):  If you've already created a 1-element Tensor that contains > 0.0
            if the optimizer's current gradients have an inf/nan and 0.0 if they don't (by, for example, an earlier
            call to ``optimizer.check_inf()``) you may pass this as ``found_inf``.  By default, you don't need to pass
            ``found_inf`` manually.  :func:`unscale_and_step` will inspect the gradients for you, create a
            ``found_inf`` tensor, and return it.  Leaving ``found_inf=None`` and letting :func:`unscale_and_step`
            create it for you is typical.
        scaling_enabled (bool, optional):  If ``False``, :func:`unscale_and_step` simply calls ``self.step()`` and
            returns ``opt_ret, None, None``, where opt_ret is the default return value of ``self.step()``.

    Returns:
        A tuple ``opt_ret, found_inf, recommended_scale``.  ``found_inf`` is a ``torch.cuda.FloatTensor`` that
        contains > 0 if an inf/nan was found during unscaling, and 0.0 otherwise.  ``recommended_scale`` is a
        ``torch.cuda.FloatTensor`` containing the recommended gradient scale to use next iteration.  ``opt_ret``
        is the default return value of the underlying ``self.step()`` call.  If ``scaling_enabled=False``, the return
        values will be ``opt_ret, None, None``.
    """
    if not scaling_enabled:
        return self.step(closure), None, None

    if closure is not None:
        raise ValueError("Closure use is not currently supported.  It's tricky, but not impossible, and we're trying "
                         "to decide if it's worth implementing.  If you require closure use, please comment on "
                         "https://github.com/pytorch/pytorch/issues/25081, which will help us gauge demand.")

    found, recommended_scale = self.unscale(current_scale,
                                            scale_growth_factor=scale_growth_factor,
                                            scale_backoff_factor=scale_backoff_factor,
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


def _next_recommended_scale(current_scale,
                            found_inf,
                            scale_growth_factor,
                            scale_backoff_factor):
    r"""
    The kernel carries out
    if found_inf:
        current_scale = current_scale*_scale_backoff_factor
    else:
        current_scale = current_scale*_scale_growth_factor
    In principle I could do this with primitive ops, while remaining asynchronous (a fun exercise).  But given dispatch
    and Cuda launch overhead, there's no point spending 20ish microseconds on what could be a single op.
    """
    return torch._amp_update_scale(current_scale, found_inf, scale_growth_factor, scale_backoff_factor)


def unscale(self,
            current_scale,
            scale_growth_factor=_default_scale_growth_factor,
            scale_backoff_factor=_default_scale_backoff_factor,
            scale_scheduler=None,
            scaling_enabled=True):
    r"""
    :func:`unscale` is patched onto an optimizer instance by :func:`add_amp_attributes`.  Since :func:`unscale` becomes
    a method, ``self`` is passed automatically.  :func:`unscale` should only be called as an attribute of its optimizer
    instance (e.g. ``opt.unscale()``), not as a free function.

    :func:`unscale` divides the optimizer's owned gradients by ``current_scale``, and returns two Tensors indicating
    whether or not the gradients contained infs/nans and a recommended scale to use for next iteration.

    By default, the new recommended scale is computed as follows::

       if gradients have inf/nan:
           recommended scale  = current_scale*scale_backoff_factor
       else:
           recommended_scale = current_scale*scale_growth_factor

    Default values for ``scale_backoff_factor`` and ``scale_growth_factor`` are 0.5 and 1.001 respectively.  This means
    that if no infs/nans were found, the new recommended scale is slightly increased from the current scale, and if
    infs/nans were found, the new recommended scale is half the current scale.

    By default, all operations in :func:`unscale()` are asynchronous (this is why :func:`unscale` returns Tensors,
    as opposed to Python floats).  If you wish to inspect whether an inf/nan was found, or the recommended next
    scale value, query the ``.item()`` attributes of the returned Tensors.  In general this should be done sparingly
    (not every iteration), because ``.item()`` incurs a CPU-GPU sync which can impair GPU utilization.

    .. warning::
        :func:`unscale` should only be called after all gradients for this optimizer's upcoming ``step`` have been
        populated.

    Arguments:
        current_scale (float or torch.cuda.FloatTensor):  The value most recently used to scale gradients (by e.g. a
            call to :func:`amp.scale_outputs`).
        scale_growth_factor (float, optional, default=1.001):  A Python float to be used as the scale growth factor.
            If ``scale_scheduler`` is supplied, this argument will be ignored.
        scale_backoff_factor (float, optional, default=0.5):  A Python float to be used as the scale backoff factor.
            If ``scale_scheduler`` is supplied, this argument will be ignored.
        scale_scheduler (callable, optional):  Function to customize how the recommended scale is updated.
            If supplied, ``scale_scheduler`` should take 2 arguments.  Both will be one-element
            ``torch.cuda.FloatTensors``.  The first will contain the current scale value.  The second will contain > 0
            if gradients currently contain inf or nan, and 0.0 otherwise.  The scheduler should use these arguments to
            determine the next recommended scale, and return a one-element ``torch.cuda.FloatTensor`` containing the
            next recommended scale.
        scaling_enabled (bool, optional):  If ``False``, :func:`unscale` becomes a no-op and returns ``None, None``.

    Returns:
        A tuple ``found_inf, recommended_scale``.  ``found_inf`` is a ``torch.cuda.FloatTensor`` that contains > 0 if
        an inf/nan was found during unscaling, and 0.0 otherwise.  ``recommended_scale`` is a
        ``torch.cuda.FloatTensor`` containing the recommended gradient scale to use next iteration.  If
        ``scaling_enabled=False``, the return values will be ``None, None``.
    """
    if not scaling_enabled:
        return None, None

    # Don't want to use torch.zeros() here because that gives the backend leeway to potentially call memset,
    # and I've heard ominous rumours memset does not play well with cuda graphs.
    found_inf = torch.full((1,), 0.0, dtype=torch.float32, device="cuda")

    if isinstance(current_scale, float):
        current_scale = torch.full((1,), current_scale, dtype=torch.float32, device="cuda")

    # Eventually, we'd like this to become a multi-tensor apply call dispatched via NestedTensor.
    # In the meantime, compute the reciprocal of the current scale to use for all grads.
    # FP32 division can be imprecise for certain compile options, so we carry out the reciprocal in FP64.
    # These ops are cheap and only need to be done once for all the params.
    rscale = current_scale.double().reciprocal().float()

    for group in self.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                if param.grad.dtype == torch.float16:
                    raise ValueError("Attempting to unscale FP16 gradients.  If you want to check for infs/nans  "
                                     "without unscaling, use optimizer.check_infs() instead.")
                else:
                    torch._amp_unscale_inf_check_(param.grad, rscale, found_inf)

    if scale_scheduler is None:
        next_recommended_scale = _next_recommended_scale(current_scale, found_inf,
                                                         scale_growth_factor, scale_backoff_factor)
    else:
        next_recommended_scale = scale_scheduler(current_scale, found_inf)

    return found_inf, next_recommended_scale


def check_inf(self, scaling_enabled=True):
    r"""
    :func:`check_inf` is patched onto an optimizer instance by :func:`add_amp_attributes`.  Since :func:`check_inf`
    becomes a method, ``self`` is passed automatically.  :func:`check_inf` should only be called as an attribute of
    its optimizer instance (e.g. ``opt.check_inf()``), not as a free function.

    Check the gradients currently owned by this optimizer for infs/nans, without unscaling.  For typical
    implementations of gradient unscaling in your script, it's not necessary to call this function; prefer
    :func:`unscale_and_step` or the :func:`unscale`+:func:`step_after_unscale` combination as shown in the Examples.

    Arguments:
        scaling_enabled (bool, optional):  If ``False``, :func:`check_inf` becomes a no-op and returns ``None``.

    Returns:
        ``found_inf``, a ``torch.cuda.FloatTensor`` that contains > 0 if an inf/nan is present in the current gradients,
        and 0.0 otherwise.  If ``scaling_enabled=False``, the return value will be ``None``.
    """
    if not scaling_enabled:
        return None

    found_inf = torch.full((1,), 0.0, dtype=torch.float32, device="cuda")

    # This duplicates code from unscale.  In principle I could just call into unscale(1.0), but that would error
    # if gradients are fp16, and I want check_inf to be usable on fp16 gradients.  I also don't want to add an argument
    # to "unscale" that permits unscaling in fp16.
    for group in self.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                torch._amp_unscale_inf_check_(param.grad,
                                              torch.full((1,), 1.0, dtype=torch.float32, device="cuda"),
                                              found_inf)

    return found_inf


def step_after_unscale(self,
                       closure=None,
                       found_inf=None,
                       skip_if_inf=True,
                       scaling_enabled=True):
    r"""
    :func:`step_after_unscale` is patched onto an optimizer instance by :func:`add_amp_attributes`.  Since
    :func:`step_after_unscale` becomes a method, ``self`` is passed automatically.  :func:`step_after_unscale`
    should only be called as an attribute of its optimizer instance (e.g. ``opt.step_after_unscale()``),
    not as a free function.

    ``found_inf`` should be the result of an earlier call to ``optimizer.unscale`` called on the same optimizer
    instance after all of its gradients have been populated.
    By default, :func:`step_after_unscale` checks the value contained by ``found_inf``.  If ``found_inf`` contains 0.0
    (in other words if the gradients do not contain infs/nans), :func:`step_after_unscale` runs the underlying
    ``self.step()``.  If ``found_inf`` contains > 0, indicating that the gradients do contain infs or nans,
    it skips the step().

    Arguments:
        found_inf (torch.cuda.FloatTensor):  1-element Tensor produced by an earlier call to ``optimizer.unscale()``
            that should contain > 0 if the optimizer's current gradients have an inf/nan and 0.0 if they don't.
        skip_if_inf (bool, optional, default=True):  If False, call ``step()`` without checking the value of
            ``found_inf``.  If you're manually using the same gradient scale every iteration ("static gradient scaling"
            as opposed to dynamic gradient scaling) and you're fairly
            certain no inf/nan gradients will be encountered, setting ``skip_if_inf=False`` can provide a minor
            performance improvement, because for typical optimizers it avoids copying ``found_inf``'s value back to
            the CPU.  Another case where it makes sense to supply ``skip_if_inf=False`` is batch replay, where you
            replay this iteration until inf/nan-free gradients are created.  By the time you reach the
            ``optimizer.step_after_unscale()`` call, you're sure the gradients do not contain inf/nan, so it's safe
            to elide the check.  For other (typical) use, the default ``skip_if_inf=True`` is strongly recommended.
        scaling_enabled (bool, optional):  If ``False``, :func:`step_after_unscale` simply calls ``self.step()``
            and returns ``opt_ret, None, None``, where opt_ret is the default return value of ``self.step()``.

    Returns:
        If ``found_inf`` contains 0.0, :func:`step_after_unscale` returns the default return value of the
        ``self.step()`` call.  If ``found_inf`` contains > 0, :func:`step_after_unscale` returns None.
    """
    if not scaling_enabled:
        return self.step(closure)

    if closure is not None:
        raise ValueError("Closure use is not currently supported.  It's tricky, but not impossible, and we're trying "
                         "to decide if it's worth implementing.  If you require closure use, please comment on "
                         "https://github.com/pytorch/pytorch/issues/25081, which will help us gauge demand.")

    if skip_if_inf:
        if found_inf.item():
            return None
        else:
            return self.step()
    else:
        return self.step()


def _add_amp_attributes(optimizer):
    if not hasattr(optimizer, "unscale_and_step"):
        optimizer.unscale_and_step = types.MethodType(unscale_and_step, optimizer)

    if not hasattr(optimizer, "step_after_unscale"):
        optimizer.step_after_unscale = types.MethodType(step_after_unscale, optimizer)

    if not hasattr(optimizer, "unscale"):
        optimizer.unscale = types.MethodType(unscale, optimizer)

    if not hasattr(optimizer, "check_inf"):
        optimizer.check_inf = types.MethodType(check_inf, optimizer)


def add_amp_attributes(optimizers):
    r"""
    Adds convenience methods for gradient unscaling and inf/nan checking to an optimizer instance, and returns
    a Tensor containing the recommended initial gradient scale (which you are free to use, or substitute your own).

    The methods added are :func:`unscale`, :func:`check_inf`, :func:`unscale_and_step`, and :func`step_after_unscale`.

    If an optimizer some or all of these methods defined already, :func:`add_amp_attributes` won't touch them, allowing
    you to customize their behavior (if desired) by defining them yourself as part of your optimizer class.
    Custom implementations of  :func:`unscale`, :func:`check_inf`, :func:`unscale_and_step`,
    and :func:`step_after_unscale`, if present, should have the same interface (arguments + return values)
    as the default implementations described here.  Defining the methods yourself is not required and the
    default implementations should work with any optimizer that defines ``step``.

    Arguments:
        optimizers (torch.optim.Optimizer or iterable of torch.optim.Optimizers):  Optimizer(s) for which to
            add gradient scaling methods.

    Returns:
        A torch.cuda.FloatTensor containing the recommended initial gradient scale value.
    """
    optimizers = (optimizers,) if isinstance(optimizers, torch.optim.Optimizer) else tuple(optimizers)
    if len(optimizers) == 0:
        raise ValueError("add_amp_attributes received an empty optimizer list.")

    for optimizer in optimizers:
        _add_amp_attributes(optimizer)

    return _recommend_init_scale()
