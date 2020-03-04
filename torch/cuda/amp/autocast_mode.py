import torch
import functools
import warnings
import numpy as np
from torch._six import container_abcs, string_classes


class autocast(object):
    r"""
    Instances of :class:`autocast` serve as context managers or decorators that
    allow regions of your script to run in mixed precision.

    In autocast-enabled regions, the backend automatically chooses the precision
    for GPU operations to improve performance while maintaining accuracy.

    When entering an autocast-enabled region, Tensors may be any type.  It is not necessary or
    recommended to call ``.half()`` on your model(s) or data to use autocasting.

    :class:`autocast` should wrap the forward pass(es) of your network::

        # Creates model and optimizer in default precision (float32)
        model = Net().cuda()
        optimizer = optim.SGD(model.parameters(), ...)

        for input, target in data:
            optimizer.zero_grad()

            # Enables autocasting for the forward pass (model + loss)
            with autocast():
                output = model(input)
                loss = loss_fn(output, target)

            # Exits the context manager before backward()
            # Backward passes under autocast are not necessary or recommended.
            # Backward ops run in the same type that autocast used for corresponding forward ops.
            loss.backward()
            optimizer.step()

    :class:`autocast` can also be used as a decorator, e.g., on the ``forward`` method of your model::

        class AutocastModel(nn.Module):
            ...
            @autocast()
            def forward(self, input):
                ...

    :class:`autocast` is nestable.  If you want to force particular ops to run in ``float32``,
    you can nest ``autocast(enabled=False)`` regions in a surrounding autocast-enabled region::

        mat0 = torch.rand((8, 8), device="cuda", dtype.torch.float32)
        mat1 = torch.rand((8, 8), device="cuda", dtype.torch.float32)
        mat2 = torch.rand((8, 8), device="cuda", dtype.torch.float32)
        mat3_float16 = torch.rand((8, 8), device="cuda", dtype.torch.float16)

        with autocast():
            # torch.mm is on autocast's list of ops that should run in float16..
            # Inputs are float32, but the op runs in float16 and produces float16 output.
            # No manual casts are required.
            tmp_float16 = torch.mm(mat0, mat1)

            with autocast(enabled=False):
                # Here torch.mm behaves normally.
                # To force float32 execution, ensure the inputs are float32.
                # The output type matches the input types.
                tmp_float32 = torch.mm(tmp_float16.float(), mat2)

            # No manual casts are required when re-entering the autocast-enabled region.
            # torch.mm again runs in float16 and produces float16 output, regardless of input types.
            # Note that mismatched input types are transparently handled.
            float16_result = torch.mm(tmp_float32, mat3_float16)

    Arguments:
        enabled(bool, optional, default=True):  Whether autocasting should be enabled in this region.

    .. note::
        Tensors produced in an autocast-enabled region may be ``float16``.  After returning to an
        autocast-disabled region, using them along with ``float32`` tensors may cause type mismatch errors.
        If so, simply call ``.float()`` on the offending tensor(s).

        Type mismatch errors *in* an autocast-enabled region are a bug; if this is what you observe,
        please file an issue.

    .. note::
        Autocast only affects GPU operations (operations running on CUDA Tensors).

    .. note::
        The autocast state is thread-local.  If you want it enabled in a new thread, the context manager or decorator
        must be invoked in that thread.  This affects :class:`torch.nn.DataParallel`, which spawns
        new threads to run ``forward`` on each device.  See the :ref:`DataParallel example<amp-dataparallel>`
        for best practices.

    .. note::
        Currently, autocast only affects out-of-place operations.  In-place ops still work in autocast-enabled
        regions, but won't be autocasted (e.g., ``a.addmm(b)`` is guaranteed to run in ``float16``, but
        ``a.addmm_(b)` may not).  For best performance and accuracy, prefer out-of-place ops if possible.
    """
    def __init__(self, enabled=True):
        if enabled and not torch.cuda.is_available():
            warnings.warn("torch.cuda.amp.autocast only affects CUDA ops, but CUDA is not available.  Disabling.")
            self._enabled = False
        else:
            self._enabled = enabled

    def __enter__(self):
        self.prev = torch.is_autocast_enabled()
        torch.set_autocast_enabled(self._enabled)
        torch.autocast_increment_nesting()

    def __exit__(self, *args):
        # Drop the cache when we exit to a nesting level that's outside any instance of autocast.
        if torch.autocast_decrement_nesting() == 0:
            torch.clear_autocast_cache()
        torch.set_autocast_enabled(self.prev)
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_autocast(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorate_autocast


# Casts Tensors and containers of Tensors.  Special-cases passthroughs for strings and np.ndarrays, which
# may be falsely detected as "Iterables."
def _cast(value, dtype):
    if isinstance(value, torch.Tensor):
        return value.to(dtype) if (value.is_floating_point() and value.is_cuda) else value
    elif isinstance(value, string_classes):
        return value
    elif isinstance(value, np.ndarray):
        return value
    elif isinstance(value, container_abcs.Mapping):
        return {_cast(k, dtype): _cast(v, dtype) for k, v in value.items()}
    elif isinstance(value, container_abcs.Iterable):
        return type(value)(_cast(v, dtype) for v in value)
    else:
        return value


# custom_fwd is a decorator that may or may not be used with arguments, following
# https://github.com/dabeaz/python-cookbook/tree/master/src/9/defining_a_decorator_that_takes_an_optional_argument.
# this works:
#     @custom_fwd
#     def forward(...):
# this also works:
#     @custom_fwd(cast_inputs=torch.float)
#     def forward(...):
def custom_fwd(fwd=None, *, cast_inputs=None):
    """
    Helper decorator for ``forward`` methods of custom autograd functions (subclasses of
    :class:`torch.autograd.Function`).  See the :ref:`example page<amp-custom-examples>` for more detail.

    Arguments:
        cast_inputs (:class:`torch.dtype` or None, optional, default=None):  If not ``None``, casts incoming
            floating-point Tensors to the target dtype and causes ``forward`` to execute with autocast disabled.
            If ``None``, inputs are not cast and ``forward`` executes with whatever autocast state surrounds the
            point-of-use.
    """
    if fwd is None:
        return functools.partial(custom_fwd, cast_inputs=cast_inputs)

    @functools.wraps(fwd)
    def decorate_fwd(*args, **kwargs):
        if cast_inputs is None:
            args[0]._fwd_used_autocast = torch.is_autocast_enabled()
            return fwd(*args, **kwargs)
        else:
            args[0]._fwd_used_autocast = False
            with autocast(enabled=False):
                return fwd(*_cast(args, cast_inputs), **_cast(kwargs, cast_inputs))
    return decorate_fwd


# Autograd ensures incoming gradients are the same type as forward outputs.  Allowing a separate
# cast_inputs_to argument on custom_bwd is unnecessary and could cause errors if it doesn't match
# cast_inputs_to supplied to custom_fwd.
def custom_bwd(bwd):
    """
    Helper decorator for backward methods of custom autograd functions (subclasses of
    :class:`torch.autograd.Function`).
    Ensures that ``backward`` executes with the same autocast state as ``forward``.
    See the :ref:`example page<amp-custom-examples>` for more detail.
    """
    @functools.wraps(bwd)
    def decorate_bwd(*args, **kwargs):
        with autocast(args[0]._fwd_used_autocast):
            return bwd(*args, **kwargs)
    return decorate_bwd
