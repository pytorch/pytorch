import torch
import functools


class autocast(object):
    r"""
    Instances of :class:`autocast` serve as context managers or decorators that
    allow regions of your script to run in mixed precision.

    Within autocast-enabled regions, the backend automatically chooses the precision
    for GPU operations to improve performance while maintaining accuracy.

    When entering an autocast-enabled region, Tensors may be any type.  It is not necessary or
    recommended to call ``.half()`` on your model or data to use autocasting.

    :class:`autocast` should be used to wrap the forward pass of your model::

        # Creates model in default precision (torch.float32)
        model = Net()

        for input, target in data:
            optimizer.zero_grad()

            # Enables autocasting for the forward pass (model + loss)
            with torch.cuda.amp.autocast():
                output = model(input)
                loss = loss_fn(output, target)

            # Exits the context manager before backward()
            # Running backward() under autocast is not necessary or recommended.
            # Autograd correctly handles any casts that occurred during the forward pass.
            loss.backward()
            optimizer.step()

    :class:`autocast` is nestable.  If you want to force particular ops to run in ``torch.float32``,
    you can nest autocast-disabled regions within a surrounding autocast-enabled region::

        mat0 = torch.rand((8,8), device="cuda", dtype.torch.float32)
        mat1 = torch.rand((8,8), device="cuda", dtype.torch.float32)
        mat2 = torch.rand((8,8), device="cuda", dtype.torch.float32)
        mat3 = torch.rand((8,8), device="cuda", dtype.torch.float32)

        with torch.cuda.amp.autocast():
            # Here torch.mm autocasts.
            # The inputs are float32, but the op runs in float16 and produces float16 output.
            # No manual casts are required.
            fp16_tensor = torch.mm(mat0, mat1)

            with torch.cuda.amp.autocast(enabled=False):
                # Here torch.mm does not autocast.
                # To force float32 execution, we ensure the inputs are float32.
                # The output type matches the input types as usual.
                fp32_tensor = torch.mm(fp16_tensor.float(), mat2)

            # No manual casts are required when re-entering the autocast-enabled region.
            fp16_result = torch.mm(fp32_tensor, mat3)

    Autocast-enabled regions transparently handle ops with mismatched floating-point types::

        mat0 = torch.rand((8,8), device="cuda", dtype.torch.float16)
        mat1 = torch.rand((8,8), device="cuda", dtype.torch.float32)

        with torch.cuda.amp.autocast():
            fp16_tensor = torch.mm(mat0, mat1)

    Arguments:
        enabled(bool, optional, default=True):  Whether autocasting should be enabled within this region.

    .. note::
        Tensors produced within an autocast-enabled region may be ``torch.float16``.  After returning to an
        autocast-disabled region, using them along with ``torch.float32`` tensors may cause type mismatch errors.
        If so, simply call ``.float()`` on the offending tensor(s).

        Type mismatch errors *within* an autocast-enabled region are a bug; if this is what you observe,
        please file an issue.
    """
    def __init__(self, enabled=True):
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
