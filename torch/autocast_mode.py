import torch
import functools
import warnings

class autocast(object):
    r"""
    Instances of :class:`autocast` serve as context managers or decorators that
    allow regions of your script to run in mixed precision.

    In these regions, ops run in an op-specific dtype chosen by autocast
    to improve performance while maintaining accuracy.
    See the :ref:`Autocast Op Reference<autocast-op-reference>` for details.

    When entering an autocast-enabled region, Tensors may be any type.
    You should not call ``half()`` or ``bfloat16()`` on your model(s) or inputs when using autocasting.

    :class:`autocast` should wrap only the forward pass(es) of your network, including the loss
    computation(s).  Backward passes under autocast are not recommended.
    Backward ops run in the same type that autocast used for corresponding forward ops.

    Example for CUDA Devices::

        # Creates model and optimizer in default precision
        model = Net().cuda()
        optimizer = optim.SGD(model.parameters(), ...)

        for input, target in data:
            optimizer.zero_grad()

            # Enables autocasting for the forward pass (model + loss)
            with autocast():
                output = model(input)
                loss = loss_fn(output, target)

            # Exits the context manager before backward()
            loss.backward()
            optimizer.step()

    See the :ref:`Automatic Mixed Precision examples<amp-examples>` for usage (along with gradient scaling)
    in more complex scenarios (e.g., gradient penalty, multiple models/losses, custom autograd functions).

    :class:`autocast` can also be used as a decorator, e.g., on the ``forward`` method of your model::

        class AutocastModel(nn.Module):
            ...
            @autocast()
            def forward(self, input):
                ...

    Floating-point Tensors produced in an autocast-enabled region may be ``float16``.
    After returning to an autocast-disabled region, using them with floating-point
    Tensors of different dtypes may cause type mismatch errors.  If so, cast the Tensor(s)
    produced in the autocast region back to ``float32`` (or other dtype if desired).
    If a Tensor from the autocast region is already ``float32``, the cast is a no-op,
    and incurs no additional overhead.
    CUDA Example::

        # Creates some tensors in default dtype (here assumed to be float32)
        a_float32 = torch.rand((8, 8), device="cuda")
        b_float32 = torch.rand((8, 8), device="cuda")
        c_float32 = torch.rand((8, 8), device="cuda")
        d_float32 = torch.rand((8, 8), device="cuda")

        with autocast():
            # torch.mm is on autocast's list of ops that should run in float16.
            # Inputs are float32, but the op runs in float16 and produces float16 output.
            # No manual casts are required.
            e_float16 = torch.mm(a_float32, b_float32)
            # Also handles mixed input types
            f_float16 = torch.mm(d_float32, e_float16)

        # After exiting autocast, calls f_float16.float() to use with d_float32
        g_float32 = torch.mm(d_float32, f_float16.float())

    CPU Example::

        # Creates some tensors in default dtype (here assumed to be float32)
        a_float32 = torch.rand((8, 8), device="cpu")
        b_float32 = torch.rand((8, 8), device="cpu")
        c_float32 = torch.rand((8, 8), device="cpu")
        d_float32 = torch.rand((8, 8), device="cpu")

        with autocast(dtype=torch.bfloat16, device_type="cpu"):
            # torch.mm is on autocast's list of ops that should run in bfloat16.
            # Inputs are float32, but the op runs in bfloat16 and produces bfloat16 output.
            # No manual casts are required.
            e_bfloat16 = torch.mm(a_float32, b_float32)
            # Also handles mixed input types
            f_bfloat16 = torch.mm(d_float32, e_bfloat16)

        # After exiting autocast, calls f_float16.float() to use with d_float32
        g_float32 = torch.mm(d_float32, f_bfloat16.float())

    Type mismatch errors *in* an autocast-enabled region are a bug; if this is what you observe,
    please file an issue.

    ``autocast(enabled=False)`` subregions can be nested in autocast-enabled regions.
    Locally disabling autocast can be useful, for example, if you want to force a subregion
    to run in a particular ``dtype``.  Disabling autocast gives you explicit control over
    the execution type.  In the subregion, inputs from the surrounding region
    should be cast to ``dtype`` before use::

        # Creates some tensors in default dtype (here assumed to be float32)
        a_float32 = torch.rand((8, 8), device="cuda")
        b_float32 = torch.rand((8, 8), device="cuda")
        c_float32 = torch.rand((8, 8), device="cuda")
        d_float32 = torch.rand((8, 8), device="cuda")

        with autocast():
            e_float16 = torch.mm(a_float32, b_float32)
            with autocast(enabled=False):
                # Calls e_float16.float() to ensure float32 execution
                # (necessary because e_float16 was created in an autocasted region)
                f_float32 = torch.mm(c_float32, e_float16.float())

            # No manual casts are required when re-entering the autocast-enabled region.
            # torch.mm again runs in float16 and produces float16 output, regardless of input types.
            g_float16 = torch.mm(d_float32, f_float32)

    The autocast state is thread-local.  If you want it enabled in a new thread, the context manager or decorator
    must be invoked in that thread.  This affects :class:`torch.nn.DataParallel` and
    :class:`torch.nn.parallel.DistributedDataParallel` when used with more than one GPU per process
    (see :ref:`Working with Multiple GPUs<amp-multigpu>`).

    Args:
        device_type(string, required):  Whether to use 'cuda' or 'cpu' device
        enabled(bool, optional, default=True):  Whether autocasting should be enabled in the region.
        dtype(torch_dtype, optional):  Whether to use torch.float16 or torch.bfloat16.
        cache_enabled(bool, optional, default=True):  Whether the weight cache inside autocast should be enabled.
    """
    def __init__(self, device_type, enabled=True, **kwargs):
        self.device = device_type
        if self.device == 'cuda':
            self.fast_dtype = torch.get_autocast_gpu_dtype()
        elif self.device == 'cpu':
            self.fast_dtype = torch.get_autocast_cpu_dtype()
        else:
            raise RuntimeError('User specified autocast device_type must be \'cuda\' or \'cpu\'')
        self._cache_enabled = torch.is_autocast_cache_enabled()
        if torch.cuda.amp.common.amp_definitely_not_available() and self.device == 'cuda':
            warnings.warn('User provided device_type of \'cuda\', but CUDA is not available. Disabling')
            enabled = False
        for key, value in kwargs.items():
            if key == 'dtype':
                self.fast_dtype = value
            if key == 'cache_enabled':
                self._cache_enabled = value
            if not ((key == 'dtype') or (key == 'cache_enabled')):
                raise RuntimeError('Unrecognized optional argument supplied to autocast context manager: ' + str(key))

        if self.device == 'cpu':
            supported_dtype = [torch.bfloat16]
            if self.fast_dtype not in supported_dtype:
                error_message = 'In CPU autocast, but the target dtype is not supported. Disabling autocast.\n'
                error_message += 'CPU Autocast only supports dtype of torch.bfloat16 currently.'
                warnings.warn(error_message)
                enabled = False
        if self.device == 'cuda':
            if self.fast_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                raise RuntimeError('Current CUDA Device does not support bfloat16. Please switch dtype to float16.')
        self._enabled = enabled

    def __enter__(self):
        self.prev_cache_enabled = torch.is_autocast_cache_enabled()
        if self.device == 'cpu':
            self.prev = torch.is_autocast_cpu_enabled()
            self.prev_fastdtype = torch.get_autocast_cpu_dtype()
            torch.set_autocast_cpu_enabled(self._enabled)
            torch.set_autocast_cpu_dtype(self.fast_dtype)
            torch.autocast_increment_nesting()
        else:
            self.prev = torch.is_autocast_enabled()
            self.prev_fastdtype = torch.get_autocast_gpu_dtype()
            torch.set_autocast_gpu_dtype(self.fast_dtype)
            torch.set_autocast_enabled(self._enabled)
            torch.autocast_increment_nesting()
        torch.set_autocast_cache_enabled(self._cache_enabled)

    def __exit__(self, *args):
        # Drop the cache when we exit to a nesting level that's outside any instance of autocast.
        if self.device == 'cpu':
            if torch.autocast_decrement_nesting() == 0:
                torch.clear_autocast_cache()
            torch.set_autocast_cpu_enabled(self.prev)
            torch.set_autocast_cpu_dtype(self.prev_fastdtype)
        else:
            if torch.autocast_decrement_nesting() == 0:
                torch.clear_autocast_cache()
            torch.set_autocast_enabled(self.prev)
            torch.set_autocast_gpu_dtype(self.prev_fastdtype)
        torch.set_autocast_cache_enabled(self.prev_cache_enabled)
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_autocast(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorate_autocast
