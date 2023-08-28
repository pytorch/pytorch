import functools
import warnings

from typing import Any, Optional

import torch
from torch.types import _dtype

__all__ = ["autocast_decorator", "autocast"]


def autocast_decorator(autocast_instance, func):
    @functools.wraps(func)
    def decorate_autocast(*args, **kwargs):
        with autocast_instance:
            return func(*args, **kwargs)

    decorate_autocast.__script_unsupported = "@autocast() decorator is not supported in script mode"  # type: ignore[attr-defined]
    return decorate_autocast


class autocast:
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
            with torch.autocast(device_type="cuda"):
                output = model(input)
                loss = loss_fn(output, target)

            # Exits the context manager before backward()
            loss.backward()
            optimizer.step()

    See the :ref:`CUDA Automatic Mixed Precision examples<amp-examples>` for usage (along with gradient scaling)
    in more complex scenarios (e.g., gradient penalty, multiple models/losses, custom autograd functions).

    :class:`autocast` can also be used as a decorator, e.g., on the ``forward`` method of your model::

        class AutocastModel(nn.Module):
            ...
            @torch.autocast(device_type="cuda")
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

        with torch.autocast(device_type="cuda"):
            # torch.mm is on autocast's list of ops that should run in float16.
            # Inputs are float32, but the op runs in float16 and produces float16 output.
            # No manual casts are required.
            e_float16 = torch.mm(a_float32, b_float32)
            # Also handles mixed input types
            f_float16 = torch.mm(d_float32, e_float16)

        # After exiting autocast, calls f_float16.float() to use with d_float32
        g_float32 = torch.mm(d_float32, f_float16.float())

    CPU Training Example::

        # Creates model and optimizer in default precision
        model = Net()
        optimizer = optim.SGD(model.parameters(), ...)

        for epoch in epochs:
            for input, target in data:
                optimizer.zero_grad()

                # Runs the forward pass with autocasting.
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    output = model(input)
                    loss = loss_fn(output, target)

                loss.backward()
                optimizer.step()


    CPU Inference Example::

        # Creates model in default precision
        model = Net().eval()

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            for input in data:
                # Runs the forward pass with autocasting.
                output = model(input)

    CPU Inference Example with Jit Trace::

        class TestModel(nn.Module):
            def __init__(self, input_size, num_classes):
                super().__init__()
                self.fc1 = nn.Linear(input_size, num_classes)
            def forward(self, x):
                return self.fc1(x)

        input_size = 2
        num_classes = 2
        model = TestModel(input_size, num_classes).eval()

        # For now, we suggest to disable the Jit Autocast Pass,
        # As the issue: https://github.com/pytorch/pytorch/issues/75956
        torch._C._jit_set_autocast_mode(False)

        with torch.cpu.amp.autocast(cache_enabled=False):
            model = torch.jit.trace(model, torch.randn(1, input_size))
        model = torch.jit.freeze(model)
        # Models Run
        for _ in range(3):
            model(torch.randn(1, input_size))

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

        with torch.autocast(device_type="cuda"):
            e_float16 = torch.mm(a_float32, b_float32)
            with torch.autocast(device_type="cuda", enabled=False):
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
        device_type(str, required):  Device type to use. Possible values are: 'cuda', 'cpu', 'xpu' and 'hpu'.
                                     The type is the same as the `type` attribute of a :class:`torch.device`.
                                     Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
        enabled(bool, optional):  Whether autocasting should be enabled in the region.
            Default: ``True``
        dtype(torch_dtype, optional):  Whether to use torch.float16 or torch.bfloat16.
        cache_enabled(bool, optional):  Whether the weight cache inside autocast should be enabled.
            Default: ``True``
    """

    def __init__(
        self,
        device_type: str,
        dtype: Optional[_dtype] = None,
        enabled: bool = True,
        cache_enabled: Optional[bool] = None,
    ):
        if torch._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = device_type
            self.fast_dtype = dtype
            # TODO: support get_autocast_gpu/cpu_dtype
            assert dtype is not None
            return
        self.device = device_type
        self.custom_backend_name = torch._C._get_privateuse1_backend_name()
        if self.device == "cuda":
            self.fast_dtype = torch.get_autocast_gpu_dtype()
        elif self.device == "cpu":
            self.fast_dtype = torch.get_autocast_cpu_dtype()
        elif self.device == "xpu":
            self.fast_dtype = torch.xpu.get_autocast_xpu_dtype()  # type: ignore[attr-defined]
        elif self.device == "ipu":
            self.fast_dtype = torch.get_autocast_ipu_dtype()  # type: ignore[attr-defined]
        elif self.device == "hpu":
            self.fast_dtype = torch.hpu.get_autocast_hpu_dtype()  # type: ignore[attr-defined]
        elif self.device == "xla":
            self.fast_dtype = torch.get_autocast_xla_dtype()  # type: ignore[attr-defined]
        elif self.device == self.custom_backend_name:
            necessary_funcs = [
                "is_autocast_enabled",
                "set_autocast_enabled",
                "get_autocast_dtype",
                "set_autocast_dtype",
                "get_amp_supported_dtype",
            ]
            message = f"Tried to use AMP with the `{self.custom_backend_name}` backend, but the backend has not "
            message += "registered a module or  the module miss some necessary funcs. The backend should register "
            message += "a module by `torch._register_device_module`, and the module must have these funcs: \n"
            message += "`is_autocast_enabled() -> bool`, `set_autocast_enabled(bool) -> None`, "
            message += "`get_autocast_dtype() -> torch.dtype`, `set_autocast_dtype(torch.dtype) "
            message += (
                "-> None` and `get_amp_supported_dtype() -> List[torch.dtype]`. \n"
            )

            assert hasattr(torch, self.custom_backend_name), message
            self.custom_device_mod = getattr(torch, self.custom_backend_name)
            for func in necessary_funcs:
                assert hasattr(self.custom_device_mod, func), (
                    message + f"But the func `{func}` is missing. \n"
                )

            self.fast_dtype = self.custom_device_mod.get_autocast_dtype()
        else:
            raise RuntimeError(
                f"User specified an unsupported autocast device_type '{self.device}'"
            )
        self._cache_enabled = torch.is_autocast_cache_enabled()
        if (
            enabled
            and torch.cuda.amp.common.amp_definitely_not_available()
            and self.device == "cuda"
        ):
            warnings.warn(
                "User provided device_type of 'cuda', but CUDA is not available. Disabling"
            )
            enabled = False
        if dtype is not None:
            self.fast_dtype = dtype
        if cache_enabled is not None:
            self._cache_enabled = cache_enabled

        if self.device == "cpu":
            supported_dtype = [torch.bfloat16]
            if self.fast_dtype not in supported_dtype:
                error_message = "In CPU autocast, but the target dtype is not supported. Disabling autocast.\n"
                error_message += (
                    "CPU Autocast only supports dtype of torch.bfloat16 currently."
                )
                warnings.warn(error_message)
                enabled = False
        elif self.device == "xpu":
            supported_dtype = [torch.bfloat16, torch.float16]
            if self.fast_dtype not in supported_dtype:
                error_message = "In XPU autocast, but the target dtype is not supported. Disabling autocast.\n"
                error_message += "XPU Autocast only supports dtypes of torch.bfloat16 and torch.float16 currently."
                warnings.warn(error_message)
                enabled = False
        elif self.device == "ipu":
            supported_dtypes = [torch.bfloat16, torch.float16]
            if self.fast_dtype not in supported_dtypes:
                error_message = "In IPU autocast, but the target dtype is not supported. Disabling autocast.\n"
                error_message += "IPU Autocast only supports dtypes of torch.bfloat16 and torch.float16 currently."
                warnings.warn(error_message)
                enabled = False
        elif self.device == "hpu":
            supported_dtype = [torch.bfloat16, torch.float16]
            if self.fast_dtype not in supported_dtype:
                error_message = "In HPU autocast, but the target dtype is not supported. Disabling autocast.\n"
                error_message += "HPU Autocast only supports dtypes of torch.bfloat16 and torch.float16 currently."
                warnings.warn(error_message)
                enabled = False
        elif self.device == self.custom_backend_name:
            supported_dtype = self.custom_device_mod.get_amp_supported_dtype()
            if self.fast_dtype not in supported_dtype:
                error_message = f"In {self.custom_backend_name} autocast, but the target dtype is not supported. "
                error_message += f"Disabling autocast.\n {self.custom_backend_name} Autocast only supports dtypes of "
                error_message += (
                    ", ".join(str(dtype) for dtype in supported_dtype) + " currently."
                )
                warnings.warn(error_message)
                enabled = False
        elif self.device == "cuda":
            if (
                enabled
                and self.fast_dtype == torch.bfloat16
                and not torch.cuda.is_bf16_supported()
            ):
                raise RuntimeError(
                    "Current CUDA Device does not support bfloat16. Please switch dtype to float16."
                )
        elif self.device == "xla":
            supported_dtype = [torch.bfloat16, torch.float16]
            if self.fast_dtype not in supported_dtype:
                error_message = "In XLA autocast, but the target dtype is not supported. Disabling autocast.\n"
                error_message += (
                    "XLA Autocast only supports dtype of torch.bfloat16 and torch.float16 currently."
                )
                warnings.warn(error_message)
                enabled = False
        self._enabled = enabled

    def __enter__(self):
        if torch._jit_internal.is_scripting():
            assert self.fast_dtype is not None
            return self

        self.prev_cache_enabled = torch.is_autocast_cache_enabled()
        if self.device == "cpu":
            self.prev = torch.is_autocast_cpu_enabled()
            self.prev_fastdtype = torch.get_autocast_cpu_dtype()
            torch.set_autocast_cpu_enabled(self._enabled)
            torch.set_autocast_cpu_dtype(self.fast_dtype)  # type: ignore[arg-type]
            torch.autocast_increment_nesting()
        elif self.device == "xpu":
            self.prev = torch.xpu.is_autocast_xpu_enabled()  # type: ignore[attr-defined]
            self.prev_fastdtype = torch.xpu.get_autocast_xpu_dtype()  # type: ignore[attr-defined]
            torch.xpu.set_autocast_xpu_enabled(self._enabled)  # type: ignore[attr-defined]
            torch.xpu.set_autocast_xpu_dtype(self.fast_dtype)  # type: ignore[attr-defined]
            torch.autocast_increment_nesting()
        elif self.device == "ipu":
            self.prev = torch.is_autocast_ipu_enabled()  # type: ignore[attr-defined]
            self.prev_fastdtype = torch.get_autocast_ipu_dtype()  # type: ignore[attr-defined]
            torch.set_autocast_ipu_enabled(self._enabled)  # type: ignore[attr-defined]
            torch.set_autocast_ipu_dtype(self.fast_dtype)  # type: ignore[attr-defined]
            torch.autocast_increment_nesting()
        elif self.device == "hpu":
            self.prev = torch.hpu.is_autocast_hpu_enabled()  # type: ignore[attr-defined]
            self.prev_fastdtype = torch.hpu.get_autocast_hpu_dtype()  # type: ignore[attr-defined]
            torch.hpu.set_autocast_hpu_enabled(self._enabled)  # type: ignore[attr-defined]
            torch.hpu.set_autocast_hpu_dtype(self.fast_dtype)  # type: ignore[attr-defined]
            torch.autocast_increment_nesting()
        elif self.device == "xla":
            self.prev = torch.is_autocast_xla_enabled()  # type: ignore[attr-defined]
            self.prev_fastdtype = torch.get_autocast_xla_dtype()  # type: ignore[attr-defined]
            torch.set_autocast_xla_enabled(self._enabled)  # type: ignore[attr-defined]
            torch.set_autocast_xla_dtype(self.fast_dtype)  # type: ignore[attr-defined]
            torch.autocast_increment_nesting()
        elif self.device == self.custom_backend_name:
            self.prev = self.custom_device_mod.is_autocast_enabled()
            self.prev_fastdtype = self.custom_device_mod.get_autocast_dtype()
            self.custom_device_mod.set_autocast_enabled(self._enabled)
            self.custom_device_mod.set_autocast_dtype(self.fast_dtype)
            torch.autocast_increment_nesting()
        else:
            self.prev = torch.is_autocast_enabled()
            self.prev_fastdtype = torch.get_autocast_gpu_dtype()
            torch.set_autocast_gpu_dtype(self.fast_dtype)  # type: ignore[arg-type]
            torch.set_autocast_enabled(self._enabled)
            torch.autocast_increment_nesting()
        torch.set_autocast_cache_enabled(self._cache_enabled)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        if torch._jit_internal.is_scripting():
            return

        # Drop the cache when we exit to a nesting level that's outside any instance of autocast.
        if self.device == "cpu":
            if torch.autocast_decrement_nesting() == 0:
                torch.clear_autocast_cache()
            torch.set_autocast_cpu_enabled(self.prev)
            torch.set_autocast_cpu_dtype(self.prev_fastdtype)
        elif self.device == "xpu":
            if torch.autocast_decrement_nesting() == 0:
                torch.clear_autocast_cache()
            torch.xpu.set_autocast_xpu_enabled(self.prev)  # type: ignore[attr-defined]
            torch.xpu.set_autocast_xpu_dtype(self.prev_fastdtype)  # type: ignore[attr-defined]
        elif self.device == "ipu":
            if torch.autocast_decrement_nesting() == 0:
                torch.clear_autocast_cache()
            torch.set_autocast_ipu_enabled(self.prev)  # type: ignore[attr-defined]
            torch.set_autocast_ipu_dtype(self.prev_fastdtype)  # type: ignore[attr-defined]
        elif self.device == "hpu":
            if torch.autocast_decrement_nesting() == 0:
                torch.clear_autocast_cache()
            torch.hpu.set_autocast_hpu_enabled(self.prev)  # type: ignore[attr-defined]
            torch.hpu.set_autocast_hpu_dtype(self.prev_fastdtype)  # type: ignore[attr-defined]
        elif self.device == "xla":
            if torch.autocast_decrement_nesting() == 0:
                torch.clear_autocast_cache()
            torch.set_autocast_xla_enabled(self.prev)  # type: ignore[attr-defined]
            torch.set_autocast_xla_dtype(self.prev_fastdtype)  # type: ignore[attr-defined]
        elif self.device == self.custom_backend_name:
            if torch.autocast_decrement_nesting() == 0:
                torch.clear_autocast_cache()
            self.custom_device_mod.set_autocast_enabled(self.prev)
            self.custom_device_mod.set_autocast_dtype(self.prev_fastdtype)
        else:
            if torch.autocast_decrement_nesting() == 0:
                torch.clear_autocast_cache()
            torch.set_autocast_enabled(self.prev)
            torch.set_autocast_gpu_dtype(self.prev_fastdtype)
        torch.set_autocast_cache_enabled(self.prev_cache_enabled)
        return False

    def __call__(self, func):
        if torch._jit_internal.is_scripting():
            return func
        return autocast_decorator(self, func)


# These functions aren't meant for public usage.
# They are what we trace into a graph during pre_dispatch tracing
# when we encounter an autocast context manager.
def _enter_autocast(*vals):
    # For pre-dispatch tracing, if a TorchFunction mode is active, we'll want to trace this into a graph.
    if torch._C._is_torch_function_mode_enabled():
        return torch.overrides.handle_torch_function(
            torch.amp._enter_autocast, [], *vals
        )
    mode = torch.amp.autocast(*vals)
    mode.__enter__()
    return mode


def _exit_autocast(mode):
    if torch._C._is_torch_function_mode_enabled():
        return torch.overrides.handle_torch_function(torch.amp._exit_autocast, [], mode)
    mode.__exit__(None, None, None)
