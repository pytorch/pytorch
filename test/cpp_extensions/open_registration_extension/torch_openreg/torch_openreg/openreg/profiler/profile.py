import torch
from torch.autograd.profiler import (
    _disable_profiler,
    _enable_profiler,
    _run_on_profiler_start,
    _run_on_profiler_stop,
    ProfilerConfig,
    ProfilerState,
    _ExperimentalConfig,
)

class emit_openreg:
    """Context manager that makes every autograd operation emit a PRIVATEUSE1 ("OpenReg") range.

    OpenReg here is a *simulated* accelerator that piggybacks on CPU timing; it does **not** have
    its own native profiler tooling. This class exists as an integration example showing how a
    third-party / custom device backend could hook into the autograd profiler via
    ``ProfilerState.PRIVATEUSE1``. The emitted ranges can be viewed using the standard PyTorch
    profiler APIs (e.g. ``torch.profiler.profile``) and will appear with the PRIVATEUSE1 state.

    Compared to real accelerator integrations (e.g. ``emit_nvtx`` or ``emit_itt``), this context
    manager:
    - Records timestamps using the approximate CPU clock.
    - Optionally captures input tensor shapes for each op if ``record_shapes=True``.
    - Does not emit hardware-specific markers or synchronize any real device stream; any
      synchronization is a no-op placeholder unless a concrete backend is registered as
      ``privateuseone``.

    .. note:: Use this as a template: replace the placeholder calls and extend the C++ observer
              (see the OpenReg observer implementation) when adding a genuine accelerator.

    .. warning::
        This context manager is not re-entrant; nesting will raise an error.

    Args:
        enabled (bool, optional): If ``False`` the context is a no-op. Default: ``True``.
        record_shapes (bool, optional): If ``True``, each emitted range stores a list of input
            tensor dimension sizes: ``[[arg0.dim0, arg0.dim1, ...], [...], ...]``. Non-tensor or
            non-shape-bearing arguments are ``[]``. Ordering reflects backend operator argument
            order and may differ from Python call order. Shape collection adds minor overhead.

    Example:
        >>> # xdoctest: +SKIP("undefined variables")
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD_PROFILER)
        >>> with torch.openreg.profiler.emit_openreg(record_shapes=True):
        ...     model(x)
    """

    def __init__(self, enabled=True, record_shapes=False):
        self.enabled = enabled
        self.entered = False
        self.record_shapes = record_shapes

    def __enter__(self):
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError("OpenReg annotation context manager is not reentrant")
        self.entered = True
        if hasattr(torch, "openreg"):
            torch.accelerator.synchronize()
        _run_on_profiler_start()
        _enable_profiler(
            ProfilerConfig(
                ProfilerState.PRIVATEUSE1,
                self.record_shapes,
                False,
                False,
                False,
                False,
                _ExperimentalConfig(),
            ),
            set(),
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        if hasattr(torch, "openreg"):
            torch.accelerator.synchronize()
        _disable_profiler()
        _run_on_profiler_stop()
        return False
