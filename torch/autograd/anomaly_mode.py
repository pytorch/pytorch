# mypy: allow-untyped-defs
r"""Autograd anomaly mode."""

import warnings

import torch


__all__ = ["detect_anomaly", "set_detect_anomaly"]


class detect_anomaly:
    r"""Context-manager that enable anomaly detection for the autograd engine.

    This does two things:

    - Running the forward pass with detection enabled will allow the backward
      pass to print the traceback of the forward operation that created the failing
      backward function.
    - If ``check_nan`` is ``True``, any backward computation that generate "nan"
      value will raise an error. Default ``True``.
    - If ``mixed_stack`` is ``True``, the stack traces will show the combined Python/C++
      traceback.

    .. warning::
        This mode should be enabled only for debugging as the different tests
        will slow down your program execution.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ANOMALY)
        >>> import torch
        >>> from torch import autograd
        >>> class MyFunc(autograd.Function):
        ...     @staticmethod
        ...     def forward(ctx, inp):
        ...         return inp.clone()
        ...
        ...     @staticmethod
        ...     def backward(ctx, gO):
        ...         # Error during the backward pass
        ...         raise RuntimeError("Some error in backward")
        ...         return gO.clone()
        >>> def run_fn(a):
        ...     out = MyFunc.apply(a)
        ...     return out.sum()
        >>> inp = torch.rand(10, 10, requires_grad=True)
        >>> out = run_fn(inp)
        >>> out.backward()
            Traceback (most recent call last):
              File "<stdin>", line 1, in <module>
              File "/your/pytorch/install/torch/_tensor.py", line 93, in backward
                torch.autograd.backward(self, gradient, retain_graph, create_graph)
              File "/your/pytorch/install/torch/autograd/__init__.py", line 90, in backward
                allow_unreachable=True)  # allow_unreachable flag
              File "/your/pytorch/install/torch/autograd/function.py", line 76, in apply
                return self._forward_cls.backward(self, *args)
              File "<stdin>", line 8, in backward
            RuntimeError: Some error in backward
        >>> with autograd.detect_anomaly():
        ...     inp = torch.rand(10, 10, requires_grad=True)
        ...     out = run_fn(inp)
        ...     out.backward()
            Traceback of forward call that caused the error:
              File "tmp.py", line 53, in <module>
                out = run_fn(inp)
              File "tmp.py", line 44, in run_fn
                out = MyFunc.apply(a)
            Traceback (most recent call last):
              File "<stdin>", line 4, in <module>
              File "/your/pytorch/install/torch/_tensor.py", line 93, in backward
                torch.autograd.backward(self, gradient, retain_graph, create_graph)
              File "/your/pytorch/install/torch/autograd/__init__.py", line 90, in backward
                allow_unreachable=True)  # allow_unreachable flag
              File "/your/pytorch/install/torch/autograd/function.py", line 76, in apply
                return self._forward_cls.backward(self, *args)
              File "<stdin>", line 8, in backward
            RuntimeError: Some error in backward

    """

    def __init__(self, check_nan: bool = True, mixed_stack: bool = False) -> None:  # noqa: D107
        self.prev = torch.is_anomaly_enabled()
        self.check_nan = check_nan
        self.mixed_stack = mixed_stack
        self.prev_check_nan = torch.is_anomaly_check_nan_enabled()
        self.prev_mixed_stack = torch.is_anomaly_mixed_stack_enabled()
        # If we don't check nan and use mixed stack, the overhead is minimal, so removing
        # the warning in that case.
        if not mixed_stack and check_nan:
            warnings.warn(
                "Anomaly Detection has been enabled. "
                "This mode will increase the runtime "
                "and should only be enabled for debugging.",
                stacklevel=2,
            )

    def __enter__(self) -> None:  # noqa: D105
        torch.set_anomaly_enabled(True, self.check_nan, self.mixed_stack)

    def __exit__(self, *args: object) -> None:  # noqa: D105
        torch.set_anomaly_enabled(self.prev, self.prev_check_nan, self.prev_mixed_stack)


class set_detect_anomaly:
    r"""Context-manager that sets the anomaly detection for the autograd engine on or off.

    ``set_detect_anomaly`` will enable or disable the autograd anomaly detection
    based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    See ``detect_anomaly`` above for details of the anomaly detection behaviour.

    Args:
        mode (bool): Flag whether to enable anomaly detection (``True``),
                     or disable (``False``).
        check_nan (bool): Flag whether to raise an error when the backward
                          generate "nan"
        mixed_stack (bool): Flag whether to capture forward stack traces using
                            the combined Python/C++ traceback format.

    """

    def __init__(
        self, mode: bool, check_nan: bool = True, mixed_stack: bool = False
    ) -> None:  # noqa: D107
        self.prev = torch.is_anomaly_enabled()
        self.prev_check_nan = torch.is_anomaly_check_nan_enabled()
        self.prev_mixed_stack = torch.is_anomaly_mixed_stack_enabled()
        torch.set_anomaly_enabled(mode, check_nan, mixed_stack)

    def __enter__(self) -> None:  # noqa: D105
        pass

    def __exit__(self, *args: object) -> None:  # noqa: D105
        torch.set_anomaly_enabled(self.prev, self.prev_check_nan, self.prev_mixed_stack)
