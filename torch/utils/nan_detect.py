"""Forward-pass NaN/Inf detection via TorchDispatchMode.

Usage::

    with torch.utils.nan_detect.NanDetectMode():
        out = model(x)
    # RuntimeError raised immediately when any op produces NaN

Based on the prototype at https://github.com/albanD/subclass_zoo/blob/main/nan_detect.py
"""

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


class NanDetectMode(TorchDispatchMode):
    """Detect NaN (and optionally Inf) in the output of every operation.

    When enabled, every ATen operation is followed by a check of its outputs.
    If any floating-point output tensor contains NaN (or non-finite values when
    ``check_inf=True``), a ``RuntimeError`` is raised immediately with the name
    of the offending operation.

    This complements :func:`torch.autograd.detect_anomaly`, which only checks
    for NaN during the backward pass.

    Args:
        check_inf (bool): If ``True``, also raise on ``±Inf`` values.
            Default: ``False`` (only check for NaN).

    Example::

        >>> # xdoctest: +SKIP(NanDetectMode raises on NaN)
        >>> with NanDetectMode():
        ...     x = torch.tensor([1.0, float('nan')])
        ...     y = x + 1  # raises RuntimeError
    """

    def __init__(self, *, check_inf: bool = False) -> None:
        super().__init__()
        self.check_inf = check_inf

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        res = func(*args, **kwargs)
        flat_res, _ = tree_flatten(res)
        for t in flat_res:
            if not isinstance(t, torch.Tensor):
                continue
            if not t.is_floating_point() or t.numel() == 0:
                continue
            try:
                if self.check_inf:
                    if not torch.isfinite(t).all():
                        raise RuntimeError(
                            f"Function {func} returned non-finite values"
                        )
                elif torch.isnan(t).any():
                    raise RuntimeError(f"Function {func} returned NaN values")
            except NotImplementedError:
                pass
        return res
