from __future__ import annotations

from typing import Any, TYPE_CHECKING
from typing_extensions import Self

import torch
from torch import Tensor
from torch.autograd import Function


if TYPE_CHECKING:
    from torch._ops import OpOverload
    from torch._prims_common import DeviceLikeType
    from torch.autograd.function import FunctionCtx


class ComplexTensor(Tensor):
    """A class that decomposes all ops on complex Tensors into their real and imaginary parts."""

    _re: Tensor
    _im: Tensor

    # NOTE: aliasing
    # ComplexTensor intentionally does not preserve aliasing between the
    # wrapper and its real/imag parts (e.g. view_as_real produces a copy via
    # torch.stack, not a true view). This is safe because the pass runs
    # post-functionalization -- there are no mutations in the graph -- and
    # Inductor will re-establish any required aliasing in the final output.
    # For the same reason we do not track conj/neg bit flags on the wrapper:
    # conj and neg are decomposed at the op level into arithmetic on the
    # real/imag parts.

    def __new__(
        cls,
        real: Tensor,
        imag: Tensor,
        /,
    ) -> Self:
        """Initialize a ComplexTensor from its real and imaginary parts."""
        from ._ops.common import REAL_TO_COMPLEX

        # TODO (hameerabbasi): `torch.compile` sometimes fails here without making these
        # contiguous. Why?
        real = real.contiguous()
        imag = imag.contiguous()

        dtype = REAL_TO_COMPLEX.get(real.dtype)
        if dtype is None:
            raise TypeError(
                "Unsupported dtype for constituent tensors. Supported dtypes are: "
                f"{set(REAL_TO_COMPLEX.keys())!r}."
            )
        storage_offset = real.storage_offset()
        strides = real.stride()
        layout = real.layout
        pin_memory = real.is_pinned()

        if real.shape != imag.shape:
            raise AssertionError(f"Expected imag shape {real.shape}, got {imag.shape}")
        if real.device != imag.device:
            raise AssertionError(
                f"Expected imag device {real.device}, got {imag.device}"
            )
        if real.dtype != imag.dtype:
            raise AssertionError(f"Expected imag dtype {real.dtype}, got {imag.dtype}")
        if real.is_pinned() != imag.is_pinned():
            raise AssertionError(
                f"Expected imag pinning {real.is_pinned()}, got {imag.is_pinned()}"
            )

        res = Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            real.shape,
            device=real.device,
            dtype=dtype,
            storage_offset=storage_offset,
            strides=strides,
            pin_memory=pin_memory,
            layout=layout,
            requires_grad=False,
        )
        res._re = real.detach()
        res._im = imag.detach()

        return res

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__()

    @property
    def re(self) -> Tensor:
        return self._re

    @property
    def im(self) -> Tensor:
        return self._im

    @property
    def real(self) -> Tensor:  # type: ignore[bad-override]
        return self.re

    @real.setter
    def real(self, value: Tensor) -> None:
        self.re[...] = value

    @property
    def imag(self) -> Tensor:  # type: ignore[bad-override]
        return self.im

    @imag.setter
    def imag(self, value: Tensor) -> None:
        self.im[...] = value

    @classmethod
    def __torch_dispatch__(  # type: ignore[bad-override]
        cls,
        func: OpOverload,
        types: tuple[type, ...],
        # pyrefly: ignore [implicit-any]
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        from ._ops.common import lookup_complex

        kwargs = {} if kwargs is None else kwargs

        impl = lookup_complex(func, *args, **kwargs)
        if impl is None:
            return NotImplemented

        return impl(*args, **kwargs)

    @staticmethod
    def from_interleaved(t: Tensor) -> ComplexTensor:
        re = torch.real(t)
        im = torch.imag(t) if t.dtype.is_complex else torch.zeros_like(t)
        return Complex.apply(re, im)

    def as_interleaved(self) -> Tensor:
        return torch.complex(self.re, self.im)

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, Tensor],
        meta: Any,
        outer_size: tuple[int, ...],
        outer_stride: tuple[int, ...],
    ) -> ComplexTensor:
        re, im = inner_tensors["_re"], inner_tensors["_im"]
        return ComplexTensor(re, im)

    def __tensor_flatten__(self) -> tuple[list[str], Any]:
        return ["_re", "_im"], None

    def __repr__(self, *, tensor_contents: object | None = None) -> str:
        return f"ComplexTensor({self._re!r}, {self._im!r})"

    def is_pinned(self, device: DeviceLikeType | None = None) -> bool:
        return self._re.is_pinned(device)


class Complex(Function):
    @staticmethod
    def forward(  # type: ignore[bad-override]
        ctx: FunctionCtx, real: Tensor, imag: Tensor
    ) -> ComplexTensor:
        return ComplexTensor(real, imag)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: ComplexTensor) -> tuple[Tensor, Tensor]:  # type: ignore[bad-override]
        return grad_output.real, grad_output.imag
