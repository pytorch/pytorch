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

    def __new__(
        cls,
        real: Tensor,
        imag: Tensor,
        /,
        *,
        neg_flag: bool = False,
        conj_flag: bool = False,
    ) -> Self:
        """Initialize a ComplexTensor from its real and imaginary parts."""
        from ._ops.common import REAL_TO_COMPLEX

        # TODO (hameerabbasi): `torch.compile` sometimes fails here without making these
        # contiguous. Why?
        real = real.contiguous()
        imag = imag.contiguous()

        if real.is_neg():
            neg_flag = not neg_flag
            conj_flag = not conj_flag
            real = torch._neg_view(real)

        if imag.is_neg():
            conj_flag = not conj_flag
            imag = torch._neg_view(imag)


        # TODO (hameerabbasi):
        # What should we do with dtype?
        # We could convert to the complex type (float32 -> complex64), but we
        # can't use that model for say `bfloat16` which does not have a
        # corresponding complex dtype.
        # If we want to support this complex rep using any float type (see
        # https://github.com/pytorch/pytorch/issues/95100)
        # We either need to:
        # 1) add the complex types for say `complexbf32`, knowing they can't really be used anywhere
        #    else.
        # 2) We use the real float dtype here, and it is up to the user to know
        #    that dtype=float<size> here really means complex<2xSize> with dtype
        #    matching that of re/im parts alone
        # I'm going with 1 for now, so that I can make gradcheck and some complex
        # ops work properly, but might want to discuss this in the RFP.
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
            raise AssertionError(f"Expected imag device {real.device}, got {imag.device}")
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
        torch._C._set_conj(res, conj_flag)
        torch._C._set_neg(res, neg_flag)

        return res

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__()

    @property
    def re(self) -> Tensor:
        negate = self.is_neg()
        real = self._re
        if negate:
            real = torch._neg_view(real)
        return real

    @property
    def im(self) -> Tensor:
        negate = self.is_neg() != self.is_conj()
        imag = self._im
        if negate:
            imag = torch._neg_view(imag)
        return imag

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
        return ComplexTensor(
            re, im, neg_flag=meta["neg_flag"], conj_flag=meta["conj_flag"]
        )

    def __tensor_flatten__(self) -> tuple[list[str], Any]:
        return ["_re", "_im"], {"neg_flag": self.is_neg(), "conj_flag": self.is_conj()}

    def __repr__(self, *, tensor_contents: object | None = None) -> str:
        return f"ComplexTensor({self._re!r}, {self._im!r}, conj_flag={self.is_conj()!r}, neg_flag={self.is_neg()!r})"

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
