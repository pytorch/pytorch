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
    ) -> Self:
        """Initialize a ComplexTensor from its real and imaginary parts."""
        from ._ops.common import REAL_TO_COMPLEX

        # TODO (hameerabbasi): `torch.compile` sometimes fails here without making these
        # contiguous. Why?
        real = real.contiguous()
        imag = imag.contiguous()

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
            raise AssertionError(
                f"Expected imag device {real.device}, got {imag.device}"
            )
        if real.dtype != imag.dtype:
            raise AssertionError(f"Expected imag dtype {real.dtype}, got {imag.dtype}")
        if real.is_pinned() != imag.is_pinned():
            raise AssertionError(
                f"Expected imag pinning {real.is_pinned()}, got {imag.is_pinned()}"
            )

        neg = real.is_neg()
        conj = neg != imag.is_neg()
        if neg:
            real = torch.ops.aten._neg_view(real)
        if imag.is_neg():
            imag = torch.ops.aten._neg_view(imag)

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
        torch._C._set_neg(res, neg)
        torch._C._set_conj(res, conj)

        return res

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__()

    @property
    def re(self) -> Tensor:
        if self.is_neg():
            return torch._neg_view(self._re)

        return self._re

    @property
    def im(self) -> Tensor:
        if self.is_neg() != self.is_conj():
            return torch._neg_view(self._im)

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
        out = ComplexTensor(re, im)
        torch._C._set_conj(out, meta["conj"])
        torch._C._set_neg(out, meta["neg"])
        return out

    def __tensor_flatten__(self) -> tuple[list[str], Any]:
        return ["_re", "_im"], {"conj": self.is_conj(), "neg": self.is_neg()}

    def __repr__(self, *, tensor_contents: object | None = None) -> str:
        return f"ComplexTensor({self._re!r}, {self._im!r})"

    def is_pinned(self, device: DeviceLikeType | None = None) -> bool:
        return self._re.is_pinned(device)


class InterleavedComplexTensor(Tensor):
    """A complex Tensor that keeps the interleaved layout, viewing the original input's storage.

    Ops are computed via `ComplexTensor`; mutations are written back into the original storage.
    """

    _data: Tensor

    def __new__(cls, t: Tensor, /) -> Self:
        from ._ops.common import COMPLEX_TO_REAL

        if t.dtype not in COMPLEX_TO_REAL:
            raise TypeError(
                f"Unsupported dtype {t.dtype}. Supported dtypes are: {set(COMPLEX_TO_REAL.keys())!r}."
            )
        t = torch.ops.aten.alias(t)

        neg = False
        conj = False

        if t.is_neg():
            neg = True
            t = torch.ops.aten._neg_view(t)

        if t.is_conj():
            conj = True
            t = torch.ops.aten._conj(t)

        res = Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            t.shape,
            device=t.device,
            dtype=t.dtype,
            storage_offset=t.storage_offset(),
            strides=t.stride(),
            pin_memory=t.is_pinned(),
            layout=t.layout,
            requires_grad=False,
        )
        res._data = torch.view_as_real(t.detach())
        torch._C._set_neg(res, neg)
        torch._C._set_conj(res, conj)
        return res

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__()

    @property
    def re(self) -> Tensor:
        if self.is_neg():
            return torch._neg_view(self._data[..., 0])

        return self._data[..., 0]

    @property
    def im(self) -> Tensor:
        if self.is_neg() != self.is_conj():
            return torch._neg_view(self._data[..., 1])

        return self._data[..., 1]

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

    def as_interleaved(self) -> Tensor:
        out = torch.view_as_complex(self._data)
        if self.is_neg():
            out = torch.ops.aten._neg_view(out)
        if self.is_conj():
            out = torch.ops.aten._conj(out)
        return out

    def to_complex_tensor(self) -> ComplexTensor:
        return ComplexTensor(self.re, self.im)

    @classmethod
    def __torch_dispatch__(  # type: ignore[bad-override]
        cls,
        func: OpOverload,
        types: tuple[type, ...],
        # pyrefly: ignore [implicit-any]
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        from torch.utils._pytree import tree_map

        from ._ops.common import lookup_complex

        kwargs = {} if kwargs is None else kwargs

        if func.is_view and isinstance(args[0], InterleavedComplexTensor):
            out = func(args[0].as_interleaved(), *args[1:], **kwargs)
            return tree_map(
                lambda t: InterleavedComplexTensor(t) if t.dtype.is_complex else t,
                out,
            )

        impl = lookup_complex(func, *args, **kwargs)
        if impl is None:
            return NotImplemented

        return impl(*args, **kwargs)

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, Tensor],
        meta: Any,
        outer_size: tuple[int, ...],
        outer_stride: tuple[int, ...],
    ) -> InterleavedComplexTensor:
        out = InterleavedComplexTensor(torch.view_as_complex(inner_tensors["_data"]))
        torch._C._set_conj(out, meta["conj"])
        torch._C._set_neg(out, meta["neg"])
        return out

    def __tensor_flatten__(self) -> tuple[list[str], Any]:
        return ["_data"], {"conj": self.is_conj(), "neg": self.is_neg()}

    def __repr__(self, *, tensor_contents: object | None = None) -> str:
        return f"InterleavedComplexTensor({self.as_interleaved()!r})"

    def is_pinned(self, device: DeviceLikeType | None = None) -> bool:
        return self._data.is_pinned(device)


class Complex(Function):
    @staticmethod
    def forward(  # type: ignore[bad-override]
        ctx: FunctionCtx, real: Tensor, imag: Tensor
    ) -> ComplexTensor:
        return ComplexTensor(real, imag)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: ComplexTensor) -> tuple[Tensor, Tensor]:  # type: ignore[bad-override]
        return grad_output.real, grad_output.imag
