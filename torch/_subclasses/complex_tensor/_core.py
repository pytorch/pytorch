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

    _data: Tensor

    def __new__(cls, real: Tensor, imag: Tensor | None = None) -> Self:
        """Initialize a ComplexTensor from its real and imaginary parts."""
        from ._ops.common import REAL_TO_COMPLEX

        shape = real.shape
        device = real.device

        # TODO (hameerabbasi): `torch.compile` sometimes fails here without making these
        # contiguous. Why?
        real = real
        if imag is None:
            if real.dtype.is_complex:
                data = torch.view_as_real(real)
            else:
                assert real.shape[-1] == 2
                data = real

        else:
            data = torch.stack([real, imag], dim=-1)

        real = data[..., 0]
        imag = data[..., 1]

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

        if shape != imag.shape:
            raise AssertionError(f"Expected imag shape {shape}, got {imag.shape}")
        if device != imag.device:
            raise AssertionError(f"Expected imag device {device}, got {imag.device}")
        if real.dtype != imag.dtype:
            raise AssertionError(f"Expected imag dtype {real.dtype}, got {imag.dtype}")
        if pin_memory != imag.is_pinned():
            raise AssertionError(
                f"Expected imag pinning {pin_memory}, got {imag.is_pinned()}"
            )

        res = Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            shape,
            device=device,
            dtype=dtype,
            storage_offset=storage_offset,
            strides=strides,
            pin_memory=pin_memory,
            layout=layout,
            requires_grad=False,
        )
        res._data = data.detach()
        # res._re = real.detach()
        # res._im = imag.detach()

        return res

    @property
    def re(self) -> Tensor:
        return self._data[..., 0]

    @property
    def im(self) -> Tensor:
        return self._data[..., 1]

    @classmethod
    def __torch_dispatch__(  # type: ignore[bad-override]
        cls,
        func: OpOverload,
        types: tuple[type, ...],
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
        return Complex.apply(t)

    def as_interleaved(self) -> Tensor:
        return torch.view_as_complex(torch.view_as_real(self))

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, Tensor],
        meta: Any,
        outer_size: tuple[int, ...],
        outer_stride: tuple[int, ...],
    ) -> ComplexTensor:
        assert meta is None
        data = inner_tensors["_data"]
        return ComplexTensor(data[..., 0], data[..., 1])

    def __tensor_flatten__(self) -> tuple[list[str], Any]:
        return ["_data"], None

    def __repr__(self, *, tensor_contents: object | None = None) -> str:
        return f"ComplexTensor(real={self.re!r}, imag={self.im!r})"

    def is_pinned(self, device: DeviceLikeType | None = None) -> bool:
        return self.re.is_pinned(device)


class Complex(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, real: Tensor, imag: Tensor|None = None) -> ComplexTensor:  # type: ignore[bad-override]
        return ComplexTensor(real, imag)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: ComplexTensor) -> tuple[Tensor, Tensor]:  # type: ignore[bad-override]
        return grad_output.real, grad_output.imag
