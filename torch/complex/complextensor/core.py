from __future__ import annotations

from typing import Any, TYPE_CHECKING
from typing_extensions import Self

import torch
from torch import Tensor


if TYPE_CHECKING:
    from torch._ops import OpOverload
    from torch._prims_common import DeviceLikeType


class ComplexTensor(Tensor):
    """A class that decomposes all ops on complex Tensors into their real and imaginary parts."""

    _re: Tensor
    _im: Tensor

    def __new__(cls, real: Tensor, imag: Tensor) -> Self:
        """Initialize a ComplexTensor from its real and imaginary parts."""
        from .ops.common import REAL_TO_COMPLEX

        shape = real.shape
        device = real.device

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
        requires_grad = real.requires_grad
        pin_memory = real.is_pinned()

        assert shape == imag.shape, f"Expected imag shape {shape}, got {imag.shape}"
        assert device == imag.device, (
            f"Expected imag device {device}, got {imag.device}"
        )
        assert real.dtype == imag.dtype, (
            f"Expected imag dtype {real.dtype}, got {imag.dtype}"
        )
        assert layout == imag.layout, (
            f"Expected imag layout {layout}, got {imag.layout}"
        )
        assert pin_memory == imag.is_pinned(), (
            f"Expected imag pinning {pin_memory}, got {imag.is_pinned()}"
        )
        assert requires_grad == imag.requires_grad, (
            f"Expected imag requires_grad {requires_grad}, got {imag.requires_grad}"
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
        res._re = real
        res._im = imag

        return res

    @property
    def re(self) -> Tensor:
        return self._re

    @property
    def im(self) -> Tensor:
        return self._im

    @classmethod
    def __torch_dispatch__(
        cls,
        func: OpOverload,
        types: tuple[type, ...],
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        from .ops.common import DEBUG_SET, lookup_complex

        kwargs = {} if kwargs is None else kwargs

        impl = lookup_complex(func, *args, **kwargs)
        if impl is None:
            return NotImplemented

        ret = impl(*args, **kwargs)

        # Note (debugging ops): This block checks if debugging mode is enabled,
        # and if it is, checks if the current op matches the behaviour of the
        # reference op. It is useful for detecting behaviour mismatches.
        debug_set = DEBUG_SET.get()
        if debug_set is not None and all(
            disallowed_name not in str(func) for disallowed_name in ("empty", "rand")
        ):
            from torch.utils._pytree import tree_flatten, tree_map

            from .ops.common import _as_interleaved

            args_ref, kwargs_ref = tree_map(_as_interleaved, (args, kwargs))
            ret_ref = func(*args_ref, **kwargs_ref)

            ret_flat, _ = tree_flatten(ret)
            ret_ref_flat, _ = tree_flatten(ret_ref)
            if not all(
                torch.allclose(_as_interleaved(r), rr, equal_nan=True)
                for r, rr in zip(ret_flat, ret_ref_flat, strict=True)
                if isinstance(rr, Tensor)
            ):
                debug_set.add(func)

        return ret

    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def from_interleaved(t: Tensor) -> ComplexTensor:
        t_real = t.real
        t_imag = t.imag if t.dtype.is_complex else torch.zeros_like(t_real)
        return ComplexTensor(t_real, t_imag)

    def as_interleaved(self) -> Tensor:
        return torch.complex(self.re, self.im)

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, Tensor],
        meta: Any,
        outer_size: tuple[int, ...],
        outer_stride: tuple[int, ...],
    ) -> ComplexTensor:
        assert meta is None
        re, im = inner_tensors["re"], inner_tensors["im"]
        return ComplexTensor(re, im)

    def __tensor_flatten__(self) -> tuple[list[str], Any]:
        return ["re", "im"], None

    def __repr__(self, *, tensor_contents=None) -> str:
        return f"ComplexTensor(real={self.re!r}, imag={self.im!r})"

    def is_pinned(self, device: DeviceLikeType | None = None) -> bool:
        return self.re.is_pinned(device)
