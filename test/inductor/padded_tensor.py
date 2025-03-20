# Owner(s): ["module: inductor"]
#
# This file implements PaddedTensor, which is used for testing tensor subclass
# and graph partition. This is for testing only.

import torch
import torch.nn.functional as F
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
from torch.utils._python_dispatch import return_and_correct_aliasing


def get_pad(shape: torch.Size, multipliers: dict[int, int]) -> tuple[int, ...]:
    pad = [0] * (len(shape) * 2)
    for dim, multiplier in multipliers.items():
        if 2 * dim + 1 >= len(pad):
            break
        pad[2 * dim] = (shape[dim] + multiplier - 1) // multiplier * multiplier - shape[
            dim
        ]
        pad[2 * dim + 1] = 0
    return tuple(pad[::-1])


def transform(args: Any, fn: Callable[..., Any]) -> Any:
    flat, spec = pytree.tree_flatten(args)
    out_flat = [fn(o) for o in flat]
    return pytree.tree_unflatten(out_flat, spec)


class PaddedTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        outer_size: Optional[torch.Size] = None,
        outer_stride: Optional[tuple[int, ...]] = None,
        multipliers: Optional[dict[int, int]] = None,
        neutral_element: int = 0,
    ) -> torch.Tensor:
        # original size and strides before padding
        outer_size = tensor.shape if outer_size is None else outer_size
        outer_stride = tensor.stride() if outer_stride is None else outer_stride
        return torch.Tensor._make_wrapper_subclass(
            cls,
            size=outer_size,
            strides=outer_stride,
            dtype=tensor.dtype,
            device=tensor.device,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
        )

    def __init__(
        self,
        tensor: torch.Tensor,
        outer_size: Optional[torch.Size] = None,
        outer_stride: Optional[tuple[int, ...]] = None,
        multipliers: Optional[dict[int, int]] = None,
        neutral_element: int = 0,
    ):
        self.multipliers = multipliers
        self.neutral_element = neutral_element
        self.tensor = tensor

    @staticmethod
    def from_tensor(
        tensor: torch.Tensor,
        multipliers: Optional[dict[int, int]] = None,
        neutral_element: int = 0,
    ) -> PaddedTensor:
        multipliers = multipliers if multipliers is not None else {}
        padded_tensor = F.pad(
            input=tensor,
            pad=get_pad(tensor.shape, multipliers),
            mode="constant",
            value=neutral_element,
        )
        return PaddedTensor(
            padded_tensor,
            tensor.shape,
            tensor.stride(),
            multipliers,
            neutral_element,
        )

    def __repr__(self) -> str:
        return f"PadTensor(tensor:{self.tensor}, multipliers:{self.multipliers}, neutral_element:{self.neutral_element})"

    def __tensor_flatten__(self):
        return ["tensor"], {
            "multipliers": self.multipliers,
            "neutral_element": self.neutral_element,
        }

    @staticmethod
    def __tensor_unflatten__(tensor, spec, outer_size, outer_stride):
        return PaddedTensor(
            tensor["tensor"],
            outer_size,
            outer_stride,
            spec["multipliers"],
            spec["neutral_element"],
        )

    @staticmethod
    def to_tensor(inp: Any) -> torch.Tensor:
        if isinstance(inp, PaddedTensor):
            return inp.tensor
        return inp

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        with disable_proxy_modes_tracing(), FakeTensorMode():
            fake_args = transform(
                args,
                lambda t: torch.empty_strided(t.shape, t.stride())
                if isinstance(t, torch.Tensor)
                else t,
            )
            fake_kwargs = transform(
                kwargs,
                lambda t: torch.empty_strided(t.shape, t.stride())
                if isinstance(t, torch.Tensor)
                else t,
            )
            fake_out = func(*fake_args, **fake_kwargs)

        outer_size, outer_stride = fake_out.shape, fake_out.stride()

        tensor_args = transform(args, PaddedTensor.to_tensor)
        tensor_kwargs = transform(kwargs, PaddedTensor.to_tensor)
        out = func(*tensor_args, **tensor_kwargs)
        multipliers = transform(
            args, lambda t: t.multipliers if isinstance(t, PaddedTensor) else None
        )[0]
        neutral_element = transform(
            args, lambda t: t.neutral_element if isinstance(t, PaddedTensor) else None
        )[0]

        out = PaddedTensor(
            out,
            outer_size,
            outer_stride,
            multipliers,
            neutral_element,
        )

        return return_and_correct_aliasing(func, args, kwargs, out)
