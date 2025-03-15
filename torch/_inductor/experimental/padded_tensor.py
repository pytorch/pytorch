import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import return_and_correct_aliasing
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing


from typing import Any, Optional


def get_padded_shape(shape: torch.Size, multipliers: dict[int, int]) -> torch.Size:
    padded_shape = list(shape)
    for dim, multiplier in multipliers.items():
        padded_shape[dim] = (padded_shape[dim] + multiplier - 1) // multiplier * multiplier
    return torch.Size(padded_shape)


def get_pad(shape: torch.Size, multipliers: dict[int, int]) -> tuple[int,...]:
    pad = [0] * (len(shape) * 2)
    for dim, multiplier in multipliers.items():
        if 2 * dim + 1 >= len(pad): # TODO
            break
        pad[2 * dim] = (shape[dim] + multiplier - 1) // multiplier * multiplier - shape[dim]
        pad[2 * dim + 1] = 0
    return tuple(pad[::-1])


class PaddedTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        outer_size: Optional[torch.Size] = None,
        outer_stride: Optional[tuple[int, ...]] = None,
        multipliers: Optional[dict[int, int]] = None,
        neutral_element: int = 0,
    ):
        # original size and strides before padding
        size = tensor.shape if outer_size is None else outer_size
        stride = tensor.stride() if outer_stride is None else outer_stride
        return torch.Tensor._make_wrapper_subclass(
            cls,
            size=size,
            strides=stride,
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
        self.neutral_element=neutral_element
        self.tensor = tensor

    @staticmethod
    def from_tensor(
        tensor: torch.Tensor,
        multipliers: Optional[dict[int, int]] = None,
        neutral_element: int = 0,
    ):
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

    def __repr__(self):
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
            fake_args = tuple(torch.empty_strided(t.shape, t.stride()) if isinstance(t, torch.Tensor) else t for t in args)
            fake_kwargs = {k : torch.empty_strided(v.shape, v.stride()) if isinstance(v, torch.Tensor) else v for k,v in enumerate(kwargs.items())}
            fake_out = func(*fake_args, **fake_kwargs)

        outer_size, outer_stride = fake_out.shape, fake_out.stride()

        tensor_args = tuple(map(PaddedTensor.to_tensor, args))
        tensor_kwargs = {k : PaddedTensor.to_tensor(kwargs[k]) for k in kwargs}
        out = func(*tensor_args, **tensor_kwargs)
        multipliers = args[0].multipliers # TODO: support different multipliers from args
        neutral_element = args[0].neutral_element # TODO: support different neural element

        out = PaddedTensor(
            out,
            outer_size,
            outer_stride,
            multipliers,
            neutral_element,            
        )
        assert hasattr(out, "multipliers"), breakpoint()

        return return_and_correct_aliasing(func, args, kwargs, out)
