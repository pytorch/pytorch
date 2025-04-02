from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing


torch._inductor.config.graph_partition = True


def slice_nd(
    input: torch.Tensor, start_idxs: List[int], end_idxs: List[int]
) -> torch.Tensor:
    """
    Slice a tensor along multiple dimensions. This is a generalization of torch.slice,
    which only supports slicing along one dimension.
    """
    assert len(start_idxs) == len(end_idxs)

    # Check if input.shape and end_idx are identical. Skip slicing if so.
    if all(
        input.shape[dim_idx] == end_idx
        for dim_idx, end_idx in enumerate(end_idxs)
        if end_idx is not None
    ):
        return input

    # Slice the tensor
    for dim_idx, (start_idx, end_idx) in enumerate(zip(start_idxs, end_idxs)):
        if start_idx is not None and end_idx is not None:
            if end_idx != input.shape[dim_idx]:
                assert start_idx >= 0
                assert end_idx <= input.shape[dim_idx]

                if not start_idx < end_idx:
                    raise ValueError(
                        f"Invalid slice indices: {start_idx}:{end_idx} for dimension {dim_idx}"
                    )

                input = torch.ops.aten.slice(input, dim_idx, start_idx, end_idx)

    return input


def get_strides(shape: torch.Size) -> List[int]:
    strides = [1]
    for i in range(len(shape) - 1, 0, -1):
        strides.append(strides[-1] * shape[i])
    return strides[::-1]


def get_padded_shape(shape: torch.Size, multipliers: Dict[int, int]) -> torch.Size:
    padded_shape = list(shape)
    for dim, multiplier in multipliers.items():
        if dim >= len(padded_shape):
            break
        padded_shape[dim] = (
            (padded_shape[dim] + multiplier - 1) // multiplier * multiplier
        )
    return torch.Size(padded_shape)


def get_pad(shape: torch.Size, multipliers: Dict[int, int]) -> Tuple[int, ...]:
    pad = [0] * (len(shape) * 2)
    for dim, multiplier in multipliers.items():
        pad[2 * dim] = (shape[dim] + multiplier - 1) // multiplier * multiplier - shape[
            dim
        ]
        pad[2 * dim + 1] = 0
    return tuple(pad[::-1])


class PaddedTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        multipliers: Optional[Dict[int, int]],
        original_tensor: Optional[torch.Tensor] = None,
    ):
        if multipliers is None:
            multipliers = {}
        padded_shape = get_padded_shape(tensor.shape, multipliers)

        kwargs = {}
        kwargs["strides"] = get_strides(padded_shape)
        kwargs["storage_offset"] = 0
        kwargs["device"] = tensor.device
        kwargs["layout"] = tensor.layout
        kwargs["requires_grad"] = tensor.requires_grad
        kwargs["dtype"] = tensor.dtype

        out = torch.Tensor._make_wrapper_subclass(cls, padded_shape, **kwargs)
        return out

    def __init__(
        self,
        tensor: torch.Tensor,
        multipliers: Optional[Dict[int, int]],
        original_tensor: Optional[torch.Tensor] = None,
    ):
        if multipliers is None:
            multipliers = {}
        self.multipliers = multipliers
        self.tensor = tensor
        self.original_tensor = original_tensor

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

        original_tensor = torch.ones(tensor.shape, device="meta", dtype=tensor.dtype)

        return PaddedTensor(padded_tensor, multipliers, original_tensor)

    def __repr__(self):
        return f"PaddedTensor(tensor:{list(self.tensor.shape)}, original_tensor:{list(self.original_tensor.shape)}, multipliers:{self.multipliers})"

    def __tensor_flatten__(self):
        return ["tensor", "original_tensor"], {
            "multipliers": self.multipliers,
        }

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        return PaddedTensor(
            inner_tensors["tensor"],
            meta["multipliers"],
            inner_tensors["original_tensor"],
        )

    @classmethod
    def foobar(cls, func, types, args=..., kwargs=None, shapeargs_to_meta=False):
        # In: Padded tensor
        if kwargs is None:
            kwargs = {}
        tensor_args, tensor_kwargs = pytree.tree_map_only(
            PaddedTensor, lambda x: x.tensor, (args, kwargs)
        )
        out = func(*tensor_args, **tensor_kwargs)

        # In: Original shaped meta tensor
        shapes_args, shapes_kwargs = pytree.tree_map_only(
            PaddedTensor, lambda x: x.original_tensor, (args, kwargs)
        )
        if shapeargs_to_meta:
            shapes_args = pytree.tree_map_only(
                torch.Tensor, lambda x: x.to("meta"), shapes_args
            )
        out_shapes = func(*shapes_args, **shapes_kwargs)

        # In: Multipliers
        # args_multipliers, kwargs_multipliers = pytree.tree_map_only(
        #    PaddedTensor, lambda x: x.multipliers, (args, kwargs)
        # )
        # multipliers = pytree.tree_leaves(args_multipliers, lambda x: type(x) == dict)
        # multipliers = next(filter(lambda x: type(x) == dict, multipliers))
        multipliers = {n: 1 for n in range(5)}

        # Out: Padded tensor
        out_flat, spec = pytree.tree_flatten(out)
        out_shapes_flat, out_shapes_spec = pytree.tree_flatten(out_shapes)
        assert spec == out_shapes_spec

        out_flat = [
            PaddedTensor(o, multipliers, s) for o, s in zip(out_flat, out_shapes_flat)
        ]

        out = pytree.tree_unflatten(out_flat, spec)
        return out

    @classmethod
    def __torch_function__(cls, func, types, args=..., kwargs=None):
        # print("Dispatching function %s" % func.__name__)
        # print("-" * 40)
        # print("args: ", args)
        if func.__name__ in [
            "linear",
            "embedding",
            "split_with_sizes",
            "flatten",
            "scaled_dot_product_attention",
        ]:
            out = cls.foobar(func, types, args, kwargs, shapeargs_to_meta=True)
        else:
            out = super().__torch_function__(func, types, args, kwargs)
        # print("out: ", out)
        return out

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # print("Dispatching %s" % func._overloadpacket.__name__)
        # print("-" * 40)
        # print("args: ", args)
        out = cls.foobar(func, types, args, kwargs, shapeargs_to_meta=True)
        # print("out: ", out)
        return return_and_correct_aliasing(func, args, kwargs, out)

    def unpad(self) -> torch.Tensor:
        start_idxs = [0] * len(self.original_tensor.shape)
        end_idxs = list(self.original_tensor.shape)
        return slice_nd(self.tensor, start_idxs, end_idxs)
