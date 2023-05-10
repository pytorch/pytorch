import copy
from typing import Optional, Tuple

import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from .logical_schema import TensorMeta  # type: ignore[attr-defined]


__all__ = ["convert_fake_tensor_to_tensor_meta", "convert_tensor_meta_to_fake_tensor"]


def _extract_tensor_meta(result: torch.Tensor) -> TensorMeta:
    """
    Extract a TensorMeta describing `result`.
    """
    return TensorMeta(
        dtype=result.dtype,
        sizes=result.shape,
        requires_grad=result.requires_grad,
        device=result.device,
        strides=result.stride(),
        storage_offset=0,
        layout=result.layout,
    )


def convert_fake_tensor_to_tensor_meta(
    gm: torch.fx.GraphModule
) -> Tuple[torch.fx.GraphModule, Optional[ShapeEnv]]:
    """
    Replace the faketensor metadata with the tensor metadata dataclass since we
    cannot serialize faketensors
    """
    gm = copy.deepcopy(gm)
    shape_env = None
    for node in gm.graph.nodes:
        def get_shape_env(val) -> Optional[ShapeEnv]:
            val_flat, _ = pytree.tree_flatten(val)
            curr_shape_env = None
            for v in val_flat:
                if not isinstance(v, FakeTensor):
                    continue
                if curr_shape_env is None:
                    curr_shape_env = v.fake_mode.shape_env
                else:
                    assert (
                        curr_shape_env is v.fake_mode.shape_env
                    ), "Multiple shape envs detected."
            return curr_shape_env

        if (val := node.meta.get("val", None)) is not None:
            if shape_env is None:
                shape_env = get_shape_env(val)
            elif (new_shape_env := get_shape_env(val)) is not None:
                assert (
                    shape_env is new_shape_env
                ), "Multiple shape envs detected."

            node.meta["tensor_meta"] = pytree.tree_map_only(
                torch.Tensor, _extract_tensor_meta, val
            )
            del node.meta["val"]

    return gm, shape_env


def convert_tensor_meta_to_fake_tensor(gm: torch.fx.GraphModule, shape_env: ShapeEnv = None) -> torch.fx.GraphModule:
    """
    Replace (inplace) the tensor metadata with faketensor
    """
    fake_tensor_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=shape_env)
    for node in gm.graph.nodes:
        if (val := node.meta.get("tensor_meta", None)) is not None:

            def _extract_faketensor(tensor_meta: TensorMeta):
                return FakeTensor(
                    fake_tensor_mode,
                    torch.empty(
                        tensor_meta.sizes,
                        dtype=tensor_meta.dtype,
                        device="meta",
                        requires_grad=tensor_meta.requires_grad,
                    ),
                    torch.device("cpu"),
                )

            node.meta["val"] = pytree.tree_map_only(
                TensorMeta, _extract_faketensor, val
            )
    return gm
