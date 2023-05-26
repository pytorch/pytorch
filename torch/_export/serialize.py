from typing import Optional, Tuple

import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from .serde.schema import Device, Layout, ScalarType, SymInt, TensorMeta  # type: ignore[attr-defined]
from .graph_module import ExportedProgram


__all__ = ["convert_fake_tensor_to_tensor_meta", "convert_tensor_meta_to_fake_tensor"]


def _reverse_map(d):
    return {v: k for k, v in d.items()}


_SCALAR_TYPES = {
    torch.uint8: ScalarType.BYTE,
    torch.int8: ScalarType.CHAR,
    torch.int16: ScalarType.SHORT,
    torch.int32: ScalarType.INT,
    torch.int64: ScalarType.LONG,
    torch.float16: ScalarType.HALF,
    torch.float32: ScalarType.FLOAT,
    torch.float64: ScalarType.DOUBLE,
    torch.complex32: ScalarType.COMPLEXHALF,
    torch.complex64: ScalarType.COMPLEXFLOAT,
    torch.complex128: ScalarType.COMPLEXDOUBLE,
    torch.bool: ScalarType.BOOL,
    torch.bfloat16: ScalarType.BFLOAT16
}


_DTYPES = _reverse_map(_SCALAR_TYPES)


_LAYOUTS = {
    torch.sparse_coo: Layout.SparseCoo,
    torch.sparse_csr: Layout.SparseCsr,
    torch.sparse_csc: Layout.SparseCsc,
    torch.sparse_bsr: Layout.SparseBsr,
    torch.sparse_bsc: Layout.SparseBsc,
    torch._mkldnn: Layout._mkldnn,  # type: ignore[attr-defined]
    torch.strided: Layout.Strided,
}


def _extract_sym_int(s) -> SymInt:
    if isinstance(s, int):
        return SymInt.create(as_int=s)
    elif isinstance(s, torch.SymInt):
        return SymInt.create(as_symbol=str(s))
    else:
        raise ValueError(str(s))


def _extract_tensor_meta(result: torch.Tensor) -> TensorMeta:
    """
    Extract a TensorMeta describing `result`.
    """
    return TensorMeta(
        dtype=_SCALAR_TYPES[result.dtype],
        sizes=[_extract_sym_int(s) for s in result.shape],
        requires_grad=result.requires_grad,
        device=Device(type=result.device.type, index=result.device.index),
        strides=[_extract_sym_int(s) for s in result.stride()],
        storage_offset=0,
        layout=_LAYOUTS[result.layout],
    )


def convert_fake_tensor_to_tensor_meta(
    ep: ExportedProgram
) -> Tuple[ExportedProgram, Optional[ShapeEnv]]:
    """
    Replace the faketensor metadata with the tensor metadata dataclass since we
    cannot serialize faketensors
    """
    shape_env = None
    for node in ep.graph.nodes:
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

    return ep, shape_env


def convert_tensor_meta_to_fake_tensor(ep: ExportedProgram, shape_env: ShapeEnv = None) -> ExportedProgram:
    """
    Replace (inplace) the tensor metadata with faketensor
    """
    fake_tensor_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=shape_env)
    for node in ep.graph.nodes:
        if (val := node.meta.get("tensor_meta", None)) is not None:

            def _extract_faketensor(tensor_meta: TensorMeta):
                return FakeTensor(
                    fake_tensor_mode,
                    torch.empty(
                        # TODO Support dynamic shape.
                        tuple(s.as_int for s in tensor_meta.sizes),
                        dtype=_DTYPES[tensor_meta.dtype],
                        device="meta",
                        requires_grad=tensor_meta.requires_grad,
                    ),
                    torch.device("cpu"),
                )

            node.meta["val"] = pytree.tree_map_only(
                TensorMeta, _extract_faketensor, val
            )
    return ep
