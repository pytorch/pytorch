import warnings
from typing import Tuple, Union

from torch.distributed._tensor import DeviceMesh
from torch.distributed._tensor.placement_types import Placement
from torch.distributed.device_mesh import _mesh_resources
try:
    from torch._dynamo.external_utils import is_compiling as is_torchdynamo_compiling
except Exception:
    def is_torchdynamo_compiling():  # type: ignore[misc]
        return False

LayoutsType = Union[Placement, Tuple[Placement, ...]]


def _deprecate_warnings(func_name: str, extra_msg: str) -> None:
    """
    Inject common validation logics for `_prepare_input` funcs via this decorator.

    Include verifying that input needs to be either a :class:`Tensor` or :class:`DTensor`
    and only 1D :class:`DeviceMesh` is passed in.
    """
    # TODO: Will follow up with dynamo POC to make warnings.warn working with dynamo.
    if not is_torchdynamo_compiling():
        warnings.warn(f"{func_name} is deprecated and will be removed soon. {extra_msg}")


def _validate_tp_mesh_dim(
    device_mesh: DeviceMesh,
) -> None:
    """
    Check whether TP mesh dimension is valid or not.

    Args:
        device_mesh (:class:`DeviceMesh`):
            The `device_mesh` where we perform
            Tensor Parallelism on.

    Return:
        `True` if the mesh dimension
        is valid, `False` otherwise.
    """
    if device_mesh.ndim > 1:
        raise ValueError(f"Tensor Parallel only accepts a 1D DeviceMesh, but found {device_mesh.ndim}D!"
                         "If you have a 2-D or N-D device_mesh, consider passing in device_mesh[\"tp\"]")

    parent_mesh = _mesh_resources.get_parent_mesh(device_mesh)
    if parent_mesh:
        if parent_mesh.ndim != 2:
            raise RuntimeError(
                f"Found TP device_mesh has a parent mesh with dims {parent_mesh.ndim}",
                "Currently we only support 2D TP composition with DP.",
            )

        tp_mesh_dim = _mesh_resources.get_parent_mesh_dim(device_mesh)
        if tp_mesh_dim != 1:
            raise RuntimeError(
                f"Found TP device_mesh on the {tp_mesh_dim} dimension of its parent mesh.",
                "Currently we only support intranode TP and TP needs to be the innermost dimension on its parent mesh.",
            )
