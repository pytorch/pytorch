"""Local Tensor with variance tracking for distributed operations."""

from collections.abc import Callable
from typing import Optional, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from torch.distributed.tensor import DTensor

from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Reduced, Replicate
from torch.utils._pytree import tree_map

from . import _varying_collectives as vcols


# Custom variance tracking strategies for ops and autograd.Function classes
_CUSTOM_VARIANCE_TRACKING_MAP: dict[Callable | type, Callable] = {}

_CUSTOM_OPERATOR_HANDLER_MAP: dict[Callable, Callable] = {
    torch.ops._c10d_functional.all_reduce: vcols.all_reduce_invariant
}


def register_variance_tracking_strategy(target):
    """Register custom variance tracking strategy for ops or autograd.Function.

    Args:
        target: Either a list of ops, a single op, or an autograd.Function class

    The decorated function receives:
      - input_variant_dims: set[str] - union of variant dims from all LTensor inputs
      - input_reduced_dims: set[str] - intersection of reduced dims from all LTensor inputs
      - mesh: DeviceMesh - the mesh from input LTensors
      - *args, **kwargs: the original function arguments

    Returns:
      - (output_variant_dims, output_reduced_dims): tuple[set[str], set[str]]

    Example::

        @register_variance_tracking_strategy([torch.ops._c10d_functional.all_reduce])
        def _(input_variant_dims, input_reduced_dims, mesh, *args, **kwargs):
            _, _, group_name = args
            dim_name = _get_dim_name_from_group(mesh, group_name)
            return input_variant_dims - {dim_name}, input_reduced_dims
    """

    def decorator(strategy_fn):
        if isinstance(target, list):
            for op in target:
                _CUSTOM_VARIANCE_TRACKING_MAP[op] = strategy_fn
        else:
            _CUSTOM_VARIANCE_TRACKING_MAP[target] = strategy_fn
        return strategy_fn

    return decorator


def _get_dim_name_from_group(mesh: DeviceMesh, group_name: str) -> Optional[str]:
    """Get mesh dim name for a process group."""
    if mesh.mesh_dim_names:
        for dim_name in mesh.mesh_dim_names:
            if mesh.get_group(dim_name).group_name == group_name:
                return dim_name
    return None


# Note: Intentionally do not include all_gather in the invariant-output registration.
# Keeping all_gather's output as varying prevents mark_varying from being inserted in the forward,
# which would otherwise cause a redundant all_reduce to be introduced during the backward pass.
@register_variance_tracking_strategy([torch.ops._c10d_functional.all_reduce])
def _invariant_output_strategy(
    input_variant_dims, input_reduced_dims, mesh, *args, **kwargs
):
    """Collectives that make output invariant on the reduced dim."""
    _, _, group_name = args
    dim_name = _get_dim_name_from_group(mesh, group_name)
    assert dim_name is not None, (
        f"Could not find mesh dim for group {group_name}. "
        f"Available dims: {mesh.mesh_dim_names}"
    )
    return input_variant_dims - {dim_name}, input_reduced_dims


class LTensor(torch.Tensor):
    """
    ``LTensor`` (Local Tensor) tracks variance metadata through distributed operations.

    States:
    * Variant: Different values across ranks on this mesh dim (Shard, Partial)
    * Reduced: Identical values but gradient needs reduction (Reduced placement)
    * Invariant: Identical values across ranks (Replicate)

    Propagation rules:
    * Variance: Union (ANY input variant → output variant)
    * Reduced: Intersection (ALL inputs reduced → output reduced)
    """

    _local_tensor: torch.Tensor
    _variant_dims: set[str]
    _reduced_dims: set[str]
    _mesh: DeviceMesh
    _custom_function_context: bool

    def __new__(
        cls,
        local_tensor: torch.Tensor,
        variant_dims: set[str],
        reduced_dims: set[str],
        mesh: DeviceMesh,
        custom_function_context: bool = False,
    ):
        assert not (variant_dims & reduced_dims), (
            f"Dims cannot be both variant and reduced: {variant_dims=} {reduced_dims=}"
        )

        r = local_tensor.as_subclass(cls)
        r._variant_dims = variant_dims
        r._reduced_dims = reduced_dims
        r._mesh = mesh
        r._local_tensor = local_tensor
        r._custom_function_context = custom_function_context
        return r

    @staticmethod
    def compute_metadata_from_dtensor(dtensor: "DTensor"):
        """Extract metadata from DTensor placements."""
        mesh = dtensor.device_mesh
        assert mesh.mesh_dim_names is not None, (
            "DTensor's mesh must have mesh_dim_names to convert to LTensor."
        )

        variant_dims: set[str] = set()
        reduced_dims: set[str] = set()

        for idx, placement in enumerate(dtensor.placements):
            dim_name = mesh.mesh_dim_names[idx]
            if isinstance(placement, Reduced):
                reduced_dims.add(dim_name)
            elif not isinstance(placement, Replicate):
                variant_dims.add(dim_name)

        return {
            "mesh": mesh,
            "variant_dims": variant_dims,
            "reduced_dims": reduced_dims,
        }

    @property
    def variant_dims(self) -> set[str]:
        return self._variant_dims

    @property
    def reduced_dims(self) -> set[str]:
        return self._reduced_dims

    @property
    def mesh(self) -> DeviceMesh:
        return self._mesh

    @property
    def custom_function_context(self) -> bool:
        return self._custom_function_context

    @custom_function_context.setter
    def custom_function_context(self, value: bool):
        self._custom_function_context = value

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Dispatch hook: compute variance, insert mark_varying, unwrap/wrap tensors."""
        if kwargs is None:
            kwargs = {}

        # 1. Extract metadata from all LTensor inputs
        out_variant_dims: set[str] = set()
        reduced_dims_per_arg: list[set[str]] = []
        mesh: Optional[DeviceMesh] = None
        custom_function_context = False

        def _collect_metadata(t):
            nonlocal mesh, custom_function_context
            if isinstance(t, LTensor):
                out_variant_dims.update(t._variant_dims)
                reduced_dims_per_arg.append(t._reduced_dims)
                if t._custom_function_context:
                    custom_function_context = True
                if mesh is None:
                    mesh = t._mesh
                else:
                    assert mesh == t._mesh, (
                        "Cannot mix LTensors from different meshes in one operation!"
                    )

        tree_map(_collect_metadata, args)
        tree_map(_collect_metadata, kwargs)

        assert mesh is not None, "No LTensor found in arguments"
        out_reduced_dims = (
            set.intersection(*reduced_dims_per_arg) if reduced_dims_per_arg else set()
        )

        # 2. Apply custom strategy if registered
        if func in _CUSTOM_VARIANCE_TRACKING_MAP:
            out_variant_dims, out_reduced_dims = _CUSTOM_VARIANCE_TRACKING_MAP[func](
                out_variant_dims, out_reduced_dims, mesh, *args, **kwargs
            )
            assert not (out_variant_dims & out_reduced_dims), (
                f"Strategy for {func} returned overlapping dims: "
                f"variant={out_variant_dims}, reduced={out_reduced_dims}"
            )

        # 3. Get actual function (skip custom ops in custom_function_context)
        actual_func = (
            func
            if custom_function_context
            else _CUSTOM_OPERATOR_HANDLER_MAP.get(func, func)
        )

        # 4. Unwrap LTensors and insert mark_varying for missing variant dims
        def _unwrap(t):
            if not isinstance(t, LTensor):
                return t

            assert not isinstance(t, AsyncCollectiveTensor)
            local_tensor = t._local_tensor
            if custom_function_context:
                return local_tensor

            for dim_name in out_variant_dims - t._variant_dims:
                group_name = mesh.get_group(dim_name).group_name
                local_tensor = vcols.mark_varying(local_tensor, group_name=group_name)
            return local_tensor

        unwrapped_args = tree_map(_unwrap, args)
        unwrapped_kwargs = tree_map(_unwrap, kwargs)

        # 5. Call function
        result = actual_func(*unwrapped_args, **unwrapped_kwargs)

        # 6. Wrap outputs as LTensor (preserve custom_function_context)
        def _wrap(t):
            assert not isinstance(t, AsyncCollectiveTensor)
            if isinstance(t, LTensor):
                return t
            if isinstance(t, torch.Tensor):
                return LTensor(
                    t, out_variant_dims, out_reduced_dims, mesh, custom_function_context
                )
            return t

        return tree_map(_wrap, result)


def ltensor_custom_function_call(
    autograd_function: type[torch.autograd.Function], *args, **kwargs
):
    """Handler for autograd.Function with LTensor inputs.

    This function wraps calls to autograd.Function classes when LTensor inputs
    are involved. It manages the custom_function_context flag to disable
    automatic variance tracking during the function execution, allowing
    custom strategies to handle variance propagation.

    Args:
        autograd_function: The autograd.Function class to apply
        *args: Positional arguments to pass to autograd_function.apply
        **kwargs: Keyword arguments to pass to autograd_function.apply

    Returns:
        The result of autograd_function.apply with proper LTensor wrapping

    Raises:
        AssertionError: If no custom strategy exists and output is not an LTensor
    """
    # 1. Collect metadata and set custom_function_context=True on all LTensor inputs
    mesh: Optional[DeviceMesh] = None
    input_variant_dims: set[str] = set()
    input_reduced_dims_list: list[set[str]] = []

    def _set_context_true(t):
        nonlocal mesh
        if isinstance(t, LTensor):
            t._custom_function_context = True
            input_variant_dims.update(t._variant_dims)
            input_reduced_dims_list.append(t._reduced_dims)
            if mesh is None:
                mesh = t._mesh
        return t

    tree_map(_set_context_true, args)
    tree_map(_set_context_true, kwargs)

    assert mesh is not None, "No LTensor found in arguments"
    input_reduced_dims = (
        set.intersection(*input_reduced_dims_list) if input_reduced_dims_list else set()
    )

    # 2. Apply autograd_function
    result = autograd_function.apply(*args, **kwargs)

    # If custom strategy exists, call it to get new variant/reduced dims
    if autograd_function in _CUSTOM_VARIANCE_TRACKING_MAP:
        out_variant_dims, out_reduced_dims = _CUSTOM_VARIANCE_TRACKING_MAP[
            autograd_function
        ](input_variant_dims, input_reduced_dims, mesh, *args, **kwargs)

        # Wrap results with computed dims
        def _wrap(t):
            if isinstance(t, LTensor):
                t._variant_dims = out_variant_dims
                t._reduced_dims = out_reduced_dims
                return t
            if isinstance(t, torch.Tensor):
                return LTensor(t, out_variant_dims, out_reduced_dims, mesh, False)
            return t

        result = tree_map(_wrap, result)
    else:
        # Check that all tensor outputs are LTensors
        def _check_ltensor(t):
            if isinstance(t, torch.Tensor):
                assert isinstance(t, LTensor), (
                    f"autograd.Function {autograd_function.__name__} returned a "
                    f"Tensor that is not an LTensor, but no custom variance tracking "
                    f"strategy is registered. Either register a strategy with "
                    f"@register_variance_tracking_strategy({autograd_function.__name__}) "
                    f"or ensure the function propagates LTensors."
                )
            return t

        tree_map(_check_ltensor, result)

    # 3. Set custom_function_context=False on all LTensor args, kwargs, and results
    def _set_context_false(t):
        if isinstance(t, LTensor):
            t._custom_function_context = False
        return t

    tree_map(_set_context_false, args)
    tree_map(_set_context_false, kwargs)
    tree_map(_set_context_false, result)

    return result
