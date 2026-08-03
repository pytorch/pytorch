# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import functools
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.distributed as dist


if dist._is_spmd_types_available():
    import spmd_types as spmd

from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)


try:
    from torch.utils import _cxx_pytree as pytree
except ImportError:
    from torch.utils import _pytree as pytree  # type: ignore[no-redef]


__all__ = ["local_map"]

PlacementType = Sequence[Placement] | None
InputPlacements = tuple[PlacementType, ...] | None
OutputPlacements = PlacementType | tuple[PlacementType, ...]


def _placements_to_spmd_type(
    placements: PlacementType,
    grad_placements: PlacementType,
    device_mesh: DeviceMesh,
) -> dict:
    """Convert DTensor placements to a dict of MeshAxis -> spmd local type.

    Mapping: Shard -> V, Replicate -> R (or I if grad is Replicate), Partial -> P.
    Validates that grad_placements are compatible with forward placements.
    """
    result = {}
    for dim_idx, placement in enumerate(placements):  # pyrefly: ignore
        axis = spmd.MeshAxis.of(device_mesh.get_group(dim_idx))
        grad_p = grad_placements[dim_idx] if grad_placements is not None else None

        if type(placement) is Shard:
            fwd_type = spmd.V
        elif type(placement) is Replicate:
            fwd_type = spmd.I if type(grad_p) is Replicate else spmd.R
        elif type(placement) is Partial:
            fwd_type = spmd.P
        else:
            raise ValueError(
                f"local_map(spmd_types=True) does not support placement type "
                f"{type(placement).__name__}: {placement}"
            )

        if grad_p is not None:
            actual_grad_p = _dtensor_placement_if_compatible(
                fwd_type.backward_type(),  # pyrefly: ignore [missing-attribute]
                grad_p,
            )
            if actual_grad_p is None or type(actual_grad_p) is not type(grad_p):
                raise ValueError(
                    "local_map(spmd_types=True) cannot represent this "
                    "forward/backward placement pair:\n"
                    f"in_placements={placement}, in_grad_placements={grad_p}.\n\n"
                    "The local_map boundary assigns one SPMD type to each "
                    "local tensor, which also fixes the expected backward "
                    f"placement. For in_placements={placement}, valid grad "
                    f"placements are {_valid_grad_placements(placement)}.\n\n"
                    "If the forward and backward layouts intentionally "
                    "diverge in a way not represented by one SPMD type, "
                    "expand the local_map region to include that divergence, "
                    "so the local_map region boundary respects tied FWD-BWD "
                    "typing."
                )

        result[axis] = fwd_type
    return result


def _annotate_spmd_types(
    flat_local_args: list,
    in_placements: InputPlacements,
    in_grad_placements: InputPlacements,
    device_mesh: DeviceMesh,
) -> None:
    """Annotate unwrapped local tensors with spmd_types inferred from placements.

    Assumes local_map has already validated in_placements and unwrapped DTensors.
    """
    for idx, local_arg in enumerate(flat_local_args):
        if not isinstance(local_arg, torch.Tensor):
            continue
        if in_placements is None or in_placements[idx] is None:
            continue
        grad_placements = (
            in_grad_placements[idx] if in_grad_placements is not None else None
        )
        spmd_type = _placements_to_spmd_type(
            in_placements[idx], grad_placements, device_mesh
        )
        spmd.assert_type(local_arg, spmd_type)


def _valid_grad_placements(placement: Placement) -> str:
    if type(placement) is Shard:
        return "Shard (spmd.V)"
    elif type(placement) is Replicate:
        return "Partial (spmd.R) or Replicate (spmd.I)"
    elif type(placement) is Partial:
        return "Replicate (spmd.P)"
    raise ValueError(
        f"local_map(spmd_types=True) does not support placement type {type(placement).__name__}"
    )


def _dtensor_placement_if_compatible(
    local_type, placement: Placement
) -> Placement | None:
    if local_type == spmd.V:
        if type(placement) in (Shard, Partial):
            return placement
        return None
    return spmd.spmd_type_to_dtensor_placement(local_type)


def _out_spmd_types_to_grad_placements(
    flat_out: Sequence[Any],
    out_placements_tuple: tuple[PlacementType, ...],
    device_mesh: DeviceMesh,
) -> tuple[PlacementType, ...]:
    """Validate output SPMD types and return backward grad placements."""
    grad_out_placements: list[PlacementType] = []
    for out, spec in zip(flat_out, out_placements_tuple, strict=True):
        if spec is not None and not isinstance(out, torch.Tensor):
            raise ValueError(
                f"out_placements specifies {spec} but the corresponding "
                f"output is {type(out).__name__}, not a Tensor"
            )
        if spec is None:
            if isinstance(out, torch.Tensor):
                raise ValueError(
                    "out_placements has None for a Tensor output. Provide "
                    "placements for Tensor outputs and use None only for "
                    "non-Tensor outputs."
                )
            grad_out_placements.append(None)
            continue

        # validate out type
        actual_type = spmd.get_local_type(out)
        if not actual_type:
            raise ValueError(
                "Output tensor has no spmd_types annotation but out_placements "
                "expects one. Ensure the function's output is derived from "
                "annotated inputs or is explicitly annotated."
            )
        for dim_idx, placement in enumerate(spec):  # pyrefly: ignore
            axis = spmd.MeshAxis.of(device_mesh.get_group(dim_idx))
            actual = actual_type.get(axis)
            if actual is None:
                if device_mesh.mesh_dim_names is None:
                    mesh_dims: Sequence[Any] = tuple(range(device_mesh.ndim))
                else:
                    mesh_dims = device_mesh.mesh_dim_names
                annotated_dims = [
                    mesh_dims[i]
                    for i in range(device_mesh.ndim)
                    if spmd.MeshAxis.of(device_mesh.get_group(i)) in actual_type
                ]
                raise ValueError(
                    "Output tensor has no spmd_types annotation on "
                    f"DeviceMesh dimension {mesh_dims[dim_idx]} "
                    f"but out_placements expects {placement}. "
                    "Actual annotations on this DeviceMesh are: "
                    f"{annotated_dims}"
                )
            actual_placement = _dtensor_placement_if_compatible(actual, placement)
            if actual_placement is None or type(actual_placement) is not type(
                placement
            ):
                raise ValueError(
                    f"Output tensor placement mismatch on {axis}: "
                    f"out_placements={placement} but spmd_types inferred "
                    f"spmd.{actual.name}"
                )

        # converts out types -> out bwd types -> to DTensor placements, e.g.
        # P (or V) -> R -> Replicate
        # R -> P -> Partial
        # I -> I -> Replicate
        # V -> V -> Shard (shard dim consulted from out_placements)
        grad_spec = []
        for dim_idx, placement in enumerate(spec):
            axis = spmd.MeshAxis.of(device_mesh.get_group(dim_idx))
            if actual_type[axis] == spmd.V and type(placement) is Partial:
                grad_placement = Replicate()
            else:
                grad_placement = _dtensor_placement_if_compatible(
                    actual_type[axis].backward_type(), placement
                )
            assert grad_placement is not None  # noqa: S101
            grad_spec.append(grad_placement)
        grad_out_placements.append(tuple(grad_spec))

    return tuple(grad_out_placements)


def local_map(
    func: Callable | None = None,
    out_placements: OutputPlacements = None,
    in_placements: InputPlacements = None,
    in_grad_placements: InputPlacements = None,
    device_mesh: DeviceMesh | None = None,
    *,
    redistribute_inputs: bool = False,
    spmd_types: bool = False,
):
    """
    :meth:`local_map` is an experimental API that allows users to pass :class:`DTensor` s
    to a function that is written to be applied on ``torch.Tensor`` s. It is done by extracting
    the local components of :class:`DTensor`, call the function, and wrap the outputs to
    :class:`DTensor` according to the ``out_placements``.

    Args:
        func (Callable): the function to be applied on each local shard of
            :class:`DTensor` s.
        out_placements (Union[`PlacementType`, Tuple[`PlacementType`, ...]]):
            the desired placements of the :class:`DTensor` s in ``func``'s flattened output.
            If the flattened ``output`` is a single value, the ``out_placements`` should be
            of type `PlacementType`. Otherwise if the flattened ``output`` has multiple
            values, the ``out_placements`` should be a tuple of `PlacementType` values 1:1
            mapping to the flattened ``output``.
            Besides, for :class:`Tensor` output, we use `PlacementType` as its
            placements (a `Tuple[Placement]` value). For non-Tensor output, the `PlacementType`
            should be `None`.
            Note that the only exception is when no :class:`DTensor` argument is passed
            in. In this case, even if `out_placements` is not `None`, the result function
            should ignore the desired placements because the function is not running with
            :class:`DTensor` s.
        in_placements (Tuple[`PlacementType`, ...], optional):
            the required placements of the :class:`DTensor` s in the flattened inputs of ``func``.
            If ``in_placements`` is specified, :meth:`local_map` would examine whether the
            placements of each :class:`DTensor` argument is the same as the required
            placements or not. If the placements are not the same and
            ``redistribute_inputs`` is ``False``, an exception will be raised. Otherwise if
            ``redistribute_inputs`` is ``True``, the argument will be first redistributed to
            the required sharding placements before passing its local tensor to ``func``.
            The only exception is when required placements are not ``None`` and the
            argument is a :class:`torch.Tensor`. In this case, the placements examination
            will be skipped and the argument will be directly passed to ``func``.
            If ``in_placements`` is ``None``, no placements examination will be performed.
            Default: None
        in_grad_placements (Tuple[`PlacementType`, ...], optional):
            the placements hint of the :class:`DTensor` s gradient corresponds
            to the flattened input DTensor. This argument is the hint that user
            can give to :meth:`to_local` in case the gradient layout of the
            local tensor input does not match its :class:`DTensor` input layout.
            If not specified, we will assume the gradient layout of the local
            tensor input remains the same as the original :class:`DTensor` input
            and use that for gradient computation. Default: None.
        device_mesh (:class:`DeviceMesh`, optional):
            the device mesh that the output :class:`DTensor` s are placed on. If not
            specified, this will be inferred from the first input :class:`DTensor`'s device
            mesh. Default: None.

    Keyword Args:
        redistribute_inputs (bool, optional):
            the bool value indicating whether to reshard the input :class:`DTensor` s when
            their placements are different from the required input placements. If this
            value is ``False`` and some :class:`DTensor` input has a different placement,
            an exception will be raised. Default: False.
        spmd_types (bool, optional):
            if ``True``, enable runtime SPMD type checking for ``func``. Local
            tensors are annotated with SPMD types (V, R, I, P) inferred from
            ``in_placements`` and ``in_grad_placements``, and ``func`` is executed
            under strict type checking. This helps surface distributed correctness
            bugs such as whether a replicated local tensor's gradient is
            Partial or Replicate.
            See the ``spmd_types`` local SPMD types documentation for the
            meaning of V, R, I, and P and their backward types:
            https://github.com/meta-pytorch/spmd_types/blob/main/docs/local_spmd_types.md
            For Replicate input placements, if a gradient placement is not provided,
            we default to Replicate (R), instead of Invariant (I). Requires the
            ``spmd_types`` package to be installed. Default: False.

    Returns:
        A ``Callable`` that applies ``func`` to each local shard of the input :class:`DTensor`
        and returns a :class:`DTensor` constructed from the return value of ``func``.

    Raises:
        AssertionError: For any non-DTensor output, we require its corresponding
            output placement in ``out_placements`` be None. An AssertionError will be raised
            if this is not the case.

        ValueError: If ``redistribute_inputs=False`` but the input :class:`DTensor` needs
            a redistribution according to ``in_placements``.

    Example:
        >>> # xdoctest: +SKIP("distributed")
        >>> def mm_allreduce_forward(device_mesh, W, X):
        >>>     partial_sum_tensor = torch.mm(W, X)
        >>>     reduced_tensor = funcol.all_reduce(partial_sum_tensor, "sum", device_mesh)
        >>>     return reduced_tensor
        >>>
        >>> W = torch.randn(12, 8, requires_grad=False)
        >>> X = torch.randn(8, 16, requires_grad=False)
        >>> Y = torch.mm(W, X)
        >>> row_wise = [Shard(0)]  # row-wise sharding placements on 1-d mesh
        >>> col_wise = [Shard(1)]  # col-wise sharding placements on 1-d mesh
        >>>
        >>> # local_mm_allreduce_forward is the function wrapped with DTensor/Tensor conversion
        >>> local_mm_allreduce_forward = local_map(
        >>>     mm_allreduce_forward,
        >>>     out_placements=[Replicate()],
        >>>     in_placements=[col_wise, row_wise],
        >>>     device_mesh=device_mesh,
        >>> )
        >>>
        >>> W_dt = distribute_tensor(
        ...     W, device_mesh, (col_wise)
        ... )  # col-wisely sharded W tensor
        >>> X_dt = distribute_tensor(
        ...     X, device_mesh, (row_wise)
        ... )  # row-wisely sharded X tensor
        >>> Y_dt = local_mm_allreduce_forward(
        ...     device_mesh, W_dt, X_dt
        ... )  # apply local_mm_allreduce_forward to DTensors

    .. note:: This API is currently experimental and subject to change
    """

    if func is None:
        # decorator mode
        def decorated(func):
            return local_map(
                func=func,
                out_placements=out_placements,
                in_placements=in_placements,
                in_grad_placements=in_grad_placements,
                device_mesh=device_mesh,
                redistribute_inputs=redistribute_inputs,
                spmd_types=spmd_types,
            )

        return decorated

    return functools.partial(
        _local_map_wrapped,
        func,
        out_placements,
        in_placements,
        in_grad_placements,
        device_mesh,
        redistribute_inputs,
        spmd_types,
    )


def _local_map_wrapped(
    func: Callable,
    out_placements: OutputPlacements,
    in_placements: InputPlacements,
    in_grad_placements: InputPlacements,
    device_mesh: DeviceMesh | None,
    redistribute_inputs: bool,
    enable_spmd_types: bool,
    *args,
    **kwargs,
):
    # process input args
    flat_args, args_spec = pytree.tree_flatten(args)
    if in_placements is not None:
        if len(in_placements) != len(flat_args):
            raise AssertionError(
                f"in_placements length {len(in_placements)} does not match the number "
                f"of input args {len(flat_args)}!"
            )

    # we assume every DTensor object is placed on the same device mesh
    flat_local_args = []
    seen_dtensor_arg = False
    for idx, arg in enumerate(flat_args):
        if isinstance(arg, DTensor):
            # TODO: the current code doesn't consider the uneven sharding case
            # Need to think about what the consequence is when the input DTensor
            # is uneven sharded.
            if device_mesh is None:  # infer device mesh from the DTensor arg
                device_mesh = arg.device_mesh

            # this function is applied to at least one DTensor argument
            seen_dtensor_arg = True

            if in_placements is not None:
                spec = in_placements[idx]
                if spec is None:
                    raise AssertionError(
                        f"DTensor input {arg} expects placements but received {spec}!"
                    )

                if not isinstance(spec, tuple):
                    spec = tuple(spec)

                if arg.placements != spec:
                    if redistribute_inputs:
                        # redistribute to input placements
                        arg = arg.redistribute(placements=spec)
                    else:
                        raise ValueError(
                            f"arg {arg} in local_map has a mismatched placements: "
                            f"arg placements is {arg.placements} but the input "
                            f"placements is {spec}! "
                            "If redistribute_inputs is wanted, set "
                            "redistribute_inputs=True to local_map."
                        )

            if in_grad_placements is not None:
                spec = in_grad_placements[idx]
                if spec is None:
                    raise AssertionError(
                        f"DTensor input {arg} expects in grad placements but received {spec}!"
                    )
                if not isinstance(spec, tuple):
                    spec = tuple(spec)
                local_arg = arg.to_local(grad_placements=spec)
            else:
                local_arg = arg.to_local()

            if isinstance(local_arg, AsyncCollectiveTensor):
                local_arg = local_arg.wait()

            flat_local_args.append(local_arg)
        else:
            # Non-Tensor input must have None in `in_placements`
            if in_placements is not None and not isinstance(arg, torch.Tensor):
                spec = in_placements[idx]
                if spec is not None:
                    raise AssertionError(
                        f"Non-Tensor input {arg} expects None placements "
                        f"but received {spec}!"
                    )

            flat_local_args.append(arg)

    # pyrefly: ignore [bad-argument-type]
    local_args = pytree.tree_unflatten(flat_local_args, args_spec)

    if enable_spmd_types and seen_dtensor_arg:
        if not dist._is_spmd_types_available():
            raise RuntimeError(
                "spmd_types=True requires the spmd_types package to be installed"
            )
        assert device_mesh is not None  # noqa: S101
        _annotate_spmd_types(
            flat_local_args, in_placements, in_grad_placements, device_mesh
        )
        from spmd_types._checker import typecheck

        with (
            spmd.set_current_mesh(device_mesh),
            typecheck(strict_mode="strict"),
        ):
            out = func(*local_args, **kwargs)
    else:
        out = func(*local_args, **kwargs)

    if seen_dtensor_arg:
        # process output to be DTensor if we've seen DTensor inputs
        flat_out, out_spec = pytree.tree_flatten(out)

        flat_dist_out = []
        out_placements_tuple = (
            out_placements if isinstance(out_placements, tuple) else (out_placements,)
        )
        if len(flat_out) != len(out_placements_tuple):
            raise AssertionError(
                "local_map requires one PlacementType be provided for each output value,"
                f" received {len(out_placements_tuple)} out_placements but"
                f" {len(flat_out)} is expected!"
            )

        grad_out_placements = None
        if enable_spmd_types:
            assert device_mesh is not None  # noqa: S101
            grad_out_placements = _out_spmd_types_to_grad_placements(
                flat_out,
                out_placements_tuple,  # pyrefly: ignore [bad-argument-type]
                device_mesh,
            )

        for idx, (out, spec) in enumerate(zip(flat_out, out_placements_tuple)):
            if isinstance(out, torch.Tensor):
                if isinstance(out, DTensor):
                    raise AssertionError(
                        f"torch.Tensor output expected but received {type(out)}: {out}"
                    )
                grad_placements = (
                    grad_out_placements[idx]
                    if grad_out_placements is not None
                    else None
                )

                flat_dist_out.append(
                    # pyrefly: ignore [bad-argument-type]
                    DTensor.from_local(
                        out,
                        device_mesh,
                        spec,  # pyrefly: ignore[bad-argument-type]
                        run_check=False,
                        grad_placements=grad_placements,
                    )
                )
            else:
                if spec is not None:
                    raise AssertionError(
                        f"Non-tensor output {out} expects None placements but received {spec}!"
                    )

                flat_dist_out.append(out)

        # pyrefly: ignore [bad-argument-type]
        return pytree.tree_unflatten(flat_dist_out, out_spec)
    else:
        return out
