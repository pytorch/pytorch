# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import functools
from collections.abc import Callable, Sequence
from typing import Optional, Union

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed._local_tensor import maybe_run_for_local_tensor
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.distributed.tensor.placement_types import Placement, Shard


try:
    from torch.utils import _cxx_pytree as pytree
except ImportError:
    from torch.utils import _pytree as pytree  # type: ignore[no-redef]


__all__ = ["local_map"]

PlacementType = Optional[Sequence[Placement]]
InputPlacements = Optional[tuple[PlacementType, ...]]
OutputPlacements = Union[PlacementType, tuple[PlacementType, ...]]


def local_map(
    func: Callable | None = None,
    out_placements: OutputPlacements = None,
    in_placements: InputPlacements = None,
    in_grad_placements: InputPlacements = None,
    device_mesh: DeviceMesh | None = None,
    *,
    redistribute_inputs: bool = False,
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
    )


def _compute_global_shape_and_stride(
    local_tensor: torch.Tensor,
    device_mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> tuple[torch.Size, tuple[int, ...]]:
    """
    Compute the global shape and stride of a DTensor from a local tensor,
    supporting uneven sharding across ranks.

    For each Shard placement, this function gathers local shapes along the
    corresponding mesh dimension and computes the global size by summing
    the sharded dimension sizes across ranks.

    Args:
        local_tensor: The local tensor on this rank
        device_mesh: The device mesh
        placements: The placements describing how the tensor is distributed

    Returns:
        A tuple of (global_shape, global_stride)
    """
    local_shape = local_tensor.shape
    global_shape = list(local_shape)

    has_shard = any(isinstance(p, Shard) for p in placements)

    if not has_shard:
        # no sharding, so local shape == global shape
        return torch.Size(global_shape), local_tensor.stride()

    # for each Shard placement, gather shapes along that mesh dimension
    # and compute the global size for the sharded tensor dimension
    for mesh_dim, placement in enumerate(placements):
        if not isinstance(placement, Shard):
            continue

        shard_dim = placement.dim

        @maybe_run_for_local_tensor
        def _create_size_tensor(size):
            return torch.tensor([size], device=device_mesh.device_type)

        local_size_tensor = _create_size_tensor(local_shape[shard_dim])

        # gather sizes from all ranks along this mesh dimension
        # for N-D mesh, we gather only along the specific mesh dimension
        mesh_size = device_mesh.size(mesh_dim)
        gathered_sizes = [
            torch.empty_like(local_size_tensor, device=local_size_tensor.device)
            for _ in range(mesh_size)
        ]

        funcol.all_gather_inplace(
            gathered_sizes,
            local_size_tensor,
            (device_mesh, mesh_dim),
        )

        @maybe_run_for_local_tensor
        def _sum_sizes(gathered_sizes):
            total = 0
            for size_tensor in gathered_sizes:
                total += size_tensor.item()
            return total

        global_size = _sum_sizes(gathered_sizes)
        global_shape[shard_dim] = global_size

    global_shape = torch.Size(global_shape)

    global_stride = [1]
    for i in range(len(global_shape) - 1, 0, -1):
        global_stride.insert(0, global_stride[0] * global_shape[i])

    return global_shape, tuple(global_stride)


def _local_map_wrapped(
    func: Callable,
    out_placements: OutputPlacements,
    in_placements: InputPlacements,
    in_grad_placements: InputPlacements,
    device_mesh: DeviceMesh | None,
    redistribute_inputs: bool,
    *args,
    **kwargs,
):
    # process input args
    flat_args, args_spec = pytree.tree_flatten(args)
    if in_placements is not None:
        assert len(in_placements) == len(flat_args), (
            f"in_placements length {len(in_placements)} does not match the number "
            f"of input args {len(flat_args)}!"
        )

    # we assume every DTensor object is placed on the same device mesh
    flat_local_args = []
    seen_dtensor_arg = False
    for idx, arg in enumerate(flat_args):
        if isinstance(arg, DTensor):
            # Note: Uneven sharding is supported - when outputs are sharded,
            # we compute the global shape/stride via all-gather to handle
            # cases where local shards have different sizes.
            if device_mesh is None:  # infer device mesh from the DTensor arg
                device_mesh = arg.device_mesh

            # this function is applied to at least one DTensor argument
            seen_dtensor_arg = True

            if in_placements is not None:
                spec = in_placements[idx]
                assert spec is not None, (
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
                assert spec is not None, (
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
                assert spec is None, (
                    f"Non-Tensor input {arg} expects None placements "
                    f"but received {spec}!"
                )

            flat_local_args.append(arg)

    # pyrefly: ignore [bad-argument-type]
    local_args = pytree.tree_unflatten(flat_local_args, args_spec)

    out = func(*local_args, **kwargs)

    if seen_dtensor_arg:
        # process output to be DTensor if we've seen DTensor inputs
        flat_out, out_spec = pytree.tree_flatten(out)

        flat_dist_out = []
        out_placements_tuple = (
            out_placements if isinstance(out_placements, tuple) else (out_placements,)
        )
        assert len(flat_out) == len(out_placements_tuple), (
            "local_map requires one PlacementType be provided for each output value,"
            f" received {len(out_placements_tuple)} out_placements but"
            f" {len(flat_out)} is expected!"
        )
        for out, spec in zip(flat_out, out_placements_tuple):
            if isinstance(out, torch.Tensor):
                assert not isinstance(out, DTensor), (
                    f"torch.Tensor output expected but received {type(out)}: {out}"
                )

                if spec is not None and any(isinstance(p, Shard) for p in spec):
                    global_shape, global_stride = _compute_global_shape_and_stride(
                        out, device_mesh, spec
                    )
                    flat_dist_out.append(
                        # pyrefly: ignore [bad-argument-type]
                        DTensor.from_local(
                            out,
                            device_mesh,
                            spec,
                            run_check=False,
                            shape=global_shape,
                            stride=global_stride,
                        )
                    )
                else:
                    flat_dist_out.append(
                        # pyrefly: ignore [bad-argument-type]
                        DTensor.from_local(out, device_mesh, spec, run_check=False)
                    )
            else:
                assert spec is None, (
                    f"Non-tensor output {out} expects None placements but received {spec}!"
                )

                flat_dist_out.append(out)

        # pyrefly: ignore [bad-argument-type]
        return pytree.tree_unflatten(flat_dist_out, out_spec)
    else:
        return out
