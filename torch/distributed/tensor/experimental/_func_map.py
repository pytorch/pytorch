# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import functools
import warnings
from collections.abc import Callable, Sequence
from typing import cast, Optional, Union

import torch
import torch.distributed._functional_collectives as funcol
from torch._prims_common import make_contiguous_strides_for
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed._local_tensor import maybe_run_for_local_tensor
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.distributed.tensor.placement_types import Placement, Shard


try:
    from torch.utils import _cxx_pytree as pytree
except ImportError:
    from torch.utils import _pytree as pytree  # type: ignore[no-redef]


__all__ = ["local_map"]

_WARNINGS_SHOWN: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg not in _WARNINGS_SHOWN:
        _WARNINGS_SHOWN.add(msg)
        warnings.warn(msg, stacklevel=2)


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
    allow_uneven_sharding: bool = False,
    out_shapes: Sequence[torch.Size] | None = None,
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
        allow_uneven_sharding (bool, optional):
            whether to compute global shapes via all-gather. When ``False`` (default), :meth:`local_map`
            assumes even sharding and computes global shapes locally without communication. This is
            fast but will warn and produce incorrect results if inputs are unevenly sharded. When ``True``,
            :meth:`local_map` computes global shapes via all-gather of local shard sizes, correctly
            handling both even and uneven sharding at the cost of additional communication overhead.
            For zero-overhead handling of uneven sharding, use ``out_shapes`` instead. Default: False.
        out_shapes (Sequence[torch.Size], optional):
            the global shapes of output :class:`DTensor` s. If provided, these shapes will be used
            directly without any communication, enabling zero-overhead handling of uneven sharding.
            This should be a sequence matching the flattened outputs, with ``None`` for non-Tensor
            outputs. When ``out_shapes`` is provided, it takes precedence over ``allow_uneven_sharding``.
            Default: None.

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
                allow_uneven_sharding=allow_uneven_sharding,
                out_shapes=out_shapes,
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
        allow_uneven_sharding,
        out_shapes,
    )


def _infer_global_shape_local_even_sharding(
    local_tensor: torch.Tensor,
    device_mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> tuple[torch.Size, tuple[int, ...]]:
    """
    Infer global shape locally assuming even sharding (no communication).

    This is the fast path that assumes even sharding and computes the global
    size by multiplying local size by mesh size. Will produce incorrect results
    if sharding is actually uneven.

    Args:
        local_tensor: The local tensor on this rank
        device_mesh: The device mesh
        placements: The placements describing how the tensor is distributed

    Returns:
        A tuple of (global_shape, global_stride)
    """
    local_shape = list(local_tensor.shape)
    global_shape = list(local_shape)

    for mesh_dim, placement in enumerate(placements):
        if not isinstance(placement, Shard):
            continue

        global_shape[placement.dim] *= device_mesh.size(mesh_dim)

    global_shape_tuple = torch.Size(global_shape)
    return global_shape_tuple, make_contiguous_strides_for(global_shape_tuple)


def _allgather_global_shape_uneven_sharding(
    local_tensor: torch.Tensor,
    device_mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> tuple[torch.Size, tuple[int, ...]]:
    """
    Compute global shape via all-gather, supporting uneven sharding.

    For each Shard placement, this function gathers local sizes from all ranks
    along the mesh dimension and sums them to get the global size. This adds
    communication overhead but correctly handles uneven sharding.

    When the same tensor dim is sharded on multiple mesh dims, this function
    processes them sequentially, using the accumulated size from previous
    gathers as input to subsequent gathers.

    Args:
        local_tensor: The local tensor on this rank
        device_mesh: The device mesh
        placements: The placements describing how the tensor is distributed

    Returns:
        A tuple of (global_shape, global_stride)
    """
    # TODO: a CPU-side control plane for exchanging shape information would
    # avoid the GPU sync and graph break caused by this path.
    _warn_once(
        "allow_uneven_sharding=True uses all-gather to compute global shapes, "
        "which introduces communication overhead and a CPU-GPU sync that will "
        "cause a graph break under torch.compile. For better performance, use "
        "out_shapes to provide global shapes directly with zero overhead."
    )

    local_shape = local_tensor.shape
    global_shape = list(local_shape)

    has_shard = any(isinstance(p, Shard) for p in placements)

    if not has_shard:
        return torch.Size(global_shape), local_tensor.stride()

    @maybe_run_for_local_tensor
    def _create_size_tensor(size: int) -> torch.Tensor:
        return torch.tensor([size], device=device_mesh.device_type)

    @maybe_run_for_local_tensor
    def _sum_sizes(gathered_sizes: list[torch.Tensor]) -> int:
        return int(torch.stack(gathered_sizes).sum())

    for mesh_dim, placement in enumerate(placements):
        if not isinstance(placement, Shard):
            continue

        shard_dim = placement.dim

        # Use the current running size (starts as local, accumulates across mesh dims)
        current_size = global_shape[shard_dim]
        current_size_tensor = _create_size_tensor(current_size)

        mesh_size = device_mesh.size(mesh_dim)
        gathered_sizes = [
            torch.empty_like(current_size_tensor, device=current_size_tensor.device)
            for _ in range(mesh_size)
        ]

        funcol.all_gather_inplace(
            gathered_sizes,
            current_size_tensor,
            (device_mesh, mesh_dim),
        )

        global_shape[shard_dim] = _sum_sizes(gathered_sizes)

    global_shape_tuple = torch.Size(global_shape)
    return global_shape_tuple, make_contiguous_strides_for(global_shape_tuple)


def _local_map_wrapped(
    func: Callable,
    out_placements: OutputPlacements,
    in_placements: InputPlacements,
    in_grad_placements: InputPlacements,
    device_mesh: DeviceMesh | None,
    redistribute_inputs: bool,
    allow_uneven_sharding: bool,
    out_shapes: Sequence[torch.Size] | None,
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
            if device_mesh is None:  # infer device mesh from the DTensor arg
                device_mesh = arg.device_mesh

            # this function is applied to at least one DTensor argument
            seen_dtensor_arg = True

            # Warn if input is unevenly sharded and user hasn't opted in.
            # TODO: make this a hard error in the next PyTorch release.
            if not allow_uneven_sharding and out_shapes is None:
                for mesh_dim, p in enumerate(arg.placements):
                    if (
                        isinstance(p, Shard)
                        and arg.shape[p.dim] % arg.device_mesh.size(mesh_dim) != 0
                    ):
                        _warn_once(
                            f"Input DTensor has uneven sharding on dim {p.dim} "
                            f"(global size {arg.shape[p.dim]} across "
                            f"{arg.device_mesh.size(mesh_dim)} ranks). "
                            "Output global shapes will be incorrect. Use "
                            "allow_uneven_sharding=True or out_shapes to "
                            "handle uneven sharding correctly. "
                            "This will become a hard error in the next "
                            "PyTorch release."
                        )
                        break

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
        assert device_mesh is not None, (
            "device_mesh must be set when DTensor args are present"
        )
        flat_out, out_spec = pytree.tree_flatten(out)

        flat_dist_out = []
        out_placements_tuple = cast(
            tuple[PlacementType, ...],
            out_placements if isinstance(out_placements, tuple) else (out_placements,),
        )
        assert len(flat_out) == len(out_placements_tuple), (
            "local_map requires one PlacementType be provided for each output value,"
            f" received {len(out_placements_tuple)} out_placements but"
            f" {len(flat_out)} is expected!"
        )
        for out_idx, (out, spec) in enumerate(zip(flat_out, out_placements_tuple)):
            placements: Sequence[Placement] | None = spec
            if isinstance(out, torch.Tensor):
                assert not isinstance(out, DTensor), (
                    f"torch.Tensor output expected but received {type(out)}: {out}"
                )

                if placements is not None and any(
                    isinstance(p, Shard) for p in placements
                ):
                    if out_shapes is not None and out_idx < len(out_shapes):
                        # user provided shape - use it directly (zero overhead)
                        provided_shape = out_shapes[out_idx]
                        if provided_shape is not None:
                            global_shape = provided_shape
                            global_stride = make_contiguous_strides_for(provided_shape)
                        else:
                            raise RuntimeError(
                                f"out_shapes[{out_idx}] is None for a tensor output. "
                                "Please provide a valid shape."
                            )
                    elif allow_uneven_sharding:
                        # all-gather to compute global shape (handles both even and uneven)
                        global_shape, global_stride = (
                            _allgather_global_shape_uneven_sharding(
                                out,
                                device_mesh,
                                placements,
                            )
                        )
                    else:
                        # Default: assume even sharding, compute without communication
                        global_shape, global_stride = (
                            _infer_global_shape_local_even_sharding(
                                out,
                                device_mesh,
                                placements,
                            )
                        )

                    flat_dist_out.append(
                        DTensor.from_local(
                            out,
                            device_mesh,
                            placements,
                            run_check=False,
                            shape=global_shape,
                            stride=global_stride,
                        )
                    )
                else:
                    flat_dist_out.append(
                        DTensor.from_local(
                            out, device_mesh, placements, run_check=False
                        )
                    )
            else:
                assert placements is None, (
                    f"Non-tensor output {out} expects None placements but received {placements}!"
                )

                flat_dist_out.append(out)

        # pyrefly: ignore [bad-argument-type]
        return pytree.tree_unflatten(flat_dist_out, out_spec)
    else:
        return out
