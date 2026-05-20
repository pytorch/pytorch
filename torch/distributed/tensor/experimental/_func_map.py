# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import functools
from collections.abc import Callable, Sequence

import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.distributed.tensor.placement_types import Placement


try:
    from torch.utils import _cxx_pytree as pytree
except ImportError:
    from torch.utils import _pytree as pytree  # type: ignore[no-redef]


__all__ = ["local_map"]

PlacementType = Sequence[Placement] | None
InputPlacements = tuple[PlacementType, ...] | None
OutputPlacements = PlacementType | tuple[PlacementType, ...]


def _is_placement_spec(x):
    """Return ``True`` if ``x`` is a ``PlacementType`` leaf.

    A placement spec leaf is either ``None`` or a non-empty list/tuple whose
    elements are all :class:`Placement` instances. This is used as the
    ``is_leaf`` callback when pytree-flattening placement structures so that
    a placement spec for a single tensor is never recursed into.
    """
    if x is None:
        return True
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return all(isinstance(p, Placement) for p in x)
    return False


def _flatten_input_placements(placements, num_flat_args, label):
    """Flatten an input placement structure into a list of ``PlacementType``.

    ``placements`` may either be a flat tuple matching the pytree-flattened
    args (the original behavior) or a nested tuple/list mirroring the
    structure of ``args`` -- in both cases the leaves (each
    ``PlacementType``) are matched 1:1 with the flattened args.
    """
    if placements is None:
        return None
    flat, _ = pytree.tree_flatten(placements, is_leaf=_is_placement_spec)
    if len(flat) != num_flat_args:
        raise AssertionError(
            f"{label} length {len(flat)} does not match the number of "
            f"input args {num_flat_args}!"
        )
    return flat


def _flatten_output_placements(out_placements, num_flat_out):
    """Flatten an ``out_placements`` structure into a list of ``PlacementType``.

    Accepts a single ``PlacementType`` for a single-leaf output (preserving
    the existing shorthand), a flat tuple matching the pytree-flattened
    output, or a nested structure mirroring the output's tree. In all cases
    each leaf is a ``PlacementType``.
    """
    flat, _ = pytree.tree_flatten(out_placements, is_leaf=_is_placement_spec)
    if len(flat) != num_flat_out:
        raise AssertionError(
            "local_map requires one PlacementType be provided for each "
            f"output value, received {len(flat)} out_placements but "
            f"{num_flat_out} is expected!"
        )
    return flat


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
            the desired placements of the :class:`DTensor` s in ``func``'s pytree-flattened
            output. ``out_placements`` may either be a single `PlacementType` (when the
            flattened output is a single value), a flat tuple of `PlacementType` values
            matching the flattened output 1:1, or a nested structure mirroring the
            output's pytree whose leaves are each a `PlacementType`. In all cases the
            leaves are paired with the pytree-flattened output values.
            For :class:`Tensor` output, we use `PlacementType` as its placements
            (a `Tuple[Placement]` value). For non-Tensor output, the `PlacementType`
            should be `None`.
            Note that the only exception is when no :class:`DTensor` argument is passed
            in. In this case, even if `out_placements` is not `None`, the result function
            should ignore the desired placements because the function is not running with
            :class:`DTensor` s.
        in_placements (Tuple[`PlacementType`, ...], optional):
            the required placements of the :class:`DTensor` s in the pytree-flattened inputs
            of ``func``. ``in_placements`` may either be a flat tuple of `PlacementType`
            values matching the flattened inputs 1:1, or a nested tuple/list mirroring
            the structure of the positional arguments whose leaves are each a
            `PlacementType`. In both cases the leaves are paired with the
            pytree-flattened input values.
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
    flat_in_placements = _flatten_input_placements(
        in_placements, len(flat_args), "in_placements"
    )
    flat_in_grad_placements = _flatten_input_placements(
        in_grad_placements, len(flat_args), "in_grad_placements"
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

            if flat_in_placements is not None:
                spec = flat_in_placements[idx]
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

            if flat_in_grad_placements is not None:
                spec = flat_in_grad_placements[idx]
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
            if flat_in_placements is not None and not isinstance(arg, torch.Tensor):
                spec = flat_in_placements[idx]
                if spec is not None:
                    raise AssertionError(
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
        flat_out_placements = _flatten_output_placements(out_placements, len(flat_out))

        flat_dist_out = []
        for out, spec in zip(flat_out, flat_out_placements):
            if isinstance(out, torch.Tensor):
                if isinstance(out, DTensor):
                    raise AssertionError(
                        f"torch.Tensor output expected but received {type(out)}: {out}"
                    )

                flat_dist_out.append(
                    # pyrefly: ignore [bad-argument-type]
                    DTensor.from_local(out, device_mesh, spec, run_check=False)
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
