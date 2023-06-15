from functools import wraps
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from torch.distributed._tensor import DeviceMesh, DTensor, Placement

from torch.utils._pytree import tree_flatten, tree_unflatten


PlacementType = Sequence[Placement]

InputLayouts = Union[PlacementType, Tuple]
OutputLayouts = Union[PlacementType, Tuple[PlacementType, ...]]


def _process_dtensor_inputs(flat_args, input_placements):
    input_placements = (
        [None] * len(flat_args) if input_placements is None else input_placements
    )

    flat_local_args = []
    for arg, spec in zip(flat_args, input_placements):
        if isinstance(arg, DTensor):
            if spec is not None:
                # TODO: see if we should just redistribute the DTensor here
                assert (
                    arg.placements == spec
                ), "DTensor input placement does not match the input placement hints"
            flat_local_args.append(arg.to_local())
        elif isinstance(arg, torch.Tensor):
            raise RuntimeError("dmap only support DTensor as input")
        else:
            assert (
                spec is None
            ), "Non-DTensor input should not have placement hints, please specify as None"
            flat_local_args.append(arg)

    return flat_local_args


def dmap(
    func: Callable,
    mesh: DeviceMesh,
    out_placements: OutputLayouts,
    input_placements: Optional[InputLayouts] = None,
):
    """
    dmap is distributed map, ``dmap(func)`` returns a function that maps the `func` over the
    dtensor inputs to their local shards. This let user write functions in a local (per-device)
    mode, instead of the DTensor "global" mode. i.e. we can write manual collectives
    inside and mix the function call with DTensor operations.

    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return one or more Tensors.
        mesh (:class:`DeviceMesh`): DeviceMesh that encodes the mesh topology
        out_placements (:class:`OutputLayouts`): the output placements that
            describes how the output tensors should be constructed as DTensor
            on the mesh.
        input_placements (:class:`InputLayouts`, optional): the input placements
            hints that describes the layout of the input DTensors. If not provided,
            it will be inferred from the input DTensors.

    Returns:
        Returns a new function. It takes the same inputs as ``func``, except each
        DTensor input will be converted to torch.Tensor. It takes returns the same
        outputs as ``func``, except each output will be converted to DTensor
        accoring to the output placements specified by ``out_placements``.

    NOTE: This API is currently experimental and subject to change
    """

    out_placements = (
        (out_placements,) if not isinstance(out_placements, tuple) else out_placements
    )

    @wraps(func)
    def wrapped(*args, **kwargs):
        flat_args, args_spec = tree_flatten(args)

        # TODO: this only converts the DTensor to local tensor, need to think about
        # the cases where user pass in a torch.Tensor, should we simply accept it by
        # assumes a sharding layout from input_placement, or should we "split"/shard the
        # tensor according to the input_placement
        flat_local_args = _process_dtensor_inputs(flat_args, input_placements)

        local_args = tree_unflatten(flat_local_args, args_spec)

        out = func(*local_args, **kwargs)

        flatten_out, flatten_spec = tree_flatten(out)

        assert len(flatten_out) == len(
            out_placements
        ), "Number of outputs does not match the number of placements"

        outputs = []
        for out, out_placement in zip(flatten_out, out_placements):
            if isinstance(out, torch.Tensor):
                # TODO: this probably only work for evenly sharded tensors, extend from_local
                # to support uneven sharding
                assert not isinstance(
                    out, DTensor
                ), "expecting torch.Tensor but found DTensor"
                dtensor = DTensor.from_local(out, mesh, out_placement, run_check=False)
                outputs.append(dtensor)
            else:
                assert (
                    out_placement is None
                ), "Non-tensor output should not have placement, please specify as None"
                outputs.append(out)

        return tree_unflatten(outputs, flatten_spec)

    return wrapped
