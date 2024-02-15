# Copyright (c) Meta Platforms, Inc. and affiliates
import contextlib
from typing import cast, Dict, Optional, Tuple

import torch
import torch._prims_common as utils
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch import Tensor
from torch.distributed._tensor import DTensor, Replicate
from torch.distributed._tensor.ops.embedding_ops import _MaskPartial
from torch.distributed._tensor.ops.math_ops import (
    _skip_dim,
    Reduction,
    replicate_reduction_dims,
)
from torch.distributed._tensor.placement_types import Placement, TensorMeta
from torch.distributed.device_mesh import DeviceMesh

aten = torch.ops.aten


@contextlib.contextmanager
def loss_parallel():
    """
    Context manager to enable loss parallelism.
    """
    _enable_custom_loss_ops()

    yield

    _disable_custom_loss_ops()


# Currently only needs to support one dimensional DeviceMesh; in general return
# the mesh_dim with placements[mesh_dim].is_shard(dim)
def _find_all_reduce_mesh_dim(placements: Tuple[Placement, ...], dim: int) -> int:
    if not len(placements) == 1:
        raise ValueError(
            "Currently loss_parallel() only supports input on one-dimensional DeviceMesh."
        )
    if not placements[0].is_shard(dim):
        raise ValueError(
            f"loss_parallel() should be enabled only when the input tensor is sharded on dimension {dim}."
        )
    return 0


def _propagate_tensor_meta(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> TensorMeta:
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    tensor_meta = DTensor._op_dispatcher.sharding_propagator._propagate_tensor_meta(
        op_info.schema
    )
    if isinstance(tensor_meta, TensorMeta):
        return tensor_meta
    elif isinstance(tensor_meta, tuple):
        return tensor_meta[0]
    else:
        raise RuntimeError(f"Unexpected tensor meta type: {type(tensor_meta)}.")


# NOTE: The implementation follows torch._decomp.decomposition._log_softmax,
# with all_reduce manually inserted to perform distributed computation.
def _log_softmax(x, dim, half_to_float, mesh, mesh_dim):
    x = x.contiguous()
    if half_to_float:
        assert x.dtype == torch.half
    computation_dtype, result_dtype = utils.elementwise_dtypes(
        x, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    x = x.to(computation_dtype)
    if x.numel() == 0:
        shifted = x
    else:
        x_max = torch.amax(x, dim, keepdim=True)
        x_max = funcol.all_reduce(
            x_max, reduceOp=c10d.ReduceOp.MAX.name, group=(mesh, mesh_dim)
        )
        shifted = x - x_max
    shifted_sumexp = torch.sum(torch.exp(shifted), dim, keepdim=True)
    shifted_sumexp = funcol.all_reduce(
        shifted_sumexp, reduceOp=c10d.ReduceOp.SUM.name, group=(mesh, mesh_dim)
    )
    shifted_logsumexp = torch.log(shifted_sumexp)
    result = shifted - shifted_logsumexp
    if not half_to_float:
        result = result.to(result_dtype)
    return result


def _log_softmax_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> object:
    x = cast(DTensor, args[0])
    dim = cast(int, args[1])
    half_to_float = cast(bool, args[2])

    spec = x._spec
    mesh_dim = _find_all_reduce_mesh_dim(spec.placements, dim)
    res = _log_softmax(x._local_tensor, dim, half_to_float, spec.mesh, mesh_dim)

    output_tensor_meta = _propagate_tensor_meta(op_call, args, kwargs)

    return DTensor(
        res,
        spec.mesh,
        spec.placements,
        shape=output_tensor_meta.shape,
        dtype=output_tensor_meta.dtype,
        requires_grad=res.requires_grad,
        stride=output_tensor_meta.stride,
    )


# NOTE: As explained below at _nll_loss_and_log_softmax_backward, the
# _log_softmax_backward_handler does not actually do any computation.
def _log_softmax_backward_no_computation(
    grad_output: Tensor, output: Tensor, dim: int, input_dtype: torch.dtype
):
    # The log_softmax backward computation has been performed during
    # nll_loss backward in _nll_loss_and_log_softmax_backward.
    grad_input = grad_output

    if grad_output.dtype != input_dtype:
        grad_input = grad_input.to(input_dtype)
    return grad_input


def _log_softmax_backward_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> object:
    grad_output = cast(DTensor, args[0])
    output = cast(DTensor, args[1])
    dim = cast(int, args[2])
    input_dtype = cast(torch.dtype, args[3])

    spec = grad_output._spec
    res = _log_softmax_backward_no_computation(
        grad_output._local_tensor, output._local_tensor, dim, input_dtype
    )

    assert spec.tensor_meta is not None
    return DTensor(
        res,
        spec.mesh,
        spec.placements,
        shape=spec.tensor_meta.shape,
        dtype=input_dtype,
        requires_grad=res.requires_grad,
        stride=spec.tensor_meta.stride,
    )


# NOTE: The implementation follows torch._decomp.decomposition._nll_loss_forward,
# with customized communication inserted to perform distributed computation.
def _nll_loss_forward(
    x: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: int,
    channel_dim_size: int,
    mesh: DeviceMesh,
    mesh_dim: int,
) -> Tuple[Tensor, Tensor]:
    n_dims = x.dim()
    channel_dim = 1
    if n_dims < 2:
        channel_dim = 0

    if weight is not None:
        if n_dims > 1:
            shape = [
                1,
            ] * n_dims
            shape[channel_dim] = weight.shape[0]
            w = weight.view(shape)
        else:
            w = weight
        x = x * w
    safe_target = torch.where(target != ignore_index, target, 0)
    safe_target_ = safe_target.unsqueeze(channel_dim)

    # The following code block is a distributed version of
    # result = -torch.gather(self, channel_dim, safe_target_).squeeze(channel_dim)
    partial_placement = _MaskPartial(logical_dim_size=channel_dim_size)
    safe_target_ = partial_placement._partition_value(safe_target_, mesh, mesh_dim)
    result_partial = torch.gather(x, channel_dim, safe_target_)
    # an all_reduce happens here
    result_reduced = partial_placement._reduce_value(result_partial, mesh, mesh_dim)
    result = -result_reduced.squeeze(channel_dim)

    result = torch.where(target != ignore_index, result, 0)

    if reduction == Reduction.NONE.value and n_dims > 1:
        total_weight = x.new_full((), 0.0)
        return result, total_weight

    if weight is not None:
        w = w.expand(x.shape)
        wsum = torch.gather(w, channel_dim, safe_target_).squeeze(channel_dim)
        wsum = torch.where(target != ignore_index, wsum, 0)
        total_weight = wsum.sum()
    else:
        total_weight = (target != ignore_index).sum().to(x)

    if reduction == Reduction.SUM.value:
        result = result.sum()
    elif reduction == Reduction.MEAN.value:
        result = result.sum() / total_weight

    return result, total_weight


# TODO: add input shapes checking like in torch._decomp.decomposition
def _nll_loss_forward_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> object:
    x = cast(DTensor, args[0])
    target = cast(torch.Tensor, args[1])
    weight = cast(Optional[torch.Tensor], args[2])
    reduction = cast(int, args[3])
    ignore_index = cast(int, args[4])

    channel_dim = 1 if x.dim() >= 2 else 0
    channel_dim_size = x.shape[channel_dim]
    spec = x._spec
    mesh_dim = _find_all_reduce_mesh_dim(spec.placements, channel_dim)
    result, total_weight = _nll_loss_forward(
        x._local_tensor,
        target,
        weight,
        reduction,
        ignore_index,
        channel_dim_size,
        spec.mesh,
        mesh_dim,
    )

    if reduction == Reduction.NONE.value:
        output_placements = _skip_dim(
            replicate_reduction_dims(spec.placements, [channel_dim]), channel_dim
        )
    else:
        output_placements = (Replicate(),) * spec.mesh.ndim

    output_tensor_meta = _propagate_tensor_meta(op_call, args, kwargs)

    return (
        DTensor(
            result,
            spec.mesh,
            output_placements,
            shape=output_tensor_meta.shape,
            dtype=output_tensor_meta.dtype,
            requires_grad=result.requires_grad,
            stride=output_tensor_meta.stride,
        ),
        total_weight,
    )


# NOTE: The backward computation of cross_entropy goes through two steps:
# backward for nll_loss and then backward for log_softmax. In loss parallel,
# the two steps are fused into the following function (called by _nll_loss_backward_handler)
# to avoid communication when target contains class indices not class probabilities.
# Also note that the _log_softmax_backward_handler does not perform computation.
# The implementation resembles _nll_loss_backward and _log_softmax_backward_data
# from torch._decomp.decomposition.
def _nll_loss_and_log_softmax_backward(
    grad_output: Tensor,
    x: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: int,
    total_weight: Tensor,
    channel_dim_size: int,
    mesh: DeviceMesh,
    mesh_dim: int,
) -> Tensor:
    channel_dim = 0 if x.dim() < 2 else 1
    if reduction == Reduction.MEAN.value:
        grad_output = grad_output / total_weight

    target = target.unsqueeze(channel_dim)
    safe_target = torch.where(target != ignore_index, target, 0)
    grad_input = torch.zeros_like(x)

    # The following code block is a distributed version of
    # grad_input = torch.scatter(grad_input, channel_dim, safe_target, -1.0)
    partial_placement = _MaskPartial(logical_dim_size=channel_dim_size)
    safe_target = safe_target.squeeze(channel_dim).flatten()
    masked_safe_target = partial_placement._partition_value(safe_target, mesh, mesh_dim)
    # only update grad_input to -1 if not masked
    assert partial_placement.mask_buffer.data is not None
    grad_update = partial_placement.mask_buffer.data.float() - 1.0
    arange_1d = torch.arange(
        masked_safe_target.shape[0], device=masked_safe_target.device
    )
    if x.dim() <= 2:  # for aten.nll_loss_backward.default
        grad_input[arange_1d, masked_safe_target] = grad_update
    else:  # for aten.nll_loss2d_backward.default
        grad_input_t = grad_input.transpose(channel_dim, -1)
        intermidate_shape = grad_input_t.shape
        grad_input_2d = grad_input_t.reshape(-1, x.shape[channel_dim])
        grad_input_2d[arange_1d, masked_safe_target] = grad_update
        grad_input = grad_input_2d.view(intermidate_shape).transpose(channel_dim, -1)

    if grad_input.dim() > grad_output.dim() > 0:
        grad_output = grad_output.unsqueeze(channel_dim)

    if weight is not None:
        new_shape = [1 for _ in range(x.dim())]
        new_shape[channel_dim] = weight.shape[0]
        weight = weight.reshape(new_shape)
        grad_output = grad_output * weight

    grad_output = torch.where(target != ignore_index, grad_output, 0)

    # NOTE: Instead of directly returning the grad_input as grad_output for log_softmax,
    # here we perform backward computation for log_softmax altogether to avoid the
    # otherwise extra all_gather communication.
    # return grad_input * grad_output
    return (grad_input + torch.exp(x)) * grad_output


# TODO: add input shapes checking like in torch._decomp.decomposition
def _nll_loss_backward_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> object:
    grad_output = cast(DTensor, args[0])
    x = cast(DTensor, args[1])
    target = cast(torch.Tensor, args[2])
    weight = cast(Optional[torch.Tensor], args[3])
    reduction = cast(int, args[4])
    ignore_index = cast(int, args[5])
    total_weight = cast(Tensor, args[6])

    channel_dim = 1 if x.dim() >= 2 else 0
    channel_dim_size = x.shape[channel_dim]
    spec = x._spec
    mesh_dim = _find_all_reduce_mesh_dim(spec.placements, channel_dim)
    result = _nll_loss_and_log_softmax_backward(
        grad_output._local_tensor,
        x._local_tensor,
        target,
        weight,
        reduction,
        ignore_index,
        total_weight,
        channel_dim_size,
        spec.mesh,
        mesh_dim,
    )

    output_tensor_meta = _propagate_tensor_meta(op_call, args, kwargs)

    return DTensor(
        result,
        spec.mesh,
        # the output sharding is the same as input sharding: Shard(channel_dim) on mesh_dim
        spec.placements,
        shape=output_tensor_meta.shape,
        dtype=output_tensor_meta.dtype,
        requires_grad=result.requires_grad,
        stride=output_tensor_meta.stride,
    )


customized_loss_ops = {
    aten._log_softmax.default: _log_softmax_handler,
    aten._log_softmax_backward_data.default: _log_softmax_backward_handler,
    aten.nll_loss_forward.default: _nll_loss_forward_handler,
    aten.nll_loss2d_forward.default: _nll_loss_forward_handler,
    aten.nll_loss_backward.default: _nll_loss_backward_handler,
    aten.nll_loss2d_backward.default: _nll_loss_backward_handler,
}


def _enable_custom_loss_ops():
    DTensor._op_dispatcher._custom_op_handlers.update(customized_loss_ops)
    DTensor._op_dispatcher._allow_implicit_replication = True


def _disable_custom_loss_ops():
    DTensor._op_dispatcher._allow_implicit_replication = False
    for custom_op in customized_loss_ops:
        DTensor._op_dispatcher._custom_op_handlers.pop(custom_op)
