# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import contextlib
from typing import cast, Optional

import torch
import torch._prims_common as utils
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._ops._embedding_ops import _MaskPartial
from torch.distributed.tensor._ops._math_ops import (
    _skip_dim,
    Reduction,
    replicate_reduction_dims,
)
from torch.distributed.tensor.placement_types import Placement


aten = torch.ops.aten


__all__ = ["loss_parallel"]


@contextlib.contextmanager
def loss_parallel():
    """
    A context manager that enables loss parallelism, where efficient parallelized loss computation
    can be performed when the input is sharded on the class dimension. Currently only the cross-entropy
    loss is supported.

    Within this context manager, one can use :func:`~torch.nn.functional.cross_entropy` or
    :class:`~torch.nn.CrossEntropyLoss` as usual, with the following assumptions on the input parameters.
    The corresponding ``backward()`` call, if any, also needs to happen under this context manager.

    Args:
        input (:class:`DTensor`):
            Input logits. Assumed to be sharded on the class dimension.
        target (Union[:class:`torch.Tensor`, :class:`DTensor`]):
            Must be ground truth class indices (class probabilities currently not supported).
            Assumed to be replicated across the ``DeviceMesh``.
        weight (Union[:class:`torch.Tensor`, :class:`DTensor`], optional):
            If given, assumed to be replicated across the ``DeviceMesh``.
        label_smoothing:
            Currently not supported.

    Returns:
        A replicated :class:`DTensor`.

    Example:
        A sharded DTensor is manually created here to showcase the usage.
        In practice, it is usually the output of a TP module.

        >>> # xdoctest: +SKIP("distributed")
        >>> from torch.distributed.tensor.parallel import loss_parallel
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> device_mesh = init_device_mesh("cuda", (8,))
        >>> input = torch.randn(4, 16, device="cuda", requires_grad=True)
        >>> dist_input = distribute_tensor(input, device_mesh, placements=[Shard(1)])
        >>> target = torch.randint(16, (4,), device="cuda")
        >>> with loss_parallel():
        >>>     loss = F.cross_entropy(dist_input, target, reduction="mean")
        >>>     loss.backward()
        >>> ...
    """
    _enable_custom_loss_ops()

    yield

    _disable_custom_loss_ops()


# Currently only needs to support one dimensional DeviceMesh; in general return
# the mesh_dim with placements[mesh_dim].is_shard(dim)
def _find_all_reduce_mesh_dim(placements: tuple[Placement, ...], dim: int) -> int:
    if not len(placements) == 1:
        raise ValueError(
            "Currently loss_parallel() only supports input on one-dimensional DeviceMesh."
        )
    if not placements[0].is_shard(dim):
        raise ValueError(
            f"loss_parallel() should be enabled only when the input tensor is sharded on dimension {dim}."
        )
    return 0


def _cast_to_dtensor(
    tensor, placements: tuple[Placement, ...], mesh: DeviceMesh
) -> DTensor:
    if isinstance(tensor, DTensor):
        if tensor.placements == placements:
            return tensor
        else:
            raise RuntimeError(f"Expected {placements} but got {tensor.placements}.")
    elif isinstance(tensor, torch.Tensor):
        return DTensor.from_local(
            tensor, device_mesh=mesh, placements=placements, run_check=False
        )
    else:
        raise TypeError(f"Unsupported type {type(tensor)}")


def _propagate_tensor_meta(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
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
    if half_to_float:
        assert x.dtype == torch.half
    computation_dtype, result_dtype = utils.elementwise_dtypes(
        x, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    x = x.to(dtype=computation_dtype, memory_format=torch.contiguous_format)
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
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    x = cast(DTensor, args[0])
    dim = cast(int, args[1])
    half_to_float = cast(bool, args[2])

    spec = x._spec
    mesh_dim = _find_all_reduce_mesh_dim(spec.placements, dim)

    output_tensor_meta = _propagate_tensor_meta(op_call, args, kwargs)

    res = _log_softmax(x._local_tensor, dim, half_to_float, spec.mesh, mesh_dim)

    res_spec = DTensorSpec(
        spec.mesh,
        spec.placements,
        tensor_meta=output_tensor_meta,
    )

    return DTensor(
        res,
        res_spec,
        requires_grad=res.requires_grad,
    )


# NOTE: As explained below at _nll_loss_and_log_softmax_backward, the
# _log_softmax_backward_handler does not actually do any computation.
def _log_softmax_backward_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    grad_output = cast(DTensor, args[0])
    input_dtype = cast(torch.dtype, args[3])
    return grad_output.to(input_dtype)


# NOTE: The implementation follows torch._decomp.decomposition._nll_loss_forward,
# with customized communication inserted to perform distributed computation.
def _nll_loss_forward(
    x: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    local_weight: Optional[Tensor],
    reduction: int,
    ignore_index: int,
    input_shape: torch.Size,
    channel_dim: int,
    mesh: DeviceMesh,
    mesh_dim: int,
) -> tuple[Tensor, Tensor]:
    n_dims = x.dim()
    channel_dim = 1
    if n_dims < 2:
        channel_dim = 0

    def _weight_view(weight: Tensor) -> Tensor:
        if n_dims > 1:
            shape = [
                1,
            ] * n_dims
            shape[channel_dim] = weight.shape[0]
            w = weight.view(shape)
        else:
            w = weight
        return w

    if weight is not None:
        w = _weight_view(weight)
        assert local_weight is not None
        local_w = _weight_view(local_weight)
        x = x * local_w
    safe_target = torch.where(target != ignore_index, target, 0)
    safe_target_ = safe_target.unsqueeze(channel_dim)

    # The following code block is a distributed version of
    # result = -torch.gather(self, channel_dim, safe_target_).squeeze(channel_dim)
    partial_placement = _MaskPartial(offset_shape=input_shape, offset_dim=channel_dim)
    safe_target_partial_ = partial_placement._partition_value(
        safe_target_, mesh, mesh_dim
    )
    result_partial = torch.gather(x, channel_dim, safe_target_partial_)
    # an all_reduce happens here
    result_reduced = partial_placement._reduce_value(result_partial, mesh, mesh_dim)
    result = -result_reduced.squeeze(channel_dim)

    result = torch.where(target != ignore_index, result, 0)

    if reduction == Reduction.NONE.value and n_dims > 1:
        total_weight = x.new_full((), 0.0)
        return result, total_weight

    if weight is not None:
        new_shape = list(x.shape)
        new_shape[channel_dim] = -1
        w = w.expand(new_shape)
        wsum = torch.gather(w, channel_dim, safe_target_).squeeze(channel_dim)
        wsum = torch.where(target != ignore_index, wsum, 0)
        total_weight = wsum.sum()
    else:
        total_weight = (target != ignore_index).sum().to(x)

    # NOTE: this is correct only on 1D DeviceMesh; o/w additional
    #       all-reduce on result and total_weight is needed
    if reduction == Reduction.SUM.value:
        result = result.sum()
    elif reduction == Reduction.MEAN.value:
        result = result.sum() / total_weight

    return result, total_weight


def _nll_loss_forward_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    x = cast(DTensor, args[0])
    target = args[1]
    weight = args[2]
    reduction = cast(int, args[3])
    ignore_index = cast(int, args[4])

    channel_dim = 1 if x.dim() >= 2 else 0
    spec = x._spec
    mesh_dim = _find_all_reduce_mesh_dim(spec.placements, channel_dim)

    # Check user input: if target and weight are not DTensors, convert them to DTensors;
    # if they are DTensors, check that they have the desired placements.
    target_placements = _skip_dim(
        replicate_reduction_dims(spec.placements, [channel_dim]), channel_dim
    )
    all_replicate_placements = (Replicate(),) * spec.mesh.ndim
    target = _cast_to_dtensor(target, target_placements, spec.mesh)
    local_weight = None
    if weight is not None:
        weight = _cast_to_dtensor(weight, all_replicate_placements, spec.mesh)
        # For local computation, both (replicated) weight and (sharded) local_weight
        # are needed in _nll_loss_forward(). local_weight is generated here using
        # DTensor API, without incurring any communication.
        sharded_placements = [
            Shard(0) if i == mesh_dim else Replicate() for i in range(spec.mesh.ndim)
        ]
        local_weight = weight.redistribute(spec.mesh, sharded_placements)._local_tensor
        assert local_weight.shape[0] == x._local_tensor.shape[channel_dim]

    if reduction == Reduction.NONE.value:
        output_placements = target_placements
    else:
        output_placements = all_replicate_placements

    # tensor inputs to _propagate_tensor_meta need to be DTensors
    args = list(args)
    args[1], args[2] = target, weight
    output_tensor_meta = _propagate_tensor_meta(op_call, tuple(args), kwargs)

    result, total_weight = _nll_loss_forward(
        x._local_tensor,
        target._local_tensor,
        weight._local_tensor if weight is not None else None,
        local_weight,
        reduction,
        ignore_index,
        x.shape,
        channel_dim,
        spec.mesh,
        mesh_dim,
    )
    out_spec = DTensorSpec(spec.mesh, output_placements, tensor_meta=output_tensor_meta)

    return (
        DTensor(
            result,
            out_spec,
            requires_grad=result.requires_grad,
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
    input_shape: torch.Size,
    channel_dim: int,
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
    partial_placement = _MaskPartial(offset_shape=input_shape, offset_dim=channel_dim)
    safe_target = safe_target.squeeze(channel_dim).flatten()
    masked_safe_target = partial_placement._partition_value(safe_target, mesh, mesh_dim)
    # only update grad_input to -1 if not masked
    assert partial_placement.mask_buffer.data is not None
    grad_update = partial_placement.mask_buffer.data.to(grad_input.dtype) - 1.0
    arange_1d = torch.arange(
        masked_safe_target.shape[0], device=masked_safe_target.device
    )
    # The first two cases with x.dim() <= 2 are for aten.nll_loss_backward.default;
    # the last case is for aten.nll_loss2d_backward.default.
    if x.dim() == 1:
        grad_input[masked_safe_target] = grad_update
    elif x.dim() == 2:
        grad_input[arange_1d, masked_safe_target] = grad_update
    else:
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
        # In order for fused computation to work, the following line is rewritten.
        # grad_output = grad_output * weight
        new_shape = list(x.shape)
        new_shape[channel_dim] = -1
        w = weight.expand(new_shape)
        w_target = torch.gather(w, channel_dim, target)
        grad_output = grad_output * w_target

    grad_output = torch.where(target != ignore_index, grad_output, 0)

    # NOTE: Instead of directly returning the grad_input as grad_output for log_softmax,
    # here we perform backward computation for log_softmax altogether to avoid the
    # otherwise extra all_gather communication.
    # return grad_input * grad_output
    return (grad_input + torch.exp(x)) * grad_output


def _nll_loss_backward_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    grad_output = cast(DTensor, args[0])
    x = cast(DTensor, args[1])
    target = args[2]
    weight = args[3]
    reduction = cast(int, args[4])
    ignore_index = cast(int, args[5])
    total_weight = cast(Tensor, args[6])

    channel_dim = 1 if x.dim() >= 2 else 0
    spec = x._spec
    mesh_dim = _find_all_reduce_mesh_dim(spec.placements, channel_dim)

    # if target and weight are not DTensors, convert them to DTensors
    target_placements = _skip_dim(
        replicate_reduction_dims(spec.placements, [channel_dim]), channel_dim
    )
    all_replicate_placements = (Replicate(),) * spec.mesh.ndim
    target = _cast_to_dtensor(target, target_placements, spec.mesh)
    if weight is not None:
        weight = _cast_to_dtensor(weight, all_replicate_placements, spec.mesh)

    # tensor inputs to _propagate_tensor_meta need to be DTensors
    args = list(args)
    args[2], args[3] = target, weight
    args[6] = _cast_to_dtensor(total_weight, all_replicate_placements, spec.mesh)
    output_tensor_meta = _propagate_tensor_meta(op_call, tuple(args), kwargs)

    result = _nll_loss_and_log_softmax_backward(
        grad_output._local_tensor,
        x._local_tensor,
        target._local_tensor,
        weight._local_tensor if weight is not None else None,
        reduction,
        ignore_index,
        total_weight,
        x.shape,
        channel_dim,
        spec.mesh,
        mesh_dim,
    )
    # the output sharding is the same as input sharding: Shard(channel_dim) on mesh_dim
    out_spec = DTensorSpec(
        spec.mesh,
        spec.placements,
        tensor_meta=output_tensor_meta,
    )

    return DTensor(
        result,
        out_spec,
        requires_grad=result.requires_grad,
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


def _disable_custom_loss_ops():
    for custom_op in customized_loss_ops:
        DTensor._op_dispatcher._custom_op_handlers.pop(custom_op)
