# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
from typing import cast

import torch
import torch.distributed as dist
import torch.distributed.tensor._api as dtensor


aten = torch.ops.aten


def _requires_data_exchange(padding):
    # TODO: whether there requires data exchange is currently determined by padding
    return padding[1] != 0


def _is_supported(input_size, kernel_size, stride, padding, dilation):
    if dilation[1] != 1:
        raise RuntimeError("Dilation must be 1 for tensor parallel convolution.")
    if padding[1] != 0:
        if stride[1] != 1:
            raise RuntimeError(
                "Stride must be 1 when there is padding for tensor parallel convolution."
            )
        if kernel_size[3] // 2 > input_size[3]:
            raise RuntimeError(
                "kernel_size[3] // 2 should be less than or equal to input_size[3] for tensor parallel convolution."
            )
    else:
        if not (input_size[3] % stride[1] == 0 and stride[1] == kernel_size[3]):
            raise RuntimeError(
                "It requires that input_size[3] is divisible by stride[1] and stride[1] equals kernel_size[3] "
                "when there is padding for tensor parallel convolution."
            )
    return True


def _ring_send_recv_construct(in_tensor, d1, d2, left, right, rank, size):
    # dist comms and reconstruct local input tensor
    send_to_right = in_tensor[:, :, :, -d1:].contiguous()
    send_to_left = in_tensor[:, :, :, :d2].contiguous()
    recv_from_right = torch.zeros_like(send_to_left)
    recv_from_left = torch.zeros_like(send_to_right)

    send_op_right = dist.P2POp(dist.isend, send_to_right, right)
    send_op_left = dist.P2POp(dist.isend, send_to_left, left)
    recv_op_right = dist.P2POp(dist.irecv, recv_from_right, right)
    recv_op_left = dist.P2POp(dist.irecv, recv_from_left, left)

    reqs = dist.batch_isend_irecv(
        [send_op_right, send_op_left, recv_op_left, recv_op_right]
    )
    for req in reqs:
        req.wait()

    if rank == 0:
        in_tensor = torch.cat([in_tensor, recv_from_right], dim=-1)
    elif rank == size - 1:
        in_tensor = torch.cat([recv_from_left, in_tensor], dim=-1)
    else:
        in_tensor = torch.cat([recv_from_left, in_tensor, recv_from_right], dim=-1)

    return in_tensor


def _ring_send_recv_aggregate(grad_in_tensor, d1, d2, left, right, rank, size):
    # dist comms and aggregate gradients for edge pixels
    send_to_right = grad_in_tensor[:, :, :, -d2:].contiguous()
    send_to_left = grad_in_tensor[:, :, :, :d1].contiguous()
    recv_from_right = torch.zeros_like(send_to_left)
    recv_from_left = torch.zeros_like(send_to_right)

    send_op_right = dist.P2POp(dist.isend, send_to_right, right)
    send_op_left = dist.P2POp(dist.isend, send_to_left, left)
    recv_op_right = dist.P2POp(dist.irecv, recv_from_right, right)
    recv_op_left = dist.P2POp(dist.irecv, recv_from_left, left)

    reqs = dist.batch_isend_irecv(
        [send_op_right, send_op_left, recv_op_left, recv_op_right]
    )
    for req in reqs:
        req.wait()

    if rank == 0:
        grad_in_tensor = grad_in_tensor[:, :, :, :-d2]
        grad_in_tensor[:, :, :, -d1:] = torch.add(
            grad_in_tensor[:, :, :, -d1:], recv_from_right
        )
    elif rank == size - 1:
        grad_in_tensor = grad_in_tensor[:, :, :, d1:]
        grad_in_tensor[:, :, :, :d2] = torch.add(
            grad_in_tensor[:, :, :, :d2], recv_from_left
        )
    else:
        grad_in_tensor = grad_in_tensor[:, :, :, d1:-d2]
        grad_in_tensor[:, :, :, -d1:] = torch.add(
            grad_in_tensor[:, :, :, -d1:], recv_from_right
        )
        grad_in_tensor[:, :, :, :d2] = torch.add(
            grad_in_tensor[:, :, :, :d2], recv_from_left
        )


def tp_convolution(
    op_call: torch._ops.OpOverload,
    local_tensor_args: tuple[object, ...],
    local_tensor_kwargs: dict[str, object],
) -> object:
    assert op_call == aten.convolution.default
    assert len(local_tensor_args) == 9

    rank = dist.get_rank()
    size = dist.get_world_size()
    in_tensor = cast(torch.Tensor, local_tensor_args[0])
    weight = cast(torch.Tensor, local_tensor_args[1])
    stride, padding, dilation = local_tensor_args[3:6]

    assert _is_supported(in_tensor.shape, weight.shape, stride, padding, dilation)
    assert isinstance(padding, list)

    if not _requires_data_exchange(padding):
        local_results = op_call(*local_tensor_args, **local_tensor_kwargs)
        return local_results
    else:
        # step 0 compute the overlap pixels of the input tensor
        d = weight.shape[3] - 1
        d1 = d // 2
        d2 = d - d1
        assert d1 + d2 == d
        right = (rank + 1) % size
        left = (rank - 1 + size) % size

        # step1 reconstruct local input tensor
        in_tensor = _ring_send_recv_construct(
            in_tensor, d1, d2, left, right, rank, size
        )

        # step2 feed local input tensor to op_call
        local_tensor_args_list = list(local_tensor_args)
        local_tensor_args_list[0] = in_tensor
        local_tensor_args = cast(tuple[object, ...], local_tensor_args_list)
        local_results = op_call(*local_tensor_args, **local_tensor_kwargs)

        # step3 remove extra outputs from the results
        padding_w = padding[1]
        w = local_results.size(3)
        if rank == 0:
            local_results = local_results[:, :, :, : w - padding_w]
        elif rank == size - 1:
            local_results = local_results[:, :, :, padding_w:]
        else:
            local_results = local_results[:, :, :, padding_w : w - padding_w]

        return local_results


def tp_convolution_backward(
    op_call: torch._ops.OpOverload,
    local_tensor_args: tuple[object, ...],
    local_tensor_kwargs: dict[str, object],
) -> object:
    assert op_call == aten.convolution_backward.default
    assert len(local_tensor_args) == 11

    rank = dist.get_rank()
    size = dist.get_world_size()
    grad_out_tensor = cast(torch.Tensor, local_tensor_args[0])
    in_tensor = cast(torch.Tensor, local_tensor_args[1])
    weight = cast(torch.Tensor, local_tensor_args[2])
    stride, padding, dilation = local_tensor_args[4:7]

    assert _is_supported(in_tensor.shape, weight.shape, stride, padding, dilation)
    assert isinstance(padding, list)

    if not _requires_data_exchange(padding):
        local_results = op_call(*local_tensor_args, **local_tensor_kwargs)
        return local_results
    else:
        # step 0 compute the overlap pixels of the input tensor
        d = weight.shape[3] - 1
        d1 = d // 2
        d2 = d - d1
        assert d1 + d2 == d
        right = (rank + 1) % size
        left = (rank - 1 + size) % size

        # step1 reconstruct local input tensor
        in_tensor = _ring_send_recv_construct(
            in_tensor, d1, d2, left, right, rank, size
        )

        # step2 reconstruct local gradient output tensor
        padding_w = padding[1]
        if rank == 0:
            grad_out_tensor = torch.nn.functional.pad(
                grad_out_tensor, (0, padding_w), "constant", 0
            )
        elif rank == size - 1:
            grad_out_tensor = torch.nn.functional.pad(
                grad_out_tensor, (padding_w, 0), "constant", 0
            )
        else:
            grad_out_tensor = torch.nn.functional.pad(
                grad_out_tensor, (padding_w, padding_w), "constant", 0
            )

        # step3 feed local input tensor to op_call
        local_tensor_args_list = list(local_tensor_args)
        local_tensor_args_list[0] = grad_out_tensor
        local_tensor_args_list[1] = in_tensor
        local_tensor_args = cast(tuple[object, ...], local_tensor_args_list)
        local_results = op_call(*local_tensor_args, **local_tensor_kwargs)

        # step4 aggregate gradients for edge pixels
        grad_in_tensor = local_results[0]
        if grad_in_tensor is not None:
            grad_in_tensor = _ring_send_recv_aggregate(
                grad_in_tensor, d1, d2, left, right, rank, size
            )
            local_results = list(local_results)
            local_results[0] = grad_in_tensor

        local_results = cast(tuple[object, ...], local_results)

        return local_results


def convolution_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    # extract local tensor and sharding infos to a OpInfo
    op_info = dtensor.DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)

    # sharding propagation
    dtensor.DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"

    # local propagation
    local_results = tp_convolution(
        op_call, tuple(op_info.local_args), op_info.local_kwargs
    )

    return dtensor.DTensor._op_dispatcher.wrap(
        local_results, output_sharding.output_spec
    )


def convolution_backward_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    # Redistribute grad_output tensor to the same placement as input tensor
    args = list(args)
    assert isinstance(args[0], dtensor.DTensor) and isinstance(args[1], dtensor.DTensor)
    args[0] = args[0].redistribute(args[1].device_mesh, args[1].placements)
    args = tuple(args)

    # extract local tensor and sharding infos to a OpInfo
    op_info = dtensor.DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)

    # sharding propagation
    dtensor.DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"

    # local propagation
    local_results = tp_convolution_backward(
        op_call, tuple(op_info.local_args), op_info.local_kwargs
    )

    return dtensor.DTensor._op_dispatcher.wrap(
        local_results, output_sharding.output_spec
    )
