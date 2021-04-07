import torch
import torch.distributed as dist
from torch import nn


def _quantize_per_tensor_cuda(x, scale, zero_point):
    y = torch.round(x / scale) + zero_point
    y = torch.clamp(y, 0, 255).to(torch.uint8)
    return y


def _dequantize_per_tensor_cuda(y, scale, zero_point):
    x = scale * (y.to(torch.float32) - zero_point)
    return x


def _quantize_per_channel_cuda(x, scale, zero_point):
    y = torch.zeros(x.size(), device=x.device)
    for i in range(x.size()[0]):
        y[i, :] = torch.round(x[i, :] / scale[i]) + zero_point[i]
    y = torch.clamp(y, 0, 255).to(torch.uint8)
    return y


def _dequantize_per_channel_cuda(y, scale, zero_point):
    y = y.to(torch.float32).cuda(y.device)
    x = torch.zeros_like(y, device=y.device)
    for i in range(x.size()[0]):
        x[i, :] = scale[i] * (y[i, :] - zero_point[i])
    return x


def _get_allgather_out_list(all_gather_in_list, world_size):
    out_list = [
        torch.zeros_like(
            all_gather_in_list,
            device=all_gather_in_list.device,
            dtype=all_gather_in_list.dtype,
        )
        for _ in range(world_size)
    ]
    return out_list


def quantization_pertensor_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future:
    """
    Applies the ``torch.quantize_per_tensor`` logic to DDP using ``allgather``
    protocol. Workers first allgather the scale and zero point of their own
    ``GradBucket`` prior to the quantization. After all workers have that information,
    the first ``then`` callback called ``quantize_and_allgather`` quantizes worker's
    own gradient tensors, and uses ``allgather`` to communicate these accross all workers.
    The final ``then`` callback called ``dequantize_and_aggregate``, dequantizes and
    aggregates each quantized gradient tensors locally and returns the mean.

    .. warning ::
        This is experimental, and uses ``allgather`` protocol which is considerably slower than
        ``allreduce`` protocol. It works only with flattened grads.

    Example::
        >>> ddp_model.register_comm_hook(process_group, quantization_pertensor_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    rank = process_group.rank() if process_group is not None else dist.get_rank()
    world_size = group_to_use.size()

    tensor = bucket.get_tensor()

    myObserver = torch.quantization.MinMaxObserver().cuda(tensor.device)
    myObserver(tensor)

    s, z = myObserver.calculate_qparams()
    s_and_z = torch.FloatTensor([s, z]).cuda(tensor.device)

    all_ranks_s_and_z = _get_allgather_out_list(s_and_z, world_size)

    # First, allgather scale and zeros.
    fut = dist.all_gather(
        all_ranks_s_and_z, s_and_z, group=group_to_use, async_op=True
    ).get_future()

    def quantize_and_allgather(fut):
        # Store scale and zeros accross all workers.
        all_ranks_s_and_z = fut.wait()[0]
        # All workers quantize their own ``GradBucket`` tensors.
        quantized_tensor = _quantize_per_tensor_cuda(
            tensor, all_ranks_s_and_z[rank][0], all_ranks_s_and_z[rank][1]
        )
        # Allgather quantized tensors.
        fut = dist.all_gather(
            _get_allgather_out_list(quantized_tensor, world_size),
            quantized_tensor,
            group=group_to_use,
            async_op=True,
        ).get_future()

        return fut.wait()

    def dequantize_and_aggregate(fut):
        all_ranks_quantized_tensor = fut.wait()[0]

        aggregated_dequantized_tensor = torch.zeros_like(
            all_ranks_quantized_tensor[0], device=tensor.device, dtype=torch.float32
        )
        # Using previously allgathered scales and zeros, dequantize gradient tensors
        # locally and then aggregate them.
        for r, quantized_tensor in enumerate(all_ranks_quantized_tensor):
            aggregated_dequantized_tensor += _dequantize_per_tensor_cuda(
                quantized_tensor, all_ranks_s_and_z[r][0], all_ranks_s_and_z[r][1]
            )

        return [aggregated_dequantized_tensor / world_size]

    return fut.then(quantize_and_allgather).then(dequantize_and_aggregate)


def quantization_perchannel_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket, bucket_size=512
) -> torch.futures.Future:
    """
    Applies the ``torch.quantize_per_channel`` logic to DDP using ``allgather``
    protocol. Compared to pertensor, the main motivation of perchannel is
    for considerably large tensors such as a tensor that contains 6 million
    elements quantizing per a bucket size of 512 (or 128) elements may significantly
    increase the resolution.

    It first splits ``GradBucket`` tensors into multiple chunks (channels) of ``bucket_size``
    elements. Then, workers allgather the scales and zero points of their own
    ``GradBucket`` prior to the quantization. After all workers have that information,
    the first ``then`` callback called ``quantize_and_allgather`` quantizes worker's
    own gradient tensors, and uses ``allgather`` to communicate these accross all workers.
    The final ``then`` callback called ``dequantize_and_aggregate``, dequantizes, flattens, and
    aggregates each quantized gradient tensors locally and returns the mean.

    .. warning ::
        This is experimental, and uses ``allgather`` protocol which is considerably slower than
        ``allreduce`` protocol. It works only with flattened grads.

    Example::
        >>> ddp_model.register_comm_hook(process_group, quantization_perchannel_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    rank = process_group.rank() if process_group is not None else dist.get_rank()
    world_size = group_to_use.size()

    tensor = bucket.get_tensor()

    tensor_in_channels = (
        nn.functional.pad(
            input=tensor,
            pad=(0, bucket_size - len(tensor) % bucket_size),
            mode="constant",
            value=0,
        )
        .view(-1, bucket_size)
        .cuda(tensor.device)
    )

    myPerChannelObserver = torch.quantization.PerChannelMinMaxObserver().cuda(
        tensor.device
    )
    myPerChannelObserver(tensor_in_channels)

    s_ch, z_ch = myPerChannelObserver.calculate_qparams()
    s_and_z = torch.stack((s_ch, z_ch)).cuda(tensor.device)

    all_ranks_s_and_z = _get_allgather_out_list(s_and_z, world_size)
    # First, allgather scale and zeros.
    fut = dist.all_gather(
        all_ranks_s_and_z, s_and_z, group=group_to_use, async_op=True
    ).get_future()

    def quantize_and_allgather(fut):
        # Store scale and zeros accross all workers.
        all_ranks_s_and_z = fut.wait()[0]
        # All workers quantize their corresponding ``GradBucket`` tensors.
        quantized_tensor = _quantize_per_channel_cuda(
            tensor_in_channels,
            all_ranks_s_and_z[rank, 0, :],
            all_ranks_s_and_z[rank, 1, :],
        )
        # Allgather quantized tensors.
        fut = dist.all_gather(
            _get_allgather_out_list(quantized_tensor, world_size),
            quantized_tensor,
            group=group_to_use,
            async_op=True,
        ).get_future()

        return fut.wait()

    def dequantize_and_aggregate(fut):
        all_ranks_quantized_tensor = fut.wait()[0]

        aggregated_dequantized_tensor = torch.zeros_like(
            all_ranks_quantized_tensor[0], device=tensor.device, dtype=torch.float32
        )
        # Using previously allgathered scales and zeros, dequantize gradient tensors
        # locally and then aggregate them.
        for r, quantized_tensor in enumerate(all_ranks_quantized_tensor):
            aggregated_dequantized_tensor += _dequantize_per_channel_cuda(
                quantized_tensor, all_ranks_s_and_z[r][0], all_ranks_s_and_z[r][1]
            )

        return [
            torch.flatten(aggregated_dequantized_tensor).cuda(tensor.device)[
                : tensor.size()[0]
            ]
            / world_size
        ]

    return fut.then(quantize_and_allgather).then(dequantize_and_aggregate)
