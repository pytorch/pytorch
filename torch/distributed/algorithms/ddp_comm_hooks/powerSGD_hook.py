import math

import numpy as np
import torch
import torch.distributed as dist


def _orthogonalize(matrix, epsilon=1e-8):
    """Applies Gram-Schmidt procedure to orthogonalize a given 2D tensor."""
    num_cols = matrix.shape[1]
    for i in range(num_cols):
        # Normalize the i'th column.
        col = matrix[:, i : i + 1]
        # Note that col ** 2 can underflow/overflow if we use FP16.
        # May need to consder multiplying a scaling factor and divding it later, or using bfloat16 isntead.
        col /= torch.sqrt(torch.sum(col ** 2))
        # If no epsilon is added here, division by zero may be caused by vanishing gradients.
        # This epsilon is not needed if the input matrix covers the gradients of at least one entire layer in the neural network.
        if epsilon > 0:
            col += epsilon
        # Project it on the rest and remove it.
        if i + 1 < num_cols:
            rest = matrix[:, i + 1 :]
            rest -= torch.sum(col * rest, dim=0) * col


def _set_random(vector, random_seed=0):
    """
    Initializes a 1D tensor from a standard normal distribution.
    The see makes sure that the randomized vector in all the DDP replicas are the same at every step.
    """
    torch.manual_seed(np.random.RandomState(random_seed).randint(1_000_000_000))
    vector.data[:] = torch.randn(*vector.shape)


def powerSGD_hook(
    process_group: object, bucket: dist._GradBucket, rank: int = 1
) -> torch.futures.Future:
    """
    This DDP communication hook implements a simplified PowerSGD graidient compression
    algorithm described in https://arxiv.org/abs/1905.13727.
    Once gradient tensors are aggregated across all workers, this hook applies
    compression as follows:
    1) Views the input flattened 1D gradient tensor as a square-shaped tensor M with 0 paddings;
    2) Decomposes M into two low-rank tensors P and Q,
    such that M = PQ^T, where Q is initialized from a standard normal distribution;
    2) Allreduces P;
    3) Orthogonizes P;
    4) Compute Q, which is approximately equal to M^TP;
    5) Allreduces Q;
    6) Computes M, which is approximately equal to PQ^T.
    7) Truncates the input tensor to the original length.

    Example::
        PowerSGDState state(process_group, 1)
        >>> ddp_model.register_comm_hook(state, powerSGD_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = (
        process_group.size() if process_group is not None else dist.get_world_size()
    )

    # The input tensor is a flattened 1D tensor.
    input_tensor = bucket.get_tensors()[0]
    device = input_tensor.device
    total_length = input_tensor.shape[0]

    # View the input tensor as a 2D square-shape tensor, and pad 0s if necessary.
    square_side_length = math.ceil(math.sqrt(total_length))
    padded_total_length = square_side_length ** 2
    input_tensor.resize_(padded_total_length)
    input_tensor[total_length:padded_total_length].fill_(0)
    matrix = input_tensor.view(square_side_length, square_side_length)

    def create_low_rank_tensor():
        "Returns a low-rank 2D tensor and the allocated contiguous memory."
        memory = torch.empty(square_side_length * rank, device=device)
        return memory.view(square_side_length, rank), memory

    p, p_memory = create_low_rank_tensor()
    q, q_memory = create_low_rank_tensor()
    # Initialize each Q from a standard normal distribution.
    _set_random(q)

    torch.matmul(matrix, q, out=p)
    allreduce_p_fut = dist.all_reduce(
        p_memory, group=group_to_use, async_op=True
    ).get_future()

    def compute_q(fut):
        p_memory = fut.value()[0]
        _orthogonalize(p, 0)

        torch.matmul(matrix.t(), p, out=q)

        return [
            dist.all_reduce(q_memory, group=group_to_use, async_op=True)
            .get_future()
            .value()[0]
        ]

    def decompress(fut):
        q_memory = fut.value()[0].div_(world_size)
        torch.matmul(p, q.t(), out=matrix)

        ret = input_tensor.resize_(total_length)
        return [ret]

    return allreduce_p_fut.then(compute_q).then(decompress)
