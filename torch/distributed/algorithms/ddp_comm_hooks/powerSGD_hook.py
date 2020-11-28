import math

import torch
import torch.distributed as dist


def _orthogonalize(matrix, epsilon=1e-8):
    """
    Applies Gram-Schmidt procedure to orthogonalize a given 2D tensor.
    If epsilon is 0, this is equivalent to `torch.qr(matrix, out=(matrix, _))`,
    but `torch.qr` is very slow, probably because it is not optimized for a matrix that has a small number of columns.
    """
    num_cols = matrix.shape[1]
    for i in range(num_cols):
        # Normalize the i'th column.
        col = matrix[:, i : i + 1]
        # If no epsilon is added here, division by zero may be caused by vanishing gradients.
        # This epsilon is not needed if the input matrix covers the gradients of at least one entire layer in the neural network.
        if epsilon == 0:
            # Note that col ** 2 can underflow/overflow if we use FP16.
            # May need to consder multiplying a scaling factor and divding it later, or using bfloat16 isntead.
            col /= torch.sqrt(torch.sum(col ** 2))
        else:
            col /= torch.sqrt(torch.sum(col ** 2)) + epsilon
        # Project it on the rest and remove it.
        if i + 1 < num_cols:
            rest = matrix[:, i + 1 :]
            rest -= torch.sum(col * rest, dim=0) * col


class PowerSGDState(object):
    __slots__ = ["process_group", "matrix_approximation_rank"]

    def __init__(self, process_group, matrix_approximation_rank=1):
        self.process_group = process_group
        self.matrix_approximation_rank = matrix_approximation_rank


def powerSGD_hook(
    state: PowerSGDState,
    bucket: dist._GradBucket,
) -> torch.futures.Future:
    """
    This DDP communication hook implements a simplified PowerSGD gradient compression
    algorithm described in https://arxiv.org/abs/1905.13727.
    Once gradient tensors are aggregated across all workers, this hook applies
    compression as follows:
    1) Views the input flattened 1D gradient tensor as a square-shaped tensor M with 0 paddings;
    2) Decomposes M into two low-rank tensors P and Q,
    such that M = PQ^T, where Q is initialized from a standard normal distribution and orthogonalized;
    2) Allreduces P;
    3) Orthogonizes P;
    4) Compute Q, which is approximately equal to M^TP;
    5) Allreduces Q;
    6) Computes M, which is approximately equal to PQ^T.
    7) Truncates the input tensor to the original length.

    TODO(wayi@): The above procedure does two matmul+allreduce steps per iteration --
    one left multiplication and one right multiplication.
    For warm start, can take one such step at a time, and alternate between them.

    Arguments:
        state (PowerSGDState): State information to configure the compression rate and support error feedback, warm start, etc.
        bucket (dist._GradBucket): Bucket that stores a 1D flattened gradient tensor that batches multiple per-variable tensors.
            Note that since DDP comm hook only supports single process single device mode at this time,
            only exactly one tensor is stored in this bucket.
        matrix_approximation_rank (int): The low rank for matrix approximation.
            Typically only 1 or 2 is used. See https://arxiv.org/pdf/1905.13727.pdf.

    Returns:
        Future handler of the communication, which updates the gradients in place.

    Example::
        state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1)
        >>> ddp_model.register_comm_hook(state, powerSGD_hook)
    """
    process_group = state.process_group
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

    def create_low_rank_tensor(fill_random_values):
        "Returns a low-rank 2D tensor of square_side_length * matrix_approximation_rank."
        if fill_random_values:
            with torch.random.fork_rng(devices=[device]):
                # The seed makes sure that the initial random values are the same across all the DDP replicas.
                # Such seed should differ at every step.
                # Currently use the length of input tensor as the seed, which should be mostly different.
                # TODO(wayi@): Should read the random seed from the state of this hook provided by the constructor.
                torch.manual_seed(total_length)
                return torch.randn(
                    square_side_length, state.matrix_approximation_rank, device=device
                )
        else:
            return torch.empty(
                square_side_length, state.matrix_approximation_rank, device=device
            )

    p = create_low_rank_tensor(fill_random_values=False)
    q = create_low_rank_tensor(fill_random_values=True)
    _orthogonalize(q, 0)

    torch.matmul(matrix, q, out=p)
    allreduce_p_fut = dist.all_reduce(p, group=group_to_use, async_op=True).get_future()

    def compute_q(fut):
        p = fut.value()[0]
        _orthogonalize(p, 0)

        torch.matmul(matrix.t(), p, out=q)

        return [
            dist.all_reduce(q, group=group_to_use, async_op=True)
            .get_future()
            .value()[0]
        ]

    def decompress(fut):
        q = fut.value()[0].div_(world_size)
        torch.matmul(p, q.t(), out=matrix)

        ret = input_tensor.resize_(total_length)
        return [ret]

    return allreduce_p_fut.then(compute_q).then(decompress)
