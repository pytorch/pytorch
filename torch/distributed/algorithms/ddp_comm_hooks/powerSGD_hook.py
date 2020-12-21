import logging
import math

import numpy as np
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
    __slots__ = [
        "process_group",
        "matrix_approximation_rank",
        "use_error_feedback",
        "rng",
        "error_dict",
    ]

    def __init__(
        self,
        process_group,
        matrix_approximation_rank=1,
        use_error_feedback=True,
        random_seed=0,
    ):
        self.process_group = process_group
        # The low rank for matrix approximation.
        # Typically only 1 or 2 is used. See https://arxiv.org/pdf/1905.13727.pdf.
        self.matrix_approximation_rank = matrix_approximation_rank
        # Error feedback is usually crucial for both for convergence and generalization,
        # because PowerSGD is a biased compressor,
        # i.e., compressing and decompressing a random gradient does not yield the original in expectation.
        # This mechanism requires a temporary copy of the input gradients,
        # so it increases the peak memory consumption by the size of gradient tensor.
        # However, if the target matrices are known to be exactly low-ranked (instead of just low stable rank),
        # sometimes it is possible to converge to the optima without error feedback.
        # See: http://proceedings.mlr.press/v54/yurtsever17a/yurtsever17a.pdf
        self.use_error_feedback = use_error_feedback
        # The purpose of this RNG is to generate different random seeds for initializing Q across iterations,
        # but in the same order for all the DDP replicas.
        # Different random seeds across iterations indicate different 'projections' of the gradients at different SGD steps.
        # If the same random projection is used,
        # there will be differences between the gradients that are never synchronized.
        self.rng = np.random.RandomState(random_seed)
        # Since there is only a single state instance for all the input buckets,
        # need to maintain a dictionary that maps each bucket index to the local error.
        self.error_dict = {}


def powerSGD_hook(
    state: PowerSGDState,
    bucket,
) -> torch.futures.Future:
    """
    This DDP communication hook implements the original PowerSGD gradient compression
    algorithm described in https://arxiv.org/abs/1905.13727.
    Once gradient tensors are aggregated across all workers, this hook applies
    compression as follows:
    1) Views the input flattened 1D gradient tensor as two groups of per-parameter tensors:
    high-rank tensors and vector-like rank-1 tensors (for biases).
    2) Handles rank-1 tensors by allreducing them without compression:
        2.1) Allocate contiguous memory for those rank-1 tensors,
        and allreduces all the rank-1 tensors as a batch, without compression;
        2.2) Copies the indvidual rank-1 tensors from the contiguous memory back to the input tensor.
    3) Handles high-rank tensors by PowerSGD compression:
        3.1) For each high-rank tensor M, creates two low-rank tensors P and Q for decomposing M,
        such that M = PQ^T, where Q is initialized from a standard normal distribution and orthogonalized;
        3.2) Computes each P in Ps, which is equal to MQ;
        3.3) Allreduces Ps as a batch;
        3.4) Orthogonizes each P in Ps;
        3.5) Computes each Q in Qs, which is approximately equal to M^TP;
        3.6) Allreduces Qs as a batch;
        3.7) Computes each M among all the high-rank tensors, which is approximately equal to PQ^T.

    TODO(wayi@): The above procedure does two matmul+allreduce steps per iteration --
    one left multiplication and one right multiplication.
    For warm start, can take one such step at a time, and alternate between them.

    Arguments:
        state (PowerSGDState): State information to configure the compression rate and support error feedback, warm start, etc.
        bucket (dist._GradBucket): Bucket that stores a 1D flattened gradient tensor that batches multiple per-variable tensors.
            Note that since DDP comm hook only supports single process single device mode at this time,
            only exactly one tensor is stored in this bucket.

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
    dtype = input_tensor.dtype

    # Incorporate the error from the previous state into the gradients.
    bucket_index = bucket.get_index()
    input_tensor_cp = None
    total_length = input_tensor.shape[0]
    if state.use_error_feedback:
        # The buckets can be rebuilt during training.
        # In this case, the error tensor shape will not be aligned with the input tensor,
        # and the error will be re-initialized as zeros.
        if (
            bucket_index in state.error_dict
            and state.error_dict[bucket_index].shape[0] == total_length
        ):
            input_tensor.add_(state.error_dict[bucket_index])
        else:
            logging.info(
                "A zero tensor of length {} that represents local error is created.".format(
                    total_length
                )
            )
            state.error_dict[bucket_index] = torch.zeros(total_length, device=device)

        # Keep a copy of the input tensor,
        # so that we can compute the local error caused by compression later,
        # by comparing this copy and the input tensor updated after decompression.
        input_tensor_cp = torch.clone(input_tensor).detach()

    # Unflatten the input tensor into per-parameter tensors, for layer-wise compression.
    tensors = [
        input_tensor[offset : offset + length].view(sizes)
        for offset, length, sizes in zip(
            bucket.get_offsets(), bucket.get_lengths(), bucket.get_sizes_list()
        )
    ]

    # Step I: Handle rank-1 tensors.
    # Allocate contiguous memory for rank-1 tensors to allreduce them without compression efficiently.
    rank1_tensors = [tensor for tensor in tensors if tensor.ndimension() <= 1]
    rank1_tensors_memory = (
        torch.cat([tensor.view(-1) for tensor in rank1_tensors])
        if rank1_tensors
        else torch.tensor([], device=device)
    )

    # Step II: Handle high-rank tensors.
    # Allocate contiguous memory for Ps and Qs to allreduce compressed high-rank tensors efficiently.
    high_rank_tensors = [
        tensor.view(tensor.shape[0], -1)
        for tensor in tensors
        if tensor.ndimension() > 1
    ]
    total_Ps_size = 0
    ps_memory = None  # TODO(wayi): Store it in a dict of PowerState for warm-up.
    total_Qs_size = 0
    qs_memory = None  # TODO(wayi): Store it in a dict of PowerState for warm-up.
    for tensor in high_rank_tensors:
        n, m = tensor.shape
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        total_Ps_size += n * matrix_approximation_rank
        total_Qs_size += m * matrix_approximation_rank
    ps_memory = torch.empty(total_Ps_size, device=device, dtype=dtype)
    qs_memory = torch.empty(total_Qs_size, device=device, dtype=dtype)

    # Create Ps and Qs that point to the allocated memory.
    ps = []
    qs = []
    p_idx = 0
    q_idx = 0
    for tensor in high_rank_tensors:
        n, m = tensor.shape
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        ps.append(
            ps_memory[p_idx : p_idx + n * matrix_approximation_rank].view(
                n, matrix_approximation_rank
            )
        )
        qs.append(
            qs_memory[q_idx : q_idx + m * matrix_approximation_rank].view(
                m, matrix_approximation_rank
            )
        )
        p_idx += n * matrix_approximation_rank
        q_idx += m * matrix_approximation_rank

    # Initialize and then orthogonalize Qs.
    with torch.random.fork_rng(devices=[]):
        # Fork this RNG to avoid changing the seed globally and affecting the random sampling anywhere else in the training.
        # The seed makes sure that the initial random values are the same across all the DDP replicas.
        # Such seed should differ at every step.
        # Since it is very slow to fork RNG state across all the CUDA devices,
        # only fork on CPU and then move the generated tensor to the CUDA device.
        torch.manual_seed(state.rng.randint(1_000_000_000))
        for q in qs:
            q.data = torch.randn(
                *q.shape,
                device="cpu",
                dtype=dtype,
            ).to(device)
            _orthogonalize(q)

    # Compute Ps.
    for tensor, q, p in zip(high_rank_tensors, qs, ps):
        torch.matmul(tensor, q, out=p)

    # This allreduce is only applied to rank-1 tensors,
    # so it should have been kicked off before the above computation on the high-rank tensors to hide more communication costs.
    # However, this somehow requires a separate future chain at this time.
    allreduce_contiguous_rank1_tensors_fut = dist.all_reduce(
        rank1_tensors_memory, group=group_to_use, async_op=True
    ).get_future()

    def unpack_rank1_tensors_and_allreduce_ps(fut):
        rank1_tensors_memory = fut.value()[0].div_(world_size)
        idx = 0
        for tensor in rank1_tensors:
            tensor.copy_(rank1_tensors_memory[idx : idx + tensor.shape[0]])
            idx += tensor.shape[0]

        # Since these Ps will be orthogonized later, no need to divide them by world size.
        return [
            dist.all_reduce(ps_memory, group=group_to_use, async_op=True)
            .get_future()
            .wait()[0]
        ]

    def compute_qs(fut):
        ps_memory = fut.wait()[0]
        for p in ps:
            _orthogonalize(p)

        # Compute Qs.
        for tensor, p, q in zip(high_rank_tensors, ps, qs):
            torch.matmul(tensor.t(), p, out=q)

        # Allreduce Qs.
        return [
            dist.all_reduce(qs_memory, group=group_to_use, async_op=True)
            .get_future()
            .wait()[0]
        ]

    def decompress(fut):
        qs_memory = fut.wait()[0].div_(world_size)

        for p, q, tensor in zip(ps, qs, high_rank_tensors):
            torch.matmul(p, q.t(), out=tensor)
            assert not torch.any(torch.isnan(tensor))

        if state.use_error_feedback:
            # Memorize the local errors.
            state.error_dict[bucket_index] = input_tensor_cp - input_tensor
        return [input_tensor]

    return (
        allreduce_contiguous_rank1_tensors_fut.then(
            unpack_rank1_tensors_and_allreduce_ps
        )
        .then(compute_qs)
        .then(decompress)
    )


def batched_powerSGD_hook(
    state: PowerSGDState,
    bucket,
) -> torch.futures.Future:
    """
    This DDP communication hook implements a simplified PowerSGD gradient compression
    algorithm described in https://arxiv.org/abs/1905.13727.
    Once gradient tensors are aggregated across all workers, this hook applies
    compression to the flattened input tensor that batches per-parameter tensors as follows:
    1) Views the input flattened 1D gradient tensor as a square-shaped tensor M with 0 paddings;
    2) Creates two low-rank tensors P and Q for decomposing M,
    such that M = PQ^T, where Q is initialized from a standard normal distribution and orthogonalized;
    2) Computes P, which is equal to MQ;
    3) Allreduces P;
    4) Orthogonizes P;
    5) Computes Q, which is approximately equal to M^TP;
    6) Allreduces Q;
    7) Computes M, which is approximately equal to PQ^T.
    8) Truncates the input tensor to the original length.

    TODO(wayi@): The above procedure does two matmul+allreduce steps per iteration --
    one left multiplication and one right multiplication.
    For warm start, can take one such step at a time, and alternate between them.

    Arguments:
        state (PowerSGDState): State information to configure the compression rate and support error feedback, warm start, etc.
        bucket (dist._GradBucket): Bucket that stores a 1D flattened gradient tensor that batches multiple per-variable tensors.
            Note that since DDP comm hook only supports single process single device mode at this time,
            only exactly one tensor is stored in this bucket.

    Returns:
        Future handler of the communication, which updates the gradients in place.

    Example::
        state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1)
        >>> ddp_model.register_comm_hook(state, batched_powerSGD_hook)
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

    # Incorporate the error from the previous state into the gradients.
    bucket_index = bucket.get_index()
    input_tensor_cp = None
    if state.use_error_feedback:
        # The buckets can be rebuilt during training.
        # In this case, the error tensor shape will not be aligned with the input tensor,
        # and the error will be re-initialized as zeros.
        if (
            bucket_index in state.error_dict
            and state.error_dict[bucket_index].shape[0] == padded_total_length
        ):
            input_tensor.add_(state.error_dict[bucket_index])
        else:
            logging.info(
                "A zero tensor of length {} that represents local error is created.".format(
                    padded_total_length
                )
            )
            state.error_dict[bucket_index] = torch.zeros(
                padded_total_length, device=device
            )

        # Keep a copy of the input tensor,
        # so that we can compute the local error caused by compression later,
        # by comparing this copy and the input tensor updated after decompression.
        input_tensor_cp = torch.clone(input_tensor).detach()
    matrix = input_tensor.view(square_side_length, square_side_length)

    def create_low_rank_tensor(fill_random_values, rng):
        "Returns a low-rank 2D tensor of square_side_length * matrix_approximation_rank."
        if fill_random_values:
            with torch.random.fork_rng(devices=[]):
                # Fork this RNG to avoid changing the seed globally and affecting the random sampling anywhere else in the training.
                # The seed makes sure that the initial random values are the same across all the DDP replicas.
                # Such seed should differ at every step.
                # Since it is very slow to fork RNG state across all the CUDA devices,
                # only fork on CPU and then move the generated tensor to the CUDA device.
                torch.manual_seed(rng.randint(1_000_000_000))
                return torch.randn(
                    square_side_length,
                    state.matrix_approximation_rank,
                    device="cpu",
                    dtype=input_tensor.dtype,
                ).to(device)
        else:
            return torch.empty(
                square_side_length,
                state.matrix_approximation_rank,
                device=device,
                dtype=input_tensor.dtype,
            )

    p = create_low_rank_tensor(fill_random_values=False, rng=state.rng)
    q = create_low_rank_tensor(fill_random_values=True, rng=state.rng)
    _orthogonalize(q, 0)

    torch.matmul(matrix, q, out=p)
    allreduce_p_fut = dist.all_reduce(p, group=group_to_use, async_op=True).get_future()

    def compute_q(fut):
        p = fut.value()[0]
        _orthogonalize(p, 0)

        torch.matmul(matrix.t(), p, out=q)

        return [
            dist.all_reduce(q, group=group_to_use, async_op=True).get_future().wait()[0]
        ]

    def decompress(fut):
        q = fut.value()[0].div_(world_size)
        torch.matmul(p, q.t(), out=matrix)

        if state.use_error_feedback:
            # Memorize the local errors.
            state.error_dict[bucket_index] = input_tensor_cp - input_tensor
            assert not torch.any(torch.isnan(state.error_dict[bucket_index]))
        ret = input_tensor.resize_(total_length)
        return [ret]

    return allreduce_p_fut.then(compute_q).then(decompress)
