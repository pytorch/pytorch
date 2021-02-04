import logging
import math

import numpy as np
import torch
import torch.distributed as dist

from . import default_hooks as default


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
            # May need to consider multiplying a scaling factor and dividing it later, or using bfloat16 instead.
            col /= torch.norm(col)
        else:
            col /= torch.norm(col) + epsilon
        # Project it on the rest and remove it.
        if i + 1 < num_cols:
            rest = matrix[:, i + 1 :]
            rest -= torch.sum(col * rest, dim=0) * col


class PowerSGDState(object):
    """
    Stores both the gradient compression configs and the internal states for all the gradients during the training.
    Particularly, `matrix_approximation_rank` and `start_powerSGD_iter` are the main configs that need to be tuned by the user.
    Although `use_error_feedback` and `warm_start` can also be tuned by the user,
    they are typically turned on for performance.

    Note [Guidance to Tune `matrix_approximation_rank` And `start_powerSGD_iter`]
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    1) To tune `matrix_approximation_rank`, the user can increase it from 1 by factors of 2,
    until a satisfying accuracy can be reached.
    The increase of `matrix_approximation_rank` can substantially increase the computation costs of the compression.
    However, the accuracy may not be futher improved beyond a certain `matrix_approximation_rank` value.
    2) To tune `start_powerSGD_iter`, the user can typically start with 10% of total training steps,
    and increase it until a satisfying accuracy can be reached.
    Deferrring PowerSGD can effectively improve the accuracy,
    even a relatively small `matrix_approximation_rank` is used.
    This is because that, the beginning of training phase is usually very sensitive to inaccurate gradients,
    and compressing gradients too early may make the training quickly take a suboptimal trajectory,
    which can result in an irrecoverable impact on the accuracy.
    The minimum value allowed in DDP is 2, if error feedback or warm-up is enabled.
    This is because there is another internal optimization that rebuilds buckets at iteration 1 in DDP,
    and this can conflict with any tensor memorized before the rebuild process.
    """

    __slots__ = [
        "process_group",
        # The two fields below are the configs that usually need to be tuned by the user.
        "matrix_approximation_rank",
        "start_powerSGD_iter",
        # The two fields below are the configs that usually need to be turned on for performance.
        "use_error_feedback",
        "warm_start",
        # The fields below are not configs.
        "rng",
        "error_dict",
        "p_memory_dict",
        "q_memory_dict",
        "iter",
    ]

    def __init__(
        self,
        process_group,
        matrix_approximation_rank=1,
        start_powerSGD_iter=10,
        use_error_feedback=True,
        warm_start=True,
        random_seed=0,
    ):
        logging.info(
            "PowerSGD config: matrix_approximation_rank = {}; "
            "start_powerSGD_iter = {}; use_error_feedback = {}; warm_start = {}.".format(
                matrix_approximation_rank,
                start_powerSGD_iter,
                use_error_feedback,
                warm_start,
            )
        )

        self.process_group = process_group
        # The low rank for matrix approximation controls the size of compressed low-rank tensors,
        # which determines the computation ratio.
        # Typically only a small value 1-4 is used.
        # For some NLP tasks (as shown in Appendix D of the original paper
        # https://arxiv.org/pdf/1905.13727.pdf, the rank value has been increased to 32.
        # A high rank value will increase the computation costs of compression exponentially.
        # A good choice depends on how much extra computation can be hidden by the dominating communication costs.
        self.matrix_approximation_rank = matrix_approximation_rank
        # This defers PowerSGD compression util step 'start_powerSGD_iter',
        # and vanilla allreduce runs before step 'start_powerSGD_iter'.
        # This hybrid scheme of vanilla allreduce + PowerSGD can have two advantages:
        # 1) It turns out that PowerSGD may lead to a non-trivial accuracy loss,
        # even if the matrix approximation rank is increased to a large value.
        # To mitigate the accuracy loss, a simple yet effective way is mixing vanilla allreduce
        # (or a more convervative compression such as FP16 compression) with PowerSGD.
        # 2) There is an internal optimization of rebuilding buckets process in DDP,
        # in order to save the memory space.
        # This step takes place after the first iteration.
        # However, this means that the shape of input bucketized tensors is subject to change,
        # which will complicate the implementations of error feedback and warm-up.
        # Running vanilla allreduce in the first few iterations can avoid this complexity.
        if (use_error_feedback or warm_start) and start_powerSGD_iter <= 1:
            raise ValueError(
                "Expect `start_powerSGD_iter` > 1 if `use_error_feedback` or `warm_start` is enabled, "
                "because PowerSGD can only be applied after the first two iterations in DDP."
            )
        self.start_powerSGD_iter = start_powerSGD_iter
        # Error feedback is usually crucial for both for convergence and generalization,
        # because PowerSGD is a biased compressor,
        # i.e., compressing and decompressing a random gradient does not yield the original in expectation.
        # This mechanism requires a temporary copy of the input gradients,
        # so it increases the peak memory consumption by the size of the gradient tensor.
        # However, if the target matrices are known to be exactly low-ranked (instead of just low stable rank),
        # sometimes it is possible to converge to the optima without error feedback.
        # See: http://proceedings.mlr.press/v54/yurtsever17a/yurtsever17a.pdf
        self.use_error_feedback = use_error_feedback
        # Warm-start reuses P(s) and Q(s) from the previous iteration.
        # This can improve the approximation quality and hence improve the accuracy.
        # Additionally, by avoiding the initialization of these low-rank tensors at every step,
        # this can also accelerate training.
        # However, this is at the cost of extra memory.
        self.warm_start = warm_start
        # The purpose of this RNG is to generate different random seeds for initializing Q across iterations,
        # but in the same order for all the DDP replicas.
        # Different random seeds across iterations indicate different 'projections' of the gradients at different SGD steps.
        # If the same random projection is used,
        # there will be differences between the gradients that are never synchronized.
        self.rng = np.random.RandomState(random_seed)
        # Since there is only a single state instance for all the input buckets,
        # need to maintain a dictionary that maps each bucket index to the local error.
        self.error_dict = {}
        self.p_memory_dict = {}
        self.q_memory_dict = {}
        # Iteration/step in the training loop.
        self.iter = 0

    def maybe_increase_iter(self, bucket):
        # Since bucket 0 is the last bucket to allreduce in an iteration.
        # Only increase `iter` when bucket 0 is processed.
        if bucket.get_index() == 0:
            self.iter += 1

        if self.iter == self.start_powerSGD_iter:
            logging.info(
                "Start to apply PowerSGD after {} iterations.".format(self.iter)
            )


def powerSGD_hook(state: PowerSGDState, bucket) -> torch.futures.Future:
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
        2.2) Copies the individual rank-1 tensors from the contiguous memory back to the input tensor.
    3) Handles high-rank tensors by PowerSGD compression:
        3.1) For each high-rank tensor M, creates two low-rank tensors P and Q for decomposing M,
        such that M = PQ^T, where Q is initialized from a standard normal distribution and orthogonalized;
        3.2) Computes each P in Ps, which is equal to MQ;
        3.3) Allreduces Ps as a batch;
        3.4) Orthogonalizes each P in Ps;
        3.5) Computes each Q in Qs, which is approximately equal to M^TP;
        3.6) Allreduces Qs as a batch;
        3.7) Computes each M among all the high-rank tensors, which is approximately equal to PQ^T.

    Note that this communication hook enforces vanilla allreduce for the first `state.start_powerSGD_iter` iterations.
    This can not only allow the user to have a finer tuning over the tradeoff between speedup and accuracy,
    but also help abstract away some complexity of the internal optimization of DDP for future communication hook developers.

    TODO(wayi@): The above procedure does two matmul+allreduce steps per iteration --
    one left multiplication and one right multiplication.
    For warm-start, can take one such step at a time, and alternate between them.

    Args:
        state (PowerSGDState): State information to configure the compression rate and support error feedback, warm start, etc.
            To tune the compression configs, see Note [Guidance to Tune `matrix_approximation_rank` And `start_powerSGD_iter`].
        bucket (dist._GradBucket): Bucket that stores a 1D flattened gradient tensor that batches multiple per-variable tensors.
            Note that since DDP comm hook only supports single process single device mode at this time,
            only exactly one tensor is stored in this bucket.

    Returns:
        Future handler of the communication, which updates the gradients in place.

    Example::
        state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1, start_powerSGD_iter=10)
        >>> ddp_model.register_comm_hook(state, powerSGD_hook)
    """
    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    # The input tensor is a flattened 1D tensor.
    input_tensor = bucket.get_tensors()[0]

    # Run vanilla allreduce in the first `start_powerSGD_iter` iterations.
    if state.iter < state.start_powerSGD_iter:
        state.maybe_increase_iter(bucket)
        return default._allreduce_fut(group_to_use, input_tensor)

    # Apply PowerSGD after `start_powerSGD_iter` iterations.
    device = input_tensor.device
    dtype = input_tensor.dtype

    # Incorporate the error from the previous state into the gradients.
    bucket_index = bucket.get_index()
    input_tensor_cp = None
    total_length = input_tensor.shape[0]
    if state.use_error_feedback:
        if bucket_index in state.error_dict:
            input_tensor.add_(state.error_dict[bucket_index])
        else:
            logging.info(
                "A zero tensor of length {} that represents local error is created.".format(
                    total_length
                )
            )
            state.error_dict[bucket_index] = torch.zeros(
                total_length, device=device, dtype=dtype
            )

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
        else torch.tensor([], device=device, dtype=dtype)
    )

    # Step II: Handle high-rank tensors.
    # Allocate contiguous memory for Ps and Qs to allreduce compressed high-rank tensors efficiently.
    high_rank_tensors = [
        tensor.view(tensor.shape[0], -1)
        for tensor in tensors
        if tensor.ndimension() > 1
    ]
    total_Ps_size = 0
    total_Qs_size = 0
    for tensor in high_rank_tensors:
        n, m = tensor.shape
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        total_Ps_size += n * matrix_approximation_rank
        total_Qs_size += m * matrix_approximation_rank
    # If warm-start is enabled, reuse Ps and Qs from the previous iteration if possible.
    # The memory spaces of Ps and Qs need to be allocated in the first iteration when PowerSGD is applied.
    need_randomize_qs = False
    if not state.warm_start or bucket_index not in state.p_memory_dict:
        need_randomize_qs = True
        # If warm-start is disabled, low-rank tensors will be initialized at every step.
        # Only log this if warm-start to avoid spamming.
        if state.warm_start:
            logging.info(
                "Allocating contiguous memory of length {} for Ps, and of length {} for Qs, respectively.".format(
                    total_Ps_size, total_Qs_size
                )
            )
        state.p_memory_dict[bucket_index] = torch.empty(
            total_Ps_size, device=device, dtype=dtype
        )
        state.q_memory_dict[bucket_index] = torch.empty(
            total_Qs_size, device=device, dtype=dtype
        )

    # Create Ps and Qs that point to the allocated memory.
    ps = []
    qs = []
    p_idx = 0
    q_idx = 0
    for tensor in high_rank_tensors:
        n, m = tensor.shape
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        ps.append(
            state.p_memory_dict[bucket_index][
                p_idx : p_idx + n * matrix_approximation_rank
            ].view(n, matrix_approximation_rank)
        )
        qs.append(
            state.q_memory_dict[bucket_index][
                q_idx : q_idx + m * matrix_approximation_rank
            ].view(m, matrix_approximation_rank)
        )
        p_idx += n * matrix_approximation_rank
        q_idx += m * matrix_approximation_rank

    # If warm-start is enabled, reuse Qs from the previous iteration if possible and skip filling random values.
    # The exception is the first iteration when PowerSGD is applied.
    if not need_randomize_qs:
        for q in qs:
            _orthogonalize(q)
    else:
        with torch.random.fork_rng(devices=[]):
            # Fork this RNG to avoid changing the seed globally and affecting the random sampling anywhere else in the training.
            # The seed makes sure that the initial random values are the same across all the DDP replicas.
            # This seed should differ at every step.
            # Since it is very slow to fork RNG state across all the CUDA devices,
            # only fork on CPU and then move the generated tensor to the CUDA device (by overwriting q).
            torch.manual_seed(state.rng.randint(1_000_000_000))
            for q in qs:
                q.copy_(
                    torch.randn(
                        *q.shape,
                        device="cpu",
                        dtype=dtype,
                    )
                )
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

        # Since these Ps will be orthogonalized later, no need to divide them by world size.
        return [
            dist.all_reduce(
                state.p_memory_dict[bucket_index], group=group_to_use, async_op=True
            )
            .get_future()
            .wait()[0]
        ]

    def compute_qs(fut):
        state.p_memory_dict[bucket_index] = fut.value()[0]
        for p in ps:
            _orthogonalize(p)

        # Compute Qs.
        for tensor, p, q in zip(high_rank_tensors, ps, qs):
            torch.matmul(tensor.t(), p, out=q)

        # Allreduce Qs.
        return [
            dist.all_reduce(
                state.q_memory_dict[bucket_index], group=group_to_use, async_op=True
            )
            .get_future()
            .wait()[0]
        ]

    def decompress(fut):
        state.q_memory_dict[bucket_index] = fut.value()[0].div_(world_size)

        for p, q, tensor in zip(ps, qs, high_rank_tensors):
            torch.matmul(p, q.t(), out=tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        if state.use_error_feedback:
            # Memorize the local errors.
            state.error_dict[bucket_index] = input_tensor_cp - input_tensor
        if not state.warm_start:
            state.p_memory_dict.clear()
            state.q_memory_dict.clear()

        state.maybe_increase_iter(bucket)

        return [input_tensor]

    return (
        allreduce_contiguous_rank1_tensors_fut.then(
            unpack_rank1_tensors_and_allreduce_ps
        )
        .then(compute_qs)
        .then(decompress)
    )


def batched_powerSGD_hook(state: PowerSGDState, bucket) -> torch.futures.Future:
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
    4) Orthogonalizes P;
    5) Computes Q, which is approximately equal to M^TP;
    6) Allreduces Q;
    7) Computes M, which is approximately equal to PQ^T.
    8) Truncates the input tensor to the original length.

    This variant is faster than `powerSGD_hook` that runs layer-wise gradient compression,
    but it usually results in a much lower accuracy, unless `matrix_approximation_rank` in the state is 1.
    Increasing `matrix_approximation_rank` may not necessarily increase the accuracy,
    because batching per-parameter tensors without column/row alignment can destroy low-rank structure.
    Therefore, the user shoud always consider `powerSGD_hook` first,
    and only consider this variant when a satisfying accuracy can be achieved when `matrix_approximation_rank` is 1.

    Note that this communication hook enforces vanilla allreduce for the first `state.start_powerSGD_iter` iterations.
    This can not only allow the user to have a finer tuning over the tradeoff between speedup and accuracy,
    but also help abstract away some complexity of the internal optimization of DDP for future communication hook developers.

    TODO(wayi@): The above procedure does two matmul+allreduce steps per iteration --
    one left multiplication and one right multiplication.
    For warm-start, can take one such step at a time, and alternate between them.

    Args:
        state (PowerSGDState): State information to configure the compression rate and support error feedback, warm start, etc.
            To tune the compression configs, see Note [Guidance to Tune `matrix_approximation_rank` And `start_powerSGD_iter`].
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
    world_size = group_to_use.size()

    # The input tensor is a flattened 1D tensor.
    input_tensor = bucket.get_tensors()[0]

    # Run vanilla allreduce in the first `start_powerSGD_iter` iterations.
    if state.iter < state.start_powerSGD_iter:
        state.maybe_increase_iter(bucket)
        return default._allreduce_fut(group_to_use, input_tensor)

    # Apply PowerSGD after `start_powerSGD_iter` iterations.
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
        if bucket_index in state.error_dict:
            input_tensor.add_(state.error_dict[bucket_index])
        else:
            logging.info(
                "A zero tensor of length {} that represents local error is created.".format(
                    padded_total_length
                )
            )
            state.error_dict[bucket_index] = torch.zeros(
                padded_total_length, device=device, dtype=input_tensor.dtype
            )

        # Keep a copy of the input tensor,
        # so that we can compute the local error caused by compression later,
        # by comparing this copy and the input tensor updated after decompression.
        input_tensor_cp = torch.clone(input_tensor).detach()
    matrix = input_tensor.view(square_side_length, square_side_length)

    # Reuse P and Q from the previous iteration if possible.
    # The memory spaces of P and Q need to be allocated in the first iteration when PowerSGD is applied.
    if not state.warm_start or bucket_index not in state.p_memory_dict:
        # If warm-start is disabled, low-rank tensors will be initialized at every step.
        # Only log this if warm-start to avoid spamming.
        if state.warm_start:
            logging.info(
                "Initializing low-rank tensors P and Q, each of which has a shape of {} x {}.".format(
                    square_side_length, state.matrix_approximation_rank
                )
            )

        def create_low_rank_tensor(fill_random_values, rng):
            "Returns a low-rank 2D tensor of square_side_length * matrix_approximation_rank."
            if fill_random_values:
                with torch.random.fork_rng(devices=[]):
                    # Fork this RNG to avoid changing the seed globally and affecting the random sampling
                    # anywhere else in the training.
                    # The seed makes sure that the initial random values are the same across all the DDP replicas.
                    # This seed should differ at every step.
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

        state.p_memory_dict[bucket_index] = create_low_rank_tensor(
            fill_random_values=False, rng=state.rng
        )
        state.q_memory_dict[bucket_index] = create_low_rank_tensor(
            fill_random_values=True, rng=state.rng
        )
    _orthogonalize(state.q_memory_dict[bucket_index], 0)

    torch.matmul(
        matrix, state.q_memory_dict[bucket_index], out=state.p_memory_dict[bucket_index]
    )
    allreduce_p_fut = dist.all_reduce(
        state.p_memory_dict[bucket_index], group=group_to_use, async_op=True
    ).get_future()

    def compute_q(fut):
        state.p_memory_dict[bucket_index] = fut.value()[0]
        _orthogonalize(state.p_memory_dict[bucket_index], 0)

        torch.matmul(
            matrix.t(),
            state.p_memory_dict[bucket_index],
            out=state.q_memory_dict[bucket_index],
        )

        return [
            dist.all_reduce(
                state.q_memory_dict[bucket_index], group=group_to_use, async_op=True
            )
            .get_future()
            .wait()[0]
        ]

    def decompress(fut):
        state.q_memory_dict[bucket_index] = fut.value()[0].div_(world_size)
        torch.matmul(
            state.p_memory_dict[bucket_index],
            state.q_memory_dict[bucket_index].t(),
            out=matrix,
        )

        if state.use_error_feedback:
            # Memorize the local errors.
            state.error_dict[bucket_index] = input_tensor_cp - input_tensor
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        if not state.warm_start:
            state.p_memory_dict.clear()
            state.q_memory_dict.clear()
        ret = input_tensor.resize_(total_length)

        state.maybe_increase_iter(bucket)

        return [ret]

    return allreduce_p_fut.then(compute_q).then(decompress)
