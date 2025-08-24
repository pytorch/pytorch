import torch


def greedy_knapsack(
    memory: list[float], runtimes: list[float], max_memory: float
) -> tuple[float, list[int], list[int]]:
    n = len(runtimes)
    items = list(range(n))

    # Sort items based on the ratio of runtime to memory in descending order
    items = sorted(items, key=lambda i: runtimes[i] / memory[i], reverse=True)

    total_memory = 0.0
    total_runtime = 0.0
    items_to_save = []
    items_to_allow_recomputing = []

    for i in items:
        if total_memory + memory[i] <= max_memory:
            total_memory += memory[i]
            total_runtime += runtimes[i]
            items_to_save.append(i)
        else:
            items_to_allow_recomputing.append(i)
    return total_runtime, items_to_save, items_to_allow_recomputing


def ilp_knapsack(
    memory: list[float], runtimes: list[float], max_memory: float
) -> tuple[float, list[int], list[int]]:
    import numpy as np

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError:
        raise RuntimeError(
            "To use the ILP for memory budget checkpointing you need to install scipy"
        ) from None

    np_memory = np.array(memory)
    np_runtimes = np.array(runtimes)
    c = -np_runtimes  # type: ignore[operator]

    memory_constraint = LinearConstraint(A=np_memory, ub=np.array(max_memory))
    constraints = [memory_constraint]

    integrality = np.ones_like(c)
    res = milp(
        c=c, constraints=constraints, integrality=integrality, bounds=Bounds(0, 1)
    )
    if not res.success:
        raise RuntimeError("Somehow scipy solving failed")

    items_to_save = []
    items_to_allow_recomputing = []
    for idx, i in enumerate(res.x):
        if i == 1:
            items_to_save.append(idx)
        else:
            items_to_allow_recomputing.append(idx)
    return -res.fun, items_to_save, items_to_allow_recomputing


def dp_knapsack(
    memory: list[float], runtime: list[float], max_memory: float
) -> tuple[float, list[int], list[int]]:
    # Scaling factor to convert floating point weights to integers
    S = 10000

    # Quantize the memory weights
    quantized_memory = torch.tensor(
        [int(round(m * S)) for m in memory], dtype=torch.long, device="cpu"
    )
    runtimes = torch.tensor(runtime, dtype=torch.float32, device="cpu")

    # Quantized pseudopolynomial DP for 0-1 Knapsack
    quantized_max_memory = int(round(max_memory * S))

    n = len(memory)

    # Initialize the DP table
    # TODO(chilli): I think if needed, this memory can be optimized with sliding
    # window trick + Hirschberg trick:
    # https://codeforces.com/blog/entry/47247?#comment-316200
    dp = torch.zeros(
        (n + 1, quantized_max_memory + 1), dtype=torch.float32, device="cpu"
    )

    for i in range(1, n + 1):
        current_memory = quantized_memory[i - 1]
        current_runtime = runtimes[i - 1]

        # Copy the previous row
        dp[i, :] = dp[i - 1, :]

        # Update dp[i, j] for all j >= current_memory
        if current_memory == 0:
            dp[i, :] = dp[i - 1, :] + current_runtime
        else:
            dp[i, current_memory:] = torch.maximum(
                dp[i - 1, current_memory:],
                dp[i - 1, :-current_memory] + current_runtime,
            )

    # Backtrack to find the items included in the knapsack
    saved_items = []
    recomputable_items = []
    j: int = quantized_max_memory
    for i in range(n, 0, -1):
        if dp[i][j] != dp[i - 1][j]:
            saved_items.append(i - 1)  # Include this item (indexing from 0)
            j -= int(quantized_memory[i - 1].item())
        else:
            recomputable_items.append(i - 1)

    saved_items.reverse()  # To get items in the order they were added

    # The maximum runtime that can be achieved within the max_memory constraint
    max_runtime = dp[n][quantized_max_memory].item()

    return max_runtime, saved_items, recomputable_items


def dp_knapsack_sliding_hirschberg(
    memory: list[float], runtime: list[float], max_memory: float
) -> tuple[float, list[int], list[int]]:
    # Scaling factor to convert floating point weights to integers
    S = 10000

    # Quantize the memory weights
    quantized_memory = torch.tensor(
        [int(round(m * S)) for m in memory], dtype=torch.long, device="cpu"
    )
    runtimes = torch.tensor(runtime, dtype=torch.float32, device="cpu")

    # Quantized pseudopolynomial DP for 0-1 Knapsack
    quantized_max_memory = int(round(max_memory * S))

    # Hirschberg algorithm:
    # split memory, solve dp for each half and recurse with best memory budget
    saved_items, recomputable_items = dp_knapsack_recurse(
        memory_indices=list(range(len(memory))),
        quantized_max_memory=quantized_max_memory,
        quantized_memory=quantized_memory,
        runtimes=runtimes,
    )

    max_runtime = sum(runtime[i] for i in saved_items)

    recomputable_items.reverse()

    return max_runtime, saved_items, recomputable_items


def dp_knapsack_recurse(
    memory_indices, quantized_max_memory, quantized_memory, runtimes
):
    if quantized_max_memory <= 0:
        return [], memory_indices

    if len(memory_indices) == 1:
        if quantized_memory[memory_indices[0]] <= quantized_max_memory:
            return memory_indices, []
        else:
            return [], memory_indices

    memory_split = len(memory_indices) // 2
    left_part_memory = memory_indices[:memory_split]
    right_part_memory = memory_indices[memory_split:]

    # sliding window: for each memory budget, compute dp profile for a half using only a previous one
    prev_dp_profile = torch.zeros(
        quantized_max_memory + 1, dtype=torch.float32, device="cpu"
    )
    dp_profile = prev_dp_profile
    for i in left_part_memory:
        current_memory = quantized_memory[i]
        current_runtime = runtimes[i]

        if current_memory == 0:
            dp_profile = prev_dp_profile + current_runtime
        else:
            dp_profile[current_memory:] = torch.maximum(
                prev_dp_profile[current_memory:],
                prev_dp_profile[:-current_memory] + current_runtime,
            )
        prev_dp_profile = dp_profile

    left_part_dp = dp_profile

    prev_dp_profile = torch.zeros(
        quantized_max_memory + 1, dtype=torch.float32, device="cpu"
    )
    dp_profile = prev_dp_profile
    for i in right_part_memory:
        current_memory = quantized_memory[i]
        current_runtime = runtimes[i]

        if current_memory == 0:
            dp_profile = prev_dp_profile + current_runtime
        else:
            dp_profile[current_memory:] = torch.maximum(
                prev_dp_profile[current_memory:],
                prev_dp_profile[:-current_memory] + current_runtime,
            )
        prev_dp_profile = dp_profile

    right_part_dp = dp_profile.flip(-1)

    _, next_memory_split = torch.max(left_part_dp + right_part_dp, dim=0)

    saved_items_left, recomputable_items_left = dp_knapsack_recurse(
        memory_indices=left_part_memory,
        quantized_max_memory=next_memory_split,
        quantized_memory=quantized_memory,
        runtimes=runtimes,
    )

    saved_items_right, recomputable_items_right = dp_knapsack_recurse(
        memory_indices=right_part_memory,
        quantized_max_memory=quantized_max_memory - next_memory_split,
        quantized_memory=quantized_memory,
        runtimes=runtimes,
    )

    return (
        saved_items_left + saved_items_right,
        recomputable_items_left + recomputable_items_right,
    )
