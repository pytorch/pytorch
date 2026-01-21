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
        [round(m * S) for m in memory], dtype=torch.long, device="cpu"
    )
    runtimes = torch.tensor(runtime, dtype=torch.float32, device="cpu")

    # Quantized pseudopolynomial DP for 0-1 Knapsack
    quantized_max_memory = round(max_memory * S)

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

    # q_ prefix stands for quantized
    q_memory = [int(round(m * S)) for m in memory]
    runtimes = [float(v) for v in runtime]

    q_max_memory = int(round(max_memory * S))

    q_memory_length = len(q_memory)
    if q_memory_length == 0:
        return 0.0, [], []

    item_indices = list(range(q_memory_length))
    dp_profile_size = q_max_memory + 1

    # Current DP profile (row)
    dp_profile = torch.zeros(dp_profile_size, dtype=torch.float32, device="cpu")
    # Store a candidate for next dp_profile - current dp row + item
    candidate_profile = torch.empty(dp_profile_size, dtype=torch.float32, device="cpu")
    left_profile = torch.empty(dp_profile_size, dtype=torch.float32, device="cpu")
    right_profile = torch.empty(dp_profile_size, dtype=torch.float32, device="cpu")

    saved_items: list[int] = []
    recomputable_items: list[int] = []

    # Explicit stack to optimize memory and avoid recursion
    # Stack stores segments as (start index, end index, capacity for segment)
    stack: list[tuple[int, int, int]] = [(0, q_memory_length, q_max_memory)]

    # LIFO
    while stack:
        start, end, capacity = stack.pop()
        length = end - start
        if length == 0:
            continue

        # Leaf
        if length == 1:
            index = item_indices[start]
            memory_item = q_memory[index]
            runtime_item = runtimes[index]
            if memory_item <= capacity and runtime_item > 0.0:
                saved_items.append(index)
            else:
                recomputable_items.append(index)
            continue

        # Split the segment into two halves
        middle = start + (length // 2)
        left_start, left_end = middle, end
        right_start, right_end = start, middle

        # Assign items to both halves
        left_items = item_indices[left_start:left_end]
        right_items = item_indices[right_start:right_end]

        # Working only on items allowed by segment's capacity
        capacity = capacity + 1
        dp_view = dp_profile[:capacity]
        candidate_view = candidate_profile[:capacity]
        left_dp_local = left_profile[:capacity]
        right_dp_local = right_profile[:capacity]

        # Left part
        dp_view.zero_()
        for index in left_items:
            memory_item = q_memory[index]
            runtime_item = runtimes[index]

            if memory_item == 0:
                # Weight is 0, so add it to all capacities; a "free lunch", essentially
                dp_view.add_(runtime_item)
                continue

            # If item is too heavy, we skip it
            if memory_item >= capacity:
                continue

            # Add the current item so we can then pick the highest value
            dp_view_candidate = candidate_view[: capacity - memory_item]
            torch.add(dp_view[:-memory_item], runtime_item, out=dp_view_candidate)
            # Take the highest - either previous (without current) or with current
            torch.maximum(
                dp_view[memory_item:], dp_view_candidate, out=dp_view[memory_item:]
            )

        # Store the left profile
        left_dp_local.copy_(dp_view)

        # Right part
        dp_view.zero_()
        for index in right_items:
            memory_item = q_memory[index]
            runtime_item = runtimes[index]

            if memory_item == 0:
                dp_view.add_(runtime_item)
                continue

            if memory_item >= capacity:
                continue

            dp_view_candidate = candidate_view[: capacity - memory_item]
            torch.add(dp_view[:-memory_item], runtime_item, out=dp_view_candidate)
            torch.maximum(
                dp_view[memory_item:], dp_view_candidate, out=dp_view[memory_item:]
            )

        # Store the reversed right profile
        right_dp_local.copy_(dp_view.flip(-1))

        # In-place compute item-wise sum of left and right to pick the split point where the sum is highest
        left_dp_local.add_(right_dp_local)

        # Pick the index of highest value of a pair, which we then use as a split point
        best_split = int(torch.argmax(left_dp_local).item())

        left_capacity = best_split
        right_capacity = capacity - best_split

        # Clamp (might be removed if we're 100% sure that there is no edge case that will mess up the indices math)
        if left_capacity < 0:
            left_capacity = 0
        if right_capacity < 0:
            right_capacity = 0
        if left_capacity > q_max_memory:
            left_capacity = q_max_memory
        if right_capacity > q_max_memory:
            right_capacity = q_max_memory

        # Push right then left, so left is processed next
        stack.append((right_start, right_end, right_capacity))
        stack.append((left_start, left_end, left_capacity))

    saved_items = sorted(saved_items)
    recomputable_items = sorted(recomputable_items)

    max_runtime = sum(runtime[i] for i in saved_items)
    recomputable_items.reverse()
    return max_runtime, saved_items, recomputable_items
