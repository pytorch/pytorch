#!/usr/bin/env python3


"""
A set of primitive functions for performing collective ops.

Each should also handle single rank scenario.
"""

from __future__ import annotations

import importlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, cast, Generic, TYPE_CHECKING, TypeVar


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

import torch
import torch.distributed as dist


__all__: list[str] = [
    "SyncPayload",
    "broadcast",
    "all_gather",
    "all_gather_object_enforce_type",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class SyncPayload(Generic[T]):
    stage_name: str | None
    success: bool
    payload: T
    exception: Exception | None = None


def broadcast(
    data_or_fn: T | Callable[[], T],
    *,
    success: bool = True,
    stage_name: str | None = None,
    rank: int = 0,
    pg: dist.ProcessGroup | None = None,
) -> T:
    """
    Broadcasts the data payload from rank 0 to all other ranks.
    Or if a function is passed, execute it in rank 0 and broadcast result to all other ranks.

    Can be used to broadcast a failure signal to stop all ranks.

    If the function raises an exception, all ranks will raise.

    Args:
        data_or_fn: the data to broadcast or function to execute and broadcast result.
        success: False to stop all ranks.
        stage_name: the name of the logical stage for synchronization and debugging
        rank: rank to broadcast data or execute function and broadcast results.
        pg: the process group for sync
    Throws:
        RuntimeError from original exception trace
    Returns:
        the value after synchronization

    Example usage:
    >> id = broadcast(data_or_fn=allocate_id, rank=0, pg=ext_pg.my_pg)
    """

    if not success and data_or_fn is not None:
        raise AssertionError(
            "Data or Function is expected to be None if not successful"
        )

    payload: T | None = None
    exception: Exception | None = None
    # if no pg is passed then execute if rank is 0
    if (pg is None and rank == 0) or (pg is not None and pg.rank() == rank):
        # determine if it is an executable function or data payload only
        if callable(data_or_fn):
            try:
                payload = data_or_fn()
            except Exception as e:
                success = False
                exception = e
        else:
            payload = data_or_fn

    # broadcast the exception type if any to all ranks for failure categorization
    sync_obj = SyncPayload(
        stage_name=stage_name,
        success=success,
        payload=payload,
        exception=exception,
    )

    if pg is not None:
        broadcast_list = [sync_obj]
        dist.broadcast_object_list(broadcast_list, src=rank, group=pg)
        if len(broadcast_list) != 1:
            raise AssertionError(
                f"Expected broadcast_list to have exactly 1 element, got {len(broadcast_list)}"
            )
        sync_obj = broadcast_list[0]

    # failure in any rank will trigger a throw in every rank.
    if not sync_obj.success:
        error_msg = f"Rank {rank} failed"
        if stage_name is not None:
            error_msg += f": stage {sync_obj.stage_name}"
        if sync_obj.exception is not None:
            error_msg += f": exception {sync_obj.exception}"

        raise RuntimeError(error_msg) from sync_obj.exception

    return cast(T, sync_obj.payload)


def all_gather(
    data_or_fn: T | Callable[[], T],
    stage_name: str | None = None,
    pg: dist.ProcessGroup | None = None,
) -> list[T]:
    """
    A simple all_gather primitive with basic synchronization guard logic,
    by checking payload from all ranks has the same stage name.

    Args:
        data_or_fn: the data to be all gathered across ranks or function to be executed
        stage_name: the sync stage name for out-of-sync protection
        pg: the process group for sync
    Throws:
        RuntimeError from original exception trace
    Returns:
        a list of synced data from all ranks

    Example usage:
    >> all_ids = all_gather(data_or_fn=allocate_id, pg=ext_pg.my_pg)
    """
    payload: T | None = None
    exception: Exception | None = None
    success = True
    # determine if it is an executable function or data payload only
    if callable(data_or_fn):
        try:
            payload = data_or_fn()
        except Exception as e:
            success = False
            exception = e
    else:
        payload = data_or_fn

    sync_obj = SyncPayload(
        stage_name=stage_name,
        success=success,
        payload=payload,
        exception=exception,
    )

    if pg is not None:
        # List of success/failure across all ranks.
        total_list = [None] * dist.get_world_size(pg)
        all_gather_object_enforce_type(pg, total_list, sync_obj)
        # Each rank will throw RuntimeError in case of failure on any rank.
        stage_name = cast(SyncPayload[T], total_list[0]).stage_name
        exception_list: list[tuple[int, Exception]] = []
        ret_list: list[T] = []
        error_msg: str = ""

        for i, sp in enumerate(cast(list[SyncPayload[T]], total_list)):
            if sp.stage_name != stage_name:
                error_msg += (
                    f"Unexpected stage name received from rank {i}: {sp.stage_name} "
                )
                continue
            if not sp.success and sp.exception is not None:
                exception_list.append((i, sp.exception))
                continue
            ret_list.append(sp.payload)

        if len(exception_list) > 0:
            raise RuntimeError(  # type: ignore[misc]
                error_msg,
                exception_list,
            ) from exception_list[0]  # pyrefly: ignore [bad-raise, invalid-inheritance]
        return ret_list
    else:
        if not sync_obj.success:
            raise RuntimeError(
                f"all_gather failed with exception {sync_obj.exception}",
            ) from sync_obj.exception
        return [sync_obj.payload]  # type: ignore[list-item]


# Note: use Any for typing for now so users can pass in
# either a list of None or target type placeholders
# otherwise pyre would complain
def all_gather_object_enforce_type(
    pg: dist.ProcessGroup,
    # pyre-fixme[2]: Parameter must have a type that does not contain `Any`
    object_list: list[Any],
    # pyre-fixme[2]: Parameter must have a type other than `Any`
    obj: Any,
    # pyre-fixme[2]: Parameter must have a type that does not contain `Any`
    type_checker: Callable[[Any, Any], bool] = lambda x, y: type(x) is type(y),
) -> None:
    """
    Similar to plain all_gather_object but with additional type checking
    AFTER gather is done to ensure basic consistency.
    If check does not pass, all ranks will fail with exception.

    This is generally to prevent conditional logic leading to
    unexpected messages being received. This is considered fatal code error,
    but due to logic stacks this might happen implicitly in practice.

    The default check does not check sub type (considered different)
    or covariance (considered same) but users can pass in custom checker
    if more complicated check is needed.
    """
    dist.all_gather_object(object_list, obj, group=pg)

    # conservative check
    list_len = len(object_list)
    if list_len == 0:
        return
    first_obj = object_list[0]
    for i in range(1, list_len):
        if not type_checker(first_obj, object_list[i]):
            raise TypeError(
                f"Object type at index {i} is {type(object_list[i])}, "
                f"while first object type is {type(first_obj)}"
            )


def _summarize_ranks(ranks: Iterable[int]) -> str:
    ranks = sorted(ranks)
    if min(ranks) < 0:
        raise AssertionError("ranks should all be positive")
    if len(set(ranks)) != len(ranks):
        raise AssertionError("ranks should not contain duplicates")
    curr: int | range | None = None
    ranges = []
    while ranks:
        x = ranks.pop(0)
        if curr is None:
            curr = x
        elif isinstance(curr, int):
            if x == curr + 1:
                curr = range(curr, x + 1, 1)
            else:
                step = x - curr
                curr = range(curr, x + step, step)
        else:
            if not isinstance(curr, range):
                raise AssertionError("curr must be an instance of range")
            if x == curr.stop:
                curr = range(curr.start, curr.stop + curr.step, curr.step)
            else:
                ranges.append(curr)
                curr = x

    if isinstance(curr, int):
        ranges.append(range(curr, curr + 1, 1))
    elif isinstance(curr, range):
        ranges.append(curr)

    result = []
    for r in ranges:
        if len(r) == 1:
            # pyrefly: ignore [bad-argument-type]
            result.append(f"{r.start}")
        elif r.step == 1:
            # pyrefly: ignore [bad-argument-type]
            result.append(f"{r.start}:{r.stop}")
        else:
            # pyrefly: ignore [bad-argument-type]
            result.append(f"{r.start}:{r.stop}:{r.step}")
    return ",".join(result)


def _check_philox_rng_sync(
    generator: torch.Generator, group: dist.ProcessGroup
) -> tuple[dict[Any, set], str]:
    local_state = generator.get_state()
    all_states = [torch.empty_like(local_state) for _ in range(group.size())]
    torch.distributed.all_gather(all_states, local_state)
    seeds_offsets = [
        (state[:8].view(torch.uint64).item(), state[8:].view(torch.uint64).item())
        for state in all_states
    ]
    seed_offset_ranks = defaultdict(set)
    for rank, (seed, offset) in enumerate(seeds_offsets):
        seed_offset_ranks[(seed, offset)].add(rank)
    return seed_offset_ranks, "(Seed, Offset)"


def _check_cpu_rng_sync(
    generator: torch.Generator, group: dist.ProcessGroup
) -> tuple[dict[Any, set], str]:
    # seed is returned as uint64_t from C impl, so may not fit in torch int64 tensor directly.
    state_tensor = generator.get_state()
    all_state_tensors = [torch.empty_like(state_tensor) for _ in range(group.size())]
    torch.distributed.all_gather(all_state_tensors, state_tensor)
    state_ranks = defaultdict(set)
    for rank, state_tensor in enumerate(all_state_tensors):
        # Summarize the state vector of the CPU rng.
        # The properties that matter most are (1) its different if there is a state difference, (2) its printable
        # (see desync table- not viable to print whole state vector of size 5k)
        state_ranks[torch.hash_tensor(state_tensor).item()].add(rank)
    return state_ranks, "Generator state hash"


def _check_rng_sync_internal(
    generator: torch.Generator, group: dist.ProcessGroup
) -> tuple[dict[Any, set], str]:
    if generator.device.type == "cuda":
        return _check_philox_rng_sync(generator, group)
    elif generator.device.type == "cpu":
        return _check_cpu_rng_sync(generator, group)
    else:
        raise NotImplementedError(
            f"Unsupported generator device: {generator.device.type}"
        )


def _desync_table_str(tag: str, value_ranks: dict[Any, set[int]]) -> str:
    headers = ["Ranks", f"{tag} values"]
    rank_values = [
        [_summarize_ranks(ranks), str(value)] for value, ranks in value_ranks.items()
    ]
    if importlib.util.find_spec("tabulate"):
        from tabulate import tabulate

        return tabulate(rank_values, headers=headers)
    row_str = "\n".join([str(row) for row in rank_values])
    return str(f"{headers}\n{row_str}")


def _check_rng_sync(generator: torch.Generator, group: dist.ProcessGroup) -> str | None:
    value_ranks, value_header = _check_rng_sync_internal(generator, group)
    log_str = None
    if len(value_ranks) > 1:
        log_str = f"Generator desync detected:\n{_desync_table_str(value_header, value_ranks)}"
        logger.error(log_str)
    return log_str
