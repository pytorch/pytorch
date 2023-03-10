#!/usr/bin/env python3
"""
This file contains utilities for constructing collective based control flows.
"""
import logging
from dataclasses import dataclass
from typing import Any, Callable, cast, Dict, Generic, List, Optional, Tuple, TypeVar

import torch.distributed as dist


logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class SyncPayload(Generic[T]):
    stage_name: Optional[str]
    success: bool
    payload: T
    # Exception info
    error_traits: Optional[Dict[str, str]]


# Note: use Any for typing for now so users can pass in
# either a list of None or target type placeholders
# otherwise pyre would complain
def all_gather_object_enforce_type(
    pg: dist.ProcessGroup,
    # pyre-fixme[2]: Parameter must have a type that does not contain `Any`
    object_list: List[Any],
    # pyre-fixme[2]: Parameter must have a type other than `Any`
    obj: Any,
    # pyre-fixme[2]: Parameter must have a type that does not contain `Any`
    type_checker: Callable[[Any, Any], bool] = lambda x, y: type(x) == type(y),
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


def all_gather_raise_on_failure(  # noqa: C901
    sync_payload: SyncPayload[T],
    msg_any_rank_failure: str,
    pg: Optional[dist.ProcessGroup],
) -> List[T]:

    if pg is not None:
        # List of success/failure across all ranks.
        total_list = [None] * dist.get_world_size(pg)
        all_gather_object_enforce_type(pg, total_list, sync_payload)
        # Each rank will throw the following exception in case
        # of failure on any rank.
        stage_name: Optional[str] = cast(SyncPayload[T], total_list[0]).stage_name
        ret_list: List[T] = []
        # List of (rank_id, error_traits) from each rank.
        error_traits_list: List[Tuple[int, Dict[str, str]]] = []
        error_msg: str = msg_any_rank_failure

        for i, sp in enumerate(cast(List[SyncPayload[T]], total_list)):
            if sp.stage_name != stage_name:
                error_msg += (
                    f"Unexpected stage name received from rank {i}: {sp.stage_name} "
                )
                continue
            if not sp.success and sp.error_traits is not None:
                error_traits_list.append((i, sp.error_traits))
                continue
            ret_list.append(sp.payload)

        if len(error_traits_list):
            # apply truncation as upstream may log the error to scuba
            # we already log the original per-rank exceptions above
            raise ValueError(
                error_msg,
                f"error_trait: {error_traits_list}",
            ) from None
        return ret_list
    else:
        if not sync_payload.success:
            synced_error_traits = sync_payload.error_traits
            assert (
                synced_error_traits is not None
            ), "Error traits should be present if rank failure"
            logger.error(f"{msg_any_rank_failure}\n{synced_error_traits}")
            # apply truncation as upstream may log the error to scuba
            raise ValueError(
                msg_any_rank_failure,
                synced_error_traits,
            ) from None
        return [sync_payload.payload]
