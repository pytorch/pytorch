#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from datetime import timedelta
from typing import List
from contextlib import contextmanager

_NUM_MEMBERS = "/num_members"
_LAST_MEMBER_CHECKIN = "/last_member"

__all__ = ["store_timeout", "get_all", "synchronize", "barrier"]

@contextmanager
def store_timeout(store, timeout: float):
    """
    This sets the timeout and then restores the old timeout when the context
    manager exits.

    Args:
        store: the store to set the timeout on
        timeout: the timeout to set
    """

    old_timeout = store.timeout
    store.set_timeout(timedelta(seconds=timeout))
    yield
    store.set_timeout(old_timeout)


def get_all(store, rank: int, prefix: str, world_size: int):
    r"""
    Given a store and a prefix, the method goes through the array of keys
    of the following format: ``{prefix}{idx}``, where idx is in a range
    from 0 to size, and tries to retrieve the data.

    The Rank0 process waits at the end to make sure all other processes
    finished the procedure before exiting.

    Usage

    ::

     values = get_all(store, 'torchelastic/data', 3)
     value1 = values[0] # retrieves the data for key torchelastic/data0
     value2 = values[1] # retrieves the data for key torchelastic/data1
     value3 = values[2] # retrieves the data for key torchelastic/data2

    """
    data_arr = store.multi_get(
        [f"{prefix}{idx}" for idx in range(world_size)]
    )

    barrier_key = _barrier_nonblocking(
        store=store,
        world_size=world_size,
        key_prefix=f"{prefix}/finished",
    )
    if rank == 0:
        # Rank0 runs the TCPStore daemon, as a result it needs to exit last.
        # Otherwise, the barrier may timeout if rank0 process finished the work
        # before other processes finished `get_all` method
        store.get(barrier_key)

    return data_arr


def synchronize(
    store,
    data: bytes,
    rank: int,
    world_size: int,
    key_prefix: str,
    timeout: float = 300,
) -> List[bytes]:
    """
    Synchronizes ``world_size`` agents between each other using the underlying c10d store.
    The ``data`` will be available on each of the agents.

    Note: The data on the path is not deleted, as a result there can be stale data if
        you use the same key_prefix twice.

    Time complexity: O(N) per worker, O(N^2) globally.
    """
    with store_timeout(store, timeout):
        store.set(f"{key_prefix}{rank}", data)
        agent_data = get_all(store, rank, key_prefix, world_size)
        return agent_data


def _barrier_nonblocking(store, world_size: int, key_prefix: str) -> str:
    """
    Does all the non-blocking operations for a barrier and returns the final key
    that can be waited on.
    """
    num_members_key = key_prefix + _NUM_MEMBERS
    last_member_key = key_prefix + _LAST_MEMBER_CHECKIN


    idx = store.add(num_members_key, 1)
    if idx == world_size:
        store.set(last_member_key, "<val_ignored>")

    return last_member_key


def barrier(
    store, world_size: int, key_prefix: str, barrier_timeout: float = 300
) -> None:
    """
    A global lock between agents. This will pause all workers until at least
    ``world_size`` workers respond.

    This uses a fast incrementing index to assign waiting ranks and a success
    flag set by the last worker.

    Time complexity: O(1) per worker, O(N) globally.

    Note: Since the data is not removed from the store, the barrier can be used
        once per unique ``key_prefix``.
    """

    with store_timeout(store, barrier_timeout):
        last_member_key = _barrier_nonblocking(store=store, world_size=world_size, key_prefix=key_prefix)
        store.get(last_member_key)
