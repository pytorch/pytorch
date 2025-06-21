"""
Barrier implementations for synchronizing distributed checkpoint operations.

This module provides abstract and concrete barrier implementations that ensure
all ranks in a distributed training environment complete their checkpoint operations
before proceeding, which is essential for data consistency.
"""

import abc
import logging
from collections import Counter
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed.elastic.utils.store as store_util


logger = logging.getLogger()

SC_CHECKPOINT_DEFAULT_PORT: int = 3451


class Barrier(abc.ABC):
    """
    Abstract base class for synchronization barriers.

    A barrier ensures that all ranks in a distributed environment reach a certain
    point in execution before any rank proceeds further, which is essential for
    coordinating operations like checkpointing across multiple processes.
    """

    @abc.abstractmethod
    def execute_barrier(self, barrier_prefix: str, timeout_secs: int) -> None:
        """
        Execute a synchronization barrier with the given prefix and timeout.

        Args:
            barrier_prefix: A string prefix to identify this specific barrier.
            timeout_secs: Maximum time in seconds to wait for all ranks to reach the barrier.
        """


class TCPStoreBarrier(Barrier):
    """
    A barrier implementation using PyTorch's TCPStore for synchronization.

    This barrier uses a TCP-based distributed key-value store to coordinate
    synchronization across multiple processes. It maintains separate stores
    for different barrier prefixes to allow concurrent barriers.
    """

    def __init__(
        self,
        timeout_barrier_init_secs: int,
        barrier_prefix_list: list[str],
        world_size: int,
        use_checkpoint_barrier_tcpstore_libuv: bool,
        tcpstore_port: int,
        master_address: str,
        rank: int,
        local_world_size: int,
    ):
        """
        Initialize a TCPStoreBarrier.

        Args:
            timeout_barrier_init_secs: Timeout in seconds for initializing the TCPStore.
            barrier_prefix_list: List of barrier prefixes to initialize stores for.
            world_size: Total number of processes in the distributed environment.
            use_checkpoint_barrier_tcpstore_libuv: Whether to use libuv for the TCPStore.
            tcpstore_port: Port number for the TCPStore.
            master_address: Address of the master node for the TCPStore.
            rank: Rank of the current process.
            local_world_size: Number of processes on the local node.
        """
        logger.info(
            "Initializing TCPStore master_address=%s tcpstore_port=%s rank=%s "
            "world_size=%s timeout_barrier_init_secs=%s use_checkpoint_barrier_tcpstore_libuv=%s",
            master_address,
            tcpstore_port,
            rank,
            world_size,
            timeout_barrier_init_secs,
            use_checkpoint_barrier_tcpstore_libuv,
        )

        # Counter collection to track barrier seq on a per barrier prefix basis.
        self._tcp_store_barrier_seq: Counter = Counter()

        # TCPStore clients should be used independently as we expect
        # undefined behavior (and potential deadlocks) if used by multiple
        # threads. This dict tracks one TCPStore per barrier prefix.
        self._tcp_store_dict: dict[str, dist.TCPStore] = {}
        self._rank = rank
        self._world_size = world_size
        self._local_world_size = local_world_size

        # This uses the shared TCPStore and we do NOT set any rank to be the master.
        #
        # This is behind a JK, please make sure your job is enabled: https://fburl.com/justknobs/g67ztbav

        for checkpoint_type in barrier_prefix_list:
            self._tcp_store_dict[checkpoint_type] = torch.distributed.TCPStore(
                master_address,
                int(tcpstore_port),
                world_size=world_size,
                timeout=timedelta(seconds=timeout_barrier_init_secs),
                use_libuv=use_checkpoint_barrier_tcpstore_libuv,
            )

    def execute_barrier(self, barrier_prefix: str, timeout_secs: int) -> None:
        """
        Execute a synchronization barrier with the given prefix and timeout.

        The implementation uses a sequence number that is incremented every time
        a barrier is reached. The sequence number is per barrier prefix to allow
        different barriers to operate concurrently.

        Args:
            barrier_prefix: A string prefix to identify this specific barrier.
            timeout_secs: Maximum time in seconds to wait for all ranks to reach the barrier.
        """
        logger.info(
            "Executing barrier barrier_prefix=%s timeout_secs=%s",
            barrier_prefix,
            timeout_secs,
        )

        def _rank_key(rank: int) -> str:
            return f"rank{rank}"

        # Get the TCPStore client for that specific barrier_prefix (checkpoint type).
        tcp_store = self._tcp_store_dict[barrier_prefix]

        # Track which barrier sequence this rank is joining.
        tcp_store.set(
            _rank_key(self._rank), str(self._tcp_store_barrier_seq[barrier_prefix])
        )

        # Execute barrier for that sequence number (for the specific prefix).
        store_util.barrier(
            store=tcp_store,
            world_size=self._world_size,
            key_prefix=(
                barrier_prefix + str(self._tcp_store_barrier_seq[barrier_prefix])
            ),
        )
        self._tcp_store_barrier_seq[barrier_prefix] += 1
