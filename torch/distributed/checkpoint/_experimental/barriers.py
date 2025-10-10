"""
Barrier implementations for synchronizing distributed checkpoint operations.

This module provides abstract and concrete barrier implementations that ensure
all ranks in a distributed training environment complete their checkpoint operations
before proceeding, which is essential for data consistency.
"""

import abc
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Optional

import torch.distributed as dist
import torch.distributed.elastic.utils.store as store_util


logger = logging.getLogger()


# Registry of barrier types
BARRIER_REGISTRY: dict[str, type] = {}


def register_barrier(barrier_class: type) -> type:
    """Register a barrier class in the global registry."""
    if hasattr(barrier_class, "barrier_type"):
        BARRIER_REGISTRY[barrier_class.barrier_type] = barrier_class
    return barrier_class


@dataclass
class BarrierConfig:
    """
    Configuration for barrier construction.

    This class provides a flexible way to configure different barrier implementations
    with their specific constructor arguments. The barrier type will be looked up
    from a registry and instantiated with rank_info and barrier_args.

    Attributes:
        barrier_type: A string identifying the barrier type (e.g., "tcp_store").
                     If None, no barrier will be used.
        barrier_args: Dictionary of arguments to pass to the barrier constructor.
                     rank_info will be automatically injected as the first argument.

    Examples:
        # No barrier
        BarrierConfig()

        # TCPStore barrier
        BarrierConfig(
            barrier_type="tcp_store",
            barrier_args={
                'timeout_barrier_init_secs': 30,
                'barrier_prefix_list': ['checkpoint'],
                'use_checkpoint_barrier_tcpstore_libuv': False,
                'tcpstore_port': 12345,
                'master_address': 'localhost'
            }
        )
    """

    barrier_type: Optional[str] = None
    barrier_args: dict[str, Any] = field(default_factory=dict)


def create_barrier_from_config(
    barrier_config: BarrierConfig,
) -> Optional["Barrier"]:
    """
    Create a barrier instance from BarrierConfig.

    Args:
        barrier_config: Configuration for barrier construction.

    Returns:
        Barrier instance or None if no barrier type is configured.

    Raises:
        ValueError: If the barrier_type is not found in the registry.
    """
    if barrier_config.barrier_type is None:
        return None

    if barrier_config.barrier_type not in BARRIER_REGISTRY:
        raise ValueError(
            f"Unknown barrier type: {barrier_config.barrier_type}. "
            f"Available types: {list(BARRIER_REGISTRY.keys())}"
        )

    barrier_class = BARRIER_REGISTRY[barrier_config.barrier_type]
    return barrier_class(**barrier_config.barrier_args)


class Barrier(abc.ABC):
    """
    Abstract base class for synchronization barriers.

    A barrier ensures that all ranks in a distributed environment reach a certain
    point in execution before any rank proceeds further, which is essential for
    coordinating operations like checkpointing across multiple processes.
    """

    @abc.abstractmethod
    def __init__(self, **kwargs: dict[str, Any]):
        """
        Initialize a barrier.

        Args:
            **kwargs: Keyword arguments for specific barrier implementations.
                     Common arguments may include rank information, barrier prefixes,
                     timeout settings, and other barrier-specific configuration.
        """
        # No implementation needed in the abstract base class

    @abc.abstractmethod
    def execute_barrier(self) -> None:
        """
        Execute a synchronization barrier.

        This method uses the barrier_prefix provided during initialization to
        coordinate synchronization across processes.
        """


@register_barrier
class DistBarrier(Barrier):
    """
    A barrier implementation using PyTorch's distributed barrier for synchronization.

    This barrier uses the built-in torch.distributed.barrier() function to coordinate
    synchronization across multiple processes. It's simpler than TCPStoreBarrier but
    requires an initialized process group.
    """

    barrier_type = "dist_barrier"

    def __init__(
        self,
    ) -> None:
        """
        Initialize a DistBarrier.

        This barrier requires an initialized PyTorch distributed process group.
        No additional arguments are needed as it uses the current process group.

        Raises:
            AssertionError: If the distributed process group is not initialized.
        """
        assert dist.is_initialized(), (
            "DistBarrier requires an initialized process group."
        )

    def execute_barrier(self) -> None:
        """
        Execute a synchronization barrier using the prefix provided during initialization.
        """
        # Note: dist.barrier() doesn't support explicit timeouts
        # The timeout is handled by the underlying implementation
        dist.barrier()


@register_barrier
class TCPStoreBarrier(Barrier):
    """
    A barrier implementation using PyTorch's TCPStore for synchronization.

    This barrier uses a TCP-based distributed key-value store to coordinate
    synchronization across multiple processes. It uses a single TCP store
    for all barrier operations, with different prefixes to distinguish between
    different barrier types.
    """

    barrier_type = "tcp_store"

    def __init__(
        self,
        global_rank: int,
        global_world_size: int,
        barrier_prefix: str,
        timeout_barrier_init_secs: int,
        use_checkpoint_barrier_tcpstore_libuv: bool,
        tcpstore_port: int,
        master_address: str,
        timeout_secs: int,
    ):
        """
        Initialize a TCPStoreBarrier.

        Args:
            global_rank: The rank of the current process in the distributed environment.
            global_world_size: The total number of processes in the distributed environment.
            barrier_prefix: A string prefix to identify this specific barrier.
            timeout_barrier_init_secs: Timeout in seconds for initializing the TCPStore.
            use_checkpoint_barrier_tcpstore_libuv: Whether to use libuv for the TCPStore.
            tcpstore_port: Port number for the TCPStore.
            master_address: Address of the master node for the TCPStore.
            timeout_secs: Maximum time in seconds to wait for all ranks to reach the barrier.
        """
        logger.info(
            "Initializing TCPStore master_address=%s tcpstore_port=%s rank=%s "
            "world_size=%s barrier_prefix=%s timeout_barrier_init_secs=%s "
            "use_checkpoint_barrier_tcpstore_libuv=%s timeout_secs=%s",
            master_address,
            tcpstore_port,
            global_rank,
            global_world_size,
            barrier_prefix,
            timeout_barrier_init_secs,
            use_checkpoint_barrier_tcpstore_libuv,
            timeout_secs,
        )

        # Counter collection to track barrier seq on a per barrier prefix basis.
        self._tcp_store_barrier_seq: Counter = Counter()
        self._barrier_prefix = barrier_prefix

        # Store rank and world size for barrier operations
        self._global_rank = global_rank
        self._global_world_size = global_world_size
        self._timeout_secs = timeout_secs

        # Create a single TCP store for all barrier operations
        self._tcp_store = dist.TCPStore(
            master_address,
            int(tcpstore_port),
            world_size=self._global_world_size,
            timeout=timedelta(seconds=timeout_barrier_init_secs),
            is_master=(self._global_rank == 0),
        )

    def execute_barrier(self) -> None:
        """
        Execute a synchronization barrier using the prefix provided during initialization.

        The implementation uses a sequence number that is incremented every time
        a barrier is reached. The sequence number is per barrier prefix to allow
        different barriers to operate concurrently.
        """
        barrier_prefix = self._barrier_prefix

        logger.info(
            "Executing barrier barrier_prefix=%s timeout_secs=%s",
            barrier_prefix,
            self._timeout_secs,
        )

        def _rank_key(rank: int) -> str:
            return f"rank{rank}"

        # Track which barrier sequence this rank is joining.
        self._tcp_store.set(
            _rank_key(self._global_rank),
            str(self._tcp_store_barrier_seq[barrier_prefix]),
        )

        # Execute barrier for that sequence number (for the specific prefix).
        store_util.barrier(
            store=self._tcp_store,
            world_size=self._global_world_size,
            key_prefix=(
                barrier_prefix + str(self._tcp_store_barrier_seq[barrier_prefix])
            ),
        )
        self._tcp_store_barrier_seq[barrier_prefix] += 1
