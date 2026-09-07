from __future__ import annotations

import dataclasses
import datetime
import logging
import os
import threading
import time
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch._dynamo.testing import rand_strided
from torch._inductor import config


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._inductor.runtime.triton_compat import Config
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from torch._prims_common import ShapeType, StrideType

log = logging.getLogger(__name__)

_AUTOTUNE_PG: dist.ProcessGroup | None = None
_AUTOTUNE_PG_LOCK = threading.Lock()


def get_or_create_autotune_pg() -> dist.ProcessGroup | None:
    """
    Lazily create the autotuning process group. Must only be called after all
    participating ranks have confirmed via the 2PC protocol, since PG creation
    requires all member ranks to participate.
    """
    global _AUTOTUNE_PG
    if _AUTOTUNE_PG is not None:
        return _AUTOTUNE_PG

    with _AUTOTUNE_PG_LOCK:
        if _AUTOTUNE_PG is not None:
            return _AUTOTUNE_PG

        if not dist.is_available() or not dist.is_initialized():
            return None

        if config.distributed_runtime_autotune_host_only:
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 0))
            local_rank = int(os.environ.get("LOCAL_RANK", -1))
            if local_world_size <= 0 or local_rank < 0:
                log.error(
                    "Distributed autotune: LOCAL_WORLD_SIZE or LOCAL_RANK not set, "
                    "cannot initialize host-only autotune process group"
                )
                return None
            global_rank = dist.get_rank()
            base = global_rank - local_rank
            local_ranks = list(range(base, base + local_world_size))
            _AUTOTUNE_PG = dist.new_group(
                ranks=local_ranks,
                use_local_synchronization=True,
                group_desc="pt2_distributed_runtime_autotune",
            )
        else:
            _AUTOTUNE_PG = dist.new_group(
                group_desc="pt2_distributed_runtime_autotune",
            )

        return _AUTOTUNE_PG


@dataclasses.dataclass
class TensorMeta:
    """
    Contains all the metadata needed to construct a random tensor for
    benchmarking a Triton kernel.
    """

    size: ShapeType
    stride: StrideType
    dtype: torch.dtype
    offset: int | torch.SymInt

    @classmethod
    def from_irnode(
        cls,
        node: Any,
        sizevars: Any,
        hint_override: int | None = None,
    ) -> TensorMeta:
        size = sizevars.optimization_hints_with_override(node.get_size(), hint_override)
        stride = sizevars.optimization_hints_with_override(
            node.get_stride(), hint_override
        )
        offset = sizevars.optimization_hint_with_override(
            node.get_layout().offset, hint_override
        )
        dtype = node.get_dtype()

        return cls(
            size=size,
            stride=stride,
            dtype=dtype,
            offset=offset,
        )

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> TensorMeta:
        return cls(
            size=tuple(tensor.size()),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
            offset=tensor.storage_offset(),
        )

    def to_tensor(self) -> torch.Tensor:
        """
        Create a random tensor from metadata.
        """
        return rand_strided(
            self.size,
            self.stride,
            dtype=self.dtype,
            device=torch.device("cuda", torch.cuda.current_device()),
            extra_size=int(self.offset),
        )


@dataclasses.dataclass
class WorkspaceMeta(TensorMeta):
    """
    Metadata for a Workspace buffer. Produces a zeroed tensor.
    """

    @classmethod
    def from_workspace(
        cls,
        workspace_arg: Any,
        sizevars: Any,
        hint_override: int | None = None,
    ) -> WorkspaceMeta:
        count = sizevars.optimization_hint_with_override(
            workspace_arg.count, hint_override
        )
        return cls(size=(count,), stride=(1,), dtype=workspace_arg.dtype, offset=0)

    def to_tensor(self) -> torch.Tensor:
        return torch.zeros(
            self.size,
            dtype=self.dtype,
            device=torch.device("cuda", torch.cuda.current_device()),
        )


class DistributedAutotuneCoordinator:
    """
    Distributes runtime autotuning work across ranks using a Two-Phase Commit
    protocol. When enabled, each rank autotunes at most 1/world_size of the
    kernels and results are synchronized, primarily reducing total autotuning
    time, but adding the benefit that all ranks use the same config for a
    given kernel.

    Phase 1 (codegen): Each rank fires a non-collective participation vote via
    store.add(). The first rank to vote becomes the leader. NOTE that we assume
    that if a rank generates at least one tunable kernel for a given compile_id,
    then it's guaranteed to enter CachingAutotuner.run() and we'll perform phase
    2 exactly once.

    Phase 2 (runtime, on first kernel invocation): The leader polls until all
    ranks have voted or timeout, then writes a commit/abort decision. All ranks
    read the decision before proceeding to any collectives, preventing the case
    where some ranks enter a collective while others have timed out. If all
    ranks are confirmed to participate, work is distributed round-robin, each
    rank autotunes its subset, and results are synchronized via all_gather.
    Finally, the best config is applied to each CachingAutotuner. Note that
    we assume that all ranks autotune the same kernels. If that is not the
    case, we simply apply the results we get back and those kernels that get no
    result fall back to local autotuning.
    """

    def __init__(self, key):
        self.key = key

        # A lock is probably overkill, but better for future proofing.
        self._lock = threading.Lock()
        self._autotuning_done = False

        # These just help with testing: whether the distributed autotuning
        # was attempted at all and whether we were able to get results for
        # the full set of kernels.
        self._autotuning_attempted = False
        self._autotuning_full_match = False

        self._kernel_metadata: dict[str, list[TensorMeta | int]] = {}
        self._autotuners: dict[str, CachingAutotuner] = {}

        # When coordinating across ranks, we use kernel hashes instead of names
        # alone to identify the same kernel.
        self._kernel_name_to_hash: dict[str, str] = {}
        self._kernel_hash_to_name: dict[str, str] = {}

        default_pg = dist.distributed_c10d._get_default_group()
        self._store = dist.distributed_c10d._get_default_store()

        if config.distributed_runtime_autotune_host_only:
            local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = local_world_size
            self.rank = local_rank
            host_base_rank = default_pg.rank() - local_rank
            key_prefix = f"pt2_dra_host_{host_base_rank}"
        else:
            self.world_size = default_pg.size()
            self.rank = default_pg.rank()
            key_prefix = "pt2_dra"

        self.pg: dist.ProcessGroup | None = None

        # Store keys for 2PC participation protocol
        key_suffix = key.replace("/", "_")
        self._vote_key = f"{key_prefix}_vote_{key_suffix}"
        self._decision_key = f"{key_prefix}_decision_{key_suffix}"

        # Fire participation vote (2PC Phase 1) when the coordinator is created.
        # The first rank to vote is the leader for Phase 2.
        vote_count = self._store.add(self._vote_key, 1)
        self._is_leader = vote_count == 1
        if self._is_leader:
            log.debug("[%s] Rank %d is the leader", self.key, self.rank)

    def register_kernel(
        self,
        kernel_name: str,
        kernel_hash: str,
        arg_metadata: list[TensorMeta | int],
    ) -> None:
        """
        Register a kernel during codegen (when we have arg metadata).
        """
        with self._lock:
            if kernel_hash in self._kernel_hash_to_name:
                log.error(
                    "[%s] Distributed autotune: BUG: kernel hash collision: "
                    "%s and %s have the same hash",
                    self.key,
                    self._kernel_hash_to_name[kernel_hash],
                    kernel_name,
                )

            self._kernel_metadata[kernel_name] = arg_metadata
            self._kernel_name_to_hash[kernel_name] = kernel_hash
            self._kernel_hash_to_name[kernel_hash] = kernel_name

    def register_autotuner(
        self,
        kernel_name: str,
        autotuner: CachingAutotuner,
    ) -> None:
        """
        Register a CachingAutotuner instance at runtime. Must be called after we
        know the CachingAutotuner's compile_id.
        """
        with self._lock:
            # Not all compiled kernels are eligible for autotuning
            if kernel_name in self._kernel_metadata:
                assert kernel_name not in self._autotuners
                self._autotuners[kernel_name] = autotuner

    def _leader_collect_votes_and_decide(self, kernel_count: int) -> None:
        """
        Poll until all ranks have voted or timeout, then write the commit/abort
        decision. Also checks whether there are enough kernels to justify the
        coordination overhead — this check must happen here (not per-rank)
        because all ranks must agree on whether to enter the all_gather collective.
        We optimize for the case where the ranks autotune the same kernels. If
        non-leader ranks would have enough kernels, we're ok skipping autotuning.
        """
        if kernel_count < config.distributed_runtime_autotune_min_kernels:
            log.warning(
                "[%s] Distributed autotune: too few kernels (%d < %d)",
                self.key,
                kernel_count,
                config.distributed_runtime_autotune_min_kernels,
            )
            # '0' indicates too few kernels.
            self._store.set(self._decision_key, "0")
            return

        start = time.monotonic()
        deadline = start + config.distributed_runtime_autotune_max_wait_seconds
        poll_interval = 0.1

        while time.monotonic() < deadline:
            vote_count = self._store.add(self._vote_key, 0)
            if vote_count == self.world_size:
                self._store.set(self._decision_key, str(self.world_size))
                log.debug(
                    "[%s] Distributed autotune: rank %d marking "
                    "all %d ranks participating, waited %.2fs",
                    self.key,
                    self.rank,
                    self.world_size,
                    time.monotonic() - start,
                )
                return
            time.sleep(poll_interval)

        # Timeout: do a final read in case the last vote arrived between the
        # last poll and the deadline.
        vote_count = self._store.add(self._vote_key, 0)
        self._store.set(self._decision_key, str(vote_count))
        if vote_count == self.world_size:
            log.debug(
                "[%s] Distributed autotune: rank %d marking "
                "all %d ranks participating (late, after timeout)",
                self.key,
                self.rank,
                self.world_size,
            )
        else:
            log.warning(
                "[%s] Distributed autotune: rank %d marking only %d / %d ranks responding",
                self.key,
                self.rank,
                vote_count,
                self.world_size,
            )

    def _check_participation(self, kernel_count: int) -> bool:
        """
        Phase 2 (store-based only, no collectives): the leader collects votes
        and writes a commit/abort decision, then all ranks read the decision
        and lazily create the process group if needed. Returns True if all
        ranks are participating.
        """
        # Phase 2a: Leader collects votes and writes decision
        if self._is_leader:
            self._leader_collect_votes_and_decide(kernel_count)

        # Phase 2b: All ranks wait for the leader's decision. Specify a timeout so we
        # can at least get an error message. But we already account for the max wait
        # seconds when writing the decision node, so we'd only expect a timeout only
        # for a serious problem like a leader crash.
        log.debug("[%s] Distributed autotune: waiting for decision", self.key)
        try:
            self._store.wait([self._decision_key], datetime.timedelta(minutes=5))
        except RuntimeError:
            log.exception(
                "[%s] Distributed autotune: error waiting for leader decision",
                self.key,
            )
            return False

        decision = int(self._store.get(self._decision_key))
        if decision == 0:
            log.warning(
                "[%s] Distributed autotune: leader found too few kernels to autotune",
                self.key,
            )
            return False
        if decision != self.world_size:
            log.warning(
                "[%s] Distributed autotune: no consensus (%d / %d ranks participating)",
                self.key,
                decision,
                self.world_size,
            )
            return False

        # All ranks confirmed — lazily create the process group. This is safe
        # because every participating rank reaches this point.
        self.pg = get_or_create_autotune_pg()
        if self.pg is None:
            log.error(
                "[%s] Distributed autotune: failed to create process group",
                self.key,
            )
            return False

        return True

    def run_autotuning(self) -> None:
        """
        Run distributed autotuning and apply results directly to all registered
        CachingAutotuners. Called on the first kernel run(). On success, each
        autotuner receives its winning Config. On failure, autotuners fall back
        to local autotuning.
        """
        if self._autotuning_done:
            return

        self._autotuning_done = True
        log.debug("[%s] Distributed autotune: starting on rank %d", self.key, self.rank)

        kernel_hashes = self._compute_kernel_hashes()

        results: dict[str, Config] = {}
        try:
            start = time.monotonic()
            if self._check_participation(len(kernel_hashes)):
                self._autotuning_attempted = True
                results = self._perform_distributed_autotuning(kernel_hashes)
                self._autotuning_full_match = len(results) == len(kernel_hashes)
                log.debug(
                    "[%s] Distributed autotune: rank %d received %d / %d results in %.2fs",
                    self.key,
                    self.rank,
                    len(results),
                    len(kernel_hashes),
                    time.monotonic() - start,
                )
        except Exception:
            log.exception("[%s] Distributed autotune: failed", self.key)

        # Apply results to autotuners that received a distributed result.
        # Autotuners without a result will fall back to local autotuning.
        for kernel_name, autotuner in self._autotuners.items():
            result = results.get(kernel_name)
            if result is not None:
                try:
                    autotuner._apply_distributed_autotune_result(result)
                except Exception:
                    log.exception(
                        "[%s] Distributed autotune: failed to apply result for %s",
                        self.key,
                        kernel_name,
                    )

        # Clean up data structures, but deliberately do not remove the coordinator
        # from _COORDINATORS so other CachingAutotuners can check _autotuning_done.
        self.cleanup()

    def _compute_kernel_hashes(self) -> list[str]:
        """
        Compute the kernel hashes eligible for autotuning (and sort for distribution).
        """
        if self._kernel_metadata.keys() != self._autotuners.keys():
            # This should never happen, but fail safe to avoid hanging the collective
            log.error("[%s] Distributed autotune: BUG: kernel name mismatch", self.key)
            return []

        # Kernels with more than one config, or all kernels if coordinate
        # descent is enabled. Sorted for consistent round-robin assignment.
        if config.coordinate_descent_tuning:
            return sorted(self._kernel_name_to_hash.values())

        return sorted(
            kernel_hash
            for name, kernel_hash in self._kernel_name_to_hash.items()
            if self._autotuners[name].initial_num_configs > 1
        )

    def _perform_distributed_autotuning(
        self, kernel_hashes: list[str]
    ) -> dict[str, Config]:
        """
        Autotune assigned kernels round-robin and sync results via all_gather.
        Optimizes for the common case where all ranks have the same kernel list,
        but gracefully handles mismatches by using whatever results match locally.
        """
        log.debug(
            "[%s] Distributed autotune: ready with %d kernels across %d ranks",
            self.key,
            len(kernel_hashes),
            self.world_size,
        )

        # Autotune kernels assigned to this rank
        local_results: list[tuple[str, Config]] = []

        for i, kernel_hash in enumerate(kernel_hashes):
            if i % self.world_size == self.rank:
                kernel_name = self._kernel_hash_to_name[kernel_hash]
                try:
                    result = self._autotune_kernel(kernel_name)
                except Exception:
                    log.exception(
                        "[%s] Distributed autotune: failed to autotune kernel %s",
                        self.key,
                        kernel_name,
                    )
                    result = None
                if result is not None:
                    local_results.append((kernel_hash, result))

        # Sync results across all ranks (keyed by hash, not name)
        all_results: list[list[tuple[str, Config]]] = [
            [] for _ in range(self.world_size)
        ]
        dist.all_gather_object(all_results, local_results, group=self.pg)

        # Map results back to local kernel names. With matching kernel lists,
        # we get results for all kernels. With mismatched lists, we use
        # whatever matches and fall back to local autotuning for the rest.
        results: dict[str, Config] = {}
        for rank_results in all_results:
            for kernel_hash, result_config in rank_results:
                kernel_name = self._kernel_hash_to_name.get(kernel_hash)
                if kernel_name is not None:
                    results[kernel_name] = result_config

        if len(results) < len(kernel_hashes):
            log.warning(
                "[%s] Distributed autotune: received results for only %d / %d kernels",
                self.key,
                len(results),
                len(kernel_hashes),
            )

        return results

    def _autotune_kernel(self, kernel_name: str) -> Config | None:
        """
        Autotune a single kernel using stored metadata.
        """
        autotuner = self._autotuners[kernel_name]
        arg_metadata = self._kernel_metadata[kernel_name]

        log.debug(
            "[%s] Distributed autotune: locally tuning kernel %s", self.key, kernel_name
        )

        # Construct args from metadata
        args: list[Any] = []
        for meta in arg_metadata:
            if isinstance(meta, TensorMeta):
                args.append(meta.to_tensor())
            else:
                args.append(meta)

        # Precompile if needed
        if len(autotuner.launchers) == 0:
            autotuner.precompile()

        # Run autotuning
        if len(autotuner.launchers) > 1:
            autotuner.autotune_to_one_config(*args)

        # Extract winning config
        if not autotuner.launchers:
            log.warning(
                "[%s] Distributed autotune: no launcher after autotuning %s",
                self.key,
                kernel_name,
            )
            return None

        # Run coordinate descent tuning if enabled
        if not getattr(
            autotuner.launchers[0].config, "found_by_coordesc", False
        ) and autotuner.inductor_meta.get("coordinate_descent_tuning", False):
            autotuner.launchers = [
                autotuner.coordinate_descent_tuning(autotuner.launchers[0], *args)
            ]

        return autotuner.launchers[0].config

    def cleanup(self) -> None:
        self._kernel_metadata.clear()
        self._autotuners.clear()
        self._kernel_name_to_hash.clear()
        self._kernel_hash_to_name.clear()


# Per-compile coordinator instances, keyed by (compile_id, is_backward) via
# _coordinator_key(). Forward and backward passes are handled separately.
_COORDINATORS: dict[str, DistributedAutotuneCoordinator | None] = {}
_COORDINATORS_LOCK = threading.Lock()


def _coordinator_key(compile_id: str, is_backward: bool) -> str:
    return f"{compile_id}/{'bwd' if is_backward else 'fwd'}"


def get_coordinator(
    compile_id: str, is_backward: bool
) -> DistributedAutotuneCoordinator | None:
    return _COORDINATORS.get(_coordinator_key(compile_id, is_backward))


def get_or_create_coordinator(
    compile_id: str, is_backward: bool
) -> DistributedAutotuneCoordinator | None:
    key = _coordinator_key(compile_id, is_backward)
    if key in _COORDINATORS:
        return _COORDINATORS[key]

    with _COORDINATORS_LOCK:
        if key in _COORDINATORS:
            return _COORDINATORS[key]

        assert config.distributed_runtime_autotune

        if not dist.is_available() or not dist.is_initialized():
            _COORDINATORS[key] = None
            return None

        if config.distributed_runtime_autotune_host_only:
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 0))
            local_rank = int(os.environ.get("LOCAL_RANK", -1))
            if local_world_size <= 0 or local_rank < 0:
                log.error(
                    "[%s] Distributed autotune: LOCAL_WORLD_SIZE or LOCAL_RANK not set",
                    key,
                )
                _COORDINATORS[key] = None
                return None

        coordinator = DistributedAutotuneCoordinator(key)
        _COORDINATORS[key] = coordinator
        return coordinator


def try_register_kernel_from_codegen(
    kernel_name: str,
    kernel_hash: str,
    is_backward: bool,
    get_arg_metadata: Callable[[], list[TensorMeta | int]],
) -> None:
    """
    Try to register a kernel during codegen.
    """
    compile_id = torch._guards.CompileContext.current_compile_id()
    if compile_id is None:
        log.error(
            "Distributed autotune: Unable to get compile_id for kernel %s",
            kernel_name,
        )
        return

    try:
        coordinator = get_or_create_coordinator(str(compile_id), is_backward)
        if coordinator is not None:
            arg_metadata = get_arg_metadata()
            coordinator.register_kernel(kernel_name, kernel_hash, arg_metadata)
    except Exception:
        log.exception(
            "[%s] Distributed autotune: Failed to register kernel",
            _coordinator_key(str(compile_id), is_backward),
        )


def register_autotuner(
    compile_id: str,
    is_backward: bool,
    kernel_name: str,
    autotuner: CachingAutotuner,
) -> None:
    coordinator = get_coordinator(compile_id, is_backward)
    if coordinator is not None:
        coordinator.register_autotuner(kernel_name, autotuner)
