# Copyright (c) 2025-2026, Tri Dao.

from typing import Optional
from dataclasses import dataclass

import cutlass.cute as cute
from cutlass import Boolean, Int32, const_expr
from cutlass.cutlass_dsl import if_generate, and_, dsl_user_op
from cutlass._mlir.dialects import nvvm as _nvvm, llvm
from cutlass.cute.typing import AddressSpace, Int, Pointer
from cutlass.pipeline import PipelineState, PipelineUserType
from cutlass.pipeline import Agent, CooperativeGroup, PipelineOp
from cutlass.pipeline import agent_sync, alloc_reserved_mbarrier
from cutlass.pipeline import NamedBarrier as NamedBarrierOg
from cutlass.pipeline import PipelineAsync as PipelineAsyncOg
from cutlass.pipeline import PipelineCpAsync as PipelineCpAsyncOg
from cutlass.pipeline import PipelineTmaAsync as PipelineTmaAsyncOg
from cutlass.pipeline import PipelineTmaStore as PipelineTmaStoreOg
from cutlass.pipeline import PipelineTmaUmma as PipelineTmaUmmaOg
from cutlass.pipeline import PipelineUmmaAsync as PipelineUmmaAsyncOg
from cutlass.pipeline import PipelineAsyncUmma as PipelineAsyncUmmaOg


# ── Shared helpers ───────────────────────────────────────────────────────────


def _override_create(parent_cls, child_cls):
    """Create a static factory that constructs parent_cls then re-classes to child_cls."""

    @staticmethod
    def create(*args, **kwargs):
        obj = parent_cls.create(*args, **kwargs)
        # Can't assign to __class__ directly since the dataclass is frozen
        object.__setattr__(obj, "__class__", child_cls)
        return obj

    return create


@dsl_user_op
def mbarrier_arrive_release_cluster(
    mbar_ptr: Pointer, peer_cta_rank_in_cluster: Int, *, loc=None, ip=None
) -> None:
    """Arrive on a peer CTA's mbarrier with cluster-scope release semantics.

    cute.arch.mbarrier_arrive with a peer rank emits mbarrier.arrive with the default
    .release.cta semantics, which does not order this CTA's smem writes with the peer
    CTA's reads of them (e.g. a 2-CTA tcgen05.mma reading our smem over DSMEM). Emit
    fence.release.sync_restrict::shared::cta.cluster followed by
    mbarrier.arrive.relaxed.cluster.shared::cluster instead: the fence releases all smem
    writes this thread has observed at cluster scope, and a cluster-scope acquire of the
    barrier phase on the consumer side (mbarrier_test_wait_acquire_cluster) completes the
    release-acquire relation. This is the cheaper form of mbarrier.arrive.release.cluster,
    which lowers to MEMBAR.GPU.
    """
    _nvvm.fence_sync_restrict(_nvvm.MemOrderKind.RELEASE, loc=loc, ip=ip)
    remote_ptr = _nvvm.mapa(
        llvm.PointerType.get(AddressSpace.dsmem),
        mbar_ptr.to_llvm_ptr(loc=loc, ip=ip),
        Int32(peer_cta_rank_in_cluster).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    _nvvm.mbarrier_arrive(
        None,
        remote_ptr,
        count=Int32(1).ir_value(loc=loc, ip=ip),
        scope=_nvvm.MemScopeKind.CLUSTER,
        relaxed=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbarrier_acquire_cluster(mbar_ptr: Pointer, phase: Int, *, loc=None, ip=None) -> None:
    """Cluster-scope acquire observation of an already-completed mbarrier phase.

    Emits mbarrier.test_wait.parity.acquire.cluster. The regular waits
    (mbarrier.try_wait.parity) only acquire at cta scope, which cannot pair with a
    cluster-scope release from another CTA (mbarrier_arrive_release_cluster). Call this
    after a regular wait on the same (barrier, phase) has already succeeded: the test_wait
    then observes the completed phase and upgrades the observation to cluster scope.
    The result must feed control flow, otherwise the test_wait is dead-code-eliminated;
    branch to a (never-taken) blocking wait so the observation cannot be dropped.
    """
    status = Boolean(
        _nvvm.mbarrier_wait_parity(
            mbar_ptr.to_llvm_ptr(loc=loc, ip=ip),
            Int32(phase).ir_value(loc=loc, ip=ip),
            _nvvm.MBarrierWaitKind.TEST,
            scope=_nvvm.MBarrierScopeKind.CLUSTER,
            order=_nvvm.MemOrderKind.ACQUIRE,
            loc=loc,
            ip=ip,
        )
    )
    if_generate(
        status == 0,
        lambda: cute.arch.mbarrier_wait(mbar_ptr, phase, loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )


def _make_state(index: Int32, phase: Int32) -> PipelineState:
    """Construct a PipelineState from index and phase (count/stages unused by callers)."""
    return PipelineState(stages=0, count=Int32(0), index=index, phase=phase)


def _call_with_elect_one(parent_method, self, state, elect_one, syncwarp, loc, ip):
    """Optionally wrap a parent pipeline method call in sync_warp + elect_one."""
    if const_expr(elect_one):
        if const_expr(syncwarp):
            cute.arch.sync_warp()
        with cute.arch.elect_one():
            parent_method(self, state, loc=loc, ip=ip)
    else:
        parent_method(self, state, loc=loc, ip=ip)


# ── Pipeline state ──────────────────────────────────────────────────────────


class PipelineStateWAdvance(PipelineState):
    @dsl_user_op
    def advance_iters(self, num_iterations: Int32, *, loc=None, ip=None):
        self._count += Int32(num_iterations)
        new_index = self._index + Int32(num_iterations)
        # How many times did we cross the stages boundary
        num_crossings = new_index // self.stages
        self._phase ^= num_crossings
        self._index = new_index % self.stages

    # This can be overridden by derived classes
    def __new_from_mlir_values__(self, values):
        return PipelineStateWAdvance(
            self.stages, Int32(values[0]), Int32(values[1]), Int32(values[2])
        )


def make_pipeline_state(type: PipelineUserType, stages: int):
    """
    Creates a pipeline state. Producers are assumed to start with an empty buffer and have a flipped phase bit of 1.
    """
    if type is PipelineUserType.Producer:
        return PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(1))
    elif type is PipelineUserType.Consumer:
        return PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(0))
    else:
        assert False, "Error: invalid PipelineUserType specified for make_pipeline_state."


# ── Mixin: _w_index / _w_index_phase variants ───────────────────────────────


class _PipelineIndexPhaseMixin:
    """Mixin providing _w_index_phase / _w_index methods that delegate to PipelineState-based parents."""

    @dsl_user_op
    def producer_acquire_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_acquire_token: Optional[Boolean] = None,
        *,
        loc=None,
        ip=None,
    ):
        state = _make_state(index, phase)
        self.producer_acquire(state, try_acquire_token, loc=loc, ip=ip)

    @dsl_user_op
    def producer_commit_w_index(self, index: Int32, *, loc=None, ip=None):
        state = _make_state(index, Int32(0))
        self.producer_commit(state, loc=loc, ip=ip)

    @dsl_user_op
    def consumer_wait_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_wait_token: Optional[Boolean] = None,
        *,
        loc=None,
        ip=None,
    ):
        state = _make_state(index, phase)
        self.consumer_wait(state, try_wait_token, loc=loc, ip=ip)

    @dsl_user_op
    def consumer_release_w_index(self, index: Int32, *, loc=None, ip=None):
        state = _make_state(index, Int32(0))
        self.consumer_release(state, loc=loc, ip=ip)


# ── NamedBarrier ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NamedBarrier(NamedBarrierOg):
    create = _override_create(NamedBarrierOg, None)  # patched below

    @dsl_user_op
    def arrive_w_index(self, index: Int32, *, loc=None, ip=None) -> None:
        """
        The aligned flavor of arrive is used when all threads in the CTA will execute the
        same instruction. See PTX documentation.
        """
        cute.arch.barrier_arrive(
            barrier_id=self.barrier_id + index,
            number_of_threads=self.num_threads,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def arrive_and_wait_w_index(self, index: Int32, *, loc=None, ip=None) -> None:
        cute.arch.barrier(
            barrier_id=self.barrier_id + index,
            number_of_threads=self.num_threads,
            loc=loc,
            ip=ip,
        )


NamedBarrier.create = _override_create(NamedBarrierOg, NamedBarrier)


# ── PipelineAsync ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineAsync(_PipelineIndexPhaseMixin, PipelineAsyncOg):
    """
    PipelineAsync with optional elect_one for producer_commit and consumer_release.

    When elect_one_*=True (set at create time), only one elected thread per warp
    signals the barrier arrive. This is useful when the mask count is set to 1 per warp.

    Args (to create):
        elect_one_commit: If True, only elected thread signals producer_commit.
        syncwarp_before_commit: If True (default), issue syncwarp before elect_one.
        elect_one_release: If True, only elected thread signals consumer_release.
        syncwarp_before_release: If True (default), issue syncwarp before elect_one.
            Set syncwarp to False when threads are already converged (e.g. after wgmma wait_group).
    """

    _elect_one_commit: bool = False
    _syncwarp_before_commit: bool = True
    _elect_one_release: bool = False
    _syncwarp_before_release: bool = True

    @staticmethod
    def create(
        *args,
        elect_one_commit: bool = False,
        syncwarp_before_commit: bool = True,
        elect_one_release: bool = False,
        syncwarp_before_release: bool = True,
        **kwargs,
    ):
        obj = PipelineAsyncOg.create(*args, **kwargs)
        object.__setattr__(obj, "__class__", PipelineAsync)
        object.__setattr__(obj, "_elect_one_commit", elect_one_commit)
        object.__setattr__(obj, "_syncwarp_before_commit", syncwarp_before_commit)
        object.__setattr__(obj, "_elect_one_release", elect_one_release)
        object.__setattr__(obj, "_syncwarp_before_release", syncwarp_before_release)
        return obj

    @dsl_user_op
    def producer_commit(self, state: PipelineState, *, loc=None, ip=None):
        _call_with_elect_one(
            PipelineAsyncOg.producer_commit,
            self,
            state,
            self._elect_one_commit,
            self._syncwarp_before_commit,
            loc,
            ip,
        )

    @dsl_user_op
    def consumer_release(self, state: PipelineState, *, loc=None, ip=None):
        _call_with_elect_one(
            PipelineAsyncOg.consumer_release,
            self,
            state,
            self._elect_one_release,
            self._syncwarp_before_release,
            loc,
            ip,
        )

    # _w_index variants inherited from _PipelineIndexPhaseMixin, which delegate
    # to producer_commit / consumer_release above.


# ── PipelineCpAsync ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineCpAsync(_PipelineIndexPhaseMixin, PipelineCpAsyncOg):
    _elect_one_release: bool = False
    _syncwarp_before_release: bool = True

    @staticmethod
    def create(
        *args,
        barrier_storage: Optional[cute.Pointer] = None,
        elect_one_release: bool = False,
        syncwarp_before_release: bool = True,
        **kwargs,
    ):
        obj = PipelineCpAsyncOg.create(*args, barrier_storage=barrier_storage, **kwargs)
        object.__setattr__(obj, "__class__", PipelineCpAsync)
        object.__setattr__(obj, "_elect_one_release", elect_one_release)
        object.__setattr__(obj, "_syncwarp_before_release", syncwarp_before_release)
        return obj

    @dsl_user_op
    def consumer_release(self, state: PipelineState, *, loc=None, ip=None):
        _call_with_elect_one(
            PipelineCpAsyncOg.consumer_release,
            self,
            state,
            self._elect_one_release,
            self._syncwarp_before_release,
            loc,
            ip,
        )

    # _w_index variants inherited from _PipelineIndexPhaseMixin.


# ── PipelineTmaAsync ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineTmaAsync(_PipelineIndexPhaseMixin, PipelineTmaAsyncOg):
    """PipelineTmaAsync with extra_tx_count plus optional elected consumer release."""

    _elect_one_release: bool = False
    _syncwarp_before_release: bool = True

    @staticmethod
    def create(
        *args,
        elect_one_release: bool = False,
        syncwarp_before_release: bool = True,
        **kwargs,
    ):
        obj = PipelineTmaAsyncOg.create(*args, **kwargs)
        object.__setattr__(obj, "__class__", PipelineTmaAsync)
        object.__setattr__(obj, "_elect_one_release", elect_one_release)
        object.__setattr__(obj, "_syncwarp_before_release", syncwarp_before_release)
        return obj

    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        extra_tx_count: int = 0,
        *,
        loc=None,
        ip=None,
    ):
        """
        TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        if const_expr(isinstance(extra_tx_count, int) and extra_tx_count == 0):
            self.sync_object_full.arrive(state.index, self.producer_mask, loc=loc, ip=ip)
        else:
            tx_count = self.sync_object_full.tx_count + extra_tx_count
            self.sync_object_full.arrive_and_expect_tx(state.index, tx_count, loc=loc, ip=ip)

    @dsl_user_op
    def consumer_release(self, state: PipelineState, *, loc=None, ip=None):
        _call_with_elect_one(
            PipelineTmaAsyncOg.consumer_release,
            self,
            state,
            self._elect_one_release,
            self._syncwarp_before_release,
            loc,
            ip,
        )


# ── PipelineStasAsync ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineStasAsync(PipelineAsyncOg):
    """Pipeline for a symmetric all-to-all cluster exchange via st.async (STAS).

    Models the reduction-buffer exchange of persistent cluster kernels (e.g.
    RMSNorm/LayerNorm bwd): every CTA is simultaneously producer and consumer.
    Each stage, every warp's lanes < cluster_n STAS values into every peer
    CTA's buffer(s), crediting the peer's transaction (full) barrier; each CTA
    then reads its own buffer(s) and multicast-releases every peer's empty
    barrier.

    Differences from the stock split-agent pipelines:

    - The full barrier is a transaction barrier armed and credited inside the
      data path (quack.reduce.cluster_reduce issues arrive_and_expect_tx and
      the STAS stores, and waits on it with ``state.phase``), so
      producer_commit is a no-op and producer_get_barrier hands the raw
      barrier to the STAS issuer.
    - Sync slots are decoupled from data slots: a stage owns ONE (full, empty)
      barrier pair, which may guard several reduction-buffer slots — the
      issuer arms the barrier with the combined byte count and one wait covers
      them all (LayerNorm bwd reduces sum(x_hat * wdy) and sum(wdy) per stage
      through one barrier this way). Buffer-slot indexing is the kernel's
      concern; the pipeline only tracks stages and phases.
    - consumer_release arrives with one lane per warp on each peer CTA; the
      empty barriers are initialized with num_warps * cluster_n arrives to
      match. The caller must fence the async proxy (which the STAS writes
      travel) before releasing — proxy ordering is the data path's concern,
      like the tx arming.

    This is close to PipelineTmaAsync with the transaction arming moved into
    the data path and point-to-point signaling replaced by all-to-all.
    """

    cluster_n: int = 1

    @staticmethod
    def create(
        *,
        num_stages: int,
        num_warps: int,
        cluster_n: int,
        barrier_storage: Optional[Pointer] = None,
        defer_sync: bool = False,
    ) -> "PipelineStasAsync":
        """Allocates num_stages (full, empty) barrier pairs, from the reserved smem
        partition when ``barrier_storage`` is None. With ``defer_sync=True`` the
        caller must fence the barrier init and arrive + wait on the cluster
        barrier itself before first use (lets the cluster sync overlap other
        setup work)."""
        if barrier_storage is None:
            barrier_storage = alloc_reserved_mbarrier(num_stages)
        # The full barrier's single arrive is the STAS issuer's
        # arrive_and_expect_tx; the pipeline itself never arrives on it.
        producer = (PipelineOp.AsyncThread, CooperativeGroup(Agent.Thread, 1))
        consumer = (
            PipelineOp.AsyncThread,
            CooperativeGroup(Agent.Thread, num_warps * cluster_n),
        )
        sync_object_full = PipelineAsyncOg._make_sync_object(
            barrier_storage.align(min_align=8), num_stages, producer
        )
        sync_object_empty = PipelineAsyncOg._make_sync_object(
            barrier_storage.align(min_align=8) + num_stages, num_stages, consumer
        )
        if not defer_sync:
            cute.arch.mbarrier_init_fence()
            agent_sync(Agent.ThreadBlockCluster, is_relaxed=True)
        return PipelineStasAsync(
            sync_object_full,
            sync_object_empty,
            num_stages,
            None,
            None,
            cluster_n,
        )

    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        *,
        loc=None,
        ip=None,
    ):
        """Wait until every peer has released this stage. Arming the transaction
        barrier is left to the STAS issuer (see class docstring)."""
        self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip)

    @dsl_user_op
    def producer_commit(self, state: PipelineState, *, loc=None, ip=None):
        """No-op: the STAS transactions complete the full barrier."""
        pass

    def producer_get_barrier(self, state: PipelineState) -> Pointer:
        """Full (transaction) barrier of the stage, for arrive_and_expect_tx + STAS."""
        return self.sync_object_full.get_barrier(state.index)

    @dsl_user_op
    def consumer_wait(
        self,
        state: PipelineState,
        try_wait_token: Optional[Boolean] = None,
        *,
        loc=None,
        ip=None,
    ):
        """Wait for the STAS transactions of this stage. Unused when the STAS
        issuer performs the wait itself (cluster_reduce does)."""
        self.sync_object_full.wait(state.index, state.phase, loc=loc, ip=ip)

    @dsl_user_op
    def consumer_release(self, state: PipelineState, *, loc=None, ip=None):
        # One lane per warp arrives on each peer (cheaper than all-lane arrives;
        # empty arrive counts match). The caller must fence the async proxy first.
        cute.arch.sync_warp()
        lane_idx = cute.arch.lane_idx()
        if_generate(
            lane_idx < self.cluster_n,
            lambda: self.sync_object_empty.arrive(state.index, lane_idx, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def producer_tail(self, state: PipelineState, *, loc=None, ip=None):
        """Prevent the CTA from exiting (invalidating its smem barriers for peer
        STAS) while peers may still be arriving. ``state`` points at the next
        useful stage; wait on ALL stages, like PipelineAsync.producer_tail —
        the remote empty arrives are .release.cta scope, so a cluster observer
        gets no ordering between arrives to different barriers and draining
        only the last-used stage would race a still-in-flight earlier arrive."""
        for _ in range(self.num_stages):
            self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip)
            state.advance(loc=loc, ip=ip)


# ── PipelineTmaStore ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineTmaStore(PipelineTmaStoreOg):
    """PipelineTmaStore with configurable cp.async.bulk wait read flag."""

    _read: bool = True

    @staticmethod
    def create(*args, read: bool = True, **kwargs):
        obj = PipelineTmaStoreOg.create(*args, **kwargs)
        object.__setattr__(obj, "__class__", PipelineTmaStore)
        object.__setattr__(obj, "_read", read)
        return obj

    @dsl_user_op
    def producer_acquire(self, *, loc=None, ip=None) -> None:
        cute.arch.cp_async_bulk_wait_group(self.num_stages - 1, read=self._read, loc=loc, ip=ip)

    @dsl_user_op
    def producer_tail(self, *, loc=None, ip=None) -> None:
        cute.arch.cp_async_bulk_wait_group(0, read=self._read, loc=loc, ip=ip)


# ── PipelineTmaUmma ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineTmaUmma(_PipelineIndexPhaseMixin, PipelineTmaUmmaOg):
    """Override producer_acquire to take in extra_tx_count parameter."""

    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        is_tma_warp: Optional[Boolean] = True,
        extra_tx_count: int = 0,
        *,
        loc=None,
        ip=None,
    ):
        """
        TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        # This is the difference between this and PipelineTmaAsync: we could have multiple
        # warps calling this, but only 1 warp should do the arrive on the full barrier
        if const_expr(extra_tx_count == 0):
            if_generate(
                and_(self.is_leader_cta, is_tma_warp),
                lambda: self.sync_object_full.arrive(
                    state.index, self.producer_mask, loc=loc, ip=ip
                ),
                loc=loc,
                ip=ip,
            )
        else:
            tx_count = self.sync_object_full.tx_count + extra_tx_count
            if_generate(
                and_(self.is_leader_cta, is_tma_warp),
                lambda: self.sync_object_full.arrive_and_expect_tx(
                    state.index, tx_count, loc=loc, ip=ip
                ),
                loc=loc,
                ip=ip,
            )


PipelineTmaUmma.create = _override_create(PipelineTmaUmmaOg, PipelineTmaUmma)


# ── PipelineUmmaAsync ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineUmmaAsync(_PipelineIndexPhaseMixin, PipelineUmmaAsyncOg):
    """
    PipelineUmmaAsync with optional elect_one for producer_commit and
    consumer_release, mirroring PipelineAsync.
    """

    _elect_one_commit: bool = False
    _syncwarp_before_commit: bool = True
    _elect_one_release: bool = False
    _syncwarp_before_release: bool = True

    @staticmethod
    def create(
        *args,
        elect_one_commit: bool = False,
        syncwarp_before_commit: bool = True,
        elect_one_release: bool = False,
        syncwarp_before_release: bool = True,
        **kwargs,
    ):
        obj = PipelineUmmaAsyncOg.create(*args, **kwargs)
        object.__setattr__(obj, "__class__", PipelineUmmaAsync)
        object.__setattr__(obj, "_elect_one_commit", elect_one_commit)
        object.__setattr__(obj, "_syncwarp_before_commit", syncwarp_before_commit)
        object.__setattr__(obj, "_elect_one_release", elect_one_release)
        object.__setattr__(obj, "_syncwarp_before_release", syncwarp_before_release)
        return obj

    @dsl_user_op
    def producer_commit(self, state: PipelineState, *, loc=None, ip=None):
        _call_with_elect_one(
            PipelineUmmaAsyncOg.producer_commit,
            self,
            state,
            self._elect_one_commit,
            self._syncwarp_before_commit,
            loc,
            ip,
        )

    @dsl_user_op
    def consumer_release(self, state: PipelineState, *, loc=None, ip=None):
        _call_with_elect_one(
            PipelineUmmaAsyncOg.consumer_release,
            self,
            state,
            self._elect_one_release,
            self._syncwarp_before_release,
            loc,
            ip,
        )


# ── PipelineAsyncUmma ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineAsyncUmma(_PipelineIndexPhaseMixin, PipelineAsyncUmmaOg):
    """
    PipelineAsyncUmma with optional elect_one for producer_commit and
    consumer_release, mirroring PipelineAsync.
    """

    _elect_one_commit: bool = False
    _syncwarp_before_commit: bool = True
    _elect_one_release: bool = False
    _syncwarp_before_release: bool = True

    @staticmethod
    def create(
        *args,
        elect_one_commit: bool = False,
        syncwarp_before_commit: bool = True,
        elect_one_release: bool = False,
        syncwarp_before_release: bool = True,
        **kwargs,
    ):
        obj = PipelineAsyncUmmaOg.create(*args, **kwargs)
        object.__setattr__(obj, "__class__", PipelineAsyncUmma)
        object.__setattr__(obj, "_elect_one_commit", elect_one_commit)
        object.__setattr__(obj, "_syncwarp_before_commit", syncwarp_before_commit)
        object.__setattr__(obj, "_elect_one_release", elect_one_release)
        object.__setattr__(obj, "_syncwarp_before_release", syncwarp_before_release)
        return obj

    @dsl_user_op
    def producer_commit(self, state: PipelineState, *, loc=None, ip=None):
        _call_with_elect_one(
            PipelineAsyncUmmaOg.producer_commit,
            self,
            state,
            self._elect_one_commit,
            self._syncwarp_before_commit,
            loc,
            ip,
        )

    @dsl_user_op
    def consumer_release(self, state: PipelineState, *, loc=None, ip=None):
        _call_with_elect_one(
            PipelineAsyncUmmaOg.consumer_release,
            self,
            state,
            self._elect_one_release,
            self._syncwarp_before_release,
            loc,
            ip,
        )


# ── PipelineTmaCpAsync ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineTmaCpAsync(_PipelineIndexPhaseMixin, PipelineTmaAsyncOg):
    """
    PipelineTmaCpAsync is used for CpAsync + TMA producers and AsyncThread consumers.
    Compared to PipelineTmaAsync, producer_acquire gates the full-barrier arrive on is_tma_warp.
    """

    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        is_tma_warp: Optional[Boolean] = True,
        *,
        loc=None,
        ip=None,
    ):
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        # This is the difference between this and PipelineTmaAsync: we could have multiple
        # warps calling this, but only 1 warp should do the arrive on the full barrier
        if_generate(
            is_tma_warp,
            lambda: self.sync_object_full.arrive(state.index, self.producer_mask, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def producer_cpasync_commit(self, state: PipelineState, *, loc=None, ip=None):
        """We need the mbarrier to track the completion of cp.async."""
        cute.arch.cp_async_mbarrier_arrive_noinc(
            self.producer_get_barrier(state, loc=loc, ip=ip), loc=loc, ip=ip
        )


PipelineTmaCpAsync.create = _override_create(PipelineTmaAsyncOg, PipelineTmaCpAsync)


# ── PipelineTmaCpAsyncUmma ──────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineTmaCpAsyncUmma(PipelineTmaUmmaOg):
    """
    PipelineTmaCpAsync is used for CpAsync + TMA producers and UMMA consumers
    (e.g. Blackwell mainloops)
    """

    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        is_tma_warp: Optional[Boolean] = True,
        *,
        loc=None,
        ip=None,
    ):
        """
        TMA producer commit conditionally waits on buffer empty and sets the
        transaction barrier for leader threadblocks.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        # This is the difference between this and PipelineTmaAsync: we could have multiple
        # warps calling this, but only 1 warp should do the arrive on the full barrier
        if_generate(
            and_(self.is_leader_cta, is_tma_warp),
            lambda: self.sync_object_full.arrive(state.index, self.producer_mask, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def producer_cpasync_commit(self, state: PipelineState, *, loc=None, ip=None):
        """
        We need the mbarrier to track the completion of cp.async
        """
        cute.arch.cp_async_mbarrier_arrive_noinc(
            self.producer_get_barrier(state, loc=loc, ip=ip), loc=loc, ip=ip
        )


PipelineTmaCpAsyncUmma.create = _override_create(PipelineTmaUmmaOg, PipelineTmaCpAsyncUmma)
