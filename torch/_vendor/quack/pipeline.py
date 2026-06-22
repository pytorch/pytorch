# Copyright (c) 2025-2026, Tri Dao.

from typing import Optional
from dataclasses import dataclass

import cutlass.cute as cute
from cutlass import Boolean, Int32, const_expr
from cutlass.cutlass_dsl import if_generate, and_, dsl_user_op
from cutlass.pipeline import MbarrierArray, CooperativeGroup, PipelineOp
from cutlass.pipeline import PipelineState, PipelineUserType
from cutlass.pipeline import Agent, agent_sync
from cutlass.pipeline import NamedBarrier as NamedBarrierOg
from cutlass.pipeline import PipelineAsync as PipelineAsyncOg
from cutlass.pipeline import PipelineCpAsync as PipelineCpAsyncOg
from cutlass.pipeline import PipelineTmaAsync as PipelineTmaAsyncOg
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
        elect_one_release: bool = False,
        syncwarp_before_release: bool = True,
        **kwargs,
    ):
        obj = PipelineCpAsyncOg.create(*args, **kwargs)
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
    """Override producer_acquire to take in extra_tx_count parameter."""

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
        if const_expr(extra_tx_count == 0):
            self.sync_object_full.arrive(state.index, self.producer_mask, loc=loc, ip=ip)
        else:
            tx_count = self.sync_object_full.tx_count + extra_tx_count
            self.sync_object_full.arrive_and_expect_tx(state.index, tx_count, loc=loc, ip=ip)


PipelineTmaAsync.create = _override_create(PipelineTmaAsyncOg, PipelineTmaAsync)


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
    pass


PipelineUmmaAsync.create = _override_create(PipelineUmmaAsyncOg, PipelineUmmaAsync)


# ── PipelineAsyncUmma ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineAsyncUmma(_PipelineIndexPhaseMixin, PipelineAsyncUmmaOg):
    pass


PipelineAsyncUmma.create = _override_create(PipelineAsyncUmmaOg, PipelineAsyncUmma)


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


# ── MbarrierArrayWDropCount ─────────────────────────────────────────────────


class MbarrierArrayWDropCount(MbarrierArray):
    @dsl_user_op
    def __init__(
        self,
        barrier_storage: cute.Pointer,
        num_stages: int,
        agent: tuple[PipelineOp, CooperativeGroup],
        tx_count: int = 0,
        drop_count: Optional[Int32] = None,
        *,
        loc=None,
        ip=None,
    ) -> None:
        self.barrier_storage = barrier_storage
        self.tx_count = tx_count
        self.num_stages = num_stages
        self.op_type, self.cg = agent
        self.arrive_count = self.cg.size
        self.drop_count = drop_count

        if self.num_stages <= 0:
            raise ValueError("Error: Mbarrier stage count must be greater than 0.")
        if self.arrive_count <= 0:
            raise ValueError("Error: Mbarrier arrive count must be greater than 0.")
        if self.op_type is PipelineOp.TmaLoad and self.tx_count < 0:
            raise ValueError("Error: Mbarrier tx count must not be less than 0 for TMA ops.")

        if const_expr(drop_count is not None):
            self.arrive_count = self.arrive_count - drop_count

        # Store mbarrier base pointer
        self.mbarrier_base = self.barrier_storage

        # Mbarrier initialization in constructor
        self.mbarrier_init(loc=loc, ip=ip)

    def __extract_mlir_values__(self):
        return [self.barrier_storage, self.drop_count]

    def __new_from_mlir_values__(self, values):
        return MbarrierArrayWDropCount(
            values[0], self.num_stages, (self.op_type, self.cg), self.tx_count, values[1]
        )


# ── PipelineTmaCpAsyncUmma ──────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineTmaCpAsyncUmma(PipelineTmaUmmaOg):
    """
    PipelineTmaCpAsync is used for CpAsync + TMA producers and UMMA consumers
    (e.g. Blackwell mainloops)
    """

    @dsl_user_op
    @staticmethod
    def create(
        *,
        num_stages: int,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        tx_count: int,
        barrier_storage: cute.Pointer = None,
        cta_layout_vmnk: Optional[cute.Layout] = None,
        mcast_mode_mn: tuple[int, int] = (1, 1),
        defer_sync: bool = False,
        producer_drop_count: Optional[Int32] = None,
        loc=None,
        ip=None,
    ):
        """Creates and initializes a new PipelineTmaUmma instance.

        :param num_stages: Number of buffer stages for this pipeline
        :type num_stages: int
        :param producer_group: CooperativeGroup for the producer agent
        :type producer_group: CooperativeGroup
        :param consumer_group: CooperativeGroup for the consumer agent
        :type consumer_group: CooperativeGroup
        :param tx_count: Number of bytes expected to be written to the transaction barrier for one stage
        :type tx_count: int
        :param barrier_storage: Pointer to the shared memory address for this pipeline's mbarriers
        :type barrier_storage: cute.Pointer, optional
        :param cta_layout_vmnk: Layout of the cluster shape
        :type cta_layout_vmnk: cute.Layout, optional
        :param mcast_mode_mn: Tuple specifying multicast modes for m and n dimensions (each 0 or 1)
        :type mcast_mode_mn: tuple[int, int], optional
        :raises ValueError: If barrier_storage is not a cute.Pointer instance
        :return: A new PipelineTmaUmma instance configured with the provided parameters
        :rtype: PipelineTmaUmma
        """
        if not isinstance(barrier_storage, cute.Pointer):
            raise TypeError(
                f"Expected barrier_storage to be a cute.Pointer, but got {type(barrier_storage)}"
            )

        producer_type = PipelineOp.TmaLoad
        consumer_type = PipelineOp.TCGen05Mma

        producer = (producer_type, producer_group)
        consumer = (consumer_type, consumer_group)

        sync_object_full = MbarrierArrayWDropCount(
            barrier_storage.align(min_align=8),
            num_stages,
            producer,
            tx_count,
            drop_count=producer_drop_count,
            loc=loc,
            ip=ip,
        )
        sync_object_empty = PipelineTmaUmmaOg._make_sync_object(
            barrier_storage.align(min_align=8) + num_stages,
            num_stages,
            consumer,
            loc=loc,
            ip=ip,
        )

        if cta_layout_vmnk is None or cute.size(cta_layout_vmnk, loc=loc, ip=ip) == 1:
            # No mcast mask if not using clusters
            producer_mask = None
            # All threadblocks are leaders if not using clusters
            is_leader_cta = True
        else:
            producer_mask = PipelineTmaUmmaOg._compute_mcast_arrival_mask(
                cta_layout_vmnk, mcast_mode_mn, loc=loc, ip=ip
            )
            is_leader_cta = PipelineTmaUmmaOg._compute_is_leader_cta(
                cta_layout_vmnk, loc=loc, ip=ip
            )

        cta_group = (
            cute.nvgpu.tcgen05.CtaGroup.ONE
            if cta_layout_vmnk is None or cute.size(cta_layout_vmnk, mode=[0], loc=loc, ip=ip) == 1
            else cute.nvgpu.tcgen05.CtaGroup.TWO
        )

        consumer_mask = producer_mask

        if not defer_sync:
            cute.arch.mbarrier_init_fence()
            if cta_layout_vmnk is None or cute.size(cta_layout_vmnk, loc=loc, ip=ip) == 1:
                agent_sync(Agent.ThreadBlock)
            else:
                agent_sync(Agent.ThreadBlockCluster, is_relaxed=True)

        return PipelineTmaCpAsyncUmma(
            sync_object_full,
            sync_object_empty,
            num_stages,
            producer_mask,
            consumer_mask,
            is_leader_cta,
            cta_group,
        )

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
