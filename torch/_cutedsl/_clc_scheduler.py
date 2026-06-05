# Helpers for wiring Cluster Launch Control (CLC) based dynamic-persistent
# tile scheduling into our grouped MM kernel. Modeled on the pattern from
# Tri Dao's flash-attention PR #2218 ("CLC work stealing").
#
# The core idea: instead of each warp running its own StaticPersistentTileScheduler
# (which advances a per-warp counter), all consumer warps share a single CLC
# pipeline. A dedicated CLC producer warp (running on the cluster's leader CTA
# only) issues clc_query instructions; the hardware dispatches tiles to CTAs
# and writes responses into smem; all consumer warps wait on the response and
# read the same tile coords in lockstep.
#
# This file provides:
#   - SchedulingMode enum (STATIC vs CLC) for runtime selection
#   - ClcState dataclass wrapping the CuTeDSL hardware scheduler + async
#     pipeline + producer/consumer states; provides the small surface that
#     consumer warps and the dedicated producer warp need.
#
# Phase 1 of the grouped-MM CLC migration uses this scaffolding without any
# additional payload (group-info broadcast). Phase 2 will add a smem
# side-table written by one warp so consumer warps can skip the per-tile
# delinearize_z prefix-sum scan.

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as cutlass_pipeline
from cutlass.pipeline import PipelineClcFetchAsync, PipelineState
from cutlass.utils import (
    ClcDynamicPersistentTileScheduler,
    ClcDynamicPersistentTileSchedulerParams,
)


class SchedulingMode:
    """Pseudo-enum (CuTeDSL @cute.jit doesn't always play well with stdlib Enum)."""

    STATIC = 0
    CLC = 1


@dataclass
class ClcState:
    """Owns the runtime state shared by all CLC consumers + producer.

    Built once in the kernel after pipeline init; passed to every warp branch
    so that each warp's per-tile loop consumes from the same hardware CLC
    pipeline. The CLC producer warp (single warp on the leader CTA) drives
    prefetch_next_work; everyone else (including same warp position on the
    non-leader CTA in the cluster) calls consumer_wait / get_current_work /
    consumer_release.

    All four members are mutable state from cute.jit's perspective; they need
    extract / new round-tripping so the @cute.jit boundary preserves them.
    """

    _hw_scheduler: ClcDynamicPersistentTileScheduler
    _pipeline: PipelineClcFetchAsync
    _consumer_state: PipelineState
    _producer_state: PipelineState

    @staticmethod
    def create(
        *,
        hw_scheduler: ClcDynamicPersistentTileScheduler,
        pipeline: PipelineClcFetchAsync,
        consumer_state: PipelineState,
        producer_state: PipelineState,
    ) -> "ClcState":
        return ClcState(hw_scheduler, pipeline, consumer_state, producer_state)

    # Initial work tile for this CTA is just its block_idx. CLC hasn't
    # been queried yet.
    def initial_work_tile_info(self, total_num_clusters=None):
        work_tile = self._hw_scheduler.initial_work_tile_info()
        if total_num_clusters is None:
            return work_tile
        return self._mask_to_actual_problem(work_tile, total_num_clusters)

    # Read the CLC response for the current pipeline stage. Caller is
    # expected to have done consumer_wait() first.
    def get_current_work(self, total_num_clusters=None):
        work_tile = self._hw_scheduler.get_current_work()
        if total_num_clusters is None:
            return work_tile
        return self._mask_to_actual_problem(work_tile, total_num_clusters)

    def _mask_to_actual_problem(self, work_tile, total_num_clusters):
        is_valid = (
            work_tile.is_valid_tile & (work_tile.tile_idx[2] < total_num_clusters)
        )
        return cutlass.utils.WorkTileInfo(work_tile.tile_idx, is_valid)

    # Producer-side: only called by the dedicated CLC scheduler warp on the
    # leader CTA. Waits for an empty pipeline slot, then issues a CLC query
    # that hardware will fulfill into that slot.
    def prefetch_next_work(self, *, loc=None, ip=None):
        self._pipeline.producer_acquire(self._producer_state, loc=loc, ip=ip)
        mbarrier_addr = self._pipeline.producer_get_barrier(
            self._producer_state, loc=loc, ip=ip
        )
        self._hw_scheduler.advance_to_next_work(mbarrier_addr, loc=loc, ip=ip)
        self._producer_state.advance(loc=loc, ip=ip)

    # Consumer-side: wait for the next tile to be filled.
    def consumer_wait(self, *, loc=None, ip=None):
        self._pipeline.consumer_wait(self._consumer_state, loc=loc, ip=ip)

    # Consumer-side: release the stage (one arrival per consumer warp; barrier
    # advances when all consumer warps have arrived).
    def consumer_release(self, *, loc=None, ip=None):
        self._pipeline.consumer_release(self._consumer_state, loc=loc, ip=ip)
        self._consumer_state.advance(loc=loc, ip=ip)

    # Producer-side cleanup. Drains the pipeline so non-leader CTAs in the
    # cluster can finish.
    def producer_tail(self, *, loc=None, ip=None):
        self._pipeline.producer_tail(self._producer_state, loc=loc, ip=ip)

    # MLIR value round-tripping. Pattern is the standard CuTeDSL one used
    # everywhere a dataclass crosses a cute.jit boundary.
    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in (
            self._hw_scheduler,
            self._pipeline,
            self._consumer_state,
            self._producer_state,
        ):
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        rebuilt = []
        for obj, n_items in zip(
            (
                self._hw_scheduler,
                self._pipeline,
                self._consumer_state,
                self._producer_state,
            ),
            self._values_pos,
        ):
            rebuilt.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return ClcState(*rebuilt)


def make_clc_problem_shape(
    cluster_shape_mn: tuple[int, int],
    total_num_clusters,
    swizzle_size: int = 1,
    raster_along_m: bool = True,
) -> ClcDynamicPersistentTileSchedulerParams:
    """Build ClcDynamicPersistentTileSchedulerParams that match what our
    static persistent scheduler uses (PersistentTileSchedulerParams with
    problem_shape_ntile_mnl=(cluster_m, cluster_n, total_num_clusters)).

    CLC dispatches tiles into the (m, n, l) coord space; consumer warps then
    feed l_idx into delinearize_z (same as today) to recover the per-group
    coordinates.

    `swizzle_size`/`raster_along_m` (v4.5.2 CLC scheduler) control the
    dispatch order across the (M, N) cluster grid; defaults give the
    pre-v4.5.2 pass-through behavior. Useful as an L2-locality tuning knob
    when problem_shape has multi-axis cluster work.
    """
    problem_shape_ntile_mnl = (
        cluster_shape_mn[0],
        cluster_shape_mn[1],
        cutlass.Int32(total_num_clusters),
    )
    cluster_shape_mnk = (*cluster_shape_mn, 1)
    return ClcDynamicPersistentTileSchedulerParams(
        problem_shape_ntile_mnl=problem_shape_ntile_mnl,
        cluster_shape_mnk=cluster_shape_mnk,
        swizzle_size=swizzle_size,
        raster_along_m=raster_along_m,
    )


def create_clc_pipeline(
    *,
    barrier_storage: cute.Pointer,
    num_stages: int,
    num_consumer_warps: int,
    cluster_size: int,
    cta_layout_vmnk: cute.Layout,
):
    """Construct the PipelineClcFetchAsync.

    Args:
      barrier_storage: smem pointer to the pipeline mbarrier array (2 *
        num_stages Int64s)
      num_stages: number of CLC response slots (>= 1)
      num_consumer_warps: total consumer warps PER CTA (every launched warp
        that is not the CLC producer counts here)
      cluster_size: number of CTAs in the cluster
      cta_layout_vmnk: cluster layout (used by PipelineClcFetchAsync to know
        signaling mask)

    Producer count is 1 (CLC pipeline is producer-driven by a single thread
    via elect_one in advance_to_next_work). Consumer count is
    `WARP_SIZE * num_consumer_warps * cluster_size` because every consumer
    warp in every CTA arrives on the CTA-0 empty barrier per stage.
    """
    producer_group = cutlass_pipeline.CooperativeGroup(cutlass_pipeline.Agent.Thread)
    consumer_group = cutlass_pipeline.CooperativeGroup(
        cutlass_pipeline.Agent.Thread,
        cute.arch.WARP_SIZE * num_consumer_warps * cluster_size,
    )
    return cutlass_pipeline.PipelineClcFetchAsync.create(
        barrier_storage=barrier_storage,
        num_stages=num_stages,
        producer_group=producer_group,
        consumer_group=consumer_group,
        tx_count=16,  # CLC response is 4 Int32s = 16 bytes per stage
        cta_layout_vmnk=cta_layout_vmnk,
    )
