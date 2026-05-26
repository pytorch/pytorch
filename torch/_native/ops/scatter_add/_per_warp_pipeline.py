"""Per-warp TMA pipeline abstraction for CuTeDSL.

Mirrors ``cutlass.pipeline.PipelineTmaAsync``'s API one-for-one but
supports N independent pipes per CTA -- one per warp. The stock
``PipelineTmaAsync.create`` gates ``mbarrier_init`` on
``warp_idx == 0``, which means a CTA with multiple pipes only
initializes warp 0's barriers and other warps deadlock when they try
to drive their own pipe.

Differences from the stock pipeline:
  - Each warp owns its own slice of barrier storage and smem buffer.
  - Lane 0 of *every* warp initializes its own mbarriers (vs warp 0
    only).
  - ``producer_get_barrier`` returns the full-side barrier so the
    caller's ``cute.copy(..., tma_bar_ptr=...)`` performs the TMA
    arrive (matches the stock pipeline's contract).

API parity with ``cutlass.pipeline.PipelineTmaAsync`` (so kernels read
the same):
  - ``producer_acquire(state)``     wait on empty barrier
  - ``producer_commit(state)``      no-op (TMA arrives via cute.copy)
  - ``producer_get_barrier(state)`` full-barrier ptr for cute.copy
  - ``consumer_wait(state)``        wait on full barrier
  - ``consumer_release(state)``     arrive on empty barrier

Producer/consumer states use the stock ``PipelineState``: producer
starts with phase=1 so the first ``producer_acquire`` sees the empty
barrier in its newly-initialized phase=0 state and returns
immediately.
"""

from dataclasses import dataclass

import cutlass.cute as cute
from cutlass import Boolean, Int32
from cutlass.cutlass_dsl import if_generate
from cutlass.pipeline import make_pipeline_state, PipelineState, PipelineUserType


@dataclass(frozen=True)
class PerWarpTmaPipeline:
    """Per-warp TMA pipeline with `num_stages` smem buffer slots and
    matching pairs of mbarriers (full + empty) per stage. Producer = TMA
    load (implicit arrive via ``cute.copy(..., tma_bar_ptr=...)``);
    consumer = TMA bulk-reduce (``cp.reduce.async.bulk``).

    Mirrors the stock ``PipelineTmaAsync``'s ``@dataclass(frozen=True)``
    layout so cute.jit sees it as a dynamic-expression-compatible
    aggregate (otherwise its ``create`` factory can't return one from
    a jit'd context)."""

    num_stages: int
    full_ptr: cute.Pointer  # num_stages Uint64 (full-side)
    empty_ptr: cute.Pointer  # num_stages Uint64 (empty-side)
    tx_count: int

    @staticmethod
    def barrier_storage_size(num_stages: int) -> int:
        """Number of ``Uint64`` slots a single warp's pipe needs.
        Caller allocates ``warps_per_cta * barrier_storage_size(num_stages)``
        Uint64s and slices."""
        return 2 * num_stages

    @staticmethod
    def create(
        *,
        num_stages: int,
        barrier_storage,  # cute.Pointer; per-warp slice of size 2 * num_stages
        tx_count: int,
        lane_id,
    ):
        """Construct + init. Caller must:
          - allocate ``barrier_storage_size(num_stages)`` Uint64 slots
            per warp in smem;
          - pass the per-warp base pointer in ``barrier_storage``;
          - call this from every thread of the warp -- only lane 0
            actually issues the inits.

        Caller is responsible for the post-init ``mbarrier_init_fence``
        and block-wide ``sync_threads`` (we may have multiple pipes in
        the same CTA; we don't want to fence per-pipe).

        Mbarrier layout per warp:
          [0 .. num_stages)         -- full-side (TMA arrive_count = 1)
          [num_stages .. 2*num_stages) -- empty-side (consumer arrive_count = 1)
        """
        full_ptr = barrier_storage
        empty_ptr = barrier_storage + Int32(num_stages)

        def _init():
            # ``num_stages`` is a Python int -- a plain ``range`` works
            # outside the @cute.kernel preprocessor (which is what
            # rewrites ``cutlass.range_constexpr`` into a static
            # unroll). The for body still emits Int32 ops for the IR.
            for s in range(num_stages):
                cute.arch.mbarrier_init(full_ptr + Int32(s), Int32(1))
                cute.arch.mbarrier_init(empty_ptr + Int32(s), Int32(1))

        # ``if_generate`` so this works whether ``create`` is called
        # from raw Python or inside a @cute.kernel body.
        if_generate(lane_id == Int32(0), _init)
        return PerWarpTmaPipeline(num_stages, full_ptr, empty_ptr, tx_count)

    @staticmethod
    def make_producer_state(num_stages: int) -> PipelineState:
        return make_pipeline_state(PipelineUserType.Producer, num_stages)

    @staticmethod
    def make_consumer_state(num_stages: int) -> PipelineState:
        return make_pipeline_state(PipelineUserType.Consumer, num_stages)

    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: "Boolean | None" = None,
    ):
        """Wait on the empty-side barrier for this stage, then arrive
        on the full-side barrier with the expected transaction byte
        count. The subsequent ``cute.copy(..., tma_bar_ptr=...)`` is
        what actually performs the full-side arrive (TMA hardware
        updates the transaction counter).

        ``try_acquire_token``: if a non-None Boolean is passed and is
        non-zero, the empty-side wait is skipped (matches stock
        ``PipelineTmaAsync.producer_acquire`` semantics)."""
        if try_acquire_token is None:
            cute.arch.mbarrier_wait(self.empty_ptr + state.index, state.phase)
        else:
            if_generate(
                try_acquire_token == Int32(0),
                lambda: cute.arch.mbarrier_wait(
                    self.empty_ptr + state.index, state.phase
                ),
            )
        cute.arch.mbarrier_arrive_and_expect_tx(
            self.full_ptr + state.index, Int32(self.tx_count)
        )

    def producer_commit(self, state: PipelineState):
        """No-op: the TMA bulk-load instruction itself arrives on the
        full-side barrier (transaction-count completion). Kept for
        API parity with stock ``PipelineTmaAsync.producer_commit``."""
        pass  # noqa: PIE790

    def producer_get_barrier(self, state: PipelineState):
        """Pointer to the current stage's full-side barrier. Pass to
        ``cute.copy(..., tma_bar_ptr=...)`` so the TMA load arrives
        on it."""
        return self.full_ptr + state.index

    def consumer_wait(
        self,
        state: PipelineState,
        try_wait_token: "Boolean | None" = None,
    ):
        """Wait for the producer's TMA load to complete on the full
        barrier. ``try_wait_token`` matches stock semantics: when
        non-zero, skip the wait."""
        if try_wait_token is None:
            cute.arch.mbarrier_wait(self.full_ptr + state.index, state.phase)
        else:
            if_generate(
                try_wait_token == Int32(0),
                lambda: cute.arch.mbarrier_wait(
                    self.full_ptr + state.index, state.phase
                ),
            )

    def consumer_release(self, state: PipelineState):
        """Signal the empty barrier so the producer can reuse this
        slot. Caller is expected to have issued the bulk-reduce + a
        ``cp_async_bulk_commit_group`` already; the bulk-reduce reads
        smem at issue time, so the empty arrive is safe immediately
        after commit_group."""
        cute.arch.mbarrier_arrive(self.empty_ptr + state.index)

    def drain_bulk_reduces(self):
        """Block until all in-flight ``cp.reduce.async.bulk`` operations
        issued by the calling thread have completed.

        Differs from stock ``PipelineAsync.producer_tail(state)``: stock
        advances state and waits the *empty* barrier (consumer must
        finish before kernel exit). Here the consumer is the TMA
        bulk-reduce engine, not async threads, so the relevant
        completion signal is the bulk-group counter, not an mbarrier.
        Use this at end-of-kernel to keep async gmem writes from
        outliving the kernel.
        """
        cute.arch.cp_async_bulk_wait_group(0, read=False)
