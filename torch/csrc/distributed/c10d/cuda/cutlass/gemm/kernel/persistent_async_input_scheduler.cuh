/**
 * This file contains PersistentAsyncInputScheduler, a forked version of PersistentScheduler that
 * supports consuming asynchronous input. This tile scheduler introduces the following arguments:
 *
 * - tiles_per_chunk_m – Specifies the size of an M chunk. Chunks are the granularity at which the
 *   asynchronous input becomes ready. It must be an interger multiple of the size of an M tile.
 *
 * - chunk_signals – chunk_signals[i] == 1 indicates that chunk i is ready. Before returning a work
 *   tile, get_current_work() waits for the signal to ensure that the corresponding chunk is ready.
 *
 * - tile_idx_pivot_m – After applying swizzling, apply `pivot(m) => (m + tile_idx_pivot_m) %
 *   tiles_m` to `m`. In a distributed setting, this allows different ranks to process different m
 *   indices at the same time, thus avoiding communication hotspots.
 *
 * Note that this scheduler currently only supports the KernelTmaWarpSpecializedCooperative kernel
 * schedule. This is enforced via the template argument KernelSchedule.
 *
 * Usage:
 *
 * using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
 *    Shape<int, int, int, int>,
 *    CollectiveMainloop,
 *    CollectiveEpilogue,
 *    cutlass::gemm::PersistentAsyncInputScheduler<KernelSchedule>>;
 */

#pragma once
#include <cutlass/gemm/kernel/static_tile_scheduler.hpp>

namespace {

__device__ __forceinline__ void wait_signal(uint32_t* addr) {
  int ready = *addr;
  while (!ready) {
    asm volatile("ld.volatile.global.b32 %0, [%1];"
                 : "=r"(ready)
                 : "l"(addr)
                 : "memory");
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 700)
    asm volatile("nanosleep.u32 20;");
#endif
  };
}

}

////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm {

////////////////////////////////////////////////////////////////////////////////

template<
  class KernelSchedule,
  typename = cute::enable_if_t<
    cute::is_same_v<KernelSchedule, cutlass::gemm::KernelTmaWarpSpecializedCooperative>>>
struct PersistentAsyncInputScheduler {};

////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel::detail {

////////////////////////////////////////////////////////////////////////////////

class PersistentTileSchedulerSm90AsyncInputParams :
  public PersistentTileSchedulerSm90Params {
public:
  int tile_idx_pivot_m;
  int tiles_per_chunk_m = 0;
  uint32_t* chunk_signals = nullptr;
};

class PersistentTileSchedulerSm90AsyncInput {
private:
  uint64_t current_work_linear_idx_;
  uint64_t total_grid_size_;
  bool is_mainloop_producer_;

public:
  using WorkTileInfo = PersistentTileSchedulerSm90::WorkTileInfo;
  using Params = PersistentTileSchedulerSm90AsyncInputParams;
  using RasterOrder = typename Params::RasterOrder;
  using RasterOrderOptions = typename Params::RasterOrderOptions;

  struct Arguments {
    int max_swizzle_size = 1;
    RasterOrderOptions raster_order = RasterOrderOptions::Heuristic;

    // Async input specific
    int tile_idx_pivot_m = 0;
    int tiles_per_chunk_m = 0;
    uint32_t* chunk_signals = nullptr;
  };

  template <class ProblemShapeMNKL, class TileShape, class ClusterShape>
  static Params
  to_underlying_arguments(
    ProblemShapeMNKL problem_shape_mnkl,
    TileShape tile_shape,
    ClusterShape cluster_shape,
    [[maybe_unused]] KernelHardwareInfo const& hw_info,
    Arguments const& arguments,
    [[maybe_unused]] void* workspace=nullptr,
    [[maybe_unused]] const uint32_t epilogue_subtile = 1) {

    // We only need the tile and cluster shape during scheduler setup, so let FTAD do the magic
    static_assert(cute::is_static<TileShape>::value);
    static_assert(cute::is_static<ClusterShape>::value);

    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);

    Params params;
    params.initialize(
      problem_blocks,
      to_gemm_coord(cluster_shape),
      hw_info,
      arguments.max_swizzle_size,
      arguments.raster_order
    );
    params.tile_idx_pivot_m = arguments.tile_idx_pivot_m;
    params.tiles_per_chunk_m = arguments.tiles_per_chunk_m;
    params.chunk_signals = arguments.chunk_signals;

    return params;
  }

  CUTLASS_HOST_DEVICE
  static bool
  can_implement(Arguments const& args) {
    return args.raster_order == RasterOrderOptions::AlongN;
  }

  CUTLASS_HOST_DEVICE
  PersistentTileSchedulerSm90AsyncInput() { }

  CUTLASS_DEVICE explicit PersistentTileSchedulerSm90AsyncInput(Params const& params_) : params(params_) {
    // MSVC requires protecting use of CUDA-specific nonstandard syntax,
    // like blockIdx and gridDim, with __CUDA_ARCH__.
#if defined(__CUDA_ARCH__)
    if (params_.raster_order_ == RasterOrder::AlongN) {
      current_work_linear_idx_ = uint64_t(blockIdx.x) + uint64_t(blockIdx.y) * uint64_t(gridDim.x);
    }
    else {
      current_work_linear_idx_ = uint64_t(blockIdx.x) * uint64_t(gridDim.y) + uint64_t(blockIdx.y);
    }

    total_grid_size_ = uint64_t(gridDim.x) * uint64_t(gridDim.y) * uint64_t(gridDim.z);

    int warp_group_role = canonical_warp_group_idx();
    int producer_warp_group_role = canonical_warp_idx_sync() % NumWarpsPerWarpGroup;
    is_mainloop_producer_ = warp_group_role == 0 && producer_warp_group_role == 0;
#else
    CUTLASS_ASSERT(false && "This line should never be reached");
#endif
  }

  // Returns the initial work tile info that will be computed over
  template <class ClusterShape>
  CUTLASS_DEVICE
  WorkTileInfo
  initial_work_tile_info(ClusterShape cluster_shape) {
    return get_current_work();
  }

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work() const {
    return get_current_work_for_linear_idx(current_work_linear_idx_);
  }

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work_for_linear_idx(uint64_t linear_idx) const {
    if (linear_idx >= params.blocks_per_problem_) {
      return WorkTileInfo::invalid_work_tile();
    }

    // Map worker's linear index into the CTA tiled problem shape to the corresponding MNL indices
    uint64_t work_idx_l, remainder;
    params.divmod_batch_(work_idx_l, remainder, linear_idx);

    uint64_t blk_per_grid_dim = params.divmod_cluster_shape_minor_.divide(remainder);

    uint64_t cluster_id, cluster_major_offset = 0, cluster_minor_offset = 0;
    params.divmod_cluster_shape_major_(cluster_id, cluster_major_offset, blk_per_grid_dim);

    auto [cta_m_in_cluster, cta_n_in_cluster, _] = cute::block_id_in_cluster();
    if (params.raster_order_ == RasterOrder::AlongN) {
      cluster_minor_offset = cta_m_in_cluster;
    }
    else {
      cluster_minor_offset = cta_n_in_cluster;
    }

    uint64_t cluster_idx_minor, cluster_idx_major;

    uint64_t cluster_idx_minor_div_swizzle, extra, offset;

    offset = cluster_id & ((1 << params.log_swizzle_size_) - 1);
    extra = cluster_id >> params.log_swizzle_size_;

    params.divmod_cluster_blk_major_(cluster_idx_minor_div_swizzle, cluster_idx_major, extra);

    cluster_idx_minor = cluster_idx_minor_div_swizzle * (1 << params.log_swizzle_size_) + offset;

    auto minor_work_idx = static_cast<int32_t>(cluster_idx_minor * params.divmod_cluster_shape_minor_.divisor +
                                               cluster_minor_offset);
    auto major_work_idx = static_cast<int32_t>(cluster_idx_major * params.divmod_cluster_shape_major_.divisor +
                                               cluster_major_offset);

    int m, n;
    if (params.raster_order_ == RasterOrder::AlongN) {
      m = minor_work_idx;
      n = major_work_idx;
    } else {
      m = major_work_idx;
      n = minor_work_idx;
    }

    // Pivot after swizzling
    auto tiles_m = params.problem_tiles_m_ * params.cluster_shape_m_;
    m = (m + params.tile_idx_pivot_m) % tiles_m;

    if (is_mainloop_producer_) {
      if (threadIdx.x == 0) {
        size_t chunk_idx = m / params.tiles_per_chunk_m;
        wait_signal(params.chunk_signals + chunk_idx);
      }

      // An arbirary, non-default id
      constexpr int barrier_id = 8;
      arch::NamedBarrier barrier(NumThreadsPerWarp, barrier_id);
      barrier.arrive_and_wait();
    }

    return {m, n, static_cast<int32_t>(work_idx_l), true};
  }

  CUTLASS_DEVICE
  void
  advance_to_next_work(uint32_t advance_count = 1) {
    current_work_linear_idx_ += total_grid_size_ * uint64_t(advance_count);
  }

  // Given the inputs, computes the total number of output blocks over which this problem will compute.
  // Note that this is only the logical size of our grid, not the physical grid we will actually launch.
  template<class ProblemShapeMNKL, class BlockShape, class ClusterShape>
  CUTLASS_HOST_DEVICE static
  dim3
  get_tiled_cta_shape_mnl(ProblemShapeMNKL problem_shape_mnkl, BlockShape cta_shape, ClusterShape cluster_shape) {
    auto cta_m = cute::size(cute::ceil_div(cute::shape<0>(problem_shape_mnkl), cute::shape<0>(cta_shape)));
    auto cta_n = cute::size(cute::ceil_div(cute::shape<1>(problem_shape_mnkl), cute::shape<1>(cta_shape)));

    return Params::get_tiled_cta_shape_mnl(
      to_gemm_coord(problem_shape_mnkl),
      to_gemm_coord(cluster_shape),
      cta_m, cta_n
    );
  }
  // Kernel helper function to get next work ID
  template <class WorkIdPipeline, class WorkIdPipelineState>
  CUTLASS_DEVICE
  auto
  fetch_next_work(
    WorkTileInfo work_tile_info,
    WorkIdPipeline& work_id_pipeline,
    WorkIdPipelineState work_id_pipe_consumer_state) {
      WorkTileInfo new_work_tile_info;
      advance_to_next_work();
      new_work_tile_info = get_current_work();

    // Return true to indicate that the WorkID pipeline state should be advanced
    return cute::make_tuple(new_work_tile_info, true);
  }

  CUTLASS_DEVICE
  static auto
  work_tile_to_cta_coord(WorkTileInfo work_tile_info) {
    // Get every cta coord in three dimensions of the cluster
    auto [cta_m_in_cluster, cta_n_in_cluster, cta_l_in_cluster] = cute::block_id_in_cluster();
    return make_coord(
      work_tile_info.M_idx + static_cast<int32_t>(cta_m_in_cluster),
      work_tile_info.N_idx + static_cast<int32_t>(cta_n_in_cluster),
      _,
      work_tile_info.L_idx + static_cast<int32_t>(cta_l_in_cluster)
    );
  }

  // Given the inputs, computes the physical grid we should launch.
  template<class ProblemShapeMNKL, class BlockShape, class ClusterShape>
  CUTLASS_HOST_DEVICE static
  dim3
  get_grid_shape(
    ProblemShapeMNKL problem_shape_mnk,
    BlockShape cta_shape,
    ClusterShape cluster_shape,
    KernelHardwareInfo hw_info,
    Arguments arguments,
    bool truncate_by_problem_size=true) {

    auto problem_shape_mnkl = cute::append<4>(problem_shape_mnk, cute::Int<1>{});
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, cta_shape, cluster_shape);

    return Params::get_grid_shape(
      problem_blocks,
      to_gemm_coord(cluster_shape),
      hw_info,
      arguments.max_swizzle_size,
      arguments.raster_order,
      /* truncate_by_problem_size = */true
    );
  }

  // Given the inputs, computes the physical grid we should launch.
  template<class ProblemShapeMNKL, class BlockShape, class ClusterShape>
  CUTLASS_HOST_DEVICE static
  dim3
  get_grid_shape(
    Params const& params,
    ProblemShapeMNKL problem_shape_mnk,
    BlockShape cta_shape,
    ClusterShape cluster_shape,
    KernelHardwareInfo hw_info) {

    auto problem_shape_mnkl = cute::append<4>(problem_shape_mnk, cute::Int<1>{});
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, cta_shape, cluster_shape);

    Arguments args{};
    if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>) {
      args.max_swizzle_size = 1 << params.log_swizzle_size_;
    }
    args.raster_order = params.raster_order_ == RasterOrder::AlongN ? RasterOrderOptions::AlongN : RasterOrderOptions::AlongM;

    return Params::get_grid_shape(
      problem_blocks,
      to_gemm_coord(cluster_shape),
      hw_info,
      args.max_swizzle_size,
      args.raster_order,
      /* truncate_by_problem_size = */true
    );
  }

  // Convert CTA-level work tile info to cluster-level tile coord
  CUTLASS_DEVICE
  cute::Coord<int,int,int,int>
  tile_info_to_coord_mnkl(WorkTileInfo work_tile_info) const {
    // TileScheduler works at CTA-level, kernel works at cluster-level
    int m_coord = idx2crd(work_tile_info.M_idx / params.cluster_shape_m_,
                          params.problem_tiles_m_);
    int n_coord = idx2crd(work_tile_info.N_idx / params.cluster_shape_n_,
                          params.problem_tiles_n_);
    int l_coord = idx2crd(work_tile_info.L_idx,
                          params.problem_tiles_l_);
    return make_coord(m_coord, n_coord, _, l_coord);
  }

  // Returns whether the block assigned this work should compute the epilogue for the corresponding
  // output tile. For the basic tile scheduler, this is always true.
  CUTLASS_HOST_DEVICE
  static bool
  compute_epilogue(WorkTileInfo const&, Params const&) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  static bool
  compute_epilogue(WorkTileInfo const&) {
    return true;
  }

  // Performs the reduction across splits for a given output tile. Since this scheduler does
  // not split output tiles, no reduction is needed.
  template <class FrgTensorC>
  CUTLASS_DEVICE
  static void
  fixup(Params const&, WorkTileInfo const&, FrgTensorC&, uint32_t, uint32_t) {}

  // Performs the reduction across splits for a given output tile. No fixup is required for
  // work units returned by this scheduler.
  template <class FrgTensorC>
  CUTLASS_DEVICE
  void
  fixup(WorkTileInfo const&, FrgTensorC&, uint32_t, uint32_t) const { }

  // Returns whether the current WorkTileInfo passed in should continue to be used. Since
  // this scheduler only schedules work in units of single, full output tiles, the WorkTileInfo
  // passed in should not be used after having been processed.
  CUTLASS_DEVICE
  static bool
  continue_current_work(WorkTileInfo&) {
    return false;
  }

  template <class ProblemShape, class TileShape>
  CUTLASS_HOST_DEVICE
  static int
  get_work_k_tile_count(WorkTileInfo const& work_tile_info, ProblemShape problem_shape, TileShape tile_shape) {
    // All work units returned by this scheduler cover the entire K iteration
    // space of the output tile assigned to the work unit.
    return cute::size(cute::ceil_div(cute::get<2>(problem_shape), cute::get<2>(tile_shape)));
  }

  CUTLASS_HOST_DEVICE
  static uint32_t
  get_work_k_tile_start(WorkTileInfo const&) {
    // All work units returned by this scheduler start from K tile 0
    return 0u;
  }

  CUTLASS_DEVICE
  static bool
  need_separate_reduction(Params const& params) {
    return false;
  }

  CUTLASS_DEVICE
  bool
  is_work_tile_for_reduction(WorkTileInfo const& work_tile_info, Params const& params) {
    return false;
  }

  template <class FrgTensorC>
  CUTLASS_DEVICE
  void
  separate_reduction(
    Params const& params,
    WorkTileInfo const& work_tile_info,
    FrgTensorC& accumulators,
    uint32_t num_barriers,
    uint32_t barrier_idx) {
  }

  // Shares the accumulator set with peers in the global workspace
  template <class FrgTensorC>
  CUTLASS_DEVICE
  static void
  share(
    Params const& params,
    WorkTileInfo const& work_tile_info,
    FrgTensorC& accumulators,
    uint32_t num_barriers,
    uint32_t barrier_idx) {
  }

  CUTLASS_DEVICE
  static bool
  valid_warpgroup_in_work_tile(WorkTileInfo const& work_tile_info) {
    return true;
  }

  CUTLASS_DEVICE
  static bool
  requires_separate_reduction(Params const& params) {
    return false;
  }

  // The basic tile scheduler does not require any additional workspace
  template <class ProblemShape, class ElementAccumulator>
  static int
  get_workspace_size(Arguments const&, ProblemShape, KernelHardwareInfo const&, uint32_t, const uint32_t = 1) {
    return 0;
  }

  template <class ProblemShape, class ElementAccumulator>
  static cutlass::Status
  initialize_workspace(Arguments const&, void*, cudaStream_t, ProblemShape, KernelHardwareInfo const&,
    uint32_t, const uint32_t = 1) {
    return Status::kSuccess;
  }
public:
  // Sink scheduler params as a member
  Params params;
};

// Selector
template <
  class KernelSchedule,
  class TileShape,
  class ClusterShape
>
struct TileSchedulerSelector<
  PersistentAsyncInputScheduler<KernelSchedule>,
  arch::Sm90,
  TileShape,
  ClusterShape
  > {
  using Scheduler = PersistentTileSchedulerSm90AsyncInput;
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel::detail

///////////////////////////////////////////////////////////////////////////////
