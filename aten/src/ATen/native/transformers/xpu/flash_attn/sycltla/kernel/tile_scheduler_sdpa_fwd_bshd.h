#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.h"

namespace cutlass::flash_attention {

namespace kernel {

struct XeFlashIndividualTileScheduler {

  struct Params {
    dim3 grid;
    // FastDivmod divmod_num_heads;
  };

  bool valid_ = true;
  Params params;

  CUTLASS_DEVICE
  XeFlashIndividualTileScheduler(Params const &params) : params(params) {}

  template <class ProblemSize, class TileShape>
  static Params to_underlying_arguments(ProblemSize const &problem_size,
                                        KernelHardwareInfo hw_info,
                                        TileShape const &tile_shape) {
    using namespace cute;
    dim3 grid(size(ceil_div(shape<3>(problem_size),
                            shape<0>(tile_shape))), // seq_len_qo / 128
              size(shape<1>(problem_size)),         // num_heads_q
              size(shape<0>(problem_size)));        // batch
    return Params{grid};
  }

  template <int Num_SGs> static dim3 get_grid_shape(Params const &params) {
    return params.grid;
  }

  CUTLASS_DEVICE
  bool is_valid() { return valid_; }

  CUTLASS_DEVICE
  auto get_block_coord() {
    using namespace cute;
    return make_coord(BlockIdxX(), BlockIdxY(), BlockIdxZ());
  }

  CUTLASS_DEVICE
  XeFlashIndividualTileScheduler &operator++() {
    valid_ = false;
    return *this;
  }
};

struct XeFlashDecodeIndividualTileScheduler {

  struct Params {
    dim3 grid;
    FastDivmod divmod_num_heads;
  };

  bool valid_ = true;
  Params params;

  CUTLASS_DEVICE
  XeFlashDecodeIndividualTileScheduler(Params const &params) : params(params) {}

  template <class ProblemSize, class TileShape>
  static Params to_underlying_arguments(ProblemSize const &problem_size,
                                        KernelHardwareInfo hw_info,
                                        TileShape const &tile_shape) {
    using namespace cute;
    dim3 grid(
        size(ceil_div(shape<7>(problem_size), shape<1>(tile_shape))),
        size(ceil_div(shape<3>(problem_size),
                      8)), // we want to process only 8 tokens per workgroup
        size(shape<0>(problem_size) * shape<1>(problem_size)));
    return Params{grid, {shape<1>(problem_size)}};
  }

  template <int Num_SGs> static dim3 get_grid_shape(Params const &params) {
    return params.grid;
  }

  CUTLASS_DEVICE
  bool is_valid() { return valid_; }

  CUTLASS_DEVICE
  auto get_block_coord() {
    using namespace cute;
    int block_decode = BlockIdxZ();
    int bidh;
    params.divmod_num_heads(block_decode, bidh, block_decode);
    return make_coord(BlockIdxX(), BlockIdxY(), block_decode, bidh);
  }

  CUTLASS_DEVICE
  XeFlashDecodeIndividualTileScheduler &operator++() {
    valid_ = false;
    return *this;
  }
};

////////////////////////////////////////////////////////////////////////////////

struct XeFlashPersistentTileScheduler {

  struct Params {
    int num_blocks;
    FastDivmod divmod_seq_len_block;
    FastDivmod divmod_head_size_block;
    FastDivmod divmod_num_heads;

    KernelHardwareInfo hw_info;
  };

  int block_idx = 0;
  Params params;

  CUTLASS_DEVICE
  XeFlashPersistentTileScheduler(Params const &params)
      : block_idx(BlockIdxX()), params(params) {}

  template <class ProblemSize, class TileShape>
  static Params to_underlying_arguments(ProblemSize const &problem_size,
                                        KernelHardwareInfo hw_info,
                                        TileShape const &tile_shape) {
    using namespace cute;
    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST(
          "  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments "
          "KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);
    }

    CUTLASS_TRACE_HOST(
        "to_underlying_arguments(): Setting persistent grid SM count to "
        << sm_count);
    hw_info.sm_count = sm_count;

    int num_head_size_blocks =
        size(ceil_div(shape<6>(problem_size), shape<1>(tile_shape)));
    int num_seq_len_blocks =
        size(ceil_div(shape<3>(problem_size), shape<0>(tile_shape)));
    int num_blocks = num_seq_len_blocks * num_head_size_blocks *
                     size(shape<0>(problem_size) * shape<1>(problem_size));

    return Params{num_blocks,
                  {num_seq_len_blocks},
                  {num_head_size_blocks},
                  {shape<1>(problem_size)},
                  hw_info};
  }

  template <int Num_SGs> static dim3 get_grid_shape(Params const &params) {
    auto queue = compat::get_default_queue();
    auto dev = queue.get_device();
    const size_t maxSubgroups =
        dev.template get_info<sycl::info::device::max_num_sub_groups>();
    // TODO (Codeplay): revert this back to std::min(params.num_blocks,
    // params.hw_info.sm_count) once performance issue is fixed.
    dim3 grid(
        std::min(params.num_blocks,
                 ceil_div(params.hw_info.sm_count * maxSubgroups, Num_SGs)),
        1, 1);
    return grid;
  }

  CUTLASS_DEVICE
  bool is_valid() { return block_idx < params.num_blocks; }

  CUTLASS_DEVICE
  auto get_block_coord() {
    using namespace cute;
    int block_decode = block_idx;
    int seq_len_block, head_size_block, bidh;
    params.divmod_head_size_block(block_decode, head_size_block, block_decode);
    params.divmod_seq_len_block(block_decode, seq_len_block, block_decode);
    params.divmod_num_heads(block_decode, bidh, block_decode);
    return make_coord(head_size_block, seq_len_block, block_decode, bidh);
  }

  CUTLASS_DEVICE
  XeFlashPersistentTileScheduler &operator++() {
    block_idx += GridDimX();
    return *this;
  }
};

////////////////////////////////////////////////////////////////////////////////
} // namespace kernel

struct IndividualScheduler {};
struct PersistentScheduler {};
struct FlashDecodeIndividualScheduler {};

namespace detail {

template <class TileSchedulerTag, class ArchTag, class Enable = void>
struct TileSchedulerSelector {
  static_assert(cutlass::detail::dependent_false<ArchTag>,
                "Could not select a tile scheduler for given parameters.");
};

// Default (void) maps to XeFlashIndividualTileScheduler
template <class ArchTag>
struct TileSchedulerSelector<
    void, ArchTag,
    cute::enable_if_t<cute::is_same_v<ArchTag, cutlass::arch::IntelXe>>> {
  using Scheduler =
      typename TileSchedulerSelector<IndividualScheduler, ArchTag>::Scheduler;
};

template <class ArchTag>
struct TileSchedulerSelector<
    IndividualScheduler, ArchTag,
    cute::enable_if_t<cute::is_same_v<ArchTag, cutlass::arch::IntelXe>>> {
  using Scheduler = kernel::XeFlashIndividualTileScheduler;
};

template <class ArchTag>
struct TileSchedulerSelector<
    PersistentScheduler, ArchTag,
    cute::enable_if_t<cute::is_same_v<ArchTag, cutlass::arch::IntelXe>>> {
  using Scheduler = kernel::XeFlashPersistentTileScheduler;
};

template <class ArchTag>
struct TileSchedulerSelector<
    FlashDecodeIndividualScheduler, ArchTag,
    cute::enable_if_t<cute::is_same_v<ArchTag, cutlass::arch::IntelXe>>> {
  using Scheduler = kernel::XeFlashDecodeIndividualTileScheduler;
};
} // namespace detail

////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::flash_attention
