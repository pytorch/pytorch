#include <scheduler/debug_utils.h>
#include <scheduler/normalization_utils.h>
#include <utils.h>

#include <ATen/cuda/CUDAContext.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace normalization_scheduler_utils {

using scheduler_debug_utils::log;

PreferredLaunchConfig::PreferredLaunchConfig() : valid_(true) {
  initValidGdims();
  resetBdim();
  resetGdim();
}

bool PreferredLaunchConfig::isNextSmallerBdimx() const {
  return grid_dims_pos_ + 1 == (int)valid_grid_dims_.size();
}

bool PreferredLaunchConfig::canLowerBdimx() const {
  return bdimx() > kMinBdimx;
}

bool PreferredLaunchConfig::setBdimx(int bdimx, bool dry_run) {
  constexpr int block_size = 256;

  if (bdimx < kMinBdimx || bdimx > kMaxBdimx) {
    return false;
  }

  TORCH_INTERNAL_ASSERT(block_size % bdimx == 0, "Invalid bdimx: ", bdimx);
  int bdimy = block_size / bdimx;

  if (!dry_run) {
    bdimy_ = bdimy;
    bdimx_ = bdimx;
  }

  return true;
}

// Populate the list of valid gridDim configs for persistent grid
// normalization kernels in the order of increasing gridDim.y.
// Start
// with gridDim.y == 2. For example, on A100, the list would be: [(54,
// 2), (36, 3), (27, 4), (21, 5), (18, 6), (15, 7), (13, 8), (12, 9),
// (10, 10), (9, 12), (8, 13), (7, 15), (6, 18), (5, 21), (4, 27), (3,
// 36), (2, 54)].
void PreferredLaunchConfig::initValidGdims() {
  std::vector<std::pair<int, int>> grid_dims;
  const int num_sms =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int max_first_half =
      static_cast<int>(std::sqrt(static_cast<float>(num_sms)));
  for (int gdimy = 2; gdimy <= max_first_half; ++gdimy) {
    int gdimx = num_sms / gdimy;
    grid_dims.push_back(std::make_pair(gdimx, gdimy));
  }
  // Reverse the first half and swap gridDim.x and gridDim.y. That
  // list becomes the latter half
  auto latter_half = grid_dims;
  std::reverse(latter_half.begin(), latter_half.end());
  for (const auto& gdimx_gdimy : latter_half) {
    if (gdimx_gdimy.second == gdimx_gdimy.first) {
      // This is already in the first half
      continue;
    }
    grid_dims.push_back(std::make_pair(gdimx_gdimy.second, gdimx_gdimy.first));
  }
  valid_grid_dims_ = grid_dims;
}

bool PreferredLaunchConfig::moveToNextConfig() {
  if (moveToNextGdim()) {
    return true;
  }

  // Can't increase gdimy. Try bdimy next.
  if (moveToNextBdim()) {
    return true;
  }

  // No more valid config
  invalidate();
  return false;
}

bool PreferredLaunchConfig::moveToNextBdim() {
  const int new_bdimx = bdimx() / 2;
  if (setBdimx(new_bdimx)) {
    resetGdim();
    return true;
  } else {
    invalidate();
    return false;
  }
}

bool PreferredLaunchConfig::moveToNextGdim() {
  auto grid_dims_next_pos = getNextGdimsPos();
  if (grid_dims_next_pos >= 0) {
    grid_dims_pos_ = grid_dims_next_pos;
    return true;
  } else {
    return false;
  }
}

int PreferredLaunchConfig::peekNextGdimx() const {
  auto grid_dims_next_pos = getNextGdimsPos();
  if (grid_dims_next_pos >= 0) {
    return gdimxAt(grid_dims_next_pos);
  } else {
    return -1;
  }
}

int PreferredLaunchConfig::peekNextGdimy() const {
  auto grid_dims_next_pos = getNextGdimsPos();
  if (grid_dims_next_pos >= 0) {
    return gdimyAt(grid_dims_next_pos);
  } else {
    return -1;
  }
}

int PreferredLaunchConfig::getNextGdimsPos() const {
  auto grid_dims_next_pos = grid_dims_pos_ + 1;
  if (grid_dims_next_pos < (int)valid_grid_dims_.size()) {
    return grid_dims_next_pos;
  } else {
    return -1;
  }
}

namespace {

// Estimated register count available for persistent buffer. The
// available space is considered to depend on the size of the
// persistent buffer itself due to the predicate caching
int64_t getAvailableRegisterCount(int64_t persistent_buffer_factor) {
  // The thread block size is (currently) always 256, so each thread
  // can use up to 255 registers
  int64_t register_count = 255;

  // Offset a constant overhead
  register_count -= 40;

  // Allow small number of spills
  register_count += 5;

  // account for index caching, assuming each cache entry
  //  consumes one register
  // TODO: Consider relaxing this reduction. It seems likes
  //  overestimation.
  register_count -= persistent_buffer_factor;

  return register_count;
}

int64_t getMinPersistentBufferSize(
    const int64_t total_reduction_numel,
    const int64_t bdimy,
    const int64_t gdimy) {
  return ceilDiv(ceilDiv(total_reduction_numel, bdimy), gdimy);
}

// Return true if a given combination of parameters is likely to
// result in no (or little) register spilling
bool checkIfWithinRegisterSpace(
    int64_t total_reduction_numel,
    int64_t persistent_buffer_size,
    int64_t vectorize_factor,
    int64_t bdimy,
    int64_t gdimy) {
  // The extent of the persistent buffer domain
  auto pb_factor =
      getMinPersistentBufferSize(total_reduction_numel, bdimy, gdimy);

  TORCH_INTERNAL_ASSERT(pb_factor > 0);

  const auto available_reg_count = getAvailableRegisterCount(pb_factor);

  auto per_thread_persistent_buffer_size =
      ceilDiv(ceilDiv(persistent_buffer_size, bdimy), gdimy) * vectorize_factor;

  auto persistent_buffer_reg_count =
      ceilDiv(per_thread_persistent_buffer_size, sizeof(int));

  log("persistent_buffer_reg_count: ",
      persistent_buffer_reg_count,
      ", available_reg_count: ",
      available_reg_count);

  return persistent_buffer_reg_count <= available_reg_count;
}

// Calculate the factor of work of the last thread block in each of
// reductions. More specifically, use the number of serial
// iterations for the persistent buffer loop as a proxy of the
// amount of work. The rest of the blocks should execute the loop
// buffer_size times, whereas the last block only processes the
// remaining iterations.
double getLastBlockWorkRatio(
    const int64_t total_reduction_numel,
    const int64_t bdimy,
    const int64_t persistent_buffer_size) {
  auto last_block_pb =
      total_reduction_numel % (persistent_buffer_size * bdimy) / bdimy;
  return ((double)last_block_pb) / (double)persistent_buffer_size;
};

// In the current outer normalization scheduling, only the last thread
// block of each reduction group hits the fallback path of the
// unswitched loops, so it can be significantly slower than the
// rest. This is particularly problematic with grid persistence as all
// thread blocks need to synchronize, so the slowest block determines
// the performance. This could be to some extent mitigated by
// adjusting the buffer size such that the work assigned to the last
// block is relatively smaller than the work assigned to the
// rest.
//
// Here, given a valid launch config, we try to slightly adjust it so
// that the ratio of the last work becomes the smallest. We do this by
// increasing buffer sizes and in turn decreasing gdimy and picking the
// configuration that has the smallest work ratio. All of this is done
// with some bounds, e.g., the buffer size should still be within the
// register space, the decrease of gdimy should be less than 10%,
// etc. These threshold values are experimentally picked on A100 with
// the current benchmarks, but more tuning would likely lead to better
// performance.
//
// The function returns the adjusted gdimy and persistent buffer size
// as well as a bool indicating whether the work size is
// sufficiently reduced. Nullopt is returned if no adjustment is
// successfully done and the search should continue.
std::optional<std::tuple<int64_t, int64_t, bool>> reduceWorkOfLastBlock(
    const PreferredLaunchConfig& launch_cfg,
    const int64_t total_reduction_numel,
    const int64_t total_iteration_numel,
    const int64_t persistent_buffer_size,
    const int64_t vectorize_factor) {
  const auto bdimy = launch_cfg.bdimy();

  // Aim to reduce the work size of the last block to be smaller than
  // some factor of the rest of the blocks.
  const double target_last_block_work_ratio = 0.25;

  // Start with the current gdimy and buffer size. Gradually increase
  // the buffer size and in turn decrease gdimy with the bounds set as
  // below.
  auto current_gdimy = launch_cfg.gdimy();
  auto current_buffer_size =
      getMinPersistentBufferSize(total_reduction_numel, bdimy, current_gdimy);

  log("reduceWorkOfLastBlock: ", current_gdimy, ", ", current_buffer_size);

  // Threshold to stop decreasing gdimy
  const auto min_gdimy = current_gdimy * 0.9;

  // Keep track of the best gdimy and buffer size configuration
  auto optimal_size = current_buffer_size;
  auto optimal_gdimy = current_gdimy;
  double optimal_work_ratio =
      getLastBlockWorkRatio(total_reduction_numel, bdimy, current_buffer_size);

  // Find the best gdimy and buffer size configuration by lowering
  // gdimy. Stop if the minimum gdimy is hit or the register limit is
  // reached.
  while (current_gdimy >= min_gdimy &&
         checkIfWithinRegisterSpace(
             total_reduction_numel,
             persistent_buffer_size,
             vectorize_factor,
             bdimy,
             current_gdimy)) {
    auto ratio_of_last_block_work = getLastBlockWorkRatio(
        total_reduction_numel, bdimy, current_buffer_size);
    log("Ratio of last block work: ",
        ratio_of_last_block_work,
        ", persistent_buffer: ",
        current_buffer_size,
        ", gdimy: ",
        current_gdimy);

    if (ratio_of_last_block_work < optimal_work_ratio) {
      optimal_work_ratio = ratio_of_last_block_work;
      optimal_size = current_buffer_size;
      optimal_gdimy = current_gdimy;
    }

    if (ratio_of_last_block_work < target_last_block_work_ratio) {
      // Good enough config found; stop searching
      break;
    }

    // not good enough; increase persistent buffer
    ++current_buffer_size;
    // adjust gdimy (decreased as persitent_buffer is increased)
    current_gdimy =
        ceilDiv(ceilDiv(total_reduction_numel, bdimy), current_buffer_size);

    log("Next buffer size: ",
        current_buffer_size,
        ", Next gdimy: ",
        current_gdimy);
  }

  // Use the optimal ratio if it's within the threshold
  if (optimal_work_ratio < target_last_block_work_ratio) {
    log("Successfully reduced to ", optimal_work_ratio);
    return std::make_tuple(optimal_gdimy, optimal_size, true);
  }

  // Acceptable config not found. Continue searching a better config
  // by moving to the next candidate. However, if the next candidate
  // incurs a larger number of grid syncs, i.e., the serial factor of
  // the iteration domain is larger, the additional overheaad would
  // likely to outweight the benefit of potentially better block
  // specialization, so pick the best among found so far.
  auto next_gdimx = launch_cfg.peekNextGdimx();

  // If the next gdimx is negative, that means there's no more config
  // candidate or the next would decrease the bdimx, which could be a
  // large perf degradation, so stop the search then.
  if (next_gdimx < 0) {
    log("Stop as there's no more search space left for gdimx");
    return std::make_tuple(optimal_gdimy, optimal_size, false);
  }

  if (next_gdimx > 0) {
    auto remaining_iteration_factor = ceilDiv(
        ceilDiv(total_iteration_numel, vectorize_factor), launch_cfg.bdimx());
    auto current_iterration_count =
        ceilDiv(remaining_iteration_factor, launch_cfg.gdimx());
    auto next_iteration_count = ceilDiv(remaining_iteration_factor, next_gdimx);
    log("Next iteration count: ",
        next_iteration_count,
        ", next gdimx: ",
        next_gdimx,
        ", current iteration: ",
        current_iterration_count,
        ", curreng gdimx: ",
        launch_cfg.gdimx());
    if (next_iteration_count > current_iterration_count) {
      log("Still not good but stop here to avoid increase of iteration count");
      return std::make_tuple(optimal_gdimy, optimal_size, false);
    }
  }

  log("Acceptable config not found. Continue search");
  return std::nullopt;
}

} // namespace

// Iterate configurations from largest blockDim.x and smallest
// gridDim.y until the per-thread size of the persistent buffer
// becomes sufficiently small enough not to cause (significant)
// register spill.
std::optional<GridOuterNormalizationParams> getGridOuterNormalizationParams(
    int64_t total_reduction_numel,
    int64_t total_iteration_numel,
    int64_t vectorize_factor,
    int64_t persistent_buffer_size) {
  PreferredLaunchConfig launch_cfg;

  // The launch config starts with the largest blockDim.x, which may
  // be larger than the iteration size. Decrease it until it doesn't
  // exceed the iteration size.
  const auto max_bdimx = ceilDiv(total_iteration_numel, vectorize_factor);
  while (launch_cfg.bdimx() > max_bdimx) {
    if (!launch_cfg.moveToNextBdim()) {
      // The iteration size is too small. It might still be worthwhile
      // to be persistent, but it's unlikely to be performant anyway
      return std::nullopt;
    }
  }

  // Iterate candidates of launch configurations
  while (!launch_cfg.isInvalid()) {
    log("Current config: ", launch_cfg);

    // Skip if iterations are not evenly distributed among thread
    // blocks unless the remaining factor is smaller than
    // gridDim.x. However, don't skip if this is the last valid config
    // within the same blockDim config.
    auto remaining_gdimx_factor =
        ceilDiv(total_iteration_numel / vectorize_factor, launch_cfg.bdimx());
    // TODO: Needs better tuning. Probably want to allow
    // configurations that are slightly uneven
    if (remaining_gdimx_factor > launch_cfg.gdimx() &&
        remaining_gdimx_factor % launch_cfg.gdimx() != 0 &&
        !launch_cfg.isNextSmallerBdimx()) {
      log("Rejected due to uneven iteration domain");
      launch_cfg.moveToNextConfig();
      continue;
    }

    if (!checkIfWithinRegisterSpace(
            total_reduction_numel,
            persistent_buffer_size,
            vectorize_factor,
            launch_cfg.bdimy(),
            launch_cfg.gdimy())) {
      log("Rejected due to register spill");
      launch_cfg.moveToNextConfig();
      continue;
    }

    // At this point, gdimy is large enough to keep the register
    // pressure low enough.

    // In case the iteration domain is small, the gdimx and bdimx pair
    // may be too large and some threads/blocks may be idle.

    if (remaining_gdimx_factor < launch_cfg.gdimx()) {
      log("gdimx too large: ",
          remaining_gdimx_factor,
          ", vec: ",
          vectorize_factor);
      launch_cfg.moveToNextConfig();
      continue;
    }

    // If there's idle tidx threads, don't accept if there's further
    // config candidates with smaller bdimx
    if (vectorize_factor * launch_cfg.bdimx() * launch_cfg.gdimx() >
            total_iteration_numel &&
        launch_cfg.canLowerBdimx()) {
      log("Skip due to too large bdimx: ", launch_cfg.bdimx());
      launch_cfg.moveToNextBdim();
      continue;
    }

    // Adjust gdimy and buffer size for processing predicates more
    // efficiently through the block specialization, so that the last
    // block is assigned with a relatively small chunk of work.
    // For some reason, this doesn't work well on Titan RTX. It seems
    // it's just better unswitching by a small factor.
    // TODO: Test other generations of GPUs
    int64_t adjusted_gdimy = -1;
    int64_t adjusted_buffer_size = -1;
    bool last_block_work_reduced = false;
    const auto major_ver = at::cuda::getCurrentDeviceProperties()->major;
    const auto minor_ver = at::cuda::getCurrentDeviceProperties()->minor;
    if (major_ver == 7 && minor_ver == 5) {
      adjusted_gdimy = launch_cfg.gdimy();
      adjusted_buffer_size = getMinPersistentBufferSize(
          total_reduction_numel, launch_cfg.bdimy(), launch_cfg.gdimy());
      last_block_work_reduced = false;
    } else {
      auto gdimy_pb_size = reduceWorkOfLastBlock(
          launch_cfg,
          total_reduction_numel,
          total_iteration_numel,
          persistent_buffer_size,
          vectorize_factor);
      if (!gdimy_pb_size.has_value()) {
        launch_cfg.moveToNextConfig();
        continue;
      }
      std::tie(adjusted_gdimy, adjusted_buffer_size, last_block_work_reduced) =
          *gdimy_pb_size;
    }

    // Acceptable configuration found
    auto launch_params = LaunchParams(
        launch_cfg.gdimx(),
        adjusted_gdimy,
        LaunchParams::UNINITIALIZED_VAL,
        launch_cfg.bdimx(),
        launch_cfg.bdimy(),
        LaunchParams::UNINITIALIZED_VAL);

    // If the last block is sufficiently reduced, unswitch the whole
    // persistent buffer. Otherwise, unswitch by a factor of 4.
    int64_t unswitch_factor = last_block_work_reduced
        ? adjusted_buffer_size
        : std::min(4l, adjusted_buffer_size);

    GridOuterNormalizationParams params = {
        .launch_params = launch_params,
        .persistent_buffer_factor = adjusted_buffer_size,
        .unswitch_factor = unswitch_factor};
    return params;
  }

  // No valid config found. Return launch_cfg, which should be marked
  // as invalid
  TORCH_INTERNAL_ASSERT(launch_cfg.isInvalid());
  return std::nullopt;
}

} // namespace normalization_scheduler_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
