#include <c10/core/AllocatorConfig.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/Exception.h>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

using c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig;
using c10::cuda::CUDACachingAllocator::LargeBlockPlacement;

namespace {

constexpr size_t kMiB = size_t{1} << 20;

void setConf(const std::string& conf) {
  c10::CachingAllocator::setAllocatorSettings(conf);
}

// The settings are process-global and sticky, so every placement run sets both
// keys explicitly.
std::string placementConf(const std::string& placement, size_t max_size_mb) {
  return "expandable_segments:True,large_block_placement:" + placement +
      ",first_fit_max_size_mb:" + std::to_string(max_size_mb);
}

// Carves three free holes of 600, 200 and 400 MiB at ascending addresses, each
// fenced by a live 20 MiB block, then allocates `request_mib` and returns which
// hole (1-based, by address) the allocation landed in; 0 if it landed
// elsewhere.
int holeChosenFor(const std::string& conf, size_t request_mib) {
  namespace cca = c10::cuda::CUDACachingAllocator;
  setConf(conf);
  cca::emptyCache();

  constexpr std::array<size_t, 3> hole_mib{600, 200, 400};
  std::array<void*, 3> holes{};
  std::array<void*, 3> fences{};
  for (size_t i = 0; i < 3; i++) {
    holes[i] = cca::raw_alloc(hole_mib[i] * kMiB);
    fences[i] = cca::raw_alloc(20 * kMiB);
  }
  for (size_t i = 0; i < 2; i++) {
    EXPECT_LT(
        reinterpret_cast<uintptr_t>(fences[i]),
        reinterpret_cast<uintptr_t>(holes[i + 1]))
        << "holes must be laid out in ascending address order";
  }
  for (void* hole : holes) {
    cca::raw_delete(hole);
  }

  void* block = cca::raw_alloc(request_mib * kMiB);
  int chosen = 0;
  for (size_t i = 0; i < 3; i++) {
    const auto begin = reinterpret_cast<uintptr_t>(holes[i]);
    const auto addr = reinterpret_cast<uintptr_t>(block);
    if (addr >= begin && addr < begin + hole_mib[i] * kMiB) {
      chosen = static_cast<int>(i) + 1;
    }
  }

  cca::raw_delete(block);
  for (void* fence : fences) {
    cca::raw_delete(fence);
  }
  cca::emptyCache();
  return chosen;
}

bool initDeviceOrSkip() {
  if (c10::cuda::device_count() == 0) {
    return false;
  }
  c10::cuda::set_device(0);
  c10::cuda::CUDACachingAllocator::init(c10::cuda::device_count());
  return true;
}

} // namespace

// Must stay the first test: the settings are sticky, so the default can only
// be observed before any other test sets a value.
TEST(LargeBlockPlacementDeviceTest, DefaultPlacementPicksSmallestFittingHole) {
  if (!initDeviceOrSkip()) {
    GTEST_SKIP() << "no CUDA device";
  }
  if (std::getenv("PYTORCH_CUDA_ALLOC_CONF") != nullptr) {
    GTEST_SKIP()
        << "PYTORCH_CUDA_ALLOC_CONF is set, so the default is not in effect";
  }
  EXPECT_EQ(holeChosenFor("expandable_segments:True", 300), 3);
}

TEST(LargeBlockPlacementConfigTest, ParsesValidValues) {
  setConf("large_block_placement:first_fit,first_fit_max_size_mb:1024");
  EXPECT_EQ(
      CUDAAllocatorConfig::large_block_placement(),
      LargeBlockPlacement::FIRST_FIT);
  EXPECT_EQ(CUDAAllocatorConfig::first_fit_max_size(), 1024 * kMiB);

  setConf("large_block_placement:best_fit");
  EXPECT_EQ(
      CUDAAllocatorConfig::large_block_placement(),
      LargeBlockPlacement::BEST_FIT);
}

TEST(LargeBlockPlacementConfigTest, RejectsBadValuesAndKeepsState) {
  setConf("large_block_placement:first_fit,first_fit_max_size_mb:512");
  EXPECT_THROW(setConf("large_block_placement:worst_fit"), c10::ValueError);
  EXPECT_THROW(setConf("first_fit_max_size_mb:0"), c10::ValueError);
  EXPECT_ANY_THROW(setConf("first_fit_max_size_mb:abc"));
  EXPECT_EQ(
      CUDAAllocatorConfig::large_block_placement(),
      LargeBlockPlacement::FIRST_FIT);
  EXPECT_EQ(CUDAAllocatorConfig::first_fit_max_size(), 512 * kMiB);
  setConf("large_block_placement:best_fit");
}

TEST(LargeBlockPlacementDeviceTest, BestFitPicksSmallestFittingHole) {
  if (!initDeviceOrSkip()) {
    GTEST_SKIP() << "no CUDA device";
  }
  EXPECT_EQ(holeChosenFor(placementConf("best_fit", 1024), 300), 3);
}

TEST(LargeBlockPlacementDeviceTest, FirstFitPicksLowestAddressFittingHole) {
  if (!initDeviceOrSkip()) {
    GTEST_SKIP() << "no CUDA device";
  }
  EXPECT_EQ(holeChosenFor(placementConf("first_fit", 1024), 300), 1);
}

TEST(LargeBlockPlacementDeviceTest, FirstFitSkipsHolesThatAreTooSmall) {
  if (!initDeviceOrSkip()) {
    GTEST_SKIP() << "no CUDA device";
  }
  // 500 MiB only fits the 600 MiB hole, which is also the lowest address; 700
  // MiB fits no hole and must grow the segment instead.
  EXPECT_EQ(holeChosenFor(placementConf("first_fit", 1024), 500), 1);
  EXPECT_EQ(holeChosenFor(placementConf("first_fit", 1024), 700), 0);
}

TEST(LargeBlockPlacementDeviceTest, RequestsAtThresholdKeepBestFit) {
  if (!initDeviceOrSkip()) {
    GTEST_SKIP() << "no CUDA device";
  }
  EXPECT_EQ(holeChosenFor(placementConf("first_fit", 300), 300), 3);
  EXPECT_EQ(holeChosenFor(placementConf("first_fit", 301), 300), 1);
}
