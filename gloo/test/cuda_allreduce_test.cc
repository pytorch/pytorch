/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <functional>
#include <memory>
#include <vector>

#include "gloo/cuda_allreduce_ring.h"
#include "gloo/cuda_allreduce_ring_chunked.h"
#include "gloo/test/cuda_base_test.h"

namespace gloo {
namespace test {
namespace {

// Function to instantiate and run algorithm.
using Func = std::unique_ptr<::gloo::Algorithm>(
    std::shared_ptr<::gloo::Context>&,
    std::vector<float*> ptrs,
    int count,
    std::vector<cudaStream_t> streams);

// Test parameterization.
using Param = std::tuple<int, int, std::function<Func>>;

// Test case
class CudaAllreduceTest : public CudaBaseTest,
                          public ::testing::WithParamInterface<Param> {
 public:
  void assertEqual(Fixture& fixture, int expected) {
    for (const auto& ptr : fixture.getHostBuffers()) {
      for (int i = 0; i < fixture.count; i++) {
        ASSERT_EQ(expected, ptr[i]) << "Mismatch at index " << i;
      }
    }
  }
};

TEST_P(CudaAllreduceTest, SinglePointer) {
  auto size = std::get<0>(GetParam());
  auto count = std::get<1>(GetParam());
  auto fn = std::get<2>(GetParam());

  spawn(size, [&](int rank, std::shared_ptr<Context> context) {
    // Run algorithm
    auto fixture = Fixture(1, count);
    auto ptrs = fixture.getFloatPointers();
    auto algorithm = fn(context, ptrs, count, {});
    fixture.setRank(rank);
    algorithm->run();

    // Verify result
    auto logicalSize = size;
    auto expected = (logicalSize * (logicalSize - 1)) / 2;
    assertEqual(fixture, expected);
  });
}

TEST_P(CudaAllreduceTest, MultiPointer) {
  auto size = std::get<0>(GetParam());
  auto count = std::get<1>(GetParam());
  auto fn = std::get<2>(GetParam());

  spawn(size, [&](int rank, std::shared_ptr<Context> context) {
      // Run algorithm
      auto fixture = Fixture(getDeviceCount(), count);
      auto ptrs = fixture.getFloatPointers();
      auto algorithm = fn(context, ptrs, count, {});
      fixture.setRank(rank);
      algorithm->run();

      // Verify result
      auto logicalSize = getDeviceCount() * size;
      auto expected = (logicalSize * (logicalSize - 1)) / 2;
      assertEqual(fixture, expected);
    });
}

TEST_P(CudaAllreduceTest, MultiPointerAsync) {
  auto size = std::get<0>(GetParam());
  auto count = std::get<1>(GetParam());
  auto fn = std::get<2>(GetParam());

  spawn(size, [&](int rank, std::shared_ptr<Context> context) {
      // Run algorithm
      auto fixture = Fixture(getDeviceCount(), count);
      auto ptrs = fixture.getFloatPointers();
      auto streams = fixture.getCudaStreams();
      auto algorithm = fn(context, ptrs, count, streams);
      fixture.setRankAsync(rank);
      algorithm->run();

      // Verify result
      auto logicalSize = getDeviceCount() * size;
      auto expected = (logicalSize * (logicalSize - 1)) / 2;
      assertEqual(fixture, expected);
    });
}

std::vector<int> genMemorySizes() {
  std::vector<int> v;
  v.push_back(sizeof(float));
  v.push_back(100);
  v.push_back(1000);
  v.push_back(10000);
  return v;
}

static std::function<Func> allreduceRing = [](
    std::shared_ptr<::gloo::Context>& context,
    std::vector<float*> ptrs,
    int count,
    std::vector<cudaStream_t> streams) {
  return std::unique_ptr<::gloo::Algorithm>(
    new ::gloo::CudaAllreduceRing<float>(context, ptrs, count, streams));
};

INSTANTIATE_TEST_CASE_P(
    AllreduceRing,
    CudaAllreduceTest,
    ::testing::Combine(
      ::testing::Range(2, 16),
      ::testing::ValuesIn(genMemorySizes()),
      ::testing::Values(allreduceRing)));

static std::function<Func> allreduceRingChunked = [](
    std::shared_ptr<::gloo::Context>& context,
    std::vector<float*> ptrs,
    int count,
    std::vector<cudaStream_t> streams) {
  return std::unique_ptr<::gloo::Algorithm>(
    new ::gloo::CudaAllreduceRingChunked<float>(context, ptrs, count, streams));
};

INSTANTIATE_TEST_CASE_P(
    AllreduceRingChunked,
    CudaAllreduceTest,
    ::testing::Combine(
      ::testing::Range(2, 16),
      ::testing::ValuesIn(genMemorySizes()),
      ::testing::Values(allreduceRingChunked)));

} // namespace
} // namespace test
} // namespace gloo
