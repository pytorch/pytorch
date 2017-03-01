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

#include "gloo/cuda_broadcast_one_to_all.h"
#include "gloo/test/cuda_base_test.h"

namespace gloo {
namespace test {
namespace {

// Function to instantiate and run algorithm.
using Func = std::unique_ptr<::gloo::Algorithm>(
    std::shared_ptr<::gloo::Context>&,
    float* ptr,
    int count,
    int rootRank,
    cudaStream_t stream);

// Test parameterization.
using Param = std::tuple<int, int, std::function<Func>>;

// Test fixture.
class CudaBroadcastTest : public CudaBaseTest,
                          public ::testing::WithParamInterface<Param> {
 public:
  void assertEqual(Fixture& fixture, int root) {
    const auto stride = fixture.context->size_;
    for (const auto& ptr : fixture.getHostBuffers()) {
      for (int i = 0; i < fixture.count; i++) {
        ASSERT_EQ((i * stride) + root, ptr[i])
          << "Mismatch at index " << i
          << " for rank " << fixture.context->rank_;
      }
    }
  }
};

TEST_P(CudaBroadcastTest, SinglePointer) {
  auto size = std::get<0>(GetParam());
  auto count = std::get<1>(GetParam());
  auto fn = std::get<2>(GetParam());

  spawn(size, [&](std::shared_ptr<Context> context) {
      // Run with varying root
      for (int root = 0; root < 1; root++) {
        auto fixture = Fixture(context, 1, count);
        auto ptrs = fixture.getFloatPointers();
        auto algorithm = fn(context, ptrs[0], count, root, kStreamNotSet);
        fixture.assignValues();
        algorithm->run();

        // Verify result
        assertEqual(fixture, root);
      }
    });
}

TEST_P(CudaBroadcastTest, SinglePointerAsync) {
  auto size = std::get<0>(GetParam());
  auto count = std::get<1>(GetParam());
  auto fn = std::get<2>(GetParam());

  spawn(size, [&](std::shared_ptr<Context> context) {
      // Run with varying root
      for (int root = 0; root < 1; root++) {
        auto fixture = Fixture(context, 1, count);
        auto ptrs = fixture.getFloatPointers();
        auto streams = fixture.getCudaStreams();
        auto algorithm = fn(context, ptrs[0], count, root, streams[0]);
        fixture.assignValuesAsync();
        algorithm->run();

        // Verify result
        assertEqual(fixture, root);
      }
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

static std::function<Func> broadcastOneToAll = [](
    std::shared_ptr<::gloo::Context>& context,
    float* ptr,
    int count,
    int rootRank,
    cudaStream_t stream) {
  return std::unique_ptr<::gloo::Algorithm>(
    new ::gloo::CudaBroadcastOneToAll<float>(
      context, ptr, count, rootRank, stream));
};

INSTANTIATE_TEST_CASE_P(
    OneToAllBroadcast,
    CudaBroadcastTest,
    ::testing::Combine(
        ::testing::Range(2, 16),
        ::testing::ValuesIn(genMemorySizes()),
        ::testing::Values(broadcastOneToAll)));

} // namespace
} // namespace test
} // namespace gloo
