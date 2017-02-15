/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <functional>
#include <thread>
#include <vector>

#include "gloo/common/common.h"
#include "gloo/cuda_broadcast_one_to_all.h"
#include "gloo/cuda_private.h"
#include "gloo/test/base_test.h"

namespace gloo {
namespace test {
namespace {

// Function to instantiate and run algorithm.
using Func = void(
    std::shared_ptr<::gloo::Context>&,
    float* dataPtr,
    int dataSize,
    int rootRank);

// Test parameterization.
using Param = std::tuple<int, int, std::function<Func>>;

// Test fixture.
class CudaBroadcastTest : public BaseTest,
                          public ::testing::WithParamInterface<Param> {

  using CudaBuffer = std::unique_ptr<CudaMemory<float>>;
  using HostBuffer = std::unique_ptr<float[]>;

 public:
  CudaBuffer getDeviceBuffer(int rank, int count) {
    CudaBuffer out = make_unique<CudaMemory<float> >(count, rank);
    return out;
  }

  HostBuffer getHostBuffer(const CudaBuffer& in) {
    return in->copyToHost();
  }
};

TEST_P(CudaBroadcastTest, SinglePointer) {
  auto contextSize = std::get<0>(GetParam());
  auto dataSize = std::get<1>(GetParam());
  auto fn = std::get<2>(GetParam());

  spawnThreads(contextSize, [&](int contextRank) {
    auto context =
        std::make_shared<::gloo::Context>(contextRank, contextSize);
    context->connectFullMesh(*store_, device_);

    // Run with varying root
    for (int rootRank = 0; rootRank < 1; rootRank++) {
      // Run algorithm
      auto in = getDeviceBuffer(contextRank, dataSize);
      fn(context, **in, dataSize, rootRank);
      auto out = getHostBuffer(in);

      // Verify result
      for (int i = 0; i < dataSize; i++) {
        EXPECT_EQ(rootRank, out[i]) << "Mismatch at index " << i <<
          " for rank " << contextRank;
      }
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
    float* dataPtr,
    int dataSize,
    int rootRank) {
  ::gloo::CudaBroadcastOneToAll<float> algorithm(
      context, dataPtr, dataSize, rootRank);
  algorithm.run();
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
