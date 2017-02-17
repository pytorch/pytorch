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
#include <thread>
#include <vector>

#include "gloo/common/common.h"
#include "gloo/cuda_allreduce_ring.h"
#include "gloo/cuda_allreduce_ring_chunked.h"
#include "gloo/cuda_private.h"
#include "gloo/test/base_test.h"

namespace gloo {
namespace test {
namespace {

// Function to instantiate and run algorithm.
using Func = void(
    std::shared_ptr<::gloo::Context>&,
    std::vector<float*> dataPtrs,
    int count);

// Test parameterization.
using Param = std::tuple<int, int, std::function<Func>>;

// Test fixture.
class CudaAllreduceTest : public BaseTest,
                          public ::testing::WithParamInterface<Param> {

  using CudaBuffers = std::vector<std::unique_ptr<CudaMemory<float> > >;
  using HostBuffers = std::vector<std::unique_ptr<float[]> >;

 public:
  int getDeviceCount() {
    int n = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n));
    return n;
  }

  std::vector<float*> getFloatPointers(const CudaBuffers& in) {
    std::vector<float*> out;
    for (const auto& i : in) {
      out.push_back(**i);
    }
    return out;
  }

  CudaBuffers getDeviceBuffers(int rank, int size, size_t count) {
    CudaBuffers out;

    auto n = getDeviceCount();
    for (int i = 0; i < n; i++) {
      CUDA_CHECK(cudaSetDevice(i));
      out.push_back(make_unique<CudaMemory<float> >(count, (rank * n) + i));
    }

    return out;
  }

  HostBuffers getHostBuffers(const CudaBuffers& in, size_t count) {
    HostBuffers out;
    for (const auto& src : in) {
      out.push_back(src->copyToHost());
    }
    return out;
  }
};

TEST_P(CudaAllreduceTest, SinglePointer) {
  auto size = std::get<0>(GetParam());
  auto count = std::get<1>(GetParam());
  auto fn = std::get<2>(GetParam());

  spawnThreads(size, [&](int rank) {
    auto context =
        std::make_shared<::gloo::Context>(rank, size);
    context->connectFullMesh(*store_, device_);

    // Run algorithm
    auto in = getDeviceBuffers(rank, size, count);
    fn(context, getFloatPointers(in), count);
    auto out = getHostBuffers(in, count);

    // Verify result
    auto logicalSize = getDeviceCount() * size;
    auto expected = (logicalSize * (logicalSize - 1)) / 2;
    for (const auto& ptr : out) {
      for (int i = 0; i < count; i++) {
        ASSERT_EQ(expected, ptr[i]) << "Mismatch at index " << i;
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

static std::function<Func> allreduceRing = [](
    std::shared_ptr<::gloo::Context>& context,
    std::vector<float*> ptrs,
    int count) {
  ::gloo::CudaAllreduceRing<float> algorithm(context, ptrs, count);
  algorithm.run();
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
    int count) {
  ::gloo::CudaAllreduceRingChunked<float> algorithm(context, ptrs, count);
  algorithm.run();
};

INSTANTIATE_TEST_CASE_P(
    AllreduceRingChunked,
    CudaAllreduceTest,
    ::testing::Combine(
    ::testing::Range(2, 16),
    ::testing::ValuesIn(genMemorySizes()),
    ::testing::Values(allreduceRing)));

} // namespace
} // namespace test
} // namespace gloo
