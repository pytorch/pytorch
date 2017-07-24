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

#include "gloo/cuda_allreduce_halving_doubling.h"
#include "gloo/cuda_allreduce_halving_doubling_pipelined.h"
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

using Func16 = std::unique_ptr<::gloo::Algorithm>(
    std::shared_ptr<::gloo::Context>&,
    std::vector<float16*> ptrs,
    int count,
    std::vector<cudaStream_t> streams);

// Test parameterization.
using Param = std::tuple<int, int, std::function<Func>>;
using ParamHP = std::tuple<int, int, std::function<Func16>>;

// Test case
class CudaAllreduceTest : public CudaBaseTest,
                          public ::testing::WithParamInterface<Param> {
 public:
  void assertResult(CudaFixture<float>& fixture) {
    // Size is the total number of pointers across the context
    const auto size = fixture.ptrs.size() * fixture.context->size;
    // Expected is set to the expected value at ptr[0]
    const auto expected = (size * (size - 1)) / 2;
    // The stride between values at subsequent indices is equal to
    // "size", and we have "size" of them. Therefore, after
    // allreduce, the stride between expected values is "size^2".
    const auto stride = size * size;
    // Verify all buffers passed by this instance
    for (const auto& ptr : fixture.getHostBuffers()) {
      for (int i = 0; i < fixture.count; i++) {
        ASSERT_EQ((i * stride) + expected, ptr[i])
          << "Mismatch at index " << i;
      }
    }
  }
};

class CudaAllreduceTestHP : public CudaBaseTest,
                            public ::testing::WithParamInterface<ParamHP> {
 public:
  void assertResult(CudaFixture<float16>& fixture) {
    // Size is the total number of pointers across the context
    const auto size = fixture.ptrs.size() * fixture.context->size;
    // Expected is set to the expected value at ptr[0]
    const auto expected = (size * (size - 1)) / 2;
    // The stride between values at subsequent indices is equal to
    // "size", and we have "size" of them. Therefore, after
    // allreduce, the stride between expected values is "size^2".
    const auto stride = size * size;
    // Verify all buffers passed by this instance
    for (const auto& ptr : fixture.getHostBuffers()) {
      for (int i = 0; i < fixture.count; i++) {
        ASSERT_EQ((float16)((i * stride) + expected), ptr[i])
          << "Mismatch at index " << i;
      }
    }
  }
};

static std::function<Func> allreduceRing = [](
    std::shared_ptr<::gloo::Context>& context,
    std::vector<float*> ptrs,
    int count,
    std::vector<cudaStream_t> streams) {
  return std::unique_ptr<::gloo::Algorithm>(
    new ::gloo::CudaAllreduceRing<float>(context, ptrs, count, streams));
};

static std::function<Func16> allreduceRingHP = [](
    std::shared_ptr<::gloo::Context>& context,
    std::vector<float16*> ptrs,
    int count,
    std::vector<cudaStream_t> streams) {
  return std::unique_ptr<::gloo::Algorithm>(
    new ::gloo::CudaAllreduceRing<float16>(context, ptrs, count, streams));
};

static std::function<Func> allreduceRingChunked = [](
    std::shared_ptr<::gloo::Context>& context,
    std::vector<float*> ptrs,
    int count,
    std::vector<cudaStream_t> streams) {
  return std::unique_ptr<::gloo::Algorithm>(
    new ::gloo::CudaAllreduceRingChunked<float>(context, ptrs, count, streams));
};

static std::function<Func16> allreduceRingChunkedHP = [](
    std::shared_ptr<::gloo::Context>& context,
    std::vector<float16*> ptrs,
    int count,
    std::vector<cudaStream_t> streams) {
  return std::unique_ptr<::gloo::Algorithm>(
      new ::gloo::CudaAllreduceRingChunked<float16>(
          context, ptrs, count, streams));
};

static std::function<Func> allreduceHalvingDoubling = [](
    std::shared_ptr<::gloo::Context>& context,
    std::vector<float*> ptrs,
    int count,
    std::vector<cudaStream_t> streams) {
  return std::unique_ptr<::gloo::Algorithm>(
      new ::gloo::CudaAllreduceHalvingDoubling<float>(
          context, ptrs, count, streams));
};

static std::function<Func16> allreduceHalvingDoublingHP = [](
    std::shared_ptr<::gloo::Context>& context,
    std::vector<float16*> ptrs,
    int count,
    std::vector<cudaStream_t> streams) {
  return std::unique_ptr<::gloo::Algorithm>(
      new ::gloo::CudaAllreduceHalvingDoubling<float16>(
          context, ptrs, count, streams));
};

static std::function<Func> allreduceHalvingDoublingPipelined = [](
    std::shared_ptr<::gloo::Context>& context,
    std::vector<float*> ptrs,
    int count,
    std::vector<cudaStream_t> streams) {
  return std::unique_ptr<::gloo::Algorithm>(
      new ::gloo::CudaAllreduceHalvingDoublingPipelined<float>(
          context, ptrs, count, streams));
};

static std::function<Func16> allreduceHalvingDoublingPipelinedHP = [](
    std::shared_ptr<::gloo::Context>& context,
    std::vector<float16*> ptrs,
    int count,
    std::vector<cudaStream_t> streams) {
  return std::unique_ptr<::gloo::Algorithm>(
      new ::gloo::CudaAllreduceHalvingDoublingPipelined<float16>(
          context, ptrs, count, streams));
};

TEST_P(CudaAllreduceTest, SinglePointer) {
  auto size = std::get<0>(GetParam());
  auto count = std::get<1>(GetParam());
  auto fn = std::get<2>(GetParam());

  spawn(size, [&](std::shared_ptr<Context> context) {
      // Run algorithm
      auto fixture = CudaFixture<float>(context, 1, count);
      auto ptrs = fixture.getPointers();
      auto algorithm = fn(context, ptrs, count, {});
      fixture.assignValues();
      algorithm->run();

      // Verify result
      assertResult(fixture);
    });
}

TEST_P(CudaAllreduceTest, MultiPointer) {
  auto size = std::get<0>(GetParam());
  auto count = std::get<1>(GetParam());
  auto fn = std::get<2>(GetParam());

  spawn(size, [&](std::shared_ptr<Context> context) {
      // Run algorithm
      auto fixture = CudaFixture<float>(context, cudaNumDevices(), count);
      auto ptrs = fixture.getPointers();
      auto algorithm = fn(context, ptrs, count, {});
      fixture.assignValues();
      algorithm->run();

      // Verify result
      assertResult(fixture);
    });
}

TEST_P(CudaAllreduceTest, MultiPointerAsync) {
  auto size = std::get<0>(GetParam());
  auto count = std::get<1>(GetParam());
  auto fn = std::get<2>(GetParam());

  spawn(size, [&](std::shared_ptr<Context> context) {
      // Run algorithm
      auto fixture = CudaFixture<float>(context, cudaNumDevices(), count);
      auto ptrs = fixture.getPointers();
      auto streams = fixture.getCudaStreams();
      auto algorithm = fn(context, ptrs, count, streams);
      fixture.assignValuesAsync();
      algorithm->run();

      // Verify result
      fixture.synchronizeCudaStreams();
      assertResult(fixture);
    });
}

TEST_F(CudaAllreduceTest, MultipleAlgorithms) {
  auto size = 4;
  auto count = 1000;
  auto fns = {allreduceRing,
             allreduceRingChunked,
             allreduceHalvingDoubling,
             allreduceHalvingDoublingPipelined};

  spawn(size, [&](std::shared_ptr<Context> context) {
    for (const auto& fn : fns) {
      // Run algorithm
      auto fixture = CudaFixture<float>(context, 1, count);
      auto ptrs = fixture.getPointers();

      auto algorithm = fn(context, ptrs, count, {});
      fixture.assignValues();
      algorithm->run();

      // Verify result
      assertResult(fixture);

      auto algorithm2 = fn(context, ptrs, count, {});
      fixture.assignValues();
      algorithm2->run();

      // Verify result
      assertResult(fixture);
    }
  });
}

TEST_F(CudaAllreduceTestHP, HalfPrecisionTest) {
  auto size = 4;
  auto count = 128;
  auto fns = {allreduceRingHP,
             allreduceRingChunkedHP,
             allreduceHalvingDoublingHP,
             allreduceHalvingDoublingPipelinedHP};
  spawn(size, [&](std::shared_ptr<Context> context) {
      for (const auto& fn : fns) {
        // Run algorithm
        auto fixture = CudaFixture<float16>(context, 1, count);
        auto ptrs = fixture.getPointers();

        auto algorithm = fn(context, ptrs, count, {});
        fixture.assignValues();
        algorithm->run();

        // Verify result
        assertResult(fixture);
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

INSTANTIATE_TEST_CASE_P(
    AllreduceRing,
    CudaAllreduceTest,
    ::testing::Combine(
      ::testing::Range(2, 16),
      ::testing::ValuesIn(genMemorySizes()),
      ::testing::Values(allreduceRing)));

INSTANTIATE_TEST_CASE_P(
    AllreduceRingChunked,
    CudaAllreduceTest,
    ::testing::Combine(
      ::testing::Range(2, 16),
      ::testing::ValuesIn(genMemorySizes()),
      ::testing::Values(allreduceRingChunked)));

INSTANTIATE_TEST_CASE_P(
    AllreduceHalvingDoubling,
    CudaAllreduceTest,
    ::testing::Combine(
        ::testing::ValuesIn(
          std::vector<int>({2, 3, 4, 5, 6, 7, 8, 9, 13, 16, 24, 32})),
        ::testing::ValuesIn(std::vector<int>({1, 64, 1000})),
        ::testing::Values(allreduceHalvingDoubling)));

INSTANTIATE_TEST_CASE_P(
    AllreduceHalvingDoublingPipelined,
    CudaAllreduceTest,
    ::testing::Combine(
        ::testing::ValuesIn(
          std::vector<int>({2, 3, 4, 5, 6, 7, 8, 9, 13, 16, 24, 32})),
        ::testing::ValuesIn(std::vector<int>({1, 64, 1000})),
        ::testing::Values(allreduceHalvingDoublingPipelined)));

} // namespace
} // namespace test
} // namespace gloo
