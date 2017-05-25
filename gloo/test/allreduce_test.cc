/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <stdlib.h>

#include <functional>
#include <thread>
#include <vector>

#include "gloo/allreduce_ring.h"
#include "gloo/allreduce_ring_chunked.h"
#include "gloo/allreduce_halving_doubling.h"
#include "gloo/test/base_test.h"

namespace gloo {
namespace test {
namespace {

template <typename T>
struct AlignedBufferReleaser {
  void operator()(T* buffer) {
    if (buffer != nullptr) {
      free(buffer);
    }
  }
};

// RAII handle for aligned buffer
template <typename T>
using BufferHandle = std::unique_ptr<T, AlignedBufferReleaser<T>>;

// Function to instantiate and run algorithm.
using Func = void(
    std::shared_ptr<::gloo::Context>,
    std::vector<float*> dataPtrs,
    int dataSize);

using Func16 = void(
    std::shared_ptr<::gloo::Context>,
    std::vector<float16*> dataPtrs,
    int dataSize);

// Test parameterization.
using Param = std::tuple<int, int, std::function<Func>>;
using ParamHP = std::tuple<int, int, std::function<Func16>>;

template <typename Algorithm>
class AllreduceConstructorTest : public BaseTest {
};

typedef ::testing::Types<
  AllreduceRing<float>,
  AllreduceRingChunked<float> > AllreduceTypes;
TYPED_TEST_CASE(AllreduceConstructorTest, AllreduceTypes);

TYPED_TEST(AllreduceConstructorTest, InlinePointers) {
  this->spawn(2, [&](std::shared_ptr<Context> context) {
      float f = 1.0f;
      TypeParam algorithm(
        context,
        {&f},
        1);
    });
}

TYPED_TEST(AllreduceConstructorTest, SpecifyReductionFunction) {
  this->spawn(2, [&](std::shared_ptr<Context> context) {
      float f = 1.0f;
      std::vector<float*> ptrs = {&f};
      TypeParam algorithm(
        context,
        ptrs,
        ptrs.size(),
        ReductionFunction<float>::product);
    });
}

static std::function<Func> allreduceRing = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float*> dataPtrs,
    int dataSize) {
  ::gloo::AllreduceRing<float> algorithm(context, dataPtrs, dataSize);
  algorithm.run();
};

static std::function<Func16> allreduceRingHP = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float16*> dataPtrs,
    int dataSize) {
  ::gloo::AllreduceRing<float16> algorithm(context, dataPtrs, dataSize);
  algorithm.run();
};

static std::function<Func> allreduceRingChunked = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float*> dataPtrs,
    int dataSize) {
  ::gloo::AllreduceRingChunked<float> algorithm(
      context, dataPtrs, dataSize);
  algorithm.run();
};

static std::function<Func16> allreduceRingChunkedHP = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float16*> dataPtrs,
    int dataSize) {
  ::gloo::AllreduceRingChunked<float16> algorithm(
      context, dataPtrs, dataSize);
  algorithm.run();
};

static std::function<Func> allreduceHalvingDoubling = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float*> dataPtrs,
    int dataSize) {
  ::gloo::AllreduceHalvingDoubling<float> algorithm(
      context, dataPtrs, dataSize);
  algorithm.run();
};

static std::function<Func16> allreduceHalvingDoublingHP = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float16*> dataPtrs,
    int dataSize) {
  ::gloo::AllreduceHalvingDoubling<float16> algorithm(
      context, dataPtrs, dataSize);
  algorithm.run();
};

// Test fixture.
class AllreduceTest : public BaseTest,
                      public ::testing::WithParamInterface<Param> {};

class AllreduceTestHP : public BaseTest,
                        public ::testing::WithParamInterface<ParamHP> {};

TEST_P(AllreduceTest, SinglePointer) {
  auto contextSize = std::get<0>(GetParam());
  auto dataSize = std::get<1>(GetParam());
  auto fn = std::get<2>(GetParam());

  spawnThreads(contextSize, [&](int contextRank) {
    auto context =
      std::make_shared<::gloo::rendezvous::Context>(contextRank, contextSize);
    context->connectFullMesh(*store_, device_);

    BufferHandle<float> buffer{
        (float*)aligned_alloc(kBufferAlignment, dataSize * sizeof(float))};

    float* ptr = buffer.get();
    for (int i = 0; i < dataSize; i++) {
      ptr[i] = contextRank;
    }

    fn(context, std::vector<float*>{ptr}, dataSize);

    auto expected = (contextSize * (contextSize - 1)) / 2;
    for (int i = 0; i < dataSize; i++) {
      ASSERT_EQ(expected, ptr[i]) << "Mismatch at index " << i;
    }
  });
}

TEST_F(AllreduceTest, MultipleAlgorithms) {
  auto contextSize = 4;
  auto dataSize = 1000;
  auto fns = {allreduceRing, allreduceRingChunked, allreduceHalvingDoubling};

  spawnThreads(contextSize, [&](int contextRank) {
    auto context =
        std::make_shared<::gloo::rendezvous::Context>(contextRank, contextSize);
    context->connectFullMesh(*store_, device_);

    BufferHandle<float> buffer{
        (float*)aligned_alloc(kBufferAlignment, dataSize * sizeof(float))};

    float* ptr = buffer.get();
    for (const auto& fn : fns) {
      for (int i = 0; i < dataSize; i++) {
        ptr[i] = contextRank;
      }

      fn(context, std::vector<float*>{ptr}, dataSize);

      auto expected = (contextSize * (contextSize - 1)) / 2;
      for (int i = 0; i < dataSize; i++) {
        ASSERT_EQ(expected, ptr[i]) << "Mismatch at index " << i;
      }

      for (int i = 0; i < dataSize; i++) {
        ptr[i] = contextRank;
      }

      fn(context, std::vector<float*>{ptr}, dataSize);

      expected = (contextSize * (contextSize - 1)) / 2;
      for (int i = 0; i < dataSize; i++) {
        ASSERT_EQ(expected, ptr[i]) << "Mismatch at index " << i;
      }
    }
  });
}

TEST_F(AllreduceTestHP, HalfPrecisionTest) {
  int contextSize = 4;
  auto dataSize = 1024;
  auto fns = {
      allreduceRingHP, allreduceRingChunkedHP, allreduceHalvingDoublingHP};

  spawnThreads(contextSize, [&](int contextRank) {
    auto context =
        std::make_shared<::gloo::rendezvous::Context>(contextRank, contextSize);
    context->connectFullMesh(*store_, device_);

    BufferHandle<float16> buffer{
        (float16*)aligned_alloc(kBufferAlignment, dataSize * sizeof(float16))};

    float16* ptr = buffer.get();
    for (const auto& fn : fns) {
      for (int i = 0; i < dataSize; i++) {
        ptr[i] = contextRank;
      }

      fn(context, std::vector<float16*>{ptr}, dataSize);

      float16 expected(contextSize * (contextSize - 1) / 2);
      for (int i = 0; i < dataSize; i++) {
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

INSTANTIATE_TEST_CASE_P(
    AllreduceRing,
    AllreduceTest,
    ::testing::Combine(
        ::testing::Range(2, 16),
        ::testing::ValuesIn(genMemorySizes()),
        ::testing::Values(allreduceRing)));

INSTANTIATE_TEST_CASE_P(
    AllreduceRingChunked,
    AllreduceTest,
    ::testing::Combine(
        ::testing::Range(2, 16),
        ::testing::ValuesIn(genMemorySizes()),
        ::testing::Values(allreduceRingChunked)));

INSTANTIATE_TEST_CASE_P(
    AllreduceHalvingDoubling,
    AllreduceTest,
    ::testing::Combine(
        ::testing::ValuesIn(
          std::vector<int>({2, 3, 4, 5, 6, 7, 8, 9, 13, 16, 24, 32})),
        ::testing::ValuesIn(std::vector<int>({1, 64, 1000})),
        ::testing::Values(allreduceHalvingDoubling)));

} // namespace
} // namespace test
} // namespace gloo
