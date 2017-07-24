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

#include "gloo/allgather_ring.h"
#include "gloo/common/common.h"
#include "gloo/test/base_test.h"

namespace gloo {
namespace test {
namespace {

// Test parameterization.
using Param = std::tuple<int, int, int>;

// Test fixture.
class AllgatherTest : public BaseTest,
                      public ::testing::WithParamInterface<Param> {};

TEST_P(AllgatherTest, VarNumPointer) {
  auto contextSize = std::get<0>(GetParam());
  auto dataSize = std::get<1>(GetParam());
  auto numPtrs = std::get<2>(GetParam());

  spawn(contextSize, [&](std::shared_ptr<Context> context) {
    Fixture<float> inFixture(context, numPtrs, dataSize);
    inFixture.assignValues();

    std::unique_ptr<float[]> outPtr =
        gloo::make_unique<float[]>(numPtrs * dataSize * contextSize);

    AllgatherRing<float> algorithm(
        context, inFixture.getPointers(), outPtr.get(), dataSize);

    algorithm.run();

    auto stride = contextSize * numPtrs;
    for (int i = 0; i < contextSize; ++i) {
      auto val = i * numPtrs;
      for (int j = 0; j < dataSize; j++) {
        float exp = j * stride + val;
        for (int k = 0; k < numPtrs; ++k) {
          ASSERT_EQ(
              outPtr.get()[i * dataSize * numPtrs + k * dataSize + j], exp + k)
              << "Mismatch at index [" << i << ", " << j + dataSize << "]";
        }
      }
    }
  });
}

TEST_F(AllgatherTest, MultipleAlgorithms) {
  auto contextSize = 4;
  auto dataSize = 1000;
  auto numPtrs = 8;

  spawn(contextSize, [&](std::shared_ptr<Context> context) {
    Fixture<float> inFixture(context, numPtrs, dataSize);
    inFixture.assignValues();

    std::unique_ptr<float[]> outPtr =
        gloo::make_unique<float[]>(numPtrs * dataSize * contextSize);

    for (int alg = 0; alg < 2; alg++) {
      AllgatherRing<float> algorithm(
          context, inFixture.getPointers(), outPtr.get(), dataSize);
      algorithm.run();

      auto stride = contextSize * numPtrs;
      for (int i = 0; i < contextSize; ++i) {
        auto val = i * numPtrs;
        for (int j = 0; j < dataSize; j++) {
          float exp = j * stride + val;
          for (int k = 0; k < numPtrs; ++k) {
            ASSERT_EQ(
                outPtr.get()[i * dataSize * numPtrs + k * dataSize + j],
                exp + k)
                << "Mismatch at index [" << i << ", " << j + dataSize << "]";
          }
        }
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
    AllgatherRing,
    AllgatherTest,
    ::testing::Combine(
        ::testing::Range(2, 10),
        ::testing::ValuesIn(genMemorySizes()),
        ::testing::Range(1, 4)));

} // namespace
} // namespace test
} // namespace gloo
