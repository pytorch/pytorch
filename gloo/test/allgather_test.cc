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
#include "gloo/test/base_test.h"

namespace gloo {
namespace test {
namespace {

// Test parameterization.
using Param = std::tuple<int, int>;

// Test fixture.
class AllgatherTest : public BaseTest,
                      public ::testing::WithParamInterface<Param> {};

TEST_P(AllgatherTest, TwoPointer) {
  auto contextSize = std::get<0>(GetParam());
  auto dataSize = std::get<1>(GetParam());

  spawn(contextSize, [&](std::shared_ptr<Context> context) {

    Fixture inFixture(context, 2, dataSize);
    inFixture.assignValues();

    Fixture outFixture(context, contextSize, 2 * dataSize);

    AllgatherRing<float> algorithm(
        context,
        inFixture.getFloatPointers(),
        outFixture.getFloatPointers(),
        dataSize);

    algorithm.run();

    auto stride = contextSize * 2;
    for (int i = 0; i < contextSize; ++i) {
      auto val = i * 2;
      for (int j = 0; j < dataSize; j++) {
        float exp = j * stride + val;
        ASSERT_EQ(outFixture.getFloatPointers()[i][j], exp)
            << "Mismatch at index [" << i << ", " << j << "]";
        ASSERT_EQ(outFixture.getFloatPointers()[i][j + dataSize], exp + 1)
            << "Mismatch at index [" << i << ", " << j + dataSize << "]";
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
        ::testing::Range(2, 16),
        ::testing::ValuesIn(genMemorySizes())));

} // namespace
} // namespace test
} // namespace gloo
