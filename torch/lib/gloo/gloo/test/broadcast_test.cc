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

#include "gloo/broadcast_one_to_all.h"
#include "gloo/test/base_test.h"

namespace gloo {
namespace test {
namespace {

// Function to instantiate and run algorithm.
using Func = std::unique_ptr<::gloo::Algorithm>(
    std::shared_ptr<::gloo::Context>&,
    std::vector<float*> ptrs,
    int count,
    int rootRank,
    int rootPointerRank);

// Test parameterization.
using Param = std::tuple<int, int, int, std::function<Func>>;

// Test fixture.
class BroadcastTest : public BaseTest,
                      public ::testing::WithParamInterface<Param> {
 public:
  void assertResult(Fixture<float>& fixture, int root, int rootPointer) {
    // Expected is set to the expected value at ptr[0]
    const auto expected = root * fixture.srcs.size() + rootPointer;
    // Stride is difference between values at subsequent indices
    const auto stride = fixture.srcs.size() * fixture.context->size;
    // Verify all buffers passed by this instance
    for (const auto& ptr : fixture.getPointers()) {
      for (auto i = 0; i < fixture.count; i++) {
        ASSERT_EQ((i * stride) + expected, ptr[i])
          << "Mismatch at index " << i;
      }
    }
  }
};

TEST_P(BroadcastTest, Default) {
  auto processCount = std::get<0>(GetParam());
  auto pointerCount = std::get<1>(GetParam());
  auto elementCount = std::get<2>(GetParam());
  auto fn = std::get<3>(GetParam());

  spawn(processCount, [&](std::shared_ptr<Context> context) {
      auto fixture = Fixture<float>(context, pointerCount, elementCount);
      auto ptrs = fixture.getPointers();

      // Run with varying root
      // TODO(PN): go up to processCount
      for (auto rootProcessRank = 0;
           rootProcessRank < 1;
           rootProcessRank++) {
        // TODO(PN): go up to pointerCount
        for (auto rootPointerRank = 0;
             rootPointerRank < 1;
             rootPointerRank++) {
          fixture.assignValues();
          auto algorithm = fn(context,
                              ptrs,
                              elementCount,
                              rootProcessRank,
                              rootPointerRank);
          algorithm->run();

          // Verify result
          assertResult(fixture, rootProcessRank, rootPointerRank);
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
    std::vector<float*> ptrs,
    int count,
    int rootProcessRank,
    int rootPointerRank) {
  return std::unique_ptr<::gloo::Algorithm>(
    new ::gloo::BroadcastOneToAll<float>(
      context, ptrs, count, rootProcessRank, rootPointerRank));
};

INSTANTIATE_TEST_CASE_P(
    OneToAllBroadcast,
    BroadcastTest,
    ::testing::Combine(
        ::testing::Values(2, 3, 4, 5),
        ::testing::Values(1, 2),
        ::testing::ValuesIn(genMemorySizes()),
        ::testing::Values(broadcastOneToAll)));

} // namespace
} // namespace test
} // namespace gloo
