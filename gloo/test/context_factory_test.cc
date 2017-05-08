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

#include "gloo/barrier_all_to_all.h"
#include "gloo/rendezvous/context.h"
#include "gloo/test/base_test.h"

namespace gloo {
namespace test {
namespace {

// Function to instantiate and run algorithm
using Func = void(std::shared_ptr<::gloo::Context>);

// Test parameterization
using Param = std::tuple<int, int, std::function<Func>>;

// Test fixture.
class ContextStoreTest : public BaseTest,
                         public ::testing::WithParamInterface<Param> {};

TEST_P(ContextStoreTest, RunAlgo) {
  auto contextSize = std::get<0>(GetParam());
  auto repeatCount = std::get<1>(GetParam());
  auto fn = std::get<2>(GetParam());

  spawn(contextSize, [&](std::shared_ptr<Context> context) {
      auto factory = std::make_shared<::gloo::rendezvous::ContextFactory>(
        context);
      for (int i = 0; i < repeatCount; ++i) {
        auto usingContext = factory->makeContext(device_);
        fn(usingContext);
      }
    });
}

static std::function<Func> barrierAllToAll =
  [](std::shared_ptr<::gloo::Context> context) {
  ::gloo::BarrierAllToAll algorithm(context);
  algorithm.run();
};

INSTANTIATE_TEST_CASE_P(
  BarrierAllToAll,
  ContextStoreTest,
  ::testing::Combine(
    ::testing::Range(2, 4),
    ::testing::Values(10),
    ::testing::Values(barrierAllToAll)));
} // namespace
} // namespace test
} // namespace gloo
