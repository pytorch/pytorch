/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <memory>

#include "gloo/allreduce_halving_doubling.h"
#include "gloo/allreduce_ring.h"
#include "gloo/allreduce_ring_chunked.h"
#include "gloo/barrier_all_to_all.h"
#include "gloo/barrier_all_to_one.h"
#include "gloo/broadcast_one_to_all.h"
#include "gloo/common/common.h"
#include "gloo/common/logging.h"
#include "gloo/context.h"

#include "gloo/benchmark/benchmark.h"
#include "gloo/benchmark/runner.h"

using namespace gloo;
using namespace gloo::benchmark;

namespace {

template <class T>
class AllreduceBenchmark : public Benchmark {
  using Benchmark::Benchmark;

 public:
  virtual void initialize(int elements) override {
    auto ptrs = allocate(options_.inputs, elements);
    algorithm_.reset(new T(context_, ptrs, elements));
  }

  virtual void verify() override {
    // Size is the total number of pointers across the context
    const auto size = context_->size * inputs_.size();
    // Expected is set to the expected value at ptr[0]
    const auto expected = (size * (size - 1)) / 2;
    // The stride between values at subsequent indices is equal to
    // "size", and we have "size" of them. Therefore, after
    // allreduce, the stride between expected values is "size^2".
    const auto stride = size * size;
    for (const auto& input : inputs_) {
      for (int i = 0; i < input.size(); i++) {
        auto offset = i * stride;
        GLOO_ENFORCE_EQ(offset + expected, input[i], "Mismatch at index: ", i);
      }
    }
  }
};

class BarrierAllToAllBenchmark : public Benchmark {
  using Benchmark::Benchmark;

 public:
  virtual void initialize(int /* unused */) override {
    algorithm_.reset(new BarrierAllToAll(context_));
  }
};

class BarrierAllToOneBenchmark : public Benchmark {
  using Benchmark::Benchmark;

 public:
  virtual void initialize(int /* unused */) override {
    // This tool measures at rank=0, so use root=1 for the all to one
    // barrier to measure the end-to-end latency (otherwise we might
    // not account for the send-to-root part of the algorithm).
    algorithm_.reset(new BarrierAllToOne(context_, 1));
  }
};

class BroadcastOneToAllBenchmark : public Benchmark {
  using Benchmark::Benchmark;

 public:
  virtual void initialize(int elements) override {
    auto ptrs = allocate(options_.inputs, elements);
    algorithm_.reset(
        new BroadcastOneToAll<float>(context_, ptrs, elements, rootRank_));
  }

  virtual void verify() override {
    const auto stride = context_->size * inputs_.size();
    for (const auto& input : inputs_) {
      for (int i = 0; i < input.size(); i++) {
        auto offset = i * stride;
        GLOO_ENFORCE_EQ(offset + rootRank_, input[i], "Mismatch at index: ", i);
      }
    }
  }

 protected:
  const int rootRank_ = 0;
};

} // namespace

int main(int argc, char** argv) {
  auto x = benchmark::parseOptions(argc, argv);

  Runner::BenchmarkFn fn;
  if (x.benchmark == "allreduce_ring") {
    fn = [&](std::shared_ptr<Context>& context) {
      return gloo::make_unique<
        AllreduceBenchmark<
          AllreduceRing<float>>>(context, x);
    };
  } else if (x.benchmark == "allreduce_ring_chunked") {
    fn = [&](std::shared_ptr<Context>& context) {
      return gloo::make_unique<
        AllreduceBenchmark<
          AllreduceRingChunked<float>>>(context, x);
    };
  } else if (x.benchmark == "allreduce_halving_doubling") {
    fn = [&](std::shared_ptr<Context>& context) {
      return gloo::make_unique<
        AllreduceBenchmark<
          AllreduceHalvingDoubling<float>>>(context, x);
    };
  } else if (x.benchmark == "barrier_all_to_all") {
    fn = [&](std::shared_ptr<Context>& context) {
      return gloo::make_unique<BarrierAllToAllBenchmark>(context, x);
    };
  } else if (x.benchmark == "barrier_all_to_one") {
    fn = [&](std::shared_ptr<Context>& context) {
      return gloo::make_unique<BarrierAllToOneBenchmark>(context, x);
    };
  } else if (x.benchmark == "broadcast_one_to_all") {
    fn = [&](std::shared_ptr<Context>& context) {
      return gloo::make_unique<BroadcastOneToAllBenchmark>(context, x);
    };
  }

  if (!fn) {
    GLOO_ENFORCE(false, "Invalid algorithm: ", x.benchmark);
  }

  Runner r(x);
  r.run(fn);
  return 0;
}
