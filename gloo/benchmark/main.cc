/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <memory>

#include "gloo/allreduce_ring.h"
#include "gloo/allreduce_ring_chunked.h"
#include "gloo/barrier_all_to_all.h"
#include "gloo/barrier_all_to_one.h"
#include "gloo/broadcast_one_to_all.h"
#include "gloo/common/common.h"
#include "gloo/common/logging.h"
#include "gloo/context.h"

#include "gloo/benchmark/benchmark.h"
#include "gloo/benchmark/math.h"
#include "gloo/benchmark/runner.h"

using namespace gloo;
using namespace gloo::benchmark;

namespace {

class AllreduceRingBenchmark : public Benchmark {
  using Benchmark::Benchmark;

 public:
  virtual void initialize(int elements) override {
    auto ptr = allocate(elements);
    algorithm_.reset(
        new AllreduceRing<float>(context_, {ptr}, elements, &sum));
  }

  virtual bool verify() override {
    auto expected = (context_->size_ * (context_->size_ - 1)) / 2;
    for (int i = 0; i < data_.size(); i++) {
      GLOO_ENFORCE_EQ(expected, data_[i], "Mismatch at index ", i);
    }
    return true;
  }
};

class AllreduceRingChunkedBenchmark : public Benchmark {
  using Benchmark::Benchmark;

 public:
  virtual void initialize(int elements) override {
    auto ptr = allocate(elements);
    algorithm_.reset(
        new AllreduceRingChunked<float>(context_, {ptr}, elements, &sum));
  }

  virtual bool verify() override {
    auto expected = (context_->size_ * (context_->size_ - 1)) / 2;
    for (int i = 0; i < data_.size(); i++) {
      GLOO_ENFORCE_EQ(expected, data_[i], "Mismatch at index ", i);
    }
    return true;
  }
};

class BarrierAllToAllBenchmark : public Benchmark {
  using Benchmark::Benchmark;

 public:
  virtual void initialize(int /* unused */) override {
    algorithm_.reset(new BarrierAllToAll(context_));
  }

  virtual bool verify() override {
    return true;
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

  virtual bool verify() override {
    return true;
  }
};

class BroadcastOneToAllBenchmark : public Benchmark {
  using Benchmark::Benchmark;

 public:
  virtual void initialize(int elements) override {
    auto ptr = allocate(elements);
    algorithm_.reset(
        new BroadcastOneToAll<float>(context_, ptr, elements, rootRank_));
  }

  virtual bool verify() override {
    for (int i = 0; i < data_.size(); i++) {
      GLOO_ENFORCE_EQ(rootRank_, data_[i], "Mismatch at index ", i);
    }
    return true;
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
      return gloo::make_unique<AllreduceRingBenchmark>(context, x);
    };
  } else if (x.benchmark == "allreduce_ring_chunked") {
    fn = [&](std::shared_ptr<Context>& context) {
      return gloo::make_unique<AllreduceRingChunkedBenchmark>(context, x);
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
