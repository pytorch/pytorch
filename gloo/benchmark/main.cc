/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <memory>

#include "gloo/allgather_ring.h"
#include "gloo/allreduce_halving_doubling.h"
#include "gloo/allreduce_ring.h"
#include "gloo/allreduce_ring_chunked.h"
#include "gloo/barrier_all_to_all.h"
#include "gloo/barrier_all_to_one.h"
#include "gloo/broadcast_one_to_all.h"
#include "gloo/pairwise_exchange.h"
#include "gloo/common/common.h"
#include "gloo/common/logging.h"
#include "gloo/context.h"
#include "gloo/types.h"

#include "gloo/benchmark/benchmark.h"
#include "gloo/benchmark/runner.h"

using namespace gloo;
using namespace gloo::benchmark;

namespace {

template <typename T>
class AllgatherBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
 public:
  virtual void initialize(int elements) override {
    auto inPtrs = this->allocate(this->options_.inputs, elements);
    outputs_.resize(this->options_.inputs * this->context_->size * elements);
    this->algorithm_.reset(new AllgatherRing<T>(
        this->context_, inPtrs, outputs_.data(), elements));
  }

  virtual void verify() override {
    const auto stride = this->context_->size * this->inputs_.size();
    const auto elements = this->inputs_[0].size();
    for (int rank = 0; rank < this->context_->size; rank++) {
      auto val = rank * this->inputs_.size();
      for (int elem = 0; elem < elements; elem++) {
        T exp(elem * stride + val);
        for (int input = 0; input < this->inputs_.size(); input++) {
          const auto rankOffset = rank * elements * this->inputs_.size();
          const auto inputOffset = input * elements;
          GLOO_ENFORCE_EQ(
            outputs_[rankOffset + inputOffset + elem], exp + T(input),
            "Mismatch at index: [", rank, ", ", input, ", ", elem, "]");
        }
      }
    }
  }

 protected:
  std::vector<T> outputs_;
};

template <class A, typename T>
class AllreduceBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
 public:
  virtual void initialize(int elements) override {
    auto ptrs = this->allocate(this->options_.inputs, elements);
    this->algorithm_.reset(new A(this->context_, ptrs, elements));
  }

  virtual void verify() override {
    // Size is the total number of pointers across the context
    const auto size = this->context_->size * this->inputs_.size();
    // Expected is set to the expected value at ptr[0]
    const auto expected = (size * (size - 1)) / 2;
    // The stride between values at subsequent indices is equal to
    // "size", and we have "size" of them. Therefore, after
    // allreduce, the stride between expected values is "size^2".
    const auto stride = size * size;
    for (const auto& input : this->inputs_) {
      for (int i = 0; i < input.size(); i++) {
        auto offset = i * stride;
        GLOO_ENFORCE_EQ(
            T(offset + expected), input[i], "Mismatch at index: ", i);
      }
    }
  }
};

template <typename T>
class BarrierAllToAllBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
 public:
  virtual void initialize(int /* unused */) override {
    this->algorithm_.reset(new BarrierAllToAll(this->context_));
  }
};

template <typename T>
class BarrierAllToOneBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
 public:
  virtual void initialize(int /* unused */) override {
    // This tool measures at rank=0, so use root=1 for the all to one
    // barrier to measure the end-to-end latency (otherwise we might
    // not account for the send-to-root part of the algorithm).
    this->algorithm_.reset(new BarrierAllToOne(this->context_, 1));
  }
};

template <typename T>
class BroadcastOneToAllBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
 public:
  virtual void initialize(int elements) override {
    auto ptrs = this->allocate(this->options_.inputs, elements);
    this->algorithm_.reset(
        new BroadcastOneToAll<T>(this->context_, ptrs, elements, rootRank_));
  }

  virtual void verify() override {
    const auto stride = this->context_->size * this->inputs_.size();
    for (const auto& input : this->inputs_) {
      for (int i = 0; i < input.size(); i++) {
        auto offset = i * stride;
        GLOO_ENFORCE_EQ(
            T(offset + rootRank_), input[i], "Mismatch at index: ", i);
      }
    }
  }

 protected:
  const int rootRank_ = 0;
};

template <typename T>
class PairwiseExchangeBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
 public:
  virtual void initialize(int elements) override {
    this->algorithm_.reset(new PairwiseExchange(
        this->context_, elements, this->getOptions().destinations));
  }
};

} // namespace

#define RUN_BENCHMARK(T)                                                   \
  Runner::BenchmarkFn<T> fn;                                               \
  if (x.benchmark == "allgather_ring") {                                   \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<AllgatherBenchmark<T>>(context, x);         \
    };                                                                     \
  } else if (x.benchmark == "allreduce_ring") {                            \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<AllreduceBenchmark<AllreduceRing<T>, T>>(   \
          context, x);                                                     \
    };                                                                     \
  } else if (x.benchmark == "allreduce_ring_chunked") {                    \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<                                            \
          AllreduceBenchmark<AllreduceRingChunked<T>, T>>(context, x);     \
    };                                                                     \
  } else if (x.benchmark == "allreduce_halving_doubling") {                \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<                                            \
          AllreduceBenchmark<AllreduceHalvingDoubling<T>, T>>(context, x); \
    };                                                                     \
  } else if (x.benchmark == "barrier_all_to_all") {                        \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<BarrierAllToAllBenchmark<T>>(context, x);   \
    };                                                                     \
  } else if (x.benchmark == "barrier_all_to_one") {                        \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<BarrierAllToOneBenchmark<T>>(context, x);   \
    };                                                                     \
  } else if (x.benchmark == "broadcast_one_to_all") {                      \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<BroadcastOneToAllBenchmark<T>>(context, x); \
    };                                                                     \
  } else if (x.benchmark == "pairwise_exchange") {                         \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<PairwiseExchangeBenchmark<T>>(context, x);  \
    };                                                                     \
  }                                                                        \
  if (!fn) {                                                               \
    GLOO_ENFORCE(false, "Invalid algorithm: ", x.benchmark);               \
  }                                                                        \
  Runner r(x);                                                             \
  r.run(fn);

int main(int argc, char** argv) {
  auto x = benchmark::parseOptions(argc, argv);
  if (x.benchmark == "pairwise_exchange") {
    RUN_BENCHMARK(char);
  } else if (x.halfPrecision) {
    RUN_BENCHMARK(float16);
  } else {
    RUN_BENCHMARK(float);
  }
  return 0;
}
