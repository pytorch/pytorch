/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <memory>

#include "gloo/benchmark/benchmark.h"
#include "gloo/benchmark/math.h"
#include "gloo/benchmark/runner.h"
#include "gloo/cuda_allreduce_ring.h"
#include "gloo/cuda_private.h"
#include "gloo/common/common.h"
#include "gloo/common/logging.h"

using namespace gloo;
using namespace gloo::benchmark;

namespace {

class CudaAllreduceRingBenchmark : public Benchmark {
  using Benchmark::Benchmark;

 public:
  virtual void initialize(int elements) override {
    auto ptr = allocate(elements);
    algorithm_.reset(
        new CudaAllreduceRing<float>(context_, {ptr}, elements));
  }

  virtual bool verify() override {
    auto expected = (context_->size_ * (context_->size_ - 1)) / 2;
    for (int i = 0; i < data_.size(); i++) {
      GLOO_ENFORCE_EQ(expected, data_[i], "Mismatch at index ", i);
    }
    return true;
  }

 protected:
  virtual float* allocate(int elements) override {
    ptrs_.push_back(CudaMemory<float>(elements));
    ptrs_[ptrs_.size()-1].set(context_->rank_);
    return *ptrs_[ptrs_.size()-1];
  }

  std::vector<CudaMemory<float> > ptrs_;
};

} // namespace

int main(int argc, char** argv) {
  auto x = benchmark::parseOptions(argc, argv);

  Runner::BenchmarkFn fn;
  if (x.benchmark == "cuda_allreduce_ring") {
    fn = [&](std::shared_ptr<Context>& context) {
      return gloo::make_unique<CudaAllreduceRingBenchmark>(context, x);
    };
  }

  if (!fn) {
    GLOO_ENFORCE(false, "Invalid algorithm: ", x.benchmark);
  }

  Runner r(x);
  r.run(fn);
  return 0;
}
