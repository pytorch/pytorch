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
#include "gloo/benchmark/runner.h"
#include "gloo/cuda_allreduce_ring.h"
#include "gloo/cuda_allreduce_ring_chunked.h"
#include "gloo/cuda_gpudirect_allreduce_ring.h"
#include "gloo/cuda_private.h"
#include "gloo/common/common.h"
#include "gloo/common/logging.h"

using namespace gloo;
using namespace gloo::benchmark;

namespace {

int cudaNumDevices() {
  int n = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  return n;
}

class CudaBenchmark : public Benchmark {
  using Benchmark::Benchmark;

 public:
  virtual ~CudaBenchmark() {}

 protected:
  virtual std::vector<float*> allocate(int inputs, int elements) override {
    GLOO_ENFORCE_LE(inputs, cudaNumDevices());
    std::vector<float*> ptrs;

    const auto stride = context_->size * inputs;
    for (auto i = 0; i < inputs; i++) {
      CudaDeviceScope scope(i);
      auto cudaMemory = CudaMemory<float>(elements);
      cudaMemory.set((context_->rank * inputs) + i, stride);
      ptrs.push_back(*cudaMemory);
      inputs_.push_back(std::move(cudaMemory));
    }
    return ptrs;
  }

  std::vector<CudaMemory<float> > inputs_;
};

template <class T>
class CudaAllreduceBenchmark : public CudaBenchmark {
  using CudaBenchmark::CudaBenchmark;

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
      auto ptr = input.copyToHost();
      for (int i = 0; i < input.elements; i++) {
        auto offset = i * stride;
        GLOO_ENFORCE_EQ(offset + expected, ptr[i], "Mismatch at index: ", i);
      }
    }
  }
};

} // namespace

int main(int argc, char** argv) {
  auto x = benchmark::parseOptions(argc, argv);

  Runner::BenchmarkFn fn;
  if (x.benchmark == "cuda_allreduce_ring") {
    fn = [&](std::shared_ptr<Context>& context) {
      return gloo::make_unique<
        CudaAllreduceBenchmark<
          CudaAllreduceRing<float>>>(context, x);
    };
  } else if (x.benchmark == "cuda_allreduce_ring_chunked") {
    fn = [&](std::shared_ptr<Context>& context) {
      return gloo::make_unique<
        CudaAllreduceBenchmark<
          CudaAllreduceRingChunked<float>>>(context, x);
    };
  } else if (x.benchmark == "cuda_gpudirect_allreduce_ring") {
    fn = [&](std::shared_ptr<Context>& context) {
      return gloo::make_unique<
        CudaAllreduceBenchmark<
          CudaGPUDirectAllreduceRing<float>>>(context, x);
    };
  }

  if (!fn) {
    GLOO_ENFORCE(false, "Invalid algorithm: ", x.benchmark);
  }

  Runner r(x);
  r.run(fn);
  return 0;
}
