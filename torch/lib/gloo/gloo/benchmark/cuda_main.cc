/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <map>
#include <memory>

#include "gloo/benchmark/benchmark.h"
#include "gloo/benchmark/runner.h"
#include "gloo/common/logging.h"
#include "gloo/cuda_allreduce_halving_doubling.h"
#include "gloo/cuda_allreduce_halving_doubling_pipelined.h"
#include "gloo/cuda_allreduce_ring.h"
#include "gloo/cuda_allreduce_ring_chunked.h"
#include "gloo/cuda_broadcast_one_to_all.h"
#include "gloo/cuda_private.h"

using namespace gloo;
using namespace gloo::benchmark;

namespace {

int cudaNumDevices() {
  int n = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  return n;
}

template <typename T>
class CudaBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
 public:
  virtual ~CudaBenchmark() {}

 protected:
  virtual std::vector<T*> allocate(int inputs, int elements) override {
    GLOO_ENFORCE_LE(inputs, cudaNumDevices());
    std::vector<T*> ptrs;

    const auto stride = this->context_->size * inputs;
    for (auto i = 0; i < inputs; i++) {
      CudaDeviceScope scope(i);
      auto cudaMemory = CudaMemory<T>(elements);
      cudaMemory.set((this->context_->rank * inputs) + i, stride);
      ptrs.push_back(*cudaMemory);
      inputs_.push_back(std::move(cudaMemory));
    }
    return ptrs;
  }

  std::vector<CudaMemory<T>> inputs_;
};

template <class A, typename T>
class CudaAllreduceBenchmark : public CudaBenchmark<T> {
  using CudaBenchmark<T>::CudaBenchmark;
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
      auto ptr = input.copyToHost();
      for (int i = 0; i < input.elements; i++) {
        auto offset = i * stride;
        GLOO_ENFORCE_EQ(T(offset + expected), ptr[i], "Mismatch at index: ", i);
      }
    }
  }
};

template <class A, typename T>
class CudaBroadcastOneToAllBenchmark : public CudaBenchmark<T> {
  using CudaBenchmark<T>::CudaBenchmark;
 public:
  virtual void initialize(int elements) override {
    auto ptrs = this->allocate(this->options_.inputs, elements);
    this->algorithm_.reset(new A(this->context_, ptrs, elements));
  }

  virtual void verify() override {
    const auto rootOffset = rootRank_ * this->inputs_.size() + rootPointerRank_;
    const auto stride = this->context_->size * this->inputs_.size();
    for (const auto& input : this->inputs_) {
      auto ptr = input.copyToHost();
      for (int i = 0; i < input.elements; i++) {
        auto offset = i * stride;
        GLOO_ENFORCE_EQ(
            T(rootOffset + offset), ptr[i], "Mismatch at index: ", i);
      }
    }
  }

 protected:
  const int rootRank_ = 0;
  const int rootPointerRank_ = 0;
};

} // namespace

#define RUN_BENCHMARK(T)                                                    \
  std::map<std::string, Runner::BenchmarkFn<T>> hostBenchmarks = {          \
      {                                                                     \
          "cuda_allreduce_halving_doubling",                                \
          [&](std::shared_ptr<Context>& context) {                          \
            using Algorithm =                                               \
                CudaAllreduceHalvingDoubling<T, CudaHostWorkspace<T>>;      \
            using Benchmark = CudaAllreduceBenchmark<Algorithm, T>;         \
            return gloo::make_unique<Benchmark>(context, x);                \
          },                                                                \
      },                                                                    \
      {                                                                     \
          "cuda_allreduce_halving_doubling_pipelined",                      \
          [&](std::shared_ptr<Context>& context) {                          \
            using Algorithm = CudaAllreduceHalvingDoublingPipelined<        \
                T,                                                          \
                CudaHostWorkspace<T>>;                                      \
            using Benchmark = CudaAllreduceBenchmark<Algorithm, T>;         \
            return gloo::make_unique<Benchmark>(context, x);                \
          },                                                                \
      },                                                                    \
      {                                                                     \
          "cuda_allreduce_ring",                                            \
          [&](std::shared_ptr<Context>& context) {                          \
            using Algorithm = CudaAllreduceRing<T, CudaHostWorkspace<T>>;   \
            using Benchmark = CudaAllreduceBenchmark<Algorithm, T>;         \
            return gloo::make_unique<Benchmark>(context, x);                \
          },                                                                \
      },                                                                    \
      {                                                                     \
          "cuda_allreduce_ring_chunked",                                    \
          [&](std::shared_ptr<Context>& context) {                          \
            using Algorithm = CudaAllreduceRingChunked<T>;                  \
            using Benchmark = CudaAllreduceBenchmark<Algorithm, T>;         \
            return gloo::make_unique<Benchmark>(context, x);                \
          },                                                                \
      },                                                                    \
      {                                                                     \
          "cuda_broadcast_one_to_all",                                      \
          [&](std::shared_ptr<Context>& context) {                          \
            using Algorithm = CudaBroadcastOneToAll<T>;                     \
            using Benchmark = CudaBroadcastOneToAllBenchmark<Algorithm, T>; \
            return gloo::make_unique<Benchmark>(context, x);                \
          },                                                                \
      },                                                                    \
  };                                                                        \
                                                                            \
  std::map<std::string, Runner::BenchmarkFn<T>> deviceBenchmarks = {        \
      {                                                                     \
          "cuda_allreduce_halving_doubling",                                \
          [&](std::shared_ptr<Context>& context) {                          \
            using Algorithm =                                               \
                CudaAllreduceHalvingDoubling<T, CudaDeviceWorkspace<T>>;    \
            using Benchmark = CudaAllreduceBenchmark<Algorithm, T>;         \
            return gloo::make_unique<Benchmark>(context, x);                \
          },                                                                \
      },                                                                    \
      {                                                                     \
          "cuda_allreduce_halving_doubling_pipelined",                      \
          [&](std::shared_ptr<Context>& context) {                          \
            using Algorithm = CudaAllreduceHalvingDoublingPipelined<        \
                T,                                                          \
                CudaDeviceWorkspace<T>>;                                    \
            using Benchmark = CudaAllreduceBenchmark<Algorithm, T>;         \
            return gloo::make_unique<Benchmark>(context, x);                \
          },                                                                \
      },                                                                    \
      {                                                                     \
          "cuda_allreduce_ring",                                            \
          [&](std::shared_ptr<Context>& context) {                          \
            using Algorithm = CudaAllreduceRing<T, CudaDeviceWorkspace<T>>; \
            using Benchmark = CudaAllreduceBenchmark<Algorithm, T>;         \
            return gloo::make_unique<Benchmark>(context, x);                \
          },                                                                \
      },                                                                    \
      {                                                                     \
          "cuda_allreduce_ring_chunked",                                    \
          [&](std::shared_ptr<Context>& context) {                          \
            using Algorithm =                                               \
                CudaAllreduceRingChunked<T, CudaDeviceWorkspace<T>>;        \
            using Benchmark = CudaAllreduceBenchmark<Algorithm, T>;         \
            return gloo::make_unique<Benchmark>(context, x);                \
          },                                                                \
      },                                                                    \
  };                                                                        \
                                                                            \
  Runner::BenchmarkFn<T> fn;                                                \
  if (x.gpuDirect) {                                                        \
    auto it = deviceBenchmarks.find(x.benchmark);                           \
    if (it != deviceBenchmarks.end()) {                                     \
      fn = it->second;                                                      \
    }                                                                       \
  } else {                                                                  \
    auto it = hostBenchmarks.find(x.benchmark);                             \
    if (it != hostBenchmarks.end()) {                                       \
      fn = it->second;                                                      \
    }                                                                       \
  }                                                                         \
                                                                            \
  if (!fn) {                                                                \
    GLOO_ENFORCE(false, "Invalid algorithm: ", x.benchmark);                \
  }                                                                         \
                                                                            \
  Runner r(x);                                                              \
  r.run(fn);

int main(int argc, char** argv) {
  auto x = benchmark::parseOptions(argc, argv);

  if (x.halfPrecision) {
    RUN_BENCHMARK(float16);
  } else {
    RUN_BENCHMARK(float);
  }
  return 0;
}
