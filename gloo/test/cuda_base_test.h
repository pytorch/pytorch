/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "gloo/cuda_private.h"
#include "gloo/test/base_test.h"


namespace gloo {
namespace test {

void cudaSleep(cudaStream_t stream, size_t clocks);

int cudaNumDevices();

class CudaBaseTest : public BaseTest {};

template <typename T>
class CudaFixture {
 public:
  CudaFixture(const std::shared_ptr<Context> context, int devices, int count)
      : context(context),
        count(count) {
    for (int i = 0; i < devices; i++) {
      CudaDeviceScope scope(i);
      srcs.push_back(CudaMemory<T>(count));
      ptrs.push_back(
        CudaDevicePointer<T>::create(*srcs.back(), count));
      streams.push_back(CudaStream(i));
    }
  }

  CudaFixture(CudaFixture&& other) noexcept
    : context(other.context),
      count(other.count) {
    srcs = std::move(other.srcs);
    ptrs = std::move(other.ptrs);
  }

  void assignValues() {
    const auto stride = context->size * srcs.size();
    for (int i = 0; i < srcs.size(); i++) {
      srcs[i].set((context->rank * srcs.size()) + i, stride, *streams[i]);
      CUDA_CHECK(cudaStreamSynchronize(*streams[i]));
    }
  }

  void assignValuesAsync() {
    const auto stride = context->size * srcs.size();
    for (int i = 0; i < srcs.size(); i++) {
      // Insert sleep on stream to force to artificially delay the
      // kernel that actually populates the memory to surface
      // synchronization errors.
      cudaSleep(*streams[i], 100000);
      srcs[i].set((context->rank * srcs.size()) + i, stride, *streams[i]);
    }
  }

  std::vector<T*> getPointers() const {
    std::vector<T*> out;
    for (const auto& src : srcs) {
      out.push_back(*src);
    }
    return out;
  }

  std::vector<cudaStream_t> getCudaStreams() const {
    std::vector<cudaStream_t> out;
    for (const auto& stream : streams) {
      out.push_back(stream.getStream());
    }
    return out;
  }

  std::vector<std::unique_ptr<T[]> > getHostBuffers() {
    std::vector<std::unique_ptr<T[]> > out;
    for (auto& src : srcs) {
      out.push_back(src.copyToHost());
    }
    return out;
  }

  void synchronizeCudaStreams() {
    for (const auto& stream : streams) {
      CudaDeviceScope scope(stream.getDeviceID());
      CUDA_CHECK(cudaStreamSynchronize(stream.getStream()));
    }
  }

  std::shared_ptr<Context> context;
  const int count;
  std::vector<CudaDevicePointer<T> > ptrs;
  std::vector<CudaStream> streams;
  std::vector<CudaMemory<T> > srcs;
};

} // namespace test
} // namespace gloo
