/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "gloo/test/base_test.h"

#include "gloo/cuda_private.h"

namespace gloo {
namespace test {

void cudaSleep(cudaStream_t stream, size_t clocks);

class CudaBaseTest : public BaseTest {
 public:
  int getDeviceCount() {
    int n = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n));
    return n;
  }
};

class Fixture {
 public:
  Fixture(int devices, int count) : count(count) {
    for (int i = 0; i < devices; i++) {
      CudaDeviceScope scope(i);
      srcs.push_back(CudaMemory<float>(count));
      ptrs.push_back(
        CudaDevicePointer<float>::create(*srcs.back(), count));
    }
  }

  Fixture(Fixture&& other) noexcept : count(other.count) {
    srcs = std::move(other.srcs);
    ptrs = std::move(other.ptrs);
  }

  void setRank(int rank) {
    for (int i = 0; i < srcs.size(); i++) {
      const auto& stream = ptrs[i].getStream();
      srcs[i].set((rank * srcs.size()) + i, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }

  void setRankAsync(int rank) {
    for (int i = 0; i < srcs.size(); i++) {
      const auto& stream = ptrs[i].getStream();
      // Insert sleep on stream to force to artificially delay the
      // kernel that actually populates the memory to surface
      // synchronization errors.
      cudaSleep(stream, 100000);
      srcs[i].set((rank * srcs.size()) + i, stream);
    }
  }

  std::vector<float*> getFloatPointers() const {
    std::vector<float*> out;
    for (const auto& src : srcs) {
      out.push_back(*src);
    }
    return out;
  }

  std::vector<cudaStream_t> getCudaStreams() const {
    std::vector<cudaStream_t> out;
    for (const auto& ptr : ptrs) {
      out.push_back(ptr.getStream());
    }
    return out;
  }

  std::vector<std::unique_ptr<float[]> > getHostBuffers() {
    std::vector<std::unique_ptr<float[]> > out;
    for (auto& src : srcs) {
      out.push_back(src.copyToHost());
    }
    return out;
  }

  const int count;
  std::vector<CudaDevicePointer<float> > ptrs;
  std::vector<CudaMemory<float> > srcs;
};

} // namespace test
} // namespace gloo
