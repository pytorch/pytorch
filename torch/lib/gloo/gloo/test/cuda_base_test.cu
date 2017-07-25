/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/test/cuda_base_test.h"

namespace gloo {
namespace test {
namespace {

__global__ void waitClocks(const size_t count) {
  clock_t start = clock();
  clock_t offset = 0;
  while (offset < count) {
    offset = clock() - start;
  }
}

} // namespace

void cudaSleep(cudaStream_t stream, size_t clocks) {
  waitClocks<<<1, 1, 0, stream>>>(clocks);
}

int cudaNumDevices() {
  int n = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  return n;
}

} // namespace test
} // namespace gloo
