/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "caffe2/intra_op_parallel/numa_all_reduce_op_avx2.h"

#include <immintrin.h>

namespace caffe2 {

namespace intra_op_parallel {

constexpr int SIMD_WIDTH_FP32 = 8;

void stream_copy(float* dst, const float* src, size_t n) {
  size_t n_aligned = n / SIMD_WIDTH_FP32 * SIMD_WIDTH_FP32;
  for (size_t i = 0; i < n_aligned; i += SIMD_WIDTH_FP32) {
    _mm256_stream_ps(dst + i, _mm256_loadu_ps(src + i));
  }
  for (size_t i = n_aligned; i < n; ++i) {
    dst[i] = src[i];
  }
}

void stream_add(float* dst, const float* src, size_t n) {
  size_t n_aligned = n / SIMD_WIDTH_FP32 * SIMD_WIDTH_FP32;
  for (size_t i = 0; i < n_aligned; i += SIMD_WIDTH_FP32) {
    _mm256_stream_ps(
        dst + i,
        _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(src + i)));
  }
  for (size_t i = n_aligned; i < n; ++i) {
    dst[i] += src[i];
  }
}

} // namespace intra_op_parallel

} // namespace caffe2
