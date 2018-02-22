/**
 * Copyright (c) 2018-present, Facebook, Inc.
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

#define CUB_STDERR
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/pow_op.h"
#include "caffe2/utils/conversions.h"

namespace caffe2 {

// pow, log and other math functions are defined in
// CUDA math library in header file math.h
#define CUDA_POW(x, y) (pow(x, y))

template <int b_is_scalar, typename T1, typename T2, typename R>
__global__ void PowKernel(const T1* a, const T2* b, T2 e, R* out, int n) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    out[i] = CUDA_POW(a[i], ((b == NULL) ? e : b[b_is_scalar ? 0 : i]));
  }
}
template <typename T1, typename T2, typename R>
__global__ void
PowBroadcastKernel(const T1* a, const T2* b, R* out, int pre, int n) {
  CUDA_1D_KERNEL_LOOP(i, pre * n) {
    out[i] = CUDA_POW(a[i], b[i % n]);
  }
}
template <typename T1, typename T2, typename R>
__global__ void PowBroadcast2Kernel(
    const T1* a,
    const T2* b,
    R* out,
    int pre,
    int n,
    int post) {
  CUDA_1D_KERNEL_LOOP(i, pre * n * post) {
    out[i] = CUDA_POW(a[i], b[(i / post) % n]);
  }
}

struct CudaPowFunctor {
  template <bool b_is_scalar, typename T1, typename T2, typename R>
  inline void
  Run(size_t n, const T1* a, const T2* b, T2 e, R* out, CUDAContext* context) {
    PowKernel<b_is_scalar, T1, T2, R>
        <<<CAFFE_GET_BLOCKS(n),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context->cuda_stream()>>>(a, b, e, out, n);
  }
  template <typename T1, typename T2, typename R>
  void RunWithBroadcast(
      const T1* a,
      const T2* b,
      R* out,
      size_t pre,
      size_t n,
      CUDAContext* context) {
    PowBroadcastKernel<T1, T2, R>
        <<<CAFFE_GET_BLOCKS(pre * n),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context->cuda_stream()>>>(a, b, out, pre, n);
  }
  template <typename T1, typename T2, typename R>
  void RunWithBroadcast2(
      const T1* a,
      const T2* b,
      R* out,
      size_t pre,
      size_t n,
      size_t post,
      CUDAContext* context) {
    PowBroadcast2Kernel<T1, T2, R>
        <<<CAFFE_GET_BLOCKS(pre * n * post),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context->cuda_stream()>>>(a, b, out, pre, n, post);
  }
};
REGISTER_CUDA_OPERATOR(
    Pow,
    PowOp<
        TensorTypes<float> /*NumericTypes*/,
        CUDAContext,
        CudaPowFunctor,
        SameTypeAsInput>)

} // namespace caffe2
