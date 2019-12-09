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

#include "caffe2/operators/unique_ops.h"

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/version.h>
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

#if THRUST_VERSION >= 100800
namespace {
__global__ void remap_kernel(
    thrust::device_ptr<int> second_order,
    thrust::device_ptr<int> order,
    int* output,
    int N,
    int K) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= K)
    return;
  int idx = second_order[i];
  output[order[idx]] = i;
  // Maybe cuda 1D kernel?
  for (idx++; idx < N && (i == K - 1 || idx != second_order[i + 1]); idx++) {
    output[order[idx]] = i;
  }
  return;
}

} // namespace

template <>
template <typename T>
bool UniqueOp<CUDAContext>::DoRunWithType() {
  auto& inputTensor = Input(0);
  // use dim32 to enforce that it's fine to have remapping of type int
  int N = inputTensor.dim32(0);
  CAFFE_ENFORCE_EQ(inputTensor.dim(), 1, "Input should be a vector");

  int* remapping = nullptr;
  if (REMAPPING < OutputSize()) {
    auto* remappingTensor =
        Output(REMAPPING, inputTensor.sizes(), at::dtype<int>());
    remapping = remappingTensor->template mutable_data<int>();
  }

  if (N <= 0) {
    // if the input is empty, we have nothing to do, not even launch kernel.
    /* auto* uniqueTensor = */ Output(UNIQUE, {0}, at::dtype<T>());
    return true;
  }

  const T* input = inputTensor.template data<T>();
  ReinitializeTensor(&thrust_unique_buffer_, {N}, at::dtype<T>().device(CUDA));
  auto* buffer = thrust_unique_buffer_.template mutable_data<T>();
  context_.CopyItemsSameDevice(inputTensor.meta(), N, input, buffer);

  // Create two vectors of {0, 1, ..., N-1} on CUDA device
  thrust::device_vector<int> order1(N), order2(N);
  thrust::sequence(
      thrust::cuda::par.on(context_.cuda_stream()),
      order1.begin(),
      order1.end());
  thrust::sequence(
      thrust::cuda::par.on(context_.cuda_stream()),
      order2.begin(),
      order2.end());

  // Sort the input along with order vector. So now we know where each element
  // is permutated to. For example:
  //    input1 = 1,3,5,1,5,7,9
  //    order1 = 0,1,2,3,4,5,6
  // Now we have:
  //    output = 1,1,3,5,5,7,9
  //    order1 = 0,3,1,2,4,5,6
  thrust::sort_by_key(
      thrust::cuda::par.on(context_.cuda_stream()),
      buffer,
      buffer + N,
      order1.begin());

  // Use consequent unique op to get another order_buffer
  //    input2 = 1,1,3,5,5,7,9
  //    order2 = 0,1,2,3,4,5,6
  // Now we have:
  //    output = 1,3,5,7,9
  //    order2 = 0,2,3,5,6
  auto new_last = thrust::unique_by_key(
      thrust::cuda::par.on(context_.cuda_stream()),
      buffer,
      buffer + N,
      order2.begin());
  int K = new_last.first - buffer;

  auto* uniqueTensor = Output(UNIQUE, {K}, at::dtype<T>());
  T* unique = uniqueTensor->template mutable_data<T>();
  context_.CopyItemsSameDevice(thrust_unique_buffer_.meta(), K, buffer, unique);

  // Compute the remapping. For example, for the number 1, if we look at
  // order2[0] and order2[1], we know that input2[0:2) are all 1. They are all
  // remapped to 0 in final input. And from order1, we know where they come
  // from. The rest is easy.
  if (remapping != nullptr) {
    // record remap
    remap_kernel<<<
        CAFFE_GET_BLOCKS(K),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        order2.data(), order1.data(), remapping, N, K);
  }
  return true;
}

REGISTER_CUDA_OPERATOR(Unique, UniqueOp<CUDAContext>);

#endif // THRUST_VERSION >= 100800
} // namespace caffe2
