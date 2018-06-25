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

#include <cub/block/block_reduce.cuh>
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/find_op.h"
#include "hip/hip_runtime.h"

namespace caffe2 {

template <typename T>
__global__ void FindKernel(
    int num_needles, int idx_size, const T* idx, const T* needles, int* out, int missing_value)
{
    int needle_idx = hipBlockIdx_x; // One hip block per needle
    T q            = needles[needle_idx];
    int res        = (-1);
    for(int j = hipThreadIdx_x; j < idx_size; j += CAFFE_HIP_NUM_THREADS)
    {
        if(idx[j] == q)
        {
            res = max(res, j);
        }
    }
    typedef cub::BlockReduce<int, CAFFE_HIP_NUM_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int min_res = BlockReduce(temp_storage).Reduce(res, cub::Max());
    if(hipThreadIdx_x == 0)
    {
        out[needle_idx] = min_res == (-1) ? missing_value : min_res;
    }
}

template <>
template <typename T>
bool FindOp<HIPContext>::DoRunWithType()
{
    auto& idx         = Input(0);
    auto& needles     = Input(1);
    auto* res_indices = Output(0);
    res_indices->ResizeLike(needles);

    const T* idx_data     = idx.data<T>();
    const T* needles_data = needles.data<T>();
    int* res_data         = res_indices->mutable_data<int>();

    hipLaunchKernelGGL((FindKernel<T>),
                       needles.size(),
                       CAFFE_HIP_NUM_THREADS,
                       0,
                       context_.hip_stream(),
                       static_cast<int>(needles.size()),
                       static_cast<int>(idx.size()),
                       idx_data,
                       needles_data,
                       res_data,
                       missing_value_);
    return true;
}

REGISTER_HIP_OPERATOR(Find, FindOp<HIPContext>)

} // namespace caffe2
