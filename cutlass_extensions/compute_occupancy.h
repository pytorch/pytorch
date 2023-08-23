/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
#pragma once

#include <cuda_runtime_api.h>

#include "cutlass/device_kernel.h"
#include "src/fastertransformer/utils/cuda_utils.h"

namespace fastertransformer {

template<typename GemmKernel>
inline int compute_occupancy_for_kernel()
{

    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    if (smem_size > (48 << 10)) {
        cudaError_t status =
            cudaFuncSetAttribute(cutlass::Kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        if (status == cudaError::cudaErrorInvalidValue) {
            // Clear the error bit since we can ignore this.
            // This should mean that smem_size > cudaDevAttrMaxSharedMemoryPerBlockOptin. In that case, we return an
            // occupancy of 0. This will cause the heuristic to ignore this configuration.
            status = cudaGetLastError();
            return 0;
        }
        check_cuda_error(status);
    }

    int max_active_blocks = -1;
    check_cuda_error(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, cutlass::Kernel<GemmKernel>, GemmKernel::kThreadCount, smem_size));

    return max_active_blocks;
}

}  // namespace fastertransformer