/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef HIP_INCLUDE_HIP_NVCC_DETAIL_HIP_RUNTIME_H
#define HIP_INCLUDE_HIP_NVCC_DETAIL_HIP_RUNTIME_H

#include <cuda_runtime.h>

#include <hip/hip_runtime_api.h>

#define HIP_KERNEL_NAME(...) __VA_ARGS__

typedef int hipLaunchParm ;

#define hipLaunchKernel(kernelName, numblocks, numthreads, memperblock, streamId, ...) \
do {\
kernelName<<<numblocks,numthreads,memperblock,streamId>>>(0, ##__VA_ARGS__);\
} while(0)

#define hipLaunchKernelGGL(kernelName, numblocks, numthreads, memperblock, streamId, ...) \
do {\
kernelName<<<numblocks,numthreads,memperblock,streamId>>>(__VA_ARGS__);\
} while(0)

#define hipReadModeElementType cudaReadModeElementType

#ifdef __CUDA_ARCH__


    // 32-bit Atomics:
#define __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__       (__CUDA_ARCH__ >= 110)
#define __HIP_ARCH_HAS_GLOBAL_FLOAT_ATOMIC_EXCH__   (__CUDA_ARCH__ >= 110)
#define __HIP_ARCH_HAS_SHARED_INT32_ATOMICS__       (__CUDA_ARCH__ >= 120)
#define __HIP_ARCH_HAS_SHARED_FLOAT_ATOMIC_EXCH__   (__CUDA_ARCH__ >= 120)
#define __HIP_ARCH_HAS_FLOAT_ATOMIC_ADD__           (__CUDA_ARCH__ >= 200)

// 64-bit Atomics:
#define __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__       (__CUDA_ARCH__ >= 200)
#define __HIP_ARCH_HAS_SHARED_INT64_ATOMICS__       (__CUDA_ARCH__ >= 120)

// Doubles
#define __HIP_ARCH_HAS_DOUBLES__                    (__CUDA_ARCH__ >= 120)

//warp cross-lane operations:
#define __HIP_ARCH_HAS_WARP_VOTE__                  (__CUDA_ARCH__ >= 120)
#define __HIP_ARCH_HAS_WARP_BALLOT__                (__CUDA_ARCH__ >= 200)
#define __HIP_ARCH_HAS_WARP_SHUFFLE__               (__CUDA_ARCH__ >= 300)
#define __HIP_ARCH_HAS_WARP_FUNNEL_SHIFT__          (__CUDA_ARCH__ >= 350)

//sync
#define __HIP_ARCH_HAS_THREAD_FENCE_SYSTEM__        (__CUDA_ARCH__ >= 200)
#define __HIP_ARCH_HAS_SYNC_THREAD_EXT__            (__CUDA_ARCH__ >= 200)

// misc
#define __HIP_ARCH_HAS_SURFACE_FUNCS__              (__CUDA_ARCH__ >= 200)
#define __HIP_ARCH_HAS_3DGRID__                     (__CUDA_ARCH__ >= 200)
#define __HIP_ARCH_HAS_DYNAMIC_PARALLEL__           (__CUDA_ARCH__ >= 350)

#endif

#ifdef __CUDACC__




#define hipThreadIdx_x threadIdx.x
#define hipThreadIdx_y threadIdx.y
#define hipThreadIdx_z threadIdx.z

#define hipBlockIdx_x  blockIdx.x
#define hipBlockIdx_y  blockIdx.y
#define hipBlockIdx_z  blockIdx.z

#define hipBlockDim_x  blockDim.x
#define hipBlockDim_y  blockDim.y
#define hipBlockDim_z  blockDim.z

#define hipGridDim_x  gridDim.x
#define hipGridDim_y  gridDim.y
#define hipGridDim_z  gridDim.z

#define HIP_SYMBOL(X) X

/**
 * extern __shared__
 */

#define HIP_DYNAMIC_SHARED(type, var) \
    extern __shared__ type var[]; \

#define HIP_DYNAMIC_SHARED_ATTRIBUTE

#ifdef __HIP_DEVICE_COMPILE__
#define abort() {asm("trap;");}
#endif

#endif

#endif
