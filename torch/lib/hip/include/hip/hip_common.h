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

#ifndef HIP_INCLUDE_HIP_HIP_COMMON_H
#define HIP_INCLUDE_HIP_HIP_COMMON_H

// Common code included at start of every hip file.
// Auto enable __HIP_PLATFORM_HCC__ if compiling with HCC
// Other compiler (GCC,ICC,etc) need to set one of these macros explicitly
#if defined(__HCC__)
#define __HIP_PLATFORM_HCC__
#endif //__HCC__

// Auto enable __HIP_PLATFORM_NVCC__ if compiling with NVCC
#if defined(__NVCC__)
#define __HIP_PLATFORM_NVCC__
#ifdef __CUDACC__
#define __HIPCC__
#endif

#endif //__NVCC__

// Auto enable __HIP_DEVICE_COMPILE__ if compiled in HCC or NVCC device path
#if (defined(__HCC_ACCELERATOR__) && __HCC_ACCELERATOR__ != 0) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 0)
  #define __HIP_DEVICE_COMPILE__ 1
#endif

#if __HIP_DEVICE_COMPILE__ == 0
// 32-bit Atomics
#define __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__       (0)
#define __HIP_ARCH_HAS_GLOBAL_FLOAT_ATOMIC_EXCH__   (0)
#define __HIP_ARCH_HAS_SHARED_INT32_ATOMICS__       (0)
#define __HIP_ARCH_HAS_SHARED_FLOAT_ATOMIC_EXCH__   (0)
#define __HIP_ARCH_HAS_FLOAT_ATOMIC_ADD__           (0)

// 64-bit Atomics
#define __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__       (0)
#define __HIP_ARCH_HAS_SHARED_INT64_ATOMICS__       (0)

// Doubles
#define __HIP_ARCH_HAS_DOUBLES__                    (0)

// Warp cross-lane operations
#define __HIP_ARCH_HAS_WARP_VOTE__                  (0)
#define __HIP_ARCH_HAS_WARP_BALLOT__                (0)
#define __HIP_ARCH_HAS_WARP_SHUFFLE__               (0)
#define __HIP_ARCH_HAS_WARP_FUNNEL_SHIFT__          (0)

// Sync
#define __HIP_ARCH_HAS_THREAD_FENCE_SYSTEM__        (0)
#define __HIP_ARCH_HAS_SYNC_THREAD_EXT__            (0)

// Misc
#define __HIP_ARCH_HAS_SURFACE_FUNCS__              (0)
#define __HIP_ARCH_HAS_3DGRID__                     (0)
#define __HIP_ARCH_HAS_DYNAMIC_PARALLEL__           (0)
#endif

#endif
