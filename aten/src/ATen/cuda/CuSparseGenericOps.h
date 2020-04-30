#pragma once

// guard the whole file
#if !defined(_MSC_VER) && defined(__CUDACC__) && CUDART_VERSION >= 10010 // CUDA release >= 10.1 and not windows

#include <ATen/cuda/ATenCUDAGeneral.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CuSparseDescriptors.h>

#include <cusparse.h>
#include <library_types.h>

// LIMITATION (cusparseSpMM):
// The generic APIs are currently (CUDA 10.1) available for all platforms except
// Windows. Using these APIs in any other systems will result in compile-time or
// run-time failures. Their support will be extended in the next releases.

// This file provides wrappers to cusparse Generic APIs 
// See reference: https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-function-ref

namespace at { namespace cuda { namespace sparse {

// calculates DnMat(C) = alpha * SpMat(A) * DnMat(B) + beta * DnMat(C)
// Prefer to use with at::cuda::sparse:: to distinguish from the native cusparse API
template <typename valueType>
void CuSparseSpMM(
    cusparseOperation_t opa,
    cusparseOperation_t opb,
    valueType alpha,
    valueType beta,
    cusparseSpMatDescr_t descA,
    cusparseDnMatDescr_t descB,
    cusparseDnMatDescr_t descC,
    cusparseSpMMAlg_t alg) {
  auto handle = at::cuda::getCurrentCUDASparseHandle();

  // cusparseSpMM_bufferSize returns the bufferSize that can be used by cusparseSpMM
  size_t bufferSize;
  TORCH_CUDASPARSE_CHECK(cusparseSpMM_bufferSize(
      handle, opa, opb,
      &alpha,
      descA,
      descB,
      &beta,
      descC,
      CuSpValueType<valueType>().type,  /* data type in which the computation is executed */
      alg,                              /* algorithm */
      &bufferSize                       /* output */
      ));

  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(bufferSize);

  TORCH_CUDASPARSE_CHECK(cusparseSpMM(
      handle, opa, opb,
      &alpha,
      descA,
      descB,
      &beta,
      descC,
      CuSpValueType<valueType>().type,  /* data type in which the computation is executed */
      alg,                              /* algorithm */
      dataPtr.get()                     /* external buffer */
      ));
}

} // namespace sparse
} // namespace cuda
} // namespace at

#endif