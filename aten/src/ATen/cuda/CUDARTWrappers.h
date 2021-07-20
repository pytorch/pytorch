#include <ATen/cuda/Exceptions.h>
#ifdef __HIP_PLATFORM_HCC__
#include <hip/hip_version.h>
#endif

#pragma once

namespace at { namespace cuda { namespace  {

void __inline__ memcpy_and_sync(void * dst, void * src, int64_t nbytes, cudaMemcpyKind kind, cudaStream_t stream){
#if HIP_VERSION >= 301
    AT_CUDA_CHECK(hipMemcpyWithStream(dst, src, nbytes, kind, stream));
#else
    AT_CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, kind, stream));
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
}

void __inline__ stream_synchronize(cudaStream_t stream) {
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
}
}}}
