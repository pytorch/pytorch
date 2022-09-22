#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GPUCC
#define FUNCAPI __host__ __device__
#define INLINE __forceinline__
#else
#define FUNCAPI
#define INLINE inline
#endif

#if defined(_WIN32) || defined(_WIN64)
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif
