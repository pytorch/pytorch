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
// Temporarily disable __restrict on Windows,
// as it turns out not all MSVC versions are aware of it.
// #define RESTRICT __restrict
#define RESTRICT
#else
#define RESTRICT __restrict__
#endif
