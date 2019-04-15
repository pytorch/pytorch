#ifndef THC_TENSOR_MATH_MAGMA_CUH
#define THC_TENSOR_MATH_MAGMA_CUH

#ifdef USE_MAGMA
#include <magma.h>
#else
#include <THC/THCBlas.h>
#endif

#ifdef USE_MAGMA
template <typename T>
static inline T* th_magma_malloc_pinned(size_t n)
{
  void* ptr;
  if (MAGMA_SUCCESS != magma_malloc_pinned(&ptr, n * sizeof(T)))
    THError("$ Torch: not enough memory: you tried to allocate %dGB. Buy new RAM!", n/268435456);
  return reinterpret_cast<T*>(ptr);
}

#endif

#endif // THC_TENSOR_MATH_MAGMA_CUH
