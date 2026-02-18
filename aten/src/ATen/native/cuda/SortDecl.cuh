#pragma once
#include <ATen/core/TensorBase.h>
#include <c10/macros/Macros.h>

// For lightweight header file

#if defined(USE_ROCM)
  #ifdef _WIN32
    #include <hip/hip_version.h>
  #else
    #include <rocm-core/rocm_version.h>
  #endif
#else
  #include <cuda_runtime_api.h>
#endif

#if !defined(HAS_WARP_MERGE_SORT)
  #if defined(USE_ROCM)
    #define HAS_WARP_MERGE_SORT() (ROCM_VERSION >= 70000)
  #else
    #define HAS_WARP_MERGE_SORT() (CUDA_VERSION >= 110600)
  #endif
#endif

namespace at::native {

#ifndef AT_NATIVE_CUDA_SORT_IPML_CUH
  struct SmallBitonicSort {};
  struct MediumRadixSort {};

  #if HAS_WARP_MERGE_SORT()
    template<int N> struct WarpMergeSort {};
  #endif
#endif

template <typename T, typename Sorter>
void sortCommon(
    Sorter sorter,
    const TensorBase &key,
    const TensorBase &value,
    int dim,
    bool descending);

#if HAS_WARP_MERGE_SORT()
  #define DECLARE_WARP_SORT_IF_ENABLED(TYPE) \
    extern template void sortCommon<TYPE>(WarpMergeSort<128>, const TensorBase&, const TensorBase&, int, bool);
#else
  #define DECLARE_WARP_SORT_IF_ENABLED(TYPE) // do nothing
#endif

#define DECLARE_SORT_COMMON(TYPE) \
  extern template void sortCommon<TYPE>(SmallBitonicSort, const TensorBase&, const TensorBase&, int, bool); \
  extern template void sortCommon<TYPE>(MediumRadixSort, const TensorBase&, const TensorBase&, int, bool); \
  DECLARE_WARP_SORT_IF_ENABLED(TYPE)

DECLARE_SORT_COMMON(uint8_t)

DECLARE_SORT_COMMON(uint16_t)

DECLARE_SORT_COMMON(uint32_t)

DECLARE_SORT_COMMON(uint64_t)

#undef DECLARE_SORT_COMMON
#undef DECLARE_WARP_SORT_IF_ENABLED
}
