#ifndef THC_SORT_UTILS_INC
#define THC_SORT_UTILS_INC

#include <THC/THCReduceApplyUtils.cuh>
#include <THC/THCTensorTypeUtils.cuh>
#include <THC/THCNumerics.cuh>
#include <c10/macros/Macros.h>

// Collection of kernel sort routines
template <typename T, bool handleNaN = false>
struct LTComp {
  __device__ inline bool operator()(const T& a, const T& b) const {
    return (handleNaN && THCNumerics<T>::isnan(b) && !THCNumerics<T>::isnan(a)) || THCNumerics<T>::lt(a, b);
  }
};

template <typename T, bool handleNaN = false>
struct GTComp {
  __device__ inline bool operator()(const T& a, const T& b) const {
    return (handleNaN && THCNumerics<T>::isnan(a) && !THCNumerics<T>::isnan(b)) || THCNumerics<T>::gt(a, b);
  }
};

template <typename T>
__device__ inline void swapVars(T& t1, T& t2) {
  T tmp = t1;
  t1 = t2;
  t2 = tmp;
}

template <typename Comparator, typename K, typename V>
__device__ inline void bitonicSwap(K& kA, V& vA, bool& validA,
                                   K& kB, V& vB, bool& validB,
                                   bool dir,
                                   const Comparator& comp) {
  // Invalid entries always sort to the end
  bool swap = (comp(kA, kB) && validA) || !validB;
  if (swap == dir) {
    swapVars(kA, kB);
    swapVars(vA, vB);
    swapVars(validA, validB);
  }
};

template <typename Comparator, typename K, typename V,
          typename IndexType, int Power2SortSize>
__device__ inline void bitonicSort(K keys[Power2SortSize],
                                   V values[Power2SortSize],
                                   bool valid[Power2SortSize],
                                   const Comparator& comp) {
#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
  for (unsigned int size = 2; size < Power2SortSize; size *= 2) {
    bool flag = ((threadIdx.x & (size / 2)) != 0);

#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
    for (unsigned int stride = size / 2; stride > 0; stride /= 2) {

      __syncthreads();

      unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      bitonicSwap<Comparator, K, V>(
        keys[pos], values[pos], valid[pos],
        keys[pos + stride], values[pos + stride], valid[pos + stride],
        flag, comp);
    }
  }

#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
  for (unsigned int stride = Power2SortSize / 2; stride > 0; stride /= 2) {

    __syncthreads();

    unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
    bitonicSwap<Comparator, K, V>(
      keys[pos], values[pos], valid[pos],
      keys[pos + stride], values[pos + stride], valid[pos + stride],
      false, comp);
  }

  __syncthreads();

}

uint64_t nextHighestPowerOf2(uint64_t n);

#endif // THC_SORT_UTILS_INC
