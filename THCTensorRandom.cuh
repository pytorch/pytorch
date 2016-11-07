#ifndef THC_TENSOR_RANDOM_CUH
#define THC_TENSOR_RANDOM_CUH

#include "THCNumerics.cuh"
#include "THCReduceApplyUtils.cuh"
#include "THCTensorMathReduce.cuh"

// Normalizes the L1 norm of every row to 1; used by multinomial
template <typename T>
__global__ void renormRowsL1(T* dist, long rows, long cols) {
  extern __shared__ T smem[];

  for (long row = blockIdx.x; row < rows; row += gridDim.x) {
    T sum = ScalarConvert<int, T>::to(0);
    for (long col = threadIdx.x; col < cols; col += blockDim.x) {
      sum = THCNumerics<T>::add(sum, dist[row * cols + col]);
    }

    sum = reduceBlock(smem, blockDim.x, sum, ReduceAdd<T, T>(), ScalarConvert<int, T>::to(0));
    if (threadIdx.x == 0) {
      smem[0] = sum;
    }
    __syncthreads();

    sum = smem[0];
    if (THCNumerics<T>::gt(sum, ScalarConvert<int, T>::to(0))) {
      for (long col = threadIdx.x; col < cols; col += blockDim.x) {
        dist[row * cols + col] = THCNumerics<T>::div(dist[row * cols + col], sum);
      }
    }
  }
}


template <typename T>
__device__ int binarySearchForMultinomial(T* dist,
                                          int size,
                                          T val) {
  int start = 0;
  int end = size;

  while (end - start > 0) {
    int mid = start + (end - start) / 2;

    T midVal = dist[mid];
    if (THCNumerics<T>::lt(midVal, val)) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }

  if (start == size) {
    // No probability mass or precision problems; just return the
    // first element
    start = 0;
  }

  return start;
}

#endif // THC_TENSOR_RANDOM_CUH
