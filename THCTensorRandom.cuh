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
__global__ void
sampleMultinomialOnce(T* dest,
                      long distributions,
                      int categories,
                      T* dist) {
  extern __shared__ T smem[];
  T zero = ScalarConvert<int, T>::to(0);

  for (long curDist = blockIdx.x;
       curDist < distributions; curDist += gridDim.x) {
    // Each block handles one distribution
    // First pass, find the total sum of the distribution
    T sum = zero;
    for (int cat = threadIdx.x; cat < categories; cat += blockDim.x) {
      sum += dist[curDist * categories + cat];
    }

    // threadIdx.x == 0 has the sum value from this
    sum = reduceBlock(smem, blockDim.x, sum, ReduceAdd<T, T>(), zero);

    // Broadcast sum and sample value
    if (threadIdx.x == 0) {
      smem[0] = sum;
      smem[1] = dest[curDist];
    }
    __syncthreads();

    sum = smem[0];
    T sample = smem[1];
    __syncthreads();

    if (sum == zero || sample == zero) {
      // Choose the first element
      if (threadIdx.x == 0) {
        dest[curDist] = 1;
      }

      continue;
    }

    int chunks = THCCeilDiv(categories, (int) blockDim.x);
    T prevHighProb = zero;

    for (int chunk = 0; chunk < chunks; ++chunk) {
      // All threads in bounds load a value
      int cat = chunk * blockDim.x + threadIdx.x;

      T val =
        cat < categories ? THCNumerics<T>::div(dist[curDist * categories + cat], sum) : 
        zero;

      smem[threadIdx.x] = val;
      __syncthreads();

      // Perform an inclusive prefix sum of the shared memory contents
      for (int offset = 1; offset < blockDim.x; offset *= 2) {
        T val = zero;

        if (threadIdx.x >= offset) {
          val = smem[threadIdx.x - offset] + smem[threadIdx.x];
        }

        __syncthreads();
        if (threadIdx.x >= offset) {
          smem[threadIdx.x] = val;
        }
        __syncthreads();
      }

      // Each thread will check to see if the sample falls in its
      // bucket
      T curBucket =
        smem[threadIdx.x] + prevHighProb;
      T prevBucket =
        threadIdx.x == 0 ? prevHighProb : smem[threadIdx.x - 1] + prevHighProb;
      bool inBucket =
        (cat < categories) && (sample <= curBucket) && (sample > prevBucket);

      if (inBucket) {
        // We're done; we have the sample
        // Torch indices are 1-based
        // FIXME: broadcast exit flag?
        dest[curDist] = cat + TH_INDEX_BASE;
      }

      // Store the previous scan's high value for future use
      prevHighProb += smem[blockDim.x - 1];

      __syncthreads();
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
