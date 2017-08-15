#include "hip/hip_runtime.h"
#ifndef THC_TENSOR_RANDOM_CUH
#define THC_TENSOR_RANDOM_CUH

#include "THCNumerics.cuh"
#include "THCReduceApplyUtils.cuh"
#include "THCTensorMathReduce.cuh"

#include <curand_kernel.h>

#define MAX_NUM_BLOCKS 64
#define BLOCK_SIZE 256
/* Separate kernel because curand_log_normal gets extra parameters. */

template <typename T>
__global__ void generateLogNormal(curandStateMtgp32 *state, int size, T *result, double mean, double stddev)
{
  int idx = hipBlockIdx_x * BLOCK_SIZE + hipThreadIdx_x;
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {
    float x = curand_log_normal(&state[hipBlockIdx_x], mean, stddev);
    if (i < size) {
      result[i] = ScalarConvert<float, T>::to(x);
    }
  }
}

template <>
__global__ void generateLogNormal<double>(curandStateMtgp32 *state, int size, double *result, double mean, double stddev)
{
  int idx = hipBlockIdx_x * BLOCK_SIZE + hipThreadIdx_x;
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {
    double x = curand_log_normal_double(&state[hipBlockIdx_x], mean, stddev);
    if (i < size) {
      result[i] = x;
    }
  }
}

template <typename T>
__global__ void
multinomialAliasDrawKernel(int size, long *output, long *J, T *q, long K,  T *uniform, T *bernoulli){
  long idx = hipBlockIdx_x * BLOCK_SIZE + hipThreadIdx_x;
  if (idx < size) {
    long rand_ind = ScalarConvert<T, long>::to(uniform[idx]);
    T bern_uniform = bernoulli[idx];
    int _mask = (int)THCNumerics<T>::lt(bern_uniform, q[rand_ind]);
    output[idx] = J[rand_ind]*(1 -_mask) + (rand_ind+1L) * _mask;
  }  
}

template <typename T>
__global__ void
aliasMultinomialFilter(T *q, T *probs, long *smaller, long *larger, long *J_data, long *larger_short_data, long *smaller_short_data, T one, long inputsize){
  long idx = hipBlockIdx_x * BLOCK_SIZE + hipThreadIdx_x;
  if (idx < inputsize) {
    larger_short_data[idx] = 0;
    smaller_short_data[idx] = 0;
    J_data[idx]= 0;
    T val = THCNumerics<T>::mul(probs[idx], ScalarConvert<long, T>::to(inputsize));
    if (THCNumerics<T>::lt(val, one)) {
      smaller[idx] =  idx+1;
      larger[idx] = 0;
    } else {
      larger[idx] = idx+1;
      smaller[idx] = 0;
    }
    q[idx] = val;
  }
}

template <typename T>
__global__ void
condDiv(T *q, long *J, long inputsize, T q_max) {
  long idx = hipBlockIdx_x * BLOCK_SIZE + hipThreadIdx_x;
  T one = ScalarConvert<int, T>::to(1);
  if (idx < inputsize) {
    if (J[idx] <= 0) {
      q[idx] = one;
    } else {
      if (THCNumerics<T>::gt(q_max, one)) {
	q[idx] = THCNumerics<T>::div(q[idx], q_max);
      }
    }
  }
}


#undef MAX_NUM_BLOCKS
#undef BLOCK_SIZE

// Normalizes the L1 norm of every row to 1; used by multinomial
template <typename T>
__global__ void renormRowsL1(T* dist, long rows, long cols) {
  extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
  T *smem = reinterpret_cast<T *>(my_smem);

  for (long row = hipBlockIdx_x; row < rows; row += hipGridDim_x) {
    T sum = ScalarConvert<int, T>::to(0);
    for (long col = hipThreadIdx_x; col < cols; col += hipBlockDim_x) {
      sum = THCNumerics<T>::add(sum, dist[row * cols + col]);
    }

    sum = reduceBlock(smem, hipBlockDim_x, sum, ReduceAdd<T, T>(), ScalarConvert<int, T>::to(0));
    if (hipThreadIdx_x == 0) {
      smem[0] = sum;
    }
    __syncthreads();

    sum = smem[0];
    if (THCNumerics<T>::gt(sum, ScalarConvert<int, T>::to(0))) {
      for (long col = hipThreadIdx_x; col < cols; col += hipBlockDim_x) {
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

  T curVal = dist[start];
  while(start >= 1 && THCNumerics<T>::eq(dist[start - 1], curVal)) start--;

  return start;
}

template <typename T, typename AccT>
__global__ void
sampleMultinomialOnce(long* dest,
                      long distributions,
                      int categories,
                      T* sampled,
                      T* dist) {
  extern __shared__ __align__(sizeof(AccT)) unsigned char my_smem[];
  __shared__ bool found;

  // Shared Memory hold blockdim.x T for holding the cumulative sum,
  // hipBlockDim_x AccT for normalizing the probabilities,
  T *smem = reinterpret_cast<T *>(my_smem);
  AccT *asmem = reinterpret_cast<AccT *>(&my_smem[hipBlockDim_x * sizeof(T)]);

  AccT accZero = ScalarConvert<int, AccT>::to(0);
  T zero = ScalarConvert<int, T>::to(0);

  for (long curDist = hipBlockIdx_x;
       curDist < distributions; curDist += hipGridDim_x) {
    // Each block handles one distribution
    // First pass, find the total sum of the distribution
    AccT sum = accZero;
    for (int cat = hipThreadIdx_x; cat < categories; cat += hipBlockDim_x) {
      sum = THCNumerics<AccT>::add(
        sum,
        ScalarConvert<T, AccT>::to(dist[curDist * categories + cat]));
    }

    // hipThreadIdx_x == 0 has the sum value from this
    sum = reduceBlock(asmem, hipBlockDim_x, sum, ReduceAdd<AccT, AccT>(), accZero);

    // Broadcast sum and sample value
    if (hipThreadIdx_x == 0) {
      // Make sure the sum of our distribution didn't overflow
      assert(!isinf(sum));

      asmem[0] = sum;
      smem[0] = sampled[curDist];
    }
    __syncthreads();

    sum = asmem[0];
    T sample = smem[0];
    __syncthreads();

    if (THCNumerics<AccT>::eq(sum,  accZero) || THCNumerics<T>::eq(sample, zero)) {
      // Choose the first element
      if (hipThreadIdx_x == 0) {
        dest[curDist] = TH_INDEX_BASE;
      }

      continue;
    }

    int chunks = THCCeilDiv(categories, (int) hipBlockDim_x);
    T prevHighProb = zero;
    found = false;

    for (int chunk = 0; chunk < chunks && !found; ++chunk) {
      // All threads in bounds load a value
      int cat = chunk * hipBlockDim_x + hipThreadIdx_x;

      AccT val =
        cat < categories ?
          THCNumerics<AccT>::div(
              ScalarConvert<T, AccT>::to(dist[curDist * categories + cat]),
              sum) :
          accZero;

      smem[hipThreadIdx_x] = ScalarConvert<AccT, T>::to(val);
      __syncthreads();

      // Perform an inclusive prefix sum of the shared memory contents
      for (int offset = 1; offset < hipBlockDim_x; offset *= 2) {
        T val = zero;

        if (hipThreadIdx_x >= offset) {
          val = THCNumerics<T>::add(smem[hipThreadIdx_x - offset], smem[hipThreadIdx_x]);
        }

        __syncthreads();
        if (hipThreadIdx_x >= offset) {
          smem[hipThreadIdx_x] = val;
        }
        __syncthreads();
      }

      // Each thread will check to see if the sample falls in its
      // bucket
      T curBucket = THCNumerics<T>::add(smem[hipThreadIdx_x], prevHighProb);
      T prevBucket =
        hipThreadIdx_x == 0 ? prevHighProb :
        THCNumerics<T>::add(smem[hipThreadIdx_x - 1], prevHighProb);
      bool inBucket =
        (cat < categories) &&
        (!THCNumerics<T>::gt(sample, curBucket)) &&
        (THCNumerics<T>::gt(sample, prevBucket));

      if (inBucket) {
        // We're done; we have the sample
        // Torch indices are 1-based
        dest[curDist] = cat + TH_INDEX_BASE;
        found = true;
      }

      // Store the previous scan's high value for future use
      prevHighProb = THCNumerics<T>::add(prevHighProb, smem[hipBlockDim_x - 1]);

      __syncthreads();
    }

    if (hipThreadIdx_x == 0 && !found) {
      // This should address a rare bug where we don't select a valid index. This likely occurs when
      // due to floating point arithmetic rounding errors, our cumulative sum does not add up to 1, but
      // and our uniform sample is greater than this value. In this case we likely have unitialized memory
      // in dest[curDist]. So basically we will loop through the distribution and pick the largest index
      // where the distribution is non-zero. This is obviously terribly inefficient, but due to the
      // rarity in which this occurs, this should not be an issue.
      for (int cat = categories - 1; cat >= 0; --cat) {
        if (THCNumerics<T>::gt(dist[curDist * categories + cat], zero)) {
          dest[curDist] = cat + TH_INDEX_BASE;
          break;
        }
      }
    }
  }
}

template <typename T>
__global__ void
sampleMultinomialWithReplacement(curandStateMtgp32* state,
                                 int totalSamples,
                                 long* dest,
                                 long distributions,
                                 int categories,
                                 T* normDistPrefixSum) {
  // At the moment, each warp computes one sample value in the binary
  // search due to divergence. It seems possible to compute multiple
  // values and limit divergence though later on. However, no matter
  // what, all block threads must participate in the curand_uniform
  // call to update the generator state.

  // The block determines the distribution for which we generate a point
  for (long curDist = hipBlockIdx_x;
       curDist < distributions;
       curDist += hipGridDim_x) {
    for (int sampleBase = 0;
         sampleBase < totalSamples; sampleBase += hipBlockDim_y) {
      // The warp determines the sample
      int sample = sampleBase + hipThreadIdx_y;

      // All threads participate in this
      T r = ScalarConvert<float, T>::to(curand_uniform(&state[hipBlockIdx_x]));

      if (hipThreadIdx_x == 0 && sample < totalSamples) {
        // Find the bucket that a uniform sample lies in
        int choice = binarySearchForMultinomial<T>(
          normDistPrefixSum + curDist * categories,
          categories,
          r);

        // Torch indices are 1-based
        dest[curDist * totalSamples + sample] = choice + TH_INDEX_BASE;
      }
    }
  }
}

template <typename T>
__global__ void
sampleMultinomialWithoutReplacement(curandStateMtgp32* state,
                                    int totalSamples,
                                    int sample,
                                    long* dest,
                                    long distributions,
                                    int categories,
                                    T* origDist,
                                    T* normDistPrefixSum) {
  // At the moment, each warp computes one sample value in the binary
  // search due to divergence. It seems possible to compute multiple
  // values and limit divergence though later on. However, no matter
  // what, all block threads must participate in the curand_uniform
  // call to update the generator state.

  // The block and warp determines the distribution for which we
  // generate a point
  for (long curDistBase = hipBlockIdx_x * hipBlockDim_y;
       curDistBase < distributions;
       curDistBase += hipGridDim_x * hipBlockDim_y) {
    // The warp determines the distribution
    long curDist = curDistBase + hipThreadIdx_y;

    // All threads must participate in this
    T r = ScalarConvert<float, T>::to(curand_uniform(&state[hipBlockIdx_x]));

    if (hipThreadIdx_x == 0 && curDist < distributions) {
      // Find the bucket that a uniform sample lies in
      int choice = binarySearchForMultinomial<T>(
        normDistPrefixSum + curDist * categories,
        categories,
        r);

      // Torch indices are 1-based
      dest[curDist * totalSamples + sample] = choice + TH_INDEX_BASE;

      // Without replacement, so update the original probability so it
      // is not considered a second time
      origDist[curDist * categories + choice] = ScalarConvert<int, T>::to(0);
    }
  }
}

template <typename T>
__global__ void
aliasMultinomialSetup(long *J, T*q, long inputsize, long * smaller, long *larger, int small_c, int large_c) {
  T one = ScalarConvert<long, T>::to(1);
  // Loop through and create little binary mixtures that
  // appropriately allocate the larger outcomes over the
  // overall uniform mixture.
  long large = 0;
  long small = 0;
  while (small_c > 0 && large_c > 0) {
    large = larger[large_c-1]-1;
    small = smaller[small_c-1]-1;
    J[small] = large;
    T q_sub = THCNumerics<T>::sub(one, q[small]);
    q[large] = THCNumerics<T>::sub(q[large], q_sub);
    if (THCNumerics<T>::le(q[large], one)) {
      smaller[small_c-1] = large+1;
      large_c -= 1;
    } else {
      larger[large_c-1] = large+1;
      small_c -= 1;
    }
  }
}

#endif // THC_TENSOR_RANDOM_CUH
