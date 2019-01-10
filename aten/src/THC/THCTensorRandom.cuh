#ifndef THC_TENSOR_RANDOM_CUH
#define THC_TENSOR_RANDOM_CUH

#include "THCNumerics.cuh"
#include "THCReduceApplyUtils.cuh"
#include "THCTensorMathReduce.cuh"

#include <curand_kernel.h>

#define MAX_NUM_BLOCKS 200 
#define BLOCK_SIZE 256
/* Separate kernel because curand_log_normal gets extra parameters. */

template <typename T>
__global__ void generateLogNormal(curandStateMtgp32 *state, int size, T *result, double mean, double stddev)
{
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {
    float x = curand_log_normal(&state[blockIdx.x], mean, stddev);
    if (i < size) {
      result[i] = ScalarConvert<float, T>::to(x);
    }
  }
}

template <>
__global__ void generateLogNormal<double>(curandStateMtgp32 *state, int size, double *result, double mean, double stddev)
{
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {
    double x = curand_log_normal_double(&state[blockIdx.x], mean, stddev);
    if (i < size) {
      result[i] = x;
    }
  }
}

template <typename T>
__global__ void
multinomialAliasDrawKernel(int size, int64_t *output, int64_t *J, T *q, int64_t K,  T *uniform, T *bernoulli){
  int64_t idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (idx < size) {
    int64_t rand_ind = ScalarConvert<T, int64_t>::to(uniform[idx]);
    T bern_uniform = bernoulli[idx];
    int _mask = (int) THCNumerics<T>::lt(bern_uniform, q[rand_ind]);
    output[idx] = J[rand_ind]*(1 -_mask) + (rand_ind+1L) * _mask;
  }
}

template <typename T>
__global__ void
aliasMultinomialFilter(T *q, T *probs, int64_t *smaller, int64_t *larger, int64_t *J_data, int64_t *larger_short_data, int64_t *smaller_short_data, T one, int64_t inputsize){
  int64_t idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (idx < inputsize) {
    larger_short_data[idx] = 0;
    smaller_short_data[idx] = 0;
    J_data[idx]= 0;
    T val = THCNumerics<T>::mul(probs[idx], ScalarConvert<int64_t, T>::to(inputsize));
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
condDiv(T *q, int64_t *J, int64_t inputsize, T q_max) {
  int64_t idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
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
  extern __shared__  unsigned char my_smem[];
  T *smem = reinterpret_cast<T *>(my_smem);
  T zero = ScalarConvert<int, T>::to(0);
  T val;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    T sum = ScalarConvert<int, T>::to(0);
    for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
      val = dist[row * cols + col];
      assert(THCNumerics<T>::ge(val, zero));
      sum = THCNumerics<T>::add(sum, val);
    }

    sum = reduceBlock(smem, blockDim.x, sum, ReduceAdd<T, T>(), zero);
    if (threadIdx.x == 0) {
      assert(THCNumerics<T>::gt(sum, zero));
      smem[0] = sum;
    }
    __syncthreads();

    sum = smem[0];
    if (THCNumerics<T>::gt(sum, ScalarConvert<int, T>::to(0))) {
      for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
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
    // first non-zero element by setting start to size-1 here,
    // the code below will move it to the last non-zero probability
    // this actually can happen when the random number is 1
    // (github pytorch issue #4858).
    start = size - 1;
  }

  T curVal = dist[start];
  while(start >= 1 && THCNumerics<T>::eq(dist[start - 1], curVal)) start--;

  return start;
}

template <typename T, typename AccT>
__global__ void
sampleMultinomialOnce(int64_t* dest,
                      int64_t distributions,
                      int categories,
                      T* sampled,
                      T* dist,
                      int stride_dist,        // dist->stride[0]
                      int stride_categories   // dist->stride[1]
                      ) {
  extern __shared__  unsigned char my_smem[];
  __shared__ bool found;

  // Shared Memory hold blockdim.x T for holding the cumulative sum,
  // blockDim.x AccT for normalizing the probabilities,
  T *smem = reinterpret_cast<T *>(my_smem);
  AccT *asmem = reinterpret_cast<AccT *>(&my_smem[blockDim.x * sizeof(T)]);

  AccT accZero = ScalarConvert<int, AccT>::to(0);
  T zero = ScalarConvert<int, T>::to(0);

  for (int64_t curDist = blockIdx.x;
       curDist < distributions; curDist += gridDim.x) {
    // Each block handles one distribution
    // First pass, find the total sum of the distribution
    AccT sum = accZero;
    T val;
    for (int cat = threadIdx.x; cat < categories; cat += blockDim.x) {
      val = dist[curDist * stride_dist + cat * stride_categories];
      assert(THCNumerics<T>::ge(val, zero));
      sum = THCNumerics<AccT>::add(sum, ScalarConvert<T, AccT>::to(val));
    }

    // threadIdx.x == 0 has the sum value from this
    sum = reduceBlock(asmem, blockDim.x, sum, ReduceAdd<AccT, AccT>(), accZero);

    // Broadcast sum and sample value
    if (threadIdx.x == 0) {
      // Make sure the sum of our distribution didn't overflow
      assert(!isinf(sum));
      assert(THCNumerics<AccT>::gt(sum, accZero));

      asmem[0] = sum;
      smem[0] = sampled[curDist];
    }
    __syncthreads();

    sum = asmem[0];
    T sample = smem[0];
    __syncthreads();

    if (THCNumerics<AccT>::eq(sum,  accZero) || THCNumerics<T>::eq(sample, zero)) {
      // Choose the first element
      if (threadIdx.x == 0) {
        dest[curDist] = TH_INDEX_BASE;
      }

      continue;
    }

    int chunks = THCCeilDiv(categories, (int) blockDim.x);
    T prevHighProb = zero;
    found = false;

    for (int chunk = 0; chunk < chunks && !found; ++chunk) {
      // All threads in bounds load a value
      int cat = chunk * blockDim.x + threadIdx.x;

      AccT val =
        cat < categories ?
          THCNumerics<AccT>::div(
              ScalarConvert<T, AccT>::to(dist[curDist * stride_dist + cat * stride_categories]),
              sum) :
          accZero;

      smem[threadIdx.x] = ScalarConvert<AccT, T>::to(val);
      __syncthreads();

      // Perform an inclusive prefix sum of the shared memory contents
      for (int offset = 1; offset < blockDim.x; offset *= 2) {
        T val = zero;

        if (threadIdx.x >= offset) {
          val = THCNumerics<T>::add(smem[threadIdx.x - offset], smem[threadIdx.x]);
        }

        __syncthreads();
        if (threadIdx.x >= offset) {
          smem[threadIdx.x] = val;
        }
        __syncthreads();
      }

      // Each thread will check to see if the sample falls in its
      // bucket
      T curBucket = THCNumerics<T>::add(smem[threadIdx.x], prevHighProb);
      T prevBucket =
        threadIdx.x == 0 ? prevHighProb :
        THCNumerics<T>::add(smem[threadIdx.x - 1], prevHighProb);
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
      prevHighProb = THCNumerics<T>::add(prevHighProb, smem[blockDim.x - 1]);

      __syncthreads();
    }

    if (threadIdx.x == 0 && !found) {
      // This should address a rare bug where we don't select a valid index. This likely occurs when
      // due to floating point arithmetic rounding errors, our cumulative sum does not add up to 1, but
      // and our uniform sample is greater than this value. In this case we likely have unitialized memory
      // in dest[curDist]. So basically we will loop through the distribution and pick the largest index
      // where the distribution is non-zero. This is obviously terribly inefficient, but due to the
      // rarity in which this occurs, this should not be an issue.
      for (int cat = categories - 1; cat >= 0; --cat) {
        if (THCNumerics<T>::gt(dist[curDist * stride_dist + cat * stride_categories], zero)) {
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
                                 int64_t* dest,
                                 int64_t distributions,
                                 int categories,
                                 T* normDistPrefixSum) {
  // At the moment, each warp computes one sample value in the binary
  // search due to divergence. It seems possible to compute multiple
  // values and limit divergence though later on. However, no matter
  // what, all block threads must participate in the curand_uniform
  // call to update the generator state.

  // The block determines the distribution for which we generate a point
  for (int64_t curDist = blockIdx.x;
       curDist < distributions;
       curDist += gridDim.x) {
    for (int sampleBase = 0;
         sampleBase < totalSamples; sampleBase += blockDim.y) {
      // The warp determines the sample
      int sample = sampleBase + threadIdx.y;

      // All threads participate in this
      T r = ScalarConvert<float, T>::to(curand_uniform(&state[blockIdx.x]));

      if (threadIdx.x == 0 && sample < totalSamples) {
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
                                    int64_t* dest,
                                    int64_t distributions,
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
  for (int64_t curDistBase = blockIdx.x * blockDim.y;
       curDistBase < distributions;
       curDistBase += gridDim.x * blockDim.y) {
    // The warp determines the distribution
    int64_t curDist = curDistBase + threadIdx.y;

    // All threads must participate in this
    T r = ScalarConvert<float, T>::to(curand_uniform(&state[blockIdx.x]));

    if (threadIdx.x == 0 && curDist < distributions) {
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
aliasMultinomialSetup(int64_t *J, T*q, int64_t inputsize, int64_t * smaller, int64_t *larger, int small_c, int large_c) {
  T one = ScalarConvert<int64_t, T>::to(1);
  // Loop through and create little binary mixtures that
  // appropriately allocate the larger outcomes over the
  // overall uniform mixture.
  int64_t large = 0;
  int64_t small = 0;
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
