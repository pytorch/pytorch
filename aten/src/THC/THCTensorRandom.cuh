#ifndef THC_TENSOR_RANDOM_CUH
#define THC_TENSOR_RANDOM_CUH

#include <THC/THCNumerics.cuh>
#include <THC/THCReduceApplyUtils.cuh>
#include <THC/THCTensorMathReduce.cuh>

#include <curand_kernel.h>

#define MAX_NUM_BLOCKS 200
#define BLOCK_SIZE 256

template <typename T>
__global__ void
multinomialAliasDrawKernel(int size, int64_t *output, int64_t *J, T *q, int64_t K,  T *uniform, T *bernoulli){
  int64_t idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (idx < size) {
    int64_t rand_ind = ScalarConvert<T, int64_t>::to(uniform[idx]);
    T bern_uniform = bernoulli[idx];
    int _mask = (int) THCNumerics<T>::lt(bern_uniform, q[rand_ind]);
    output[idx] = J[rand_ind]*(1 -_mask) + rand_ind * _mask;
  }
}

template <typename T>
__global__ void
aliasMultinomialFilter(T *q, T *probs, int64_t *smaller, int64_t *larger, int64_t *J_data, int64_t *larger_short_data, int64_t *smaller_short_data, T one, int64_t inputsize){
  int64_t idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (idx < inputsize) {
    larger_short_data[idx] = 0;
    smaller_short_data[idx] = 0;
    J_data[idx]= -1;
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
    if (J[idx] < 0) {
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
    large = larger[large_c-1];
    small = smaller[small_c-1];
    J[small] = large;
    T q_sum = THCNumerics<T>::add(q[large], q[small]);
    q[large] = THCNumerics<T>::sub(q_sum, one);
    if (THCNumerics<T>::lt(q[large], one)) {
      smaller[small_c-1] = large;
      large_c -= 1;
    } else {
      larger[large_c-1] = large;
      small_c -= 1;
    }
  }
}

#endif // THC_TENSOR_RANDOM_CUH
