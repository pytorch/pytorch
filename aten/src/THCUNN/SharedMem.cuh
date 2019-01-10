// Based on the simpleTempltes CUDA example

#ifndef THCUNN_SHAREDMEM_H
#define THCUNN_SHAREDMEM_H

template <typename T>
struct SharedMem {
  __device__ T *getPointer()
  {
    extern __device__ void error(void);
    error();
    return NULL;
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct SharedMem<half>
{
  __device__ half *getPointer() {
    extern __shared__ half s_half[];
    return s_half;
  }
};
#endif

template <>
struct SharedMem<float>
{
  __device__ float *getPointer() {
    extern __shared__ float s_float[];
    return s_float;
  }
};

template <>
struct SharedMem<double>
{
  __device__ double *getPointer() {
    extern __shared__ double s_double[];
    return s_double;
  }
};

#endif
