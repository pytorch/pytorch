#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMath.cu"
#else

#include <algorithm>

#include "ATen/cuda/CUDAContext.h"
#include <ATen/MemoryOverlap.h>

void THCTensor_(fill)(THCState* state, THCTensor *self_, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));

  if (!THC_pointwiseApply1<scalar_t>(
        state, self_, TensorFillOp<scalar_t>(value))) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(zero)(THCState *state, THCTensor *self_)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  if (THCTensor_(isContiguous)(state, self_)) {
    THCudaCheck(cudaMemsetAsync(THCTensor_(data)(state, self_),
                                0,
                                sizeof(scalar_t) * THCTensor_(nElement)(state, self_),
                                c10::cuda::getCurrentCUDAStream()));
  } else {
    if (!THC_pointwiseApply1<scalar_t>(
          state, self_,
          TensorFillOp<scalar_t>(ScalarConvert<int, scalar_t>::to(0)))) {
      THArgCheck(false, 1, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

ptrdiff_t
THCTensor_(numel)(THCState *state, THCTensor *t)
{
  return THCTensor_(nElement)(state, t);
}

void THCTensor_(check_shape_except_dim)(THCState *state,
    THCTensor *first, THCTensor *second, int dimension);
inline void THCTensor_(check_shape_except_dim)(THCState *state,
    THCTensor *first, THCTensor *second, int dimension)
{
  int first_dims = first->dim();
  int second_dims = second->dim();
  THArgCheck(first_dims == second_dims, 0,
      "Tensors must have same number of dimensions: got %d and %d",
      first_dims, second_dims);
  for (int dim = 0; dim < first_dims; dim++) {
    if (dim == dimension) {
      continue;
    }
    int64_t first_dim_size = THCTensor_(size)(state, first, dim);
    int64_t second_dim_size = THCTensor_(size)(state, second, dim);
    THArgCheck(first_dim_size == second_dim_size, 0,
        "Sizes of tensors must match except in dimension %d. Got %lld and %lld in dimension %d",
        dimension, (long long)first_dim_size, (long long)second_dim_size, dim);
  }
}

void THCTensor_(nonzero)(THCState* state, THCudaLongTensor *tensor,
                          THCTensor *self)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self  ));
  THCAssertSameGPU(THCudaLongTensor_checkGPU(state, 1, tensor));


  using namespace thrust::placeholders;
  THCThrustAllocator thrustAlloc(state);
  self = THCTensor_(newContiguous)(state, self);
  thrust::device_ptr<scalar_t> self_data(THCTensor_(data)(state, self));

  int num_dim = THCTensor_(nDimension)(state, self);
  int num_dim_noscalars = std::max<int>(1, num_dim);
  int64_t N = THCTensor_(nElement)(state, self);

  // this is a little awkward for scalars because we run thrust to count the number of zeros
  // (which are necessary to get the correct size), but thrust just has an array API, so
  // we need to basically threat the scalar as a 1-dimensional tensor (array) for
  // the counting part.
  THCudaLongTensor_resize2d(state, tensor, N, num_dim_noscalars);
  tensor = THCudaLongTensor_newContiguous(state, tensor);
  thrust::device_ptr<int64_t> tensor_data(THCudaLongTensor_data(state, tensor));

  thrust::counting_iterator<int64_t> idxfirst(0);
  thrust::counting_iterator<int64_t> idxlast = idxfirst + N;

  typedef thrust::device_ptr<int64_t> Iter;
  strided_range<Iter> strided_tensor(tensor_data,
                                     tensor_data+N*num_dim_noscalars, num_dim_noscalars);

#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
#endif

  strided_range<Iter>::iterator dend = thrust::copy_if(
#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
    thrust::cuda::par(thrustAlloc).on(stream),
#endif
    idxfirst,
    idxlast,
    self_data,
    strided_tensor.begin(),
    NonZeroOp<scalar_t>()
  );

  int64_t num_nonzeros = thrust::distance(strided_tensor.begin(), dend);

  if (num_nonzeros > 0 && num_dim > 0) {
    int64_t div = 1;
    for (int dim = num_dim-1; dim >= 0; dim--) {
      strided_range<Iter> stride_dim(tensor_data+dim,
                                     tensor_data+N*num_dim, num_dim);
      thrust::transform(
  #if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
        thrust::cuda::par(thrustAlloc).on(stream),
  #endif
        strided_tensor.begin(),
        strided_tensor.end(),
        stride_dim.begin(),
        idx_functor(div, THTensor_(size)(self, dim))
      );
      div *= THTensor_(size)(self, dim);
    }
  }

  THCudaLongTensor_resize2d(state, tensor, num_nonzeros, num_dim);

  THCTensor_(free)(state, self);
  THCudaLongTensor_free(state, tensor);

  THCudaCheck(cudaGetLastError());
}

#if !defined(THC_REAL_IS_BOOL) /* non bool only part */

void THCTensor_(diag)(THCState *state, THCTensor *self_, THCTensor *src_, int64_t k){
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src_));
  int nDimension = THCTensor_(nDimension)(state, src_);
  THArgCheck((nDimension == 2) || (nDimension == 1), 1, "expected a matrix or a vector");
  if (nDimension == 2) {
    int64_t stride0 = THCTensor_(stride)(state, src_, 0);
    int64_t stride1 = THCTensor_(stride)(state, src_, 1);
    int64_t size0 = THCTensor_(size)(state, src_, 0);
    int64_t size1 = THCTensor_(size)(state, src_, 1);
    int64_t size = (k > 0) ? std::min((int64_t)size0, (int64_t)size1 - k)
                           : std::min((int64_t)size0 + k, (int64_t)size1);
    THCTensor_(resize1d)(state, self_, size);
    if (size > 0) {
      int64_t strideSelf = THCTensor_(stride)(state, self_, 0);
      const dim3 threads(std::min(
          (int64_t)at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock,
          (int64_t)size));
      dim3 grid(std::min(
          (int64_t)1024, (int64_t)THCCeilDiv(size, (int64_t)threads.x)));
      int64_t start = (k >= 0 ? k * stride1 : -k * stride0);
      THCTensor_copyFromDiagonal<scalar_t><<<grid, threads, 0, c10::cuda::getCurrentCUDAStream()>>>
      (THCTensor_(data)(state, self_), THCTensor_(data)(state, src_), start, size, stride0 + stride1, strideSelf);
    }
  } else {
    ptrdiff_t totalElements = THCTensor_(nElement)(state, src_);
    ptrdiff_t size = (k > 0) ? totalElements + k : totalElements - k;
    int64_t strideSrc = THTensor_(stride)(src_, 0);
    THCTensor_(resize2d)(state, self_, size, size);
    THCTensor_(zero)(state, self_);
    if (size > 0) {
      int64_t stride0 = THCTensor_(stride)(state, self_, 0);
      int64_t stride1 = THCTensor_(stride)(state, self_, 1);
      const dim3 threads(std::min(
          (int64_t)at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock,
          (int64_t)size));
      dim3 grid(std::min(
          (int64_t)1024, (int64_t)THCCeilDiv(size, (ptrdiff_t)threads.x)));
      ptrdiff_t start = (k >= 0 ? k * stride1 : -k * stride0);
      THCTensor_copyToDiagonal<scalar_t><<<grid, threads, 0, c10::cuda::getCurrentCUDAStream()>>>
      (THCTensor_(data)(state, self_), THCTensor_(data)(state, src_), start, totalElements, stride0 + stride1, strideSrc);
    }
  }
  THCudaCheck(cudaGetLastError());
}

accreal THCTensor_(trace)(THCState *state, THCTensor *src_) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, src_));
  THArgCheck((THTensor_nDimensionLegacyAll(src_) == 2), 1, "expected a matrix");
  THCTensor *diag = THCTensor_(new)(state);
  THCTensor_(diag)(state, diag, src_, 0);
  accreal trace = THCTensor_(sumall)(state, diag);
  THCTensor_(free)(state, diag);
  return trace;
}

#endif

#endif
