#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMath.cu"
#else

#include <algorithm>

#include <ATen/cuda/CUDAContext.h>
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
    THCTensor *first, THCTensor *second, int dimension, int index);
inline void THCTensor_(check_shape_except_dim)(THCState *state,
    THCTensor *first, THCTensor *second, int dimension, int index)
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
        "Sizes of tensors must match except in dimension %d. Got %lld and %lld in dimension %d (The offending index is %d)",
        dimension, (long long)first_dim_size, (long long)second_dim_size, dim, index);
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

#endif
