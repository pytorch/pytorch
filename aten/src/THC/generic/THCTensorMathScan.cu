#include <algorithm>

#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathScan.cu"
#else

#ifndef THC_REAL_IS_HALF
template<class BinaryFunction>
__host__ void THCTensor_(scanThrust)(
    THCState *state,
    THCTensor *dst,
    THCTensor *src,
    BinaryFunction binary_op)
{
  THCThrustAllocator thrustAlloc(state);
  thrust::device_ptr<scalar_t> src_data(THCTensor_(data)(state, src));
  thrust::device_ptr<scalar_t> dst_data(THCTensor_(data)(state, dst));
  ptrdiff_t size = THCTensor_(nElement)(state, src);
  thrust::inclusive_scan(
#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
      thrust::cuda::par(thrustAlloc).on(c10::cuda::getCurrentCUDAStream()),
#endif
      src_data, src_data + size, dst_data,
      binary_op);
}
#endif

template<class BinaryOp>
__host__ void THCTensor_(scanOuterDim)(THCState *state, THCTensor *tgt,
                                       THCTensor *src, int dimension,
                                       scalar_t init, BinaryOp binary_op)
{
  unsigned ndim = THCTensor_(nDimensionLegacyAll)(state, src);
  // Treat all outer dimensions (i.e. dim < dimension) as one.
  unsigned num_orows = 1;
  for (int dim = 0; dim < dimension; dim++) {
    num_orows *= THCTensor_(sizeLegacyNoScalars)(state, src, dim);
  }
  unsigned row_size = THCTensor_(sizeLegacyNoScalars)(state, src, dimension);
  // Treat all inner dimensions (i.e. dim > dimension) as one.
  unsigned num_irows = 1;
  for (unsigned dim = dimension + 1; dim < ndim; dim++) {
    num_irows *= THCTensor_(sizeLegacyNoScalars)(state, src, dim);
  }

  dim3 threads(std::min(512u, num_irows));
  unsigned maxGridDim = 1024;
  dim3 grid(
      std::min(maxGridDim, num_orows),
      std::min(maxGridDim, THCCeilDiv(num_irows, threads.x)));

  THCTensor_kernel_scanOuterDim<scalar_t><<<grid, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
    THCTensor_(data)(state, tgt), THCTensor_(data)(state, src),
    num_orows, num_irows, row_size, init, binary_op);

  THCudaCheck(cudaGetLastError());
}

template<class BinaryFunction>
__host__ void THCTensor_(scanInnermostDim)(THCState *state, THCTensor *tgt,
                                           THCTensor *src, scalar_t init,
                                           BinaryFunction binary_op)
{
  unsigned ndim = THCTensor_(nDimensionLegacyAll)(state, src);
  // Treat all outer dimensions as a single dimension.
  unsigned num_rows = 1;
  for (unsigned dim = 0; dim < ndim - 1; dim++) {
    num_rows *= THCTensor_(sizeLegacyNoScalars)(state, src, dim);
  }
  unsigned row_size = THCTensor_(sizeLegacyNoScalars)(state, src, ndim - 1);

  dim3 threads(16, 32);
  dim3 grid(std::min(1024u, THCCeilDiv(num_rows, threads.y)));

  THCTensor_kernel_scanInnermostDim<scalar_t, 16, 32><<<grid, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
    THCTensor_(data)(state, tgt), THCTensor_(data)(state, src), num_rows, row_size, init, binary_op);

  THCudaCheck(cudaGetLastError());
}

template<class BinaryFunction>
void THCTensor_(scanDim)(THCState *state, THCTensor *self_, THCTensor *src,
                         int dimension, scalar_t init, BinaryFunction binary_op)
{
  // "init" must be the identity element for binary_op
  int ndim = THCTensor_(nDimensionLegacyNoScalars)(state, src);
  THArgCheck(dimension >= 0 && dimension < ndim, 3, "dimension %d out of range",
      dimension);

  THCTensor_(resizeAs)(state, self_, src);
  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  src = THCTensor_(newContiguous)(state, src);

  if (!self->is_empty()) {
  #ifndef THC_REAL_IS_HALF
    if (ndim == 1) {
      // thrust does not take an "init"
      THCTensor_(scanThrust)(state, self, src, binary_op);
    } else
  #endif
    if (dimension == ndim - 1) {
      THCTensor_(scanInnermostDim)(state, self, src, init, binary_op);
    } else {
      THCTensor_(scanOuterDim)(state, self, src, dimension, init, binary_op);
    }
  }

  THCTensor_(free)(state, src);
  THCTensor_(freeCopyTo)(state, self, self_);
}

void THCTensor_(cumsum)(THCState *state, THCTensor *self, THCTensor *src, int dimension)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self, src));
  dimension = at::maybe_wrap_dim(dimension, src);
  return THCTensor_(scanDim)(state, self, src, dimension,
                             ScalarConvert<float, scalar_t>::to(0.0), AddOp<scalar_t>());
}

void THCTensor_(cumprod)(THCState *state, THCTensor *self, THCTensor *src, int dimension)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self, src));
  dimension = at::maybe_wrap_dim(dimension, src);
  return THCTensor_(scanDim)(state, self, src, dimension,
                             ScalarConvert<float, scalar_t>::to(1.0), MulOp<scalar_t>());
}

#endif
