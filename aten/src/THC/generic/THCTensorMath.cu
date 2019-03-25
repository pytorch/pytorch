#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMath.cu"
#else

#include "ATen/cuda/CUDAContext.h"

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
                                THCState_getCurrentStream(state)));
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

void THCTensor_(cat)(THCState *state, THCTensor *result,
		     THCTensor *ta, THCTensor *tb, int dimension)
{
  THCTensor* inputs[2];
  inputs[0] = ta;
  inputs[1] = tb;
  THCTensor_(catArray)(state, result, inputs, 2, dimension);
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

void THCTensor_(catArray)(THCState *state, THCTensor *result,
			  THCTensor **inputs, int numInputs, int dimension)
{
  // previously, size [0] tensors were the only possible empty tensors; thus, it wasn't possible
  // to cat empty tensors unless all the other tensors were 1-dimensional, so we allowed these tensors
  // to be "skipped".  We maintain this behavior for backwards compatibility, but only for this specific
  // size (i.e. other empty sizes are not skipped).
  // FIXME: warn if this is the case
  int i, j, cohortMax;
  int64_t offset;
  bool hasSkippedInput = false;
  THCTensor *notSkippedTensor = NULL;  // non-owning reference
  auto should_skip = [](THCTensor *t) { return t->is_empty() && t->dim() == 1; };
  int nDims = 0;

  for (i = 0; i < numInputs; i++)
  {
    if (should_skip(inputs[i])) {
      hasSkippedInput = true;
      continue;
    }
    nDims = inputs[i]->dim();
    notSkippedTensor = inputs[i];
  }

  // If all inputs are empty tensors, return an empty tensor
  if (notSkippedTensor == NULL) {
    return;
  }

  THArgCheck(numInputs > 0, 3, "invalid number of inputs %d", numInputs);
  THArgCheck(dimension >= 0, 4, "invalid dimension %d", dimension);

  std::vector<int64_t> size(nDims);

  // Compute size of the result in the cat dimension
  int64_t cat_dim_size = 0;
  for (int i = 0; i < numInputs; i++) {
    THCTensor *tensor = inputs[i];
    if (should_skip(tensor)) {
      continue;
    }
    THCTensor_(check_shape_except_dim)(state, notSkippedTensor, tensor, dimension);
    cat_dim_size += THCTensor_(size)(state, tensor, dimension);
  }

  // Compute the size of the result
  for (int dim = 0; dim < nDims; dim++) {
    int64_t result_dim_size = THCTensor_(size)(state, notSkippedTensor, dim);
    if (dim == dimension) {
      result_dim_size = cat_dim_size;
    }
    size[dim] = result_dim_size;
  }
  THCTensor_(resize)(state, result, size, {});

  // We parallelize the copy if all 6 conditions pass:
  //
  // 1. There is more than one input tensor
  // 2. No empty inputs
  // 3. The result tensor is 32-bit indexable
  // 4. The number of dimensions is <= 4
  // 5. All input tensors are contiguous (output tensor may be non-contig)
  // 6. All input tensors can use 32-bit indexing
  // 7. All input tensors are on the same device

  if (numInputs > 1 &&
      !hasSkippedInput &&
      result->dim() <= CAT_ARRAY_MAX_INPUT_DIMS &&
      THCTensor_canUse32BitIndexMath(state, result) &&
      THCTensor_allContiguous(state, inputs, numInputs) &&
      THCTensor_all32BitIndexable(state, inputs, numInputs) &&
      THCTensor_allSameDevice(state, inputs, numInputs)) {

    // First, let's set up our kernel parameters. We start with a raw pointer to the storage
    // for the output Tensor.
    scalar_t *data = THCTensor_(data)(state, result);

    // Kernel Parameter
    size_t tensorMetadataSize = sizeof(CatArrInputTensor<scalar_t, unsigned int>) * CAT_ARRAY_BATCH_SIZE;
    auto d_inputs = static_cast<CatArrInputTensor<scalar_t, unsigned int> *>(THCudaMalloc(state, tensorMetadataSize));

    OutputTensorSizeStride<unsigned int, CAT_ARRAY_MAX_INPUT_DIMS> param;

    // Next, let's initialize the size, stride arrays for the output Tensor.
    for (i = 0; i < nDims; ++i) {
      param.outputSize[i] = THCTensor_(size)(state, result, i);
      param.outputStride[i] = THCTensor_(stride)(state, result, i);
    }

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // Template Declarations for dim = 1, 2, 3, 4
#define HANDLE_CASE(DIMS) \
  CatArrayBatchedCopy<scalar_t, unsigned int, DIMS><<<catGrid, applyBlock, 0, stream.stream()>>>(data, d_inputs, param, dimension, param.outputStride[dimension]);

    // Now we loop
    offset = 0;
    for (i = 0; i < numInputs; i += CAT_ARRAY_BATCH_SIZE) {
      // Re-allocate stackInputs every iteration to avoid read-after-write hazard
      {
        auto stackInputs_owner = THCudaHostAlloc(state, tensorMetadataSize);
        CatArrInputTensor<scalar_t, unsigned int>* stackInputs = static_cast<CatArrInputTensor<scalar_t, unsigned int>*>(stackInputs_owner.get());
        cohortMax = 0;
        for (j = 0; j < CAT_ARRAY_BATCH_SIZE && (i+j) < numInputs; ++j) {
          int64_t dimSize = THCTensor_(size)(state, inputs[i+j], dimension);

          stackInputs[j].input = THCTensor_(data)(state, inputs[i+j]);
          stackInputs[j].offset = offset;
          stackInputs[j].dimSize = dimSize;
          stackInputs[j].nElements = THCTensor_(nElement)(state, inputs[i+j]);
          cohortMax = cohortMax > (int) stackInputs[j].nElements ? cohortMax : (int) stackInputs[j].nElements;

          // update offset
          offset += dimSize;
        }
        THCudaCheck(cudaMemcpyAsync(
            d_inputs,
            stackInputs,
            j * sizeof(CatArrInputTensor<scalar_t, unsigned int>),
            cudaMemcpyHostToDevice,
            stream.stream()));
        THCudaHostRecord(state, stackInputs);
      }

      // Next, let's consider how we set our kernel launch parameters.
      // We borrow from THCApply, which the kernel's internal indexing
      // is based on.
      dim3 applyBlock = getApplyBlock();

      //Get grid where x dim fills half gpu and y dim is number of tensors.
      //This will have cating two tensors fill the entire grid, but prevent
      //many threads from needlessly load meta data if their sizes is small.
      dim3 catGrid;
      getCatGrid(state, j, catGrid);


      switch (nDims) {
        case 1:
          HANDLE_CASE(1);
          break;
        case 2:
          HANDLE_CASE(2);
          break;
        case 3:
          HANDLE_CASE(3);
          break;
        case 4:
          HANDLE_CASE(4);
          break;
      }
      THCudaCheck(cudaGetLastError());
    }
    THCudaFree(state, d_inputs);
#undef HANDLE_CASE
  } else {
    offset = 0;
    for (j = 0; j < numInputs; j++)
    {
      if (should_skip(inputs[j])) continue;
      int64_t dimSize = THCTensor_(size)(state, inputs[j], dimension);
      THCTensor *nt = THCTensor_(newWithTensor)(state, result);
      THCTensor_(narrow)(state, nt, NULL, dimension, offset, dimSize);
      THCTensor_(copy)(state, nt, inputs[j]);
      THCTensor_(free)(state, nt);
      offset += dimSize;
    }
  }
}

#if !defined(THC_REAL_IS_BOOL) /* non bool only part */

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
  cudaStream_t stream = THCState_getCurrentStream(state);
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

void THCTensor_(diag)(THCState *state, THCTensor *self_, THCTensor *src_, int64_t k){
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src_));
  int nDimension = THCTensor_(nDimensionLegacyNoScalars)(state, src_);
  THArgCheck((nDimension == 2) || (nDimension == 1), 1, "expected a matrix or a vector");
  if (nDimension == 2) {
    int64_t stride0 = THCTensor_(stride)(state, src_, 0);
    int64_t stride1 = THCTensor_(stride)(state, src_, 1);
    int64_t size0 = THCTensor_(size)(state, src_, 0);
    int64_t size1 = THCTensor_(size)(state, src_, 1);
    int64_t size = (k > 0) ? min((int64_t)size0, (int64_t)size1 - k) : min((int64_t)size0 + k, (int64_t)size1);
    THCTensor_(resize1d)(state, self_, size);
    if (size > 0) {
      int64_t strideSelf = THCTensor_(stride)(state, self_, 0);
      const dim3 threads(min((int64_t)at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, (int64_t)size));
      dim3 grid(min((int64_t)1024, (int64_t)THCCeilDiv(size, (int64_t)threads.x)));
      int64_t start = (k >= 0 ? k * stride1 : -k * stride0);
      THCTensor_copyFromDiagonal<scalar_t><<<grid, threads, 0, THCState_getCurrentStream(state)>>>
      (THCTensor_(data)(state, self_), THCTensor_(data)(state, src_), start, size, stride0 + stride1, strideSelf);
    }
  } else {
    ptrdiff_t totalElements = THCTensor_(nElement)(state, src_);
    ptrdiff_t size = (k > 0) ? totalElements + k : totalElements - k;
    int64_t strideSrc = THTensor_strideLegacyNoScalars(src_, 0);
    THCTensor_(resize2d)(state, self_, size, size);
    THCTensor_(zero)(state, self_);
    if (size > 0) {
      int64_t stride0 = THCTensor_(stride)(state, self_, 0);
      int64_t stride1 = THCTensor_(stride)(state, self_, 1);
      const dim3 threads(min((int64_t)at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, (int64_t)size));
      dim3 grid(min((int64_t)1024, (int64_t)THCCeilDiv(size, (ptrdiff_t)threads.x)));
      ptrdiff_t start = (k >= 0 ? k * stride1 : -k * stride0);
      THCTensor_copyToDiagonal<scalar_t><<<grid, threads, 0, THCState_getCurrentStream(state)>>>
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
