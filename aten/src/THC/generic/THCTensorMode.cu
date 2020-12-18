#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMode.cu"
#else

#include <c10/cuda/CUDAException.h>
#include <thrust/iterator/constant_iterator.h>

void THCTensor_(calculateMode)(THCState *state,
                               THCTensor *values,
                               THCudaLongTensor *indices,
                               THCTensor *input,
                               THCudaLongStorage *sortBuffer,
                               int dimension,
                               THLongStorage *position) {
  THAssert(THCTensor_(isContiguous)(state, input));

  // Because the input is contiguous, we want to get a reference to the
  // location of the buffer at the innermost dimension that we are going
  // to calculate the mode for --> we do this by manually doing the stride
  // calculations to get an offset
  scalar_t *data = THCTensor_(data)(state, input);
  for (int i = 0; i < (position->nbytes() / sizeof(int64_t)); ++i) {
    data += THLongStorage_data(position)[i] * THTensor_strideLegacyNoScalars(input, i);
  }

  int64_t nElement = THCTensor_(sizeLegacyNoScalars)(state, input, THCTensor_(nDimensionLegacyAll)(state, input) - 1);
  THCThrustAllocator thrustAlloc(state);

  // Wrap input data, sortBuffer, in Thrust device vectors
  thrust::device_ptr<scalar_t> vecPtr = thrust::device_pointer_cast(data);
  thrust::device_vector<scalar_t> iter(vecPtr, vecPtr + nElement);
  thrust::device_ptr<int64_t> sbPtr = thrust::device_pointer_cast(THCudaLongStorage_data(state, sortBuffer));
  thrust::device_vector<int64_t> seq(sbPtr, sbPtr + nElement);

  // Fill sortBuffer with [0, 1, 2, ... nElement - 1]
  thrust::sequence(
#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
    thrust::cuda::par(thrustAlloc).on(c10::cuda::getCurrentCUDAStream()),
#else
    thrust::device,
#endif
    seq.begin(), seq.end());

  // Sort the input data. The original indices of the data are stored in seq
  thrust::sort_by_key(
#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
    thrust::cuda::par(thrustAlloc).on(c10::cuda::getCurrentCUDAStream()),
#else
    thrust::device,
#endif
    iter.begin(), iter.end(), seq.begin()
#if defined(THC_REAL_IS_HALF)
    , ThrustHalfLess()
#endif
  );

  // Count # of unique elements via an inner product between adjacent elements.
  // Add 1 if two neighboring element are not equal.
  int unique = 1 + thrust::inner_product(
#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
    thrust::cuda::par(thrustAlloc).on(c10::cuda::getCurrentCUDAStream()),
#else
    thrust::device,
#endif
    iter.begin(), iter.end() - 1, iter.begin() + 1, 0, thrust::plus<int>(),
#if defined(THC_REAL_IS_HALF)
    ThrustHalfNotEqualTo()
#else
    thrust::not_equal_to<scalar_t>()
#endif
  );

  // Count frequency of each element
  thrust::device_vector<scalar_t> keys(unique);
  thrust::device_vector<int> counts(unique);
  thrust::reduce_by_key(
#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
    thrust::cuda::par(thrustAlloc).on(c10::cuda::getCurrentCUDAStream()),
#else
    thrust::device,
#endif
    iter.begin(), iter.end(),
    thrust::constant_iterator<int>(1), keys.begin(), counts.begin()
#if defined(THC_REAL_IS_HALF)
    , ThrustHalfEqualTo()
#endif
  );

  // Find index of maximum count
  thrust::device_vector<int>::iterator it = thrust::max_element(
#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
    thrust::cuda::par(thrustAlloc).on(c10::cuda::getCurrentCUDAStream()),
#else
    thrust::device,
#endif
    counts.begin(), counts.end());
  scalar_t mode = keys[it - counts.begin()];

  // Find first index within which it occurs
#if defined(THC_REAL_IS_HALF)
  thrust::device_vector<scalar_t>::iterator positionIter = thrust::find_if(
#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
    thrust::cuda::par(thrustAlloc).on(c10::cuda::getCurrentCUDAStream()),
#else
    thrust::device,
#endif
    iter.begin(), iter.end(), ThrustHalfEqualToPredicate(mode));
#else
  thrust::device_vector<scalar_t>::iterator positionIter = thrust::find(
#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
    thrust::cuda::par(thrustAlloc).on(c10::cuda::getCurrentCUDAStream()),
#else
    thrust::device,
#endif
    iter.begin(), iter.end(), mode);
#endif

  THAssert(positionIter != iter.end());
  int64_t index = seq[positionIter - iter.begin()];

  // Place mode, index in output
  ptrdiff_t valuesOffset = THCTensor_(storageOffset)(state, values);
  int64_t indicesOffset = THCudaLongTensor_storageOffset(state, indices);

  for (int i = 0; i < (position->nbytes() / sizeof(int64_t)); ++i) {
    int64_t pos = THLongStorage_data(position)[i];
    valuesOffset += THTensor_strideLegacyNoScalars(values, i) * pos;
    indicesOffset += THTensor_strideLegacyNoScalars(indices, i) * pos;
  }
  THCStorage_(set)(state, THCTensor_(storage)(state, values), valuesOffset, mode);
  THCudaLongStorage_set(state, THCudaLongTensor_storage(state, indices), indicesOffset, index);
}

// this probably could be a loop, not a recursive algorithm
void THCTensor_(dimApplyMode)(THCState *state,
                              THCTensor *values,
                              THCudaLongTensor *indices,
                              THCTensor *input,
                              THCudaLongStorage *sortBuffer,
                              int dimension,
                              THLongStorage *position,
                              int curDim) {
  int64_t ndim = THCTensor_(nDimensionLegacyAll)(state, input);

  // Because we have transposed the Tensor, the data for the dimension we are mode'ing along
  // is always in the innermost dimension
  if (curDim == ndim - 1) {
    THCTensor_(calculateMode)(state, values, indices, input, sortBuffer, dimension, position);
  } else {
    // Loop through the values and recurse
    for (int i = 0; i < THCTensor_(sizeLegacyNoScalars)(state, input, curDim); ++i) {
      THLongStorage_data(position)[curDim] = i;
      THCTensor_(dimApplyMode)(state, values, indices, input, sortBuffer, dimension, position, curDim + 1);
    }
  }
}

#define MAX_GRID_SIZE  65535
#define MAX_BLOCK_SIZE 1024

void THCTensor_(mode)(THCState *state,
                      THCTensor *values,
                      THCudaLongTensor *indices,
                      THCTensor *input,
                      int dimension,
                      int keepdim) {
  THCTensor *transposed, *contiguous, *valuesTransposed;
  THLongStorage *position;
  THCudaLongStorage *sortBuffer;
  THCudaLongTensor *indicesTransposed;
  int64_t ndim, sliceSize, slices;


  THAssert(THCTensor_(checkGPU)(state, 1, values));

  // Verify they are asking for a valid dimension
  ndim = THCTensor_(nDimensionLegacyAll)(state, input);
  THArgCheck(dimension >= 0 && dimension < ndim, 4, "Dimension of out bounds");

  sliceSize = THCTensor_(sizeLegacyNoScalars)(state, input, dimension);
  slices = THCTensor_(nElement)(state, input) / sliceSize;

  // Resize output value, index Tensors to appropriate sizes (i.e. the same as
  // the input Tensor, except at dim=dimension, the size is 1)
  THCTensor_preserveReduceDimSemantics(
      state, values, ndim, dimension, keepdim);
  THCTensor_preserveReduceDimSemantics(
      state, indices, ndim, dimension, keepdim);
  std::vector<int64_t> dim = THTensor_sizesLegacyNoScalars(input);
  dim[dimension] = 1;
  THCTensor_(resize)(state, values, dim, {});
  THCudaLongTensor_resize(state, indices, dim, {});

  // If sliceSize is 1, copy input to values and set indices
  if (sliceSize == 1) {
    THCTensor_(copy)(state, values, input);
    THCudaLongTensor_fill(state, indices, 0);
    if (!keepdim) {
      THCTensor_(squeeze1d)(state, values, values, dimension);
      THCudaLongTensor_squeeze1d(state, indices, indices, dimension);
    }
    return;
  }

  // Requirements for fused kernel implementation:
  //
  // 1. sliceSize <= 2 * max threads per block
  // 2. uses one block per slice, so number of slices must be less than the maximum number of blocks for
  // a kernel launch
  // 3. Can use 32-bit index math for indexing (mainly just for implementation conciseness, could be changed)
  if (sliceSize <= MAX_BLOCK_SIZE &&
      slices <= MAX_GRID_SIZE &&
      THCTensor_canUse32BitIndexMath(state, input)) {
    // Beginning our optimized implementation. First thing we want to do is to transpose
    // the input Tensor along the sort dimension, and then make it contiguous
    transposed = THCTensor_(newTranspose)(state, input, dimension, ndim - 1);
    contiguous = THCTensor_(newContiguous)(state, transposed);

    // We also need to view the values and indices Tensors as transposed in order to
    // properly determine the offset into the underlying storage in which to place the
    // mode and index for a particular set of dimension values
    valuesTransposed = THCTensor_(newTranspose)(state, values, dimension, ndim-1);
    indicesTransposed = THCudaLongTensor_newTranspose(state, indices, dimension, ndim-1);

    // Set-up TensorInfo structs for passing to kernel
    TensorInfo<scalar_t, unsigned int> tiValues = getTensorInfo<scalar_t, THCTensor, unsigned int>(state, valuesTransposed);
    TensorInfo<int64_t, unsigned int> tiIndices = getTensorInfo<int64_t, THCudaLongTensor, unsigned int>(state, indicesTransposed);

    // The number of blocks is the number of slices that we need to calculate the mode for. Each block
    // is responsible for computing a single mode
    dim3 grid;
    THC_getGridFromTiles(slices, grid);

    // The blocksize is two elements per thread, rounded up to the nearest power of 2
    int64_t ceilPowerOf2 = nextHighestPowerOf2(sliceSize);

    // Macro that calls kernel --> note that we set the block dimensions here, and
    // the amount of shared memory
  #define HANDLE_MODE(SIZE)                                                             \
  {                                                                                     \
    const dim3 blockSize(SIZE / 2);                                                     \
    const auto memsize = (sizeof(scalar_t) * SIZE) + (2 * SIZE * sizeof(unsigned int)); \
    computeMode<scalar_t, SIZE>                                                         \
      <<<grid, blockSize, memsize, c10::cuda::getCurrentCUDAStream()>>>(                \
        THCTensor_(data)(state, contiguous), tiValues, tiIndices, sliceSize);           \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                                     \
  }

    // Tradeoff between compilation time and the number of specializations. Ideally we would have
    // one HANDLE_MODE for each power of 2
    switch(ceilPowerOf2) {
      case 2048:
        HANDLE_MODE(2048)
        break;
      case 1024:
      case 512:
      case 256:
        HANDLE_MODE(1024)
        break;
      case 128:
      case 64:
        HANDLE_MODE(128)
        break;
      case 32:
      case 16:
      case 8:
      case 4:
      case 2:
        HANDLE_MODE(32)
        break;
      case 1:
      default:
        TORCH_INTERNAL_ASSERT(false);
    }
    THCudaCheck(cudaGetLastError());

    THCTensor_(free)(state, transposed);
    THCTensor_(free)(state, contiguous);
    THCTensor_(free)(state, valuesTransposed);
    THCudaLongTensor_free(state, indicesTransposed);
  } else {
    // Beginning our naive implementation: We don't want to mutate the input Tensor, but
    // we need to be able to sort the inputs along the dimension in order to calculate the
    // mode. Additionally, its ideal if the data along the dimension is contiguous. So
    // we transpose the dimension with the innermost dimension and make a new contiguous
    // version that we can use.
    transposed = THCTensor_(newClone)(state, input);
    THCTensor_(transpose)(state, transposed, NULL, dimension, ndim - 1);
    contiguous = THCTensor_(newContiguous)(state, transposed);
    THCTensor_(free)(state, transposed);

    // We also need to view the values and indices Tensors as transposed in order to
    // properly determine the offset into the underlying storage in which to place the
    // mode and index for a particular set of dimension values
    valuesTransposed = THCTensor_(newTranspose)(state, values, dimension, ndim - 1);
    indicesTransposed = THCudaLongTensor_newTranspose(state, indices, dimension, ndim - 1);

    // Position is a Storage that will store the dimension values we are processing
    position = THLongStorage_newWithSize(ndim - 1);

    // Sort Buffer is a Storage that will be used in the internal sort required to calculate
    // the mode efficiently
    sortBuffer = THCudaLongStorage_newWithSize(state, sliceSize);

    // Call mode
    THCTensor_(dimApplyMode)(state, valuesTransposed, indicesTransposed, contiguous, sortBuffer, dimension, position, 0);

    THCTensor_(free)(state, contiguous);
    THLongStorage_free(position);
    THCTensor_(free)(state, valuesTransposed);
    THCudaLongTensor_free(state, indicesTransposed);
    THCudaLongStorage_free(state, sortBuffer);
  }

  if (!keepdim) {
    THCTensor_(squeeze1d)(state, values, values, dimension);
    THCudaLongTensor_squeeze1d(state, indices, indices, dimension);
  }
}

#undef MAX_GRID_SIZE
#undef MAX_BLOCK_SIZE

#endif
