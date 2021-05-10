#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorScatterGather.cu"
#else

#include <c10/cuda/CUDAException.h>

#define RUN(TYPE, DIMS, REAL)                                           \
  THCudaTensor_gatherKernel<TYPE, REAL, DIMS>                           \
  <<<grid, block, 0, c10::cuda::getCurrentCUDAStream(curDevice)>>>(     \
    tensorInfo, srcInfo, indexInfo, dim, (TYPE)totalElements);          \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

void THCTensor_(gather)(THCState* state, THCTensor *tensor,
                         THCTensor *src, int dim, THCudaLongTensor *index) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, tensor, src));
  THCAssertSameGPU(THCudaLongTensor_checkGPU(state, 1, index));

  dim = at::maybe_wrap_dim(dim, src);
  THArgCheck(THCudaLongTensor_nDimensionLegacyNoScalars(state, index) == THCTensor_(nDimensionLegacyNoScalars)(state, src), 4,
             "Index tensor must have same dimensions as input tensor");
  THArgCheck(tensor->sizes().equals(index->sizes()), 4,
             "Index tensor must have the same size as output tensor.");
  THArgCheck(dim >= 0 && dim < THCTensor_(nDimensionLegacyNoScalars)(state, tensor), 3,
             "Index dimension is out of bounds");
  THArgCheck(THCTensor_(nDimensionLegacyNoScalars)(state, src) == THCTensor_(nDimensionLegacyNoScalars)(state, tensor), 2,
             "Input tensor must have same dimensions as output tensor");

  for (int d = 0; d < THCTensor_(nDimensionLegacyNoScalars)(state, tensor); d++) {
    if (d != dim) {
      THArgCheck(THCTensor_(sizeLegacyNoScalars)(state, tensor, d) == THCTensor_(sizeLegacyNoScalars)(state, src, d), 2,
                 "Input tensor must have same size as output tensor apart from the specified dimension");
    }
  }

  THArgCheck(THCTensor_(nDimensionLegacyNoScalars)(state, tensor) <= MAX_CUTORCH_DIMS,
             1, CUTORCH_DIM_WARNING);


  const ptrdiff_t totalElements = THCudaLongTensor_nElement(state, index);
  const dim3 block = getApplyBlock();
  dim3 grid;
  int curDevice = -1;
  cudaGetDevice(&curDevice);
  THArgCheck(getApplyGrid(state, totalElements, grid, curDevice), 1, CUTORCH_DIM_WARNING);

  THCTensor* oldTensor = NULL;
  if (THCTensor_maybeOverlappingIndices(state, tensor)) {
    oldTensor = tensor;
    tensor = THCTensor_(newContiguous)(state, tensor);
  }

  if (totalElements > 0) {
    if (THCTensor_canUse32BitIndexMath(state, tensor) &&
        THCTensor_canUse32BitIndexMath(state, src) &&
        THCTensor_canUse32BitIndexMath(state, index)) {
      TensorInfo<scalar_t, unsigned int> tensorInfo =
        getTensorInfo<scalar_t, THCTensor, unsigned int>(state, tensor);
      TensorInfo<scalar_t, unsigned int> srcInfo =
        getTensorInfo<scalar_t, THCTensor, unsigned int>(state, src);
      TensorInfo<int64_t, unsigned int> indexInfo =
        getTensorInfo<int64_t, THCudaLongTensor, unsigned int>(state, index);

      // Specialize for a small number of dimensions.
      switch (indexInfo.dims) {
        case 1:
          RUN(unsigned int, 1, scalar_t);
          break;
        case 2:
          RUN(unsigned int, 2, scalar_t);
          break;
        case 3:
          RUN(unsigned int, 3, scalar_t);
          break;
        default:
          RUN(unsigned int, -1, scalar_t);
          break;
      }
    } else {
      TensorInfo<scalar_t, uint64_t> tensorInfo =
        getTensorInfo<scalar_t, THCTensor, uint64_t>(state, tensor);
      TensorInfo<scalar_t, uint64_t> srcInfo =
        getTensorInfo<scalar_t, THCTensor, uint64_t>(state, src);
      TensorInfo<int64_t, uint64_t> indexInfo =
        getTensorInfo<int64_t, THCudaLongTensor, uint64_t>(state, index);
      RUN(uint64_t, -1, scalar_t);
    }
  }

  if (oldTensor) {
    THCTensor_copyIgnoringOverlaps<scalar_t>(state, oldTensor, tensor);
    THCTensor_(free)(state, tensor);
    tensor = oldTensor;
  }
  THCudaCheck(cudaGetLastError());
}

#undef RUN

#endif
