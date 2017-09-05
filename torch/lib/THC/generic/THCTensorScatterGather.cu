#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorScatterGather.cu"
#else

#define RUN(TYPE, DIMS, REAL)                                           \
  THCudaTensor_gatherKernel<TYPE, REAL, DIMS>                                \
  <<<grid, block, 0, THCState_getCurrentStream(state)>>>(               \
    tensorInfo, srcInfo, indexInfo, dim, (TYPE)totalElements);

void THCTensor_(gather)(THCState* state, THCTensor *tensor,
                         THCTensor *src, int dim, THCudaLongTensor *index) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, tensor, src));
  THCAssertSameGPU(THCudaLongTensor_checkGPU(state, 1, index));

  THArgCheck(THCTensor_(nDimension)(state, src) == THCTensor_(nDimension)(state, tensor), 2,
             "Input tensor must have same dimensions as output tensor");
  THArgCheck(dim >= 0 && dim < THCTensor_(nDimension)(state, tensor), 3,
             "Index dimension is out of bounds");
  THArgCheck(THCudaLongTensor_nDimension(state, index) == THCTensor_(nDimension)(state, src), 4,
             "Index tensor must have same dimensions as input tensor");
  THLongStorage *indexSize = THCudaLongTensor_newSizeOf(state, index);
  THArgCheck(THCTensor_(isSize)(state, tensor, indexSize), 4,
             "Index tensor must have the same size as output tensor.");
  THLongStorage_free(indexSize);

  for (int d = 0; d < THCTensor_(nDimension)(state, tensor); d++) {
    if (d != dim) {
      THArgCheck(THCTensor_(size)(state, tensor, d) == THCTensor_(size)(state, src, d), 2,
                 "Input tensor must have same size as output tensor apart from the specified dimension");
    }
  }

  THArgCheck(THCTensor_(nDimension)(state, tensor) <= MAX_CUTORCH_DIMS,
             1, CUTORCH_DIM_WARNING);


  const ptrdiff_t totalElements = THCudaLongTensor_nElement(state, index);
  const dim3 block = getApplyBlock();
  dim3 grid;
  THArgCheck(getApplyGrid(state, totalElements, grid), 1, CUTORCH_DIM_WARNING);

  THCTensor* oldTensor = NULL;
  if (TensorUtils<THCTensor>::overlappingIndices(state, tensor)) {
    oldTensor = tensor;
    tensor = THCTensor_(newContiguous)(state, tensor);
  }

  if (TensorUtils<THCTensor>::canUse32BitIndexMath(state, tensor) &&
      TensorUtils<THCTensor>::canUse32BitIndexMath(state, src) &&
      TensorUtils<THCudaLongTensor>::canUse32BitIndexMath(state, index)) {
    TensorInfo<real, unsigned int> tensorInfo =
      getTensorInfo<THCTensor, unsigned int>(state, tensor);
    TensorInfo<real, unsigned int> srcInfo =
      getTensorInfo<THCTensor, unsigned int>(state, src);
    TensorInfo<long, unsigned int> indexInfo =
      getTensorInfo<THCudaLongTensor, unsigned int>(state, index);

    // Specialize for a small number of dimensions.
    switch (indexInfo.dims) {
      case 1:
        RUN(unsigned int, 1, real);
        THCudaCheck(cudaGetLastError());
        break;
      case 2:
        RUN(unsigned int, 2, real);
        THCudaCheck(cudaGetLastError());
        break;
      case 3:
        RUN(unsigned int, 3, real);
        THCudaCheck(cudaGetLastError());
        break;
      default:
        RUN(unsigned int, -1, real);
        THCudaCheck(cudaGetLastError());
        break;
    }
  } else {
    TensorInfo<real, unsigned long> tensorInfo =
      getTensorInfo<THCTensor, unsigned long>(state, tensor);
    TensorInfo<real, unsigned long> srcInfo =
      getTensorInfo<THCTensor, unsigned long>(state, src);
    TensorInfo<long, unsigned long> indexInfo =
      getTensorInfo<THCudaLongTensor, unsigned long>(state, index);
    RUN(unsigned long, -1, real);
    THCudaCheck(cudaGetLastError());
  }

  if (oldTensor) {
    TensorUtils<THCTensor>::copyIgnoringOverlaps(state, oldTensor, tensor);
    THCTensor_(free)(state, tensor);
    tensor = oldTensor;
  }
  THCudaCheck(cudaGetLastError());
}

#undef RUN


#define RUN(TYPE, DIMS, REAL)                                           \
  THCudaTensor_scatterKernel<TYPE, REAL, DIMS>                               \
  <<<grid, block, 0, THCState_getCurrentStream(state)>>>(               \
    tensorInfo, srcInfo, indexInfo, dim, (TYPE)totalElements);

void THCTensor_(scatter)(THCState* state, THCTensor *tensor, int dim, THCudaLongTensor *index, THCTensor *src) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, tensor, src));
  THCAssertSameGPU(THCudaLongTensor_checkGPU(state, 1, index));

  THArgCheck(dim >= 0 && dim < THCTensor_(nDimension)(state, tensor), 2,
             "Index dimension is out of bounds");
  THArgCheck(THCudaLongTensor_nDimension(state, index) == THCTensor_(nDimension)(state, src), 3,
             "Index tensor must have same dimensions as input tensor");
  THArgCheck(THCTensor_(nDimension)(state, src) == THCTensor_(nDimension)(state, tensor), 4,
             "Input tensor must have same dimensions as output tensor");
  THLongStorage *indexDims = THCudaLongTensor_newSizeOf(state, index);
  THArgCheck(THCTensor_(isSize)(state, src, indexDims), 3,
             "Index tensor must have the same size as input tensor.");
  THLongStorage_free(indexDims);

  for (int d = 0; d < THCTensor_(nDimension)(state, tensor); d++) {
    if (d != dim) {
      THArgCheck(THCTensor_(size)(state, tensor, d) == THCTensor_(size)(state, src, d), 4,
                 "Input tensor must have same size as output tensor apart from the specified dimension");
    }
  }

  THArgCheck(THCTensor_(nDimension)(state, tensor) <= MAX_CUTORCH_DIMS,
             1, CUTORCH_DIM_WARNING);

  const ptrdiff_t totalElements = THCudaLongTensor_nElement(state, index);
  const dim3 block = getApplyBlock();
  dim3 grid;
  THArgCheck(getApplyGrid(state, totalElements, grid), 1, CUTORCH_DIM_WARNING);

  THCTensor* oldTensor = NULL;
  if (TensorUtils<THCTensor>::overlappingIndices(state, tensor)) {
    oldTensor = tensor;
    tensor = THCTensor_(newContiguous)(state, tensor);
  }

  if (TensorUtils<THCTensor>::canUse32BitIndexMath(state, tensor) &&
      TensorUtils<THCTensor>::canUse32BitIndexMath(state, src) &&
      TensorUtils<THCudaLongTensor>::canUse32BitIndexMath(state, index)) {
    TensorInfo<real, unsigned int> tensorInfo =
      getTensorInfo<THCTensor, unsigned int>(state, tensor);
    TensorInfo<real, unsigned int> srcInfo =
      getTensorInfo<THCTensor, unsigned int>(state, src);
    TensorInfo<long, unsigned int> indexInfo =
      getTensorInfo<THCudaLongTensor, unsigned int>(state, index);

    // Specialize for a small number of dimensions.
    switch (indexInfo.dims) {
      case 1:
        RUN(unsigned int, 1, real);
        break;
      case 2:
        RUN(unsigned int, 2, real);
        break;
      case 3:
        RUN(unsigned int, 3, real);
        break;
      default:
        RUN(unsigned int, -1, real);
        break;
    }
  } else {
    TensorInfo<real, unsigned long> tensorInfo =
      getTensorInfo<THCTensor, unsigned long>(state, tensor);
    TensorInfo<real, unsigned long> srcInfo =
      getTensorInfo<THCTensor, unsigned long>(state, src);
    TensorInfo<long, unsigned long> indexInfo =
      getTensorInfo<THCudaLongTensor, unsigned long>(state, index);

    RUN(unsigned long, -1, real)
  }

  if (oldTensor) {
    TensorUtils<THCTensor>::copyIgnoringOverlaps(state, oldTensor, tensor);
    THCTensor_(free)(state, tensor);
    tensor = oldTensor;
  }
  THCudaCheck(cudaGetLastError());
}

#undef RUN

#define RUN(TYPE, DIMS, REAL)                                           \
  THCudaTensor_scatterAddKernel<TYPE, REAL, DIMS>                               \
  <<<grid, block, 0, THCState_getCurrentStream(state)>>>(               \
    tensorInfo, srcInfo, indexInfo, dim, (TYPE)totalElements);

void THCTensor_(scatterAdd)(THCState* state, THCTensor *tensor, int dim, THCudaLongTensor *index, THCTensor *src) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, tensor, src));
  THCAssertSameGPU(THCudaLongTensor_checkGPU(state, 1, index));

  THArgCheck(dim >= 0 && dim < THCTensor_(nDimension)(state, tensor), 2,
             "Index dimension is out of bounds");
  THArgCheck(THCudaLongTensor_nDimension(state, index) == THCTensor_(nDimension)(state, src), 3,
             "Index tensor must have same dimensions as input tensor");
  THArgCheck(THCTensor_(nDimension)(state, src) == THCTensor_(nDimension)(state, tensor), 4,
             "Input tensor must have same dimensions as output tensor");
  THLongStorage *indexDims = THCudaLongTensor_newSizeOf(state, index);
  THArgCheck(THCTensor_(isSize)(state, src, indexDims), 3,
             "Index tensor must have the same size as input tensor.");
  THLongStorage_free(indexDims);

  for (int d = 0; d < THCTensor_(nDimension)(state, tensor); d++) {
    if (d != dim) {
      THArgCheck(THCTensor_(size)(state, tensor, d) == THCTensor_(size)(state, src, d), 4,
                 "Input tensor must have same size as output tensor apart from the specified dimension");
    }
  }

  THArgCheck(THCTensor_(nDimension)(state, tensor) <= MAX_CUTORCH_DIMS,
             1, CUTORCH_DIM_WARNING);

  const ptrdiff_t totalElements = THCudaLongTensor_nElement(state, index);
  const dim3 block = getApplyBlock();
  dim3 grid;
  THArgCheck(getApplyGrid(state, totalElements, grid), 1, CUTORCH_DIM_WARNING);

  THCTensor* oldTensor = NULL;
  if (TensorUtils<THCTensor>::overlappingIndices(state, tensor)) {
    oldTensor = tensor;
    tensor = THCTensor_(newContiguous)(state, tensor);
  }

  if (TensorUtils<THCTensor>::canUse32BitIndexMath(state, tensor) &&
      TensorUtils<THCTensor>::canUse32BitIndexMath(state, src) &&
      TensorUtils<THCudaLongTensor>::canUse32BitIndexMath(state, index)) {
    TensorInfo<real, unsigned int> tensorInfo =
      getTensorInfo<THCTensor, unsigned int>(state, tensor);
    TensorInfo<real, unsigned int> srcInfo =
      getTensorInfo<THCTensor, unsigned int>(state, src);
    TensorInfo<long, unsigned int> indexInfo =
      getTensorInfo<THCudaLongTensor, unsigned int>(state, index);

    // Specialize for a small number of dimensions.
    switch (indexInfo.dims) {
      case 1:
        RUN(unsigned int, 1, real);
        break;
      case 2:
        RUN(unsigned int, 2, real);
        break;
      case 3:
        RUN(unsigned int, 3, real);
        break;
      default:
        RUN(unsigned int, -1, real);
        break;
    }
  } else {
    TensorInfo<real, unsigned long> tensorInfo =
      getTensorInfo<THCTensor, unsigned long>(state, tensor);
    TensorInfo<real, unsigned long> srcInfo =
      getTensorInfo<THCTensor, unsigned long>(state, src);
    TensorInfo<long, unsigned long> indexInfo =
      getTensorInfo<THCudaLongTensor, unsigned long>(state, index);

    RUN(unsigned long, -1, real)
  }

  if (oldTensor) {
    TensorUtils<THCTensor>::copyIgnoringOverlaps(state, oldTensor, tensor);
    THCTensor_(free)(state, tensor);
    tensor = oldTensor;
  }
  THCudaCheck(cudaGetLastError());
}

#undef RUN

#define RUN(TYPE, DIMS, REAL)                                           \
  THCudaTensor_scatterFillKernel<TYPE, REAL, DIMS>                           \
      <<<grid, block, 0, THCState_getCurrentStream(state)>>>(      \
          tensorInfo, indexInfo, value, dim, (TYPE)totalElements);

void
THCTensor_(scatterFill)(THCState* state, THCTensor *tensor,
                         int dim, THCudaLongTensor *index, real value) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, tensor));
  THCAssertSameGPU(THCudaLongTensor_checkGPU(state, 1, index));

  THArgCheck(dim >= 0 && dim < THCTensor_(nDimension)(state, tensor), 2,
             "Index dimension is out of bounds");
  THArgCheck(THCudaLongTensor_nDimension(state, index) ==
             THCTensor_(nDimension)(state, tensor), 3,
             "Index tensor must have same dimensions as output tensor");

  for (int d = 0; d < THCTensor_(nDimension)(state, tensor); d++) {
    if (d != dim) {
      THArgCheck(THCTensor_(size)(state, tensor, d) ==
                 THCudaLongTensor_size(state, index, d), 4,
                 "Index tensor must have same size as output tensor apart from the specified dimension");
    }
  }

  THArgCheck(THCTensor_(nDimension)(state, tensor) <= MAX_CUTORCH_DIMS,
             1, CUTORCH_DIM_WARNING);

  const ptrdiff_t totalElements = THCudaLongTensor_nElement(state, index);
  const dim3 block = getApplyBlock();
  dim3 grid;
  THArgCheck(getApplyGrid(state, totalElements, grid), 1, CUTORCH_DIM_WARNING);

  THCTensor* oldTensor = NULL;
  if (TensorUtils<THCTensor>::overlappingIndices(state, tensor)) {
    oldTensor = tensor;
    tensor = THCTensor_(newContiguous)(state, tensor);
  }

  if (TensorUtils<THCTensor>::canUse32BitIndexMath(state, tensor) &&
      TensorUtils<THCudaLongTensor>::canUse32BitIndexMath(state, index)) {
    TensorInfo<real, unsigned int> tensorInfo =
      getTensorInfo<THCTensor, unsigned int>(state, tensor);
    TensorInfo<long, unsigned int> indexInfo =
      getTensorInfo<THCudaLongTensor, unsigned int>(state, index);

    // Specialize for a small number of dimensions.
    switch (indexInfo.dims) {
      case 1:
        RUN(unsigned int, 1, real);
        break;
      case 2:
        RUN(unsigned int, 2, real);
        break;
      case 3:
        RUN(unsigned int, 3, real);
        break;
      default:
        RUN(unsigned int, -1, real);
        break;
    }
  } else {
    TensorInfo<real, unsigned long> tensorInfo =
      getTensorInfo<THCTensor, unsigned long>(state, tensor);
    TensorInfo<long, unsigned long> indexInfo =
      getTensorInfo<THCudaLongTensor, unsigned long>(state, index);

    RUN(unsigned long, -1, real);
  }

  if (oldTensor) {
    TensorUtils<THCTensor>::copyIgnoringOverlaps(state, oldTensor, tensor);
    THCTensor_(free)(state, tensor);
    tensor = oldTensor;
  }
  THCudaCheck(cudaGetLastError());
}

#undef RUN

#endif
