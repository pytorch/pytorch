#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorKthValue.cu"
#else

THC_API void THCTensor_(kthvalue)(THCState* state,
                                  THCTensor* kthValue,
                                  THCudaLongTensor* indices,
                                  THCTensor* input,
                                  int64_t k, int dim, int keepDim) {
  THAssert(kthValue != NULL && indices != NULL && input != NULL);
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, kthValue, indices, input));
  THArgCheck(THCTensor_(nDimension)(state, kthValue) <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  int64_t dims = THCudaLongTensor_nDimension(state, indices);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 3, CUTORCH_DIM_WARNING);
  int numDims = THCTensor_(nDimension)(state, input);
  THArgCheck(numDims <= MAX_CUTORCH_DIMS, 4, CUTORCH_DIM_WARNING);

  THArgCheck(dim >= 0 && dim < numDims, 6, "dim not in range");

  int64_t sliceSize = THCTensor_(size)(state, input, dim);
  THArgCheck(k > 0 && k <= sliceSize, 5, "k not in range for dimension");

  // Build the output size, which is the dim being selected set to
  // size 1
  THLongStorage* kthValueSize = THCTensor_(newSizeOf)(state, input);
  THLongStorage_set(kthValueSize, dim, 1);
  THCTensor_(resize)(state, kthValue, kthValueSize, NULL);
  THCudaLongTensor_resize(state, indices, kthValueSize, NULL);
  THLongStorage_free(kthValueSize);

  #define RUN_K(INDEX_T, DIM)                                             \
    gatherKthValue<real, INDEX_T, DIM>                                    \
      <<<grid, block, 0, THCState_getCurrentStream(state)>>>(             \
        inputInfo,                                                        \
        sliceSize,                                                        \
        k,                                                                \
        inputSlices,                                                      \
        /* The actual dimension that the k-selection is running in */     \
        /* may have changed from collapseDims() */                        \
        inputInfo.strides[collapseInputDim],                              \
        kthValueInfo,                                                     \
        indicesInfo)

  #define RUN_DIM(INDEX_T)                        \
    if (allDims == 1) {                           \
      RUN_K(INDEX_T, 1);                          \
    } else if (allDims == 2) {                    \
      RUN_K(INDEX_T, 2);                          \
    } else if (allDims == 3) {                    \
      RUN_K(INDEX_T, 3);                          \
    } else {                                      \
      RUN_K(INDEX_T, -1);                         \
    }

  #define RUN_T(INDEX_T)                                                  \
    TensorInfo<real, INDEX_T> inputInfo =                                 \
      getTensorInfo<real, THCTensor, INDEX_T>(state, input);              \
    TensorInfo<real, INDEX_T> kthValueInfo =                              \
      getTensorInfo<real, THCTensor, INDEX_T>(state, kthValue);           \
    TensorInfo<int64_t, INDEX_T> indicesInfo =                            \
      getTensorInfo<int64_t, THCudaLongTensor, INDEX_T>(state, indices);  \
                                                                          \
    /* We use these structures solely to find the offset to */            \
    /* each slice we are operating on */                                  \
    inputInfo.sizes[dim] = 1;                                             \
    kthValueInfo.sizes[dim] = 1;                                          \
    indicesInfo.sizes[dim] = 1;                                           \
                                                                          \
    /* Collapse all other dims */                                         \
    int collapseInputDim = inputInfo.collapseDims(dim);                   \
    int collapseKthValueDim = kthValueInfo.collapseDims(dim);             \
    int collapseIndicesDim = indicesInfo.collapseDims(dim);               \
                                                                          \
    int64_t inputSlices = 1;                                              \
    for (int i = 0; i < inputInfo.dims; ++i) {                            \
      inputSlices *= inputInfo.sizes[i];                                  \
    }                                                                     \
    int64_t kthValueSlices = 1;                                           \
    for (int i = 0; i < kthValueInfo.dims; ++i) {                         \
      kthValueSlices *= kthValueInfo.sizes[i];                            \
    }                                                                     \
                                                                          \
    dim3 grid;                                                            \
    if (!THC_getGridFromTiles(inputSlices, grid)) {                       \
      THError("Slice to select is too large");                            \
    }                                                                     \
                                                                          \
    dim3 block(std::min(THCRoundUp(sliceSize, (int64_t) 32), (int64_t) 1024)); \
                                                                          \
    /* This is used as a template parameter to calculate indices. */      \
    /* We only specialize it if all collapsed dim sizes are the */        \
    /* same; otherwise, we use -1 which is the specialization */          \
    /* parameter for arbitrary dimensions */                              \
    int allDims = inputInfo.dims;                                         \
    if (kthValueInfo.dims != allDims || indicesInfo.dims != allDims) {    \
      allDims = -1;                                                       \
    }                                                                     \
                                                                          \
    RUN_DIM(INDEX_T);

  // Based on required index size, run the algorithm with the
  // appropriate index type
  if (THCTensor_canUse32BitIndexMath(state, input) &&
      THCTensor_canUse32BitIndexMath(state, kthValue) &&
      THCTensor_canUse32BitIndexMath(state, indices)) {
    RUN_T(uint32_t);
  } else {
    RUN_T(uint64_t);
  }
  #undef RUN_T
  #undef RUN_DIM
  #undef RUN_K

  if (!keepDim) {
    THCTensor_(select)(state, kthValue, NULL, dim, 0);
    THCudaLongTensor_select(state, indices, NULL, dim, 0);
  }

  THCudaCheck(cudaGetLastError());
}

#endif // THC_GENERIC_FILE
