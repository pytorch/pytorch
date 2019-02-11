#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorTopK.cu"
#else

void THCTensor_(topk)(THCState* state,
                      THCTensor *topK,
                      THCudaLongTensor *indices,
                      THCTensor *input_,
                      int64_t k, int dim, int dir, int sorted) {
  THAssert(topK != NULL && indices != NULL && input_ != NULL);
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, topK, indices, input_));
  THArgCheck(THCTensor_(nDimensionLegacyNoScalars)(state, topK) <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  int64_t dims = THCudaLongTensor_nDimensionLegacyNoScalars(state, indices);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 3, CUTORCH_DIM_WARNING);
  int numDims = THCTensor_(nDimensionLegacyNoScalars)(state, input_);
  THArgCheck(numDims <= MAX_CUTORCH_DIMS, 4, CUTORCH_DIM_WARNING);

  THArgCheck(dim >= 0 && dim < numDims, 6, "dim not in range");

  int64_t sliceSize = THCTensor_(sizeLegacyNoScalars)(state, input_, dim);
  THArgCheck(k >= 0 && k <= sliceSize, 5, "k not in range for dimension");

  THCTensor *input = THCTensor_(newContiguous)(state, input_);

  // Build the output size, which is the dim being selected set to
  // size k
  std::vector<int64_t> topKSize = THTensor_sizesLegacyNoScalars(input);
  topKSize[dim] = k;
  THCTensor_(resize)(state, topK, topKSize, {});
  THCudaLongTensor_resize(state, indices, topKSize, {});

  // static_cast is required to ensure that the correct type (INDEX_T)
  // is provided to the kernel for the arguments.

#define RUN_K(INDEX_T, DIM, DIR)                                        \
  gatherTopK<scalar_t, INDEX_T, DIM, DIR>                                   \
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(             \
      inputInfo,                                                        \
      static_cast<INDEX_T>(sliceSize),                                  \
      static_cast<INDEX_T>(k),                                          \
      static_cast<INDEX_T>(inputSlices),                                \
      /* The actual dimension that the k-selection is running in */     \
      /* may have changed from collapseDims() */                        \
      static_cast<INDEX_T>(inputInfo.strides[collapseInputDim]),        \
      topKInfo,                                                         \
      static_cast<INDEX_T>(topKSlices),                                 \
      static_cast<INDEX_T>(topKInfo.strides[collapseTopKDim]),          \
      indicesInfo,                                                      \
      static_cast<INDEX_T>(indicesInfo.strides[collapseIndicesDim]))

#define RUN_DIR(INDEX_T, DIM)                   \
  if (dir) {                                    \
    RUN_K(INDEX_T, DIM, true);                  \
  } else {                                      \
    RUN_K(INDEX_T, DIM, false);                 \
  }

#define RUN_DIM(INDEX_T)                        \
  if (allDims == 1) {                           \
    RUN_DIR(INDEX_T, 1);                        \
  } else if (allDims == 2) {                    \
    RUN_DIR(INDEX_T, 2);                        \
  } else if (allDims == 3) {                    \
    RUN_DIR(INDEX_T, 3);                        \
  } else {                                      \
    RUN_DIR(INDEX_T, -1);                       \
  }

#ifdef __HIP_PLATFORM_HCC__
#define TOPK_WARP_SIZE 64
#else
#define TOPK_WARP_SIZE 32
#endif

#define RUN_T(INDEX_T)                                                  \
  TensorInfo<scalar_t, INDEX_T> inputInfo =                                 \
    getTensorInfo<scalar_t, THCTensor, INDEX_T>(state, input);              \
  TensorInfo<scalar_t, INDEX_T> topKInfo =                                  \
    getTensorInfo<scalar_t, THCTensor, INDEX_T>(state, topK);               \
  TensorInfo<int64_t, INDEX_T> indicesInfo =                            \
    getTensorInfo<int64_t, THCudaLongTensor, INDEX_T>(state, indices);  \
                                                                        \
  /* We use these structures solely to find the offset to */            \
  /* each slice we are operating on */                                  \
  inputInfo.sizes[dim] = 1;                                             \
  topKInfo.sizes[dim] = 1;                                              \
  indicesInfo.sizes[dim] = 1;                                           \
                                                                        \
  /* Collapse all other dims */                                         \
  int collapseInputDim = inputInfo.collapseDims(dim);                   \
  int collapseTopKDim = topKInfo.collapseDims(dim);                     \
  int collapseIndicesDim = indicesInfo.collapseDims(dim);               \
                                                                        \
  int64_t inputSlices = 1;                                              \
  for (int i = 0; i < inputInfo.dims; ++i) {                            \
    inputSlices *= inputInfo.sizes[i];                                  \
  }                                                                     \
  int64_t topKSlices = 1;                                               \
  for (int i = 0; i < topKInfo.dims; ++i) {                             \
    topKSlices *= topKInfo.sizes[i];                                    \
  }                                                                     \
                                                                        \
  dim3 grid;                                                            \
  if (!THC_getGridFromTiles(inputSlices, grid)) {                       \
    THError("Slice to sort is too large");                              \
  }                                                                     \
                                                                        \
  dim3 block(std::min(THCRoundUp(sliceSize, (int64_t) TOPK_WARP_SIZE), (int64_t) 1024)); \
                                                                        \
  /* This is used as a template parameter to calculate indices. */      \
  /* We only specialize it if all collapsed dim sizes are the */        \
  /* same; otherwise, we use -1 which is the specialization */          \
  /* parameter for arbitrary dimensions */                              \
  int allDims = inputInfo.dims;                                         \
  if (topKInfo.dims != allDims || indicesInfo.dims != allDims) {        \
    allDims = -1;                                                       \
  }                                                                     \
                                                                        \
  RUN_DIM(INDEX_T);

  if (THCTensor_nElement(state, input) > 0) {
    // Based on required index size, run the algorithm with the
    // appropriate index type
    if (THCTensor_canUse32BitIndexMath(state, input) &&
        THCTensor_canUse32BitIndexMath(state, topK) &&
        THCTensor_canUse32BitIndexMath(state, indices)) {
      RUN_T(uint32_t);
    } else {
      RUN_T(uint64_t);
    }
  }
#undef RUN_T
#undef RUN_DIM
#undef RUN_DIR
#undef RUN_K
#undef TOPK_WARP_SIZE

  // Sort the results if the user wants them sorted, since our
  // selection routine does not ensure sorting
  if (sorted) {
    // FIXME: the k/v inplace sort along slice only works for size <=
    // 2048 at the moment
    if (sliceSize <= 2048) {
      // This avoids any memory allocations and performs all sorting
      // work inplace along the slice
      THCTensor_(sortKeyValueInplace)(state, topK, indices, dim, dir);
    } else {
      // Depend upon the backup sort that returns indices, which we
      // can use in conjunction with gather to produce the original
      // indices.
      // This is not the most efficient implementation, especially since
      // there are memory allocations performed here. If the user desires
      // greater performance, they should torch.gather() the results
      // themselves using the reported indices, providing previously
      // allocated tensors to receive the results.
      THCTensor* sortedTopK = THCTensor_(new)(state);
      THCudaLongTensor* sortedIndices = THCudaLongTensor_new(state);
      THCTensor_(sort)(state, sortedTopK, sortedIndices, topK, dim, dir);

      THCudaLongTensor* sortedTopKIndices = THCudaLongTensor_new(state);

      THCudaLongTensor_resizeAs(state, sortedTopKIndices, indices);
      THCudaLongTensor_gather(state, sortedTopKIndices, indices, dim, sortedIndices);

      THCTensor_(freeCopyTo)(state, sortedTopK, topK);
      THCudaLongTensor_freeCopyTo(state, sortedTopKIndices, indices);
      THCudaLongTensor_free(state, sortedIndices);
    }
  }

  THCudaLongTensor_free(state, input);

  THCudaCheck(cudaGetLastError());
}

#endif // THC_GENERIC_FILE
