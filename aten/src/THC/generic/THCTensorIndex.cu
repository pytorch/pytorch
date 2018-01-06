#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorIndex.cu"
#else

// Check tensor dimensions for index operations, and return the slice size.
// src can be nullptr in case of indexFill: in that case it is ignored.
static ptrdiff_t THCTensor_(getSliceSize)(THCState *state, THCTensor *dst,
                                          int dim,
                                          THCudaLongTensor *index,
                                          THCTensor *src)
{
  int dstDims = THCTensor_(nDimension)(state, dst);
  int srcDims = (src == nullptr) ? dstDims : THCTensor_(nDimension)(state, src);

  THArgCheck(THCudaLongTensor_nDimension(state, index) == 1, 4,
             "expecting vector of indices");
  THArgCheck(dim >= 0 && dim < dstDims, 2, "Indexing dim is out of bounds");

  ptrdiff_t dstSliceSize = 1;
  for (int d = 0; d < dstDims; d++) {
    if (d != dim) {
      dstSliceSize *= dst->size[d];
    }
  }

  if (src == nullptr) return dstSliceSize;

  THArgCheck(dim < srcDims, 3, "Indexing dim is out of bounds");
  THArgCheck(THCudaLongTensor_nElement(state, index) == src->size[dim], 4,
             "length of src.size[dim] is not equal to length of indices");

  ptrdiff_t srcSliceSize = 1;
  bool mismatch = false;

  if (dstDims != srcDims) mismatch = true;

  for (int d = 0; d < srcDims; d++) {
    if (d != dim) {
      srcSliceSize *= src->size[d];
      if (!mismatch && dst->size[d] != src->size[d]) mismatch = true;
    }
  }

  THArgCheck(dstSliceSize == srcSliceSize, 2,
             "Source/destination tensor have different slice sizes (%ld vs %ld)",
             dstSliceSize, srcSliceSize);

  if (mismatch) {
    static bool warningShown = false;
    if (!warningShown) {
      warningShown = true;
      fprintf(stderr,
              "Warning: source/destination slices have same size but different "
              "shape for an index operation.  This behavior is deprecated.\n");
    }
  }

  return dstSliceSize;
}

void THCTensor_(indexCopy_long)(THCState *state, THCTensor *dst, int dim, THLongTensor *indices, THCTensor *src)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, dst, src));

  THCudaLongTensor *indices_ = THCudaLongTensor_newWithSize1d(state, indices->size[0]);
  THCudaLongTensor_copyLong(state, indices_, indices);

  THCTensor_(indexCopy)(state, dst, dim, indices_, src);

  THCudaLongTensor_free(state, indices_);
}

void THCTensor_(indexCopy)(THCState *state, THCTensor *dst, int dim, THCudaLongTensor *indices, THCTensor *src)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, dst, src));
  THCAssertSameGPU(THCudaLongTensor_checkGPU(state, 1, indices));

  int dims = THCTensor_(nDimension)(state, dst);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  dims = THCTensor_(nDimension)(state, src);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 5, CUTORCH_DIM_WARNING);
  dims = THCudaLongTensor_nDimension(state, indices);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 4, CUTORCH_DIM_WARNING);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  ptrdiff_t sliceSize = THCTensor_(getSliceSize)(state, dst, dim, indices, src);
  ptrdiff_t srcTotalSize = THCTensor_(nElement)(state, src);
  int64_t dstCopyDimSize = THCTensor_(size)(state, dst, dim);

  ptrdiff_t numIndices = THCudaLongTensor_nElement(state, indices);
  cudaStream_t stream = THCState_getCurrentStream(state);
  int indContig = THCudaLongTensor_isContiguous(state, indices);

  int mpc = THCState_getCurrentDeviceProperties(state)->multiProcessorCount;

#define SMALL_INDEX(TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM) \
  indexCopySmallIndex<TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM>       \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(           \
      dstInfo, srcInfo, indicesInfo,                            \
      dstCopyDim, srcCopyDim, sliceSize, dstCopyDimSize);

#define LARGE_INDEX(TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM) \
  indexCopyLargeIndex<TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM>       \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(           \
      dstInfo, srcInfo, indicesInfo,                            \
      dstCopyDim, srcCopyDim, sliceSize, dstCopyDimSize);

  dim3 smallIndexGrid(std::min(THCCeilDiv(sliceSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 smallIndexBlock(std::min(sliceSize, (ptrdiff_t)128));

  dim3 largeIndexGrid(std::min(THCCeilDiv(srcTotalSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 largeIndexBlock(std::min(srcTotalSize, (ptrdiff_t)128));

  if (TensorUtils<THCTensor>::canUse32BitIndexMath(state, dst) &&
      TensorUtils<THCTensor>::canUse32BitIndexMath(state, src) &&
      TensorUtils<THCudaLongTensor>::canUse32BitIndexMath(state, indices)) {
    TensorInfo<real, unsigned int> dstInfo =
      getTensorInfo<THCTensor, unsigned int>(state, dst);
    int dstCopyDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstCopyDim);

    TensorInfo<real, unsigned int> srcInfo =
      getTensorInfo<THCTensor, unsigned int>(state, src);
    int srcCopyDim = srcInfo.collapseDims(dim);
    srcInfo.reduceDim(srcCopyDim);

    TensorInfo<int64_t, unsigned int> indicesInfo =
      getTensorInfo<THCudaLongTensor, unsigned int>(state, indices);
    indicesInfo.collapseDims();

    // A reasonable choice for when to have each thread iterate over
    // indices to choose
    if (numIndices <= 16) {
      if (dstInfo.dims == 1 && srcInfo.dims == 1 && indContig) {
        SMALL_INDEX(real, unsigned int, 1, 1, -2);
      } else if (dstInfo.dims == 2 && srcInfo.dims == 2 && indContig) {
        SMALL_INDEX(real, unsigned int, 2, 2, -2);
      } else if (dstInfo.dims == 3 && srcInfo.dims == 3 && indContig) {
        SMALL_INDEX(real, unsigned int, 3, 3, -2);
      } else {
        SMALL_INDEX(real, unsigned int, -1, -1, -1);
      }
    } else {
      if (dstInfo.dims == 1 && srcInfo.dims == 1 && indContig) {
        LARGE_INDEX(real, unsigned int, 1, 1, -2);
      } else if (dstInfo.dims == 2 && srcInfo.dims == 2 && indContig) {
        LARGE_INDEX(real, unsigned int, 2, 2, -2);
      } else if (dstInfo.dims == 3 && srcInfo.dims == 3 && indContig) {
        LARGE_INDEX(real, unsigned int, 3, 3, -2);
      } else {
        LARGE_INDEX(real, unsigned int, -1, -1, -1);
      }
    }
  } else {
    TensorInfo<real, uint64_t> dstInfo =
      getTensorInfo<THCTensor, uint64_t>(state, dst);
    int dstCopyDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstCopyDim);

    TensorInfo<real, uint64_t> srcInfo =
      getTensorInfo<THCTensor, uint64_t>(state, src);
    int srcCopyDim = srcInfo.collapseDims(dim);
    srcInfo.reduceDim(srcCopyDim);

    TensorInfo<int64_t, uint64_t> indicesInfo =
      getTensorInfo<THCudaLongTensor, uint64_t>(state, indices);
    indicesInfo.collapseDims();

    LARGE_INDEX(real, uint64_t, -1, -1, -1);
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
}

void THCTensor_(take)(THCState *state, THCTensor *dst, THCTensor *src, THCudaLongTensor *index)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, dst, src));
  THCAssertSameGPU(THCudaLongTensor_checkGPU(state, 1, index));

  THArgCheck(THCTensor_(nDimension)(state, src) <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  THArgCheck(THCTensor_(nDimension)(state, dst) <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  THArgCheck(THCudaLongTensor_nDimension(state, index) <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  THArgCheck(!(THCTensor_(nDimension)(state, src) == 0 && THCudaLongTensor_nDimension(state, index) != 0), 2,
             "tried to take from an empty tensor");

  THCTensor_(resizeNd)(state, dst, index->nDimension, index->size, NULL);

  // dispatchTakePut only handles non-empty tensors;
  if (index->nDimension > 0) {
    dispatchTakePut<real, TensorTakeOp>(state, src, dst, index);
  }
}

static void THCTensor_(sort_indices)(THCState *state, THCudaLongTensor *index, THCTensor *src) {
  THCThrustAllocator thrustAlloc(state);

  auto index_iter = thrust::device_ptr<int64_t>(THCudaLongTensor_data(state, index));
  auto src_iter = thrust::device_ptr<real>(THCTensor_(data)(state, src));
  auto numel = THCTensor_(numel)(state, src);

  thrust::sort_by_key(
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
    index_iter, index_iter + numel,
    src_iter, ThrustLTOp<int64_t>());
}

void THCTensor_(put)(THCState *state, THCTensor *dst, THCudaLongTensor *index, THCTensor *src, int accumulate)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, dst, src));
  THCAssertSameGPU(THCudaLongTensor_checkGPU(state, 1, index));

  ptrdiff_t dstSize = THCTensor_(nElement)(state, dst);
  ptrdiff_t numIndices = THCudaLongTensor_nElement(state, index);
  THArgCheck(THCTensor_(nElement)(state, src) == numIndices,
    3, "src should have the same number of elements as index");

  THArgCheck(THCTensor_(nDimension)(state, dst) <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  THArgCheck(THCTensor_(nDimension)(state, src) <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  THArgCheck(THCudaLongTensor_nDimension(state, index) <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);

  if (numIndices == 0) {
    return;
  }

  if (accumulate) {
    // wrap indices so to replace negative indices
    THCudaLongTensor* sorted_index = THCudaLongTensor_new(state);
    THCudaLongTensor_resizeAs(state, sorted_index, index);
    THC_pointwiseApply2(state, sorted_index, index, WrapIndexOp(dstSize));

    THCTensor* sorted_src = THCTensor_(newClone)(state, src);

    THCTensor_(sort_indices)(state, sorted_index, sorted_src);
    dispatchTakePut<real, TensorPutAccumulateOp>(state, dst, sorted_src, sorted_index);

    THCTensor_(free)(state, sorted_src);
    THCudaLongTensor_free(state, sorted_index);
  } else {
    dispatchTakePut<real, TensorPutOp>(state, dst, src, index);
  }
}

void THCTensor_(indexAdd_long)(THCState *state, THCTensor *dst, int dim, THLongTensor *indices, THCTensor *src)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, dst, src));

  THCudaLongTensor *indices_ = THCudaLongTensor_newWithSize1d(state, indices->size[0]);
  THCudaLongTensor_copyLong(state, indices_, indices);

  THCTensor_(indexAdd)(state, dst, dim, indices_, src);

  THCudaLongTensor_free(state, indices_);
}

void THCTensor_(indexAdd)(THCState *state, THCTensor *dst, int dim, THCudaLongTensor *indices, THCTensor *src)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, dst, src));
  THCAssertSameGPU(THCudaLongTensor_checkGPU(state, 1, indices));

  int dims = THCTensor_(nDimension)(state, dst);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  dims = THCTensor_(nDimension)(state, src);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 5, CUTORCH_DIM_WARNING);
  dims = THCudaLongTensor_nDimension(state, indices);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 4, CUTORCH_DIM_WARNING);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  ptrdiff_t sliceSize = THCTensor_(getSliceSize)(state, dst, dim, indices, src);
  ptrdiff_t srcTotalSize = THCTensor_(nElement)(state, src);
  int64_t dstAddDimSize = THCTensor_(size)(state, dst, dim);

  ptrdiff_t numIndices = THCudaLongTensor_nElement(state, indices);
  cudaStream_t stream = THCState_getCurrentStream(state);
  int indContig = THCudaLongTensor_isContiguous(state, indices);

  int mpc = THCState_getCurrentDeviceProperties(state)->multiProcessorCount;

#define SMALL_INDEX(TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM) \
  indexAddSmallIndex<TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM> \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(   \
      dstInfo, srcInfo, indicesInfo,                    \
      dstAddDim, srcAddDim, sliceSize, dstAddDimSize);

#define LARGE_INDEX(TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM) \
  indexAddLargeIndex<TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM> \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(   \
      dstInfo, srcInfo, indicesInfo,                    \
      dstAddDim, srcAddDim, sliceSize, dstAddDimSize);

  dim3 smallIndexGrid(std::min(THCCeilDiv(sliceSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 smallIndexBlock(std::min(sliceSize, (ptrdiff_t)128));

  dim3 largeIndexGrid(std::min(THCCeilDiv(srcTotalSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 largeIndexBlock(std::min(srcTotalSize, (ptrdiff_t)128));

  if (TensorUtils<THCTensor>::canUse32BitIndexMath(state, dst) &&
      TensorUtils<THCTensor>::canUse32BitIndexMath(state, src) &&
      TensorUtils<THCudaLongTensor>::canUse32BitIndexMath(state, indices)) {
    TensorInfo<real, unsigned int> dstInfo =
      getTensorInfo<THCTensor, unsigned int>(state, dst);
    int dstAddDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstAddDim);

    TensorInfo<real, unsigned int> srcInfo =
      getTensorInfo<THCTensor, unsigned int>(state, src);
    int srcAddDim = srcInfo.collapseDims(dim);
    srcInfo.reduceDim(srcAddDim);

    TensorInfo<int64_t, unsigned int> indicesInfo =
      getTensorInfo<THCudaLongTensor, unsigned int>(state, indices);
    indicesInfo.collapseDims();

    // A reasonable choice for when to have each thread iterate over
    // indices to choose
    if (numIndices <= 16) {
      if (dstInfo.dims == 1 && srcInfo.dims == 1 && indContig) {
        SMALL_INDEX(real, unsigned int, 1, 1, -2);
      } else if (dstInfo.dims == 2 && srcInfo.dims == 2 && indContig) {
        SMALL_INDEX(real, unsigned int, 2, 2, -2);
      } else if (dstInfo.dims == 3 && srcInfo.dims == 3 && indContig) {
        SMALL_INDEX(real, unsigned int, 3, 3, -2);
      } else {
        SMALL_INDEX(real, unsigned int, -1, -1, -1);
      }
    } else {
      if (dstInfo.dims == 1 && srcInfo.dims == 1 && indContig) {
        LARGE_INDEX(real, unsigned int, 1, 1, -2);
      } else if (dstInfo.dims == 2 && srcInfo.dims == 2 && indContig) {
        LARGE_INDEX(real, unsigned int, 2, 2, -2);
      } else if (dstInfo.dims == 3 && srcInfo.dims == 3 && indContig) {
        LARGE_INDEX(real, unsigned int, 3, 3, -2);
      } else {
        LARGE_INDEX(real, unsigned int, -1, -1, -1);
      }
    }
  } else {
    TensorInfo<real, uint64_t> dstInfo =
      getTensorInfo<THCTensor, uint64_t>(state, dst);
    int dstAddDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstAddDim);

    TensorInfo<real, uint64_t> srcInfo =
      getTensorInfo<THCTensor, uint64_t>(state, src);
    int srcAddDim = srcInfo.collapseDims(dim);
    srcInfo.reduceDim(srcAddDim);

    TensorInfo<int64_t, uint64_t> indicesInfo =
      getTensorInfo<THCudaLongTensor, uint64_t>(state, indices);
    indicesInfo.collapseDims();

    LARGE_INDEX(real, uint64_t, -1, -1, -1);
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
}

void THCTensor_(indexFill_long)(THCState *state, THCTensor *dst, int dim, THLongTensor *indices, real val)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, dst));

  THCudaLongTensor *indices_ = THCudaLongTensor_newWithSize1d(state, indices->size[0]);
  THCudaLongTensor_copyLong(state, indices_, indices);

  THCTensor_(indexFill)(state, dst, dim, indices_, val);

  THCudaLongTensor_free(state, indices_);
}

void THCTensor_(indexFill)(THCState *state, THCTensor *dst, int dim, THCudaLongTensor *indices, real val)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, dst));
  THCAssertSameGPU(THCudaLongTensor_checkGPU(state, 1, indices));
  int dims = THCTensor_(nDimension)(state, dst);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  dims = THCudaLongTensor_nDimension(state, indices);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 4, CUTORCH_DIM_WARNING);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  ptrdiff_t sliceSize =
    THCTensor_(getSliceSize)(state, dst, dim, indices, nullptr);
  ptrdiff_t dstTotalSize = THCTensor_(nElement)(state, dst);
  int64_t dstFillDimSize = THCTensor_(size)(state, dst, dim);

  ptrdiff_t numIndices = THCudaLongTensor_nElement(state, indices);
  cudaStream_t stream = THCState_getCurrentStream(state);
  int indContig = THCudaLongTensor_isContiguous(state, indices);

  int mpc = THCState_getCurrentDeviceProperties(state)->multiProcessorCount;

#define SMALL_INDEX(TENSOR_TYPE, TYPE, DST_DIM, IDX_DIM)  \
  indexFillSmallIndex<TENSOR_TYPE, TYPE, DST_DIM, IDX_DIM> \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(   \
      dstInfo, indicesInfo,                             \
      dstFillDim, sliceSize, dstFillDimSize, val);

#define LARGE_INDEX(TENSOR_TYPE, TYPE, DST_DIM, IDX_DIM)  \
  indexFillLargeIndex<TENSOR_TYPE, TYPE, DST_DIM, IDX_DIM> \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(   \
      dstInfo, indicesInfo,                             \
      dstFillDim, sliceSize, dstFillDimSize, val);

  dim3 smallIndexGrid(std::min(THCCeilDiv(sliceSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 smallIndexBlock(std::min(sliceSize, (ptrdiff_t)128));

  dim3 largeIndexGrid(std::min(THCCeilDiv(dstTotalSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 largeIndexBlock(std::min(dstTotalSize, (ptrdiff_t)128));

  if (TensorUtils<THCTensor>::canUse32BitIndexMath(state, dst) &&
      TensorUtils<THCudaLongTensor>::canUse32BitIndexMath(state, indices)) {
    TensorInfo<real, unsigned int> dstInfo =
      getTensorInfo<THCTensor, unsigned int>(state, dst);
    int dstFillDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstFillDim);

    TensorInfo<int64_t, unsigned int> indicesInfo =
      getTensorInfo<THCudaLongTensor, unsigned int>(state, indices);
    indicesInfo.collapseDims();

    // A reasonable choice for when to have each thread iterate over
    // indices to choose
    if (numIndices <= 16) {
      if (dstInfo.dims == 1 && indContig) {
        SMALL_INDEX(real, unsigned int, 1, -2);
      } else if (dstInfo.dims == 2 && indContig) {
        SMALL_INDEX(real, unsigned int, 2, -2);
      } else if (dstInfo.dims == 3 && indContig) {
        SMALL_INDEX(real, unsigned int, 3, -2);
      } else {
        SMALL_INDEX(real, unsigned int, -1, -1);
      }
    } else {
      if (dstInfo.dims == 1 && indContig) {
        LARGE_INDEX(real, unsigned int, 1, -2);
      } else if (dstInfo.dims == 2 && indContig) {
        LARGE_INDEX(real, unsigned int, 2, -2);
      } else if (dstInfo.dims == 3 && indContig) {
        LARGE_INDEX(real, unsigned int, 3, -2);
      } else {
        LARGE_INDEX(real, unsigned int, -1, -1);
      }
    }
  } else {
    TensorInfo<real, uint64_t> dstInfo =
      getTensorInfo<THCTensor, uint64_t>(state, dst);
    int dstFillDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstFillDim);

    TensorInfo<int64_t, uint64_t> indicesInfo =
      getTensorInfo<THCudaLongTensor, uint64_t>(state, indices);
    indicesInfo.collapseDims();

    LARGE_INDEX(real, uint64_t, -1, -1);
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
}


void THCTensor_(indexSelect_long)(THCState *state, THCTensor *dst, THCTensor *src, int dim, THLongTensor *indices)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, dst, src));
  THArgCheck(indices->nDimension <= 1, 3, "Index is supposed to be an empty tensor or a vector");

  THCudaLongTensor *indices_ = THCudaLongTensor_newWithSize1d(state, indices->size[0]);
  THCudaLongTensor_copyLong(state, indices_, indices);

  THCTensor_(indexSelect)(state, dst, src, dim, indices_);

  THCudaLongTensor_free(state, indices_);
}

void THCTensor_(indexSelect)(THCState *state, THCTensor *dst, THCTensor *src, int dim, THCudaLongTensor *indices)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, dst, src, indices));

  int dims = THCTensor_(nDimension)(state, dst);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  dims = THCTensor_(nDimension)(state, src);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 3, CUTORCH_DIM_WARNING);
  dims = THCudaLongTensor_nDimension(state, indices);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 5, CUTORCH_DIM_WARNING);

  ptrdiff_t numIndices = THCudaLongTensor_nElement(state, indices);

  int srcDims = THCTensor_(nDimension)(state, src);
  cudaStream_t stream = THCState_getCurrentStream(state);

  THArgCheck(THCudaLongTensor_nDimension(state, indices) <= 1, 3,
             "Index is supposed to be an empty tensor or a vector");
  THArgCheck(dim < srcDims, 4, "Indexing dim is out of bounds");
  THArgCheck(srcDims > 0, 2, "Source tensor is empty");

  THLongStorage *newSize;

  if (numIndices == 0) {
    newSize = THCTensor_(newSizeOf)(state, src);
    THLongStorage_set(newSize, 0, numIndices);
    THCTensor_(resize)(state, dst, newSize, NULL);
    THLongStorage_free(newSize);
    return;
  }

  newSize = THCTensor_(newSizeOf)(state, src);
  THLongStorage_set(newSize, dim, numIndices);
  THCTensor_(resize)(state, dst, newSize, NULL);
  THLongStorage_free(newSize);

  int indContig = THCudaLongTensor_isContiguous(state, indices);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  ptrdiff_t dstTotalSize = THCTensor_(nElement)(state, dst);
  int64_t srcSelectDimSize = THCTensor_(size)(state, src, dim);
  ptrdiff_t sliceSize = dstTotalSize / numIndices;

  int mpc = THCState_getCurrentDeviceProperties(state)->multiProcessorCount;

#define SMALL_INDEX(TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM) \
  indexSelectSmallIndex<TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM>     \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(           \
      dstInfo, srcInfo, indicesInfo,                            \
      dstSelectDim, srcSelectDim, sliceSize, srcSelectDimSize);

#define LARGE_INDEX(TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM)         \
  indexSelectLargeIndex<TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM>     \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(                   \
      dstInfo, srcInfo, indicesInfo,                                    \
      dstSelectDim, srcSelectDim, dstTotalSize, sliceSize, srcSelectDimSize);

  dim3 smallIndexGrid(std::min(THCCeilDiv(sliceSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 smallIndexBlock(std::min(sliceSize, (ptrdiff_t)128));

  dim3 largeIndexGrid(std::min(THCCeilDiv(dstTotalSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 largeIndexBlock(std::min(dstTotalSize, (ptrdiff_t)128));

  if (TensorUtils<THCTensor>::canUse32BitIndexMath(state, dst) &&
      TensorUtils<THCTensor>::canUse32BitIndexMath(state, src) &&
      TensorUtils<THCudaLongTensor>::canUse32BitIndexMath(state, indices)) {
    TensorInfo<real, unsigned int> dstInfo =
      getTensorInfo<THCTensor, unsigned int>(state, dst);
    int dstSelectDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstSelectDim);

    TensorInfo<real, unsigned int> srcInfo =
      getTensorInfo<THCTensor, unsigned int>(state, src);
    int srcSelectDim = srcInfo.collapseDims(dim);
    srcInfo.reduceDim(srcSelectDim);

    TensorInfo<int64_t, unsigned int> indicesInfo =
      getTensorInfo<THCudaLongTensor, unsigned int>(state, indices);
    indicesInfo.collapseDims();

    // A reasonable choice for when to have each thread iterate over
    // indices to choose
    if (numIndices <= 16) {
      if (dstInfo.dims == 1 && srcInfo.dims == 1 && indContig) {
        SMALL_INDEX(real, unsigned int, 1, 1, -2);
      } else if (dstInfo.dims == 2 && srcInfo.dims == 2 && indContig) {
        SMALL_INDEX(real, unsigned int, 2, 2, -2);
      } else if (dstInfo.dims == 3 && srcInfo.dims == 3 && indContig) {
        SMALL_INDEX(real, unsigned int, 3, 3, -2);
      } else {
        SMALL_INDEX(real, unsigned int, -1, -1, -1);
      }
    } else {
      if (dstInfo.dims == 1 && srcInfo.dims == 1 && indContig) {
        LARGE_INDEX(real, unsigned int, 1, 1, -2);
      } else if (dstInfo.dims == 2 && srcInfo.dims == 2 && indContig) {
        LARGE_INDEX(real, unsigned int, 2, 2, -2);
      } else if (dstInfo.dims == 3 && srcInfo.dims == 3 && indContig) {
        LARGE_INDEX(real, unsigned int, 3, 3, -2);
      } else {
        LARGE_INDEX(real, unsigned int, -1, -1, -1);
      }
    }
  } else {
    TensorInfo<real, uint64_t> dstInfo =
      getTensorInfo<THCTensor, uint64_t>(state, dst);
    int dstSelectDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstSelectDim);

    TensorInfo<real, uint64_t> srcInfo =
      getTensorInfo<THCTensor, uint64_t>(state, src);
    int srcSelectDim = srcInfo.collapseDims(dim);
    srcInfo.reduceDim(srcSelectDim);

    TensorInfo<int64_t, uint64_t> indicesInfo =
      getTensorInfo<THCudaLongTensor, uint64_t>(state, indices);
    indicesInfo.collapseDims();

    LARGE_INDEX(real, uint64_t, -1, -1, -1);
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
}

#define MAX_ADVINDEX_CALC_DIMS 5

void THCTensor_(calculateAdvancedIndexingOffsets)(
  THCState *state,
  THCudaLongTensor *output,
  THCTensor *indexed,
  ptrdiff_t baseOffset,
  THCudaLongTensor **indexers)
{
  int ndim = THCTensor_(nDimension)(state, indexed);
  THAssert(ndim <= MAX_ADVINDEX_CALC_DIMS);

  // Assert all Tensors are on the same GPU, and that the indexing Tensors are
  // contiguous
  for (int i = 0; i < ndim; ++i) {
    if (indexers[i] != NULL) {
      THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, output, indexers[i]));
      THAssert(THCudaLongTensor_isContiguous(state, indexers[i]));
    }
  }

  // Set grid, block dims
  ptrdiff_t nElement = THCudaLongTensor_nElement(state, output);
  const dim3 block = getApplyBlock();
  dim3 grid;
  THAssert(getApplyGrid(state, nElement, grid));

#define HANDLE_CASE(INDEX_TYPE, DIMS)                                                           \
{                                                                                               \
  LinearIndexCalcData<INDEX_TYPE, DIMS> data;                                                   \
  for (int i = 0; i < DIMS; ++i) {                                                              \
    data.baseSizes[i] = THCTensor_(size)(state, indexed, i);                                    \
    data.sizes[i] = indexers[i] != NULL ?                                                       \
      THCudaLongTensor_nElement(state, indexers[i]) :                                           \
        THCTensor_(size)(state, indexed, i);                                                    \
    data.strides[i] = THCTensor_(stride)(state, indexed, i);                                    \
    data.advIndexTensors[i] = indexers[i] != NULL ?                                             \
      THCudaLongTensor_data(state, indexers[i]) : NULL;                                         \
  }                                                                                             \
                                                                                                \
  calculateLinearIndices<INDEX_TYPE, DIMS>                                                      \
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(                                     \
    THCudaLongTensor_data(state, output),                                                       \
    nElement,                                                                                   \
    baseOffset,                                                                                 \
    data                                                                                        \
  );                                                                                            \
}

#define RUN_T(INDEX_TYPE)         \
  switch (ndim) {                 \
    case 1:                       \
      HANDLE_CASE(INDEX_TYPE, 1)  \
      break;                      \
    case 2:                       \
      HANDLE_CASE(INDEX_TYPE, 2)  \
      break;                      \
    case 3:                       \
      HANDLE_CASE(INDEX_TYPE, 3)  \
      break;                      \
    case 4:                       \
      HANDLE_CASE(INDEX_TYPE, 4)  \
      break;                      \
    case 5:                       \
      HANDLE_CASE(INDEX_TYPE, 5)  \
      break;                      \
    default:                      \
      THAssert(false);            \
}

  if (TensorUtils<THCTensor>::canUse32BitIndexMath(state, indexed)) {
    RUN_T(unsigned int);
  } else {
    RUN_T(uint64_t);
  }

#undef HANDLE_CASE
#undef RUN_T

  THCudaCheck(cudaGetLastError());
}

#undef MAX_ADVINDEX_CALC_DIMS

#endif
