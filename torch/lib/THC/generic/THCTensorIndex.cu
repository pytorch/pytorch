#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorIndex.cu"
#else

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

  long dims = THCTensor_(nDimension)(state, dst);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  dims = THCTensor_(nDimension)(state, src);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 5, CUTORCH_DIM_WARNING);
  dims = THCudaLongTensor_nDimension(state, indices);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 4, CUTORCH_DIM_WARNING);

  ptrdiff_t numIndices = THCudaLongTensor_nElement(state, indices);

  long srcDims = THCTensor_(nDimension)(state, src);
  cudaStream_t stream = THCState_getCurrentStream(state);

  THArgCheck(THCudaLongTensor_nDimension(state, indices) == 1, 3,
             "expecting vector of indices");
  THArgCheck(dim < srcDims, 4, "Indexing dim is out of bounds");
  THArgCheck(srcDims > 0, 2, "Source tensor is empty");
  THArgCheck(numIndices == src->size[dim], 4, "length of src.size[dim] is not equal to length of indices");

  int indContig = THCudaLongTensor_isContiguous(state, indices);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  ptrdiff_t srcTotalSize = THCTensor_(nElement)(state, src);
  long dstCopyDimSize = THCTensor_(size)(state, dst, dim);
  ptrdiff_t sliceSize = srcTotalSize / numIndices;

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

    TensorInfo<long, unsigned int> indicesInfo =
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
    TensorInfo<real, unsigned long> dstInfo =
      getTensorInfo<THCTensor, unsigned long>(state, dst);
    int dstCopyDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstCopyDim);

    TensorInfo<real, unsigned long> srcInfo =
      getTensorInfo<THCTensor, unsigned long>(state, src);
    int srcCopyDim = srcInfo.collapseDims(dim);
    srcInfo.reduceDim(srcCopyDim);

    TensorInfo<long, unsigned long> indicesInfo =
      getTensorInfo<THCudaLongTensor, unsigned long>(state, indices);
    indicesInfo.collapseDims();

    LARGE_INDEX(real, unsigned long, -1, -1, -1);
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
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

  long dims = THCTensor_(nDimension)(state, dst);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  dims = THCTensor_(nDimension)(state, src);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 5, CUTORCH_DIM_WARNING);
  dims = THCudaLongTensor_nDimension(state, indices);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 4, CUTORCH_DIM_WARNING);

  ptrdiff_t numIndices = THCudaLongTensor_nElement(state, indices);

  long srcDims = THCTensor_(nDimension)(state, src);
  cudaStream_t stream = THCState_getCurrentStream(state);

  THArgCheck(THCudaLongTensor_nDimension(state, indices) == 1, 3,
             "expecting vector of indices");
  THArgCheck(dim < srcDims, 4, "Indexing dim is out of bounds");
  THArgCheck(srcDims > 0, 2, "Source tensor is empty");
  THArgCheck(numIndices == src->size[dim], 4, "length of src.size[dim] is not equal to length of indices");

  int indContig = THCudaLongTensor_isContiguous(state, indices);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  ptrdiff_t srcTotalSize = THCTensor_(nElement)(state, src);
  long dstAddDimSize = THCTensor_(size)(state, dst, dim);
  ptrdiff_t sliceSize = srcTotalSize / numIndices;

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

    TensorInfo<long, unsigned int> indicesInfo =
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
    TensorInfo<real, unsigned long> dstInfo =
      getTensorInfo<THCTensor, unsigned long>(state, dst);
    int dstAddDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstAddDim);

    TensorInfo<real, unsigned long> srcInfo =
      getTensorInfo<THCTensor, unsigned long>(state, src);
    int srcAddDim = srcInfo.collapseDims(dim);
    srcInfo.reduceDim(srcAddDim);

    TensorInfo<long, unsigned long> indicesInfo =
      getTensorInfo<THCudaLongTensor, unsigned long>(state, indices);
    indicesInfo.collapseDims();

    LARGE_INDEX(real, unsigned long, -1, -1, -1);
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
  long dims = THCTensor_(nDimension)(state, dst);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  dims = THCudaLongTensor_nDimension(state, indices);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 4, CUTORCH_DIM_WARNING);

  ptrdiff_t numIndices = THCudaLongTensor_nElement(state, indices);

  long srcDims = THCTensor_(nDimension)(state, dst);
  cudaStream_t stream = THCState_getCurrentStream(state);

  THArgCheck(THCudaLongTensor_nDimension(state, indices) == 1, 3,
             "expecting vector of indices");
  THArgCheck(dim < srcDims, 4, "Indexing dim is out of bounds");
  THArgCheck(srcDims > 0, 2, "Source tensor is empty");

  int indContig = THCudaLongTensor_isContiguous(state, indices);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  ptrdiff_t dstTotalSize = THCTensor_(nElement)(state, dst);
  long dstFillDimSize = THCTensor_(size)(state, dst, dim);
  ptrdiff_t sliceSize = dstTotalSize / dstFillDimSize;

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

    TensorInfo<long, unsigned int> indicesInfo =
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
    TensorInfo<real, unsigned long> dstInfo =
      getTensorInfo<THCTensor, unsigned long>(state, dst);
    int dstFillDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstFillDim);

    TensorInfo<long, unsigned long> indicesInfo =
      getTensorInfo<THCudaLongTensor, unsigned long>(state, indices);
    indicesInfo.collapseDims();

    LARGE_INDEX(real, unsigned long, -1, -1);
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
}


void THCTensor_(indexSelect_long)(THCState *state, THCTensor *dst, THCTensor *src, int dim, THLongTensor *indices)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, dst, src));
  THArgCheck(indices->nDimension == 1, 3, "Index is supposed to be a vector");

  THCudaLongTensor *indices_ = THCudaLongTensor_newWithSize1d(state, indices->size[0]);
  THCudaLongTensor_copyLong(state, indices_, indices);

  THCTensor_(indexSelect)(state, dst, src, dim, indices_);

  THCudaLongTensor_free(state, indices_);
}

void THCTensor_(indexSelect)(THCState *state, THCTensor *dst, THCTensor *src, int dim, THCudaLongTensor *indices)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, dst, src, indices));

  long dims = THCTensor_(nDimension)(state, dst);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  dims = THCTensor_(nDimension)(state, src);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 3, CUTORCH_DIM_WARNING);
  dims = THCudaLongTensor_nDimension(state, indices);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 5, CUTORCH_DIM_WARNING);

  ptrdiff_t numIndices = THCudaLongTensor_nElement(state, indices);

  long srcDims = THCTensor_(nDimension)(state, src);
  cudaStream_t stream = THCState_getCurrentStream(state);

  THArgCheck(THCudaLongTensor_nDimension(state, indices) == 1, 3,
             "expecting vector of indices");
  THArgCheck(dim < srcDims, 4, "Indexing dim is out of bounds");
  THArgCheck(srcDims > 0, 2, "Source tensor is empty");

  THLongStorage *newSize = THCTensor_(newSizeOf)(state, src);
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
  long srcSelectDimSize = THCTensor_(size)(state, src, dim);
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

    TensorInfo<long, unsigned int> indicesInfo =
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
    TensorInfo<real, unsigned long> dstInfo =
      getTensorInfo<THCTensor, unsigned long>(state, dst);
    int dstSelectDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstSelectDim);

    TensorInfo<real, unsigned long> srcInfo =
      getTensorInfo<THCTensor, unsigned long>(state, src);
    int srcSelectDim = srcInfo.collapseDims(dim);
    srcInfo.reduceDim(srcSelectDim);

    TensorInfo<long, unsigned long> indicesInfo =
      getTensorInfo<THCudaLongTensor, unsigned long>(state, indices);
    indicesInfo.collapseDims();

    LARGE_INDEX(real, unsigned long, -1, -1, -1);
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
}

#endif
