#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorIndex.cu"
#else

#include "ATen/cuda/CUDAContext.h"

// Check tensor dimensions for index operations, and return the slice size.
// src can be nullptr in case of indexFill: in that case it is ignored.
static ptrdiff_t THCTensor_(getSliceSize)(THCState *state, THCTensor *dst,
                                          int dim,
                                          THCudaLongTensor *index,
                                          THCTensor *src)
{
  int dstDims = THCTensor_(nDimensionLegacyNoScalars)(state, dst);
  int srcDims = (src == nullptr) ? dstDims : THCTensor_(nDimensionLegacyNoScalars)(state, src);

  THArgCheck(THCudaLongTensor_nDimensionLegacyNoScalars(state, index) == 1, 4,
             "expecting vector of indices");
  THArgCheck(dim >= 0 && dim < dstDims, 2, "Indexing dim is out of bounds");

  ptrdiff_t dstSliceSize = 1;
  for (int d = 0; d < dstDims; d++) {
    if (d != dim) {
      dstSliceSize *= THTensor_sizeLegacyNoScalars(dst, d);
    }
  }

  if (src == nullptr) return dstSliceSize;

  THArgCheck(dim < srcDims, 3, "Indexing dim is out of bounds");
  THArgCheck(THCudaLongTensor_nElement(state, index) == THTensor_sizeLegacyNoScalars(src, dim), 4,
             "length of src.size[dim] is not equal to length of indices");

  ptrdiff_t srcSliceSize = 1;
  bool mismatch = false;

  if (dstDims != srcDims) mismatch = true;

  for (int d = 0; d < srcDims; d++) {
    if (d != dim) {
      srcSliceSize *= THTensor_sizeLegacyNoScalars(src, d);
      if (!mismatch && THTensor_sizeLegacyNoScalars(dst, d) != THTensor_sizeLegacyNoScalars(src, d)) mismatch = true;
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

// Compare the stride between adjacent slices (sliceStride) with strides in the
// other dimensions (i.e., strides *inside* each slice).
//
// - Returns true if some dimension inside the slice has lower stride than
//   sliceStride.  The simplest example is a 2-D contiguous tensor with sliceDim
//   == 0 (that is, each slice is a row).
//
//   In this case, we choose the CUDA kernel that processes the data in
//   "index-major order".  For example, if thread count equals slice size, then
//   all threads process slice #0 in lockstep, and then slice #1, and so on.
//
// - Otherwise (i.e., sliceStride has the lowest value), this function returns
//   false.  The simplest example is a 2-D contiguous tensor with sliceDim == 1
//   (each slice is a column).
//
//   In this case, we choose the CUDA kernel that processes the data in
//   "elementInSlice-major order".  For example, each thread can process element
//   #0 of every slice, and then element #1 of every slice, and so on.
bool THCTensor_(indexShouldBeMajor)(TensorInfo<scalar_t, unsigned int> &info,
                                    int sliceDim)
{
  // The stride between adjacent slices (e.g., between element #0 of slice #100
  // and element #0 of slice #101).
  unsigned int sliceStride = info.strides[sliceDim];

  for (int i = 0; i < info.dims; ++i) {
    if (i != sliceDim && info.sizes[i] > 1 && info.strides[i] < sliceStride) {
      return true;
    }
  }

  return false;
}

void THCTensor_(indexCopy)(THCState *state, THCTensor *dst, int dim, THCudaLongTensor *indices, THCTensor *src)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, dst, src));
  THCAssertSameGPU(THCudaLongTensor_checkGPU(state, 1, indices));

  dim  = at::maybe_wrap_dim(dim, dst);
  int dims = THCTensor_(nDimensionLegacyNoScalars)(state, dst);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  dims = THCTensor_(nDimensionLegacyNoScalars)(state, src);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 5, CUTORCH_DIM_WARNING);
  dims = THCudaLongTensor_nDimensionLegacyNoScalars(state, indices);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 4, CUTORCH_DIM_WARNING);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  ptrdiff_t sliceSize = THCTensor_(getSliceSize)(state, dst, dim, indices, src);
  ptrdiff_t srcTotalSize = THCTensor_(nElement)(state, src);
  int64_t dstCopyDimSize = THCTensor_(sizeLegacyNoScalars)(state, dst, dim);
  ptrdiff_t numIndices = THCudaLongTensor_nElement(state, indices);

  if (sliceSize == 0) {
    return;
  }

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  int indContig = THCudaLongTensor_isContiguous(state, indices);

  int mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

#define SMALL_INDEX(TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM) \
  indexCopySmallIndex<TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM>       \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(           \
      dstInfo, srcInfo, indicesInfo,                            \
      dstCopyDim, srcCopyDim, sliceSize, dstCopyDimSize);

#define LARGE_INDEX(TENSOR_TYPE, TYPE,                         \
                    DST_DIM, SRC_DIM, IDX_DIM, IDX_IS_MAJOR)   \
  indexCopyLargeIndex<TENSOR_TYPE, TYPE,                       \
                      DST_DIM, SRC_DIM, IDX_DIM, IDX_IS_MAJOR> \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(          \
      dstInfo, srcInfo, indicesInfo,                           \
      dstCopyDim, srcCopyDim, srcTotalSize,                    \
      (IDX_IS_MAJOR) ? sliceSize : numIndices,                 \
      dstCopyDimSize);

  dim3 smallIndexGrid(std::min(THCCeilDiv(sliceSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 smallIndexBlock(std::min(sliceSize, (ptrdiff_t)128));

  dim3 largeIndexGrid(std::min(THCCeilDiv(srcTotalSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 largeIndexBlock(std::min(srcTotalSize, (ptrdiff_t)128));

  if (THCTensor_canUse32BitIndexMath(state, dst) &&
      THCTensor_canUse32BitIndexMath(state, src) &&
      THCTensor_canUse32BitIndexMath(state, indices)) {
    TensorInfo<scalar_t, unsigned int> dstInfo =
      getTensorInfo<scalar_t, THCTensor, unsigned int>(state, dst);
    int dstCopyDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstCopyDim);

    TensorInfo<scalar_t, unsigned int> srcInfo =
      getTensorInfo<scalar_t, THCTensor, unsigned int>(state, src);
    int srcCopyDim = srcInfo.collapseDims(dim);
    srcInfo.reduceDim(srcCopyDim);

    TensorInfo<int64_t, unsigned int> indicesInfo =
      getTensorInfo<int64_t, THCudaLongTensor, unsigned int>(state, indices);
    indicesInfo.collapseDims();

    // A reasonable choice for when to have each thread iterate over
    // indices to choose
    if (numIndices <= 16) {
      if (dstInfo.dims == 1 && srcInfo.dims == 1 && indContig) {
        SMALL_INDEX(scalar_t, unsigned int, 1, 1, -2);
      } else if (dstInfo.dims == 2 && srcInfo.dims == 2 && indContig) {
        SMALL_INDEX(scalar_t, unsigned int, 2, 2, -2);
      } else if (dstInfo.dims == 3 && srcInfo.dims == 3 && indContig) {
        SMALL_INDEX(scalar_t, unsigned int, 3, 3, -2);
      } else {
        SMALL_INDEX(scalar_t, unsigned int, -1, -1, -1);
      }
    } else {
      bool indexIsMajor = THCTensor_(indexShouldBeMajor)(dstInfo, dstCopyDim);

      if (dstInfo.dims == 1 && srcInfo.dims == 1 && indContig) {
        LARGE_INDEX(scalar_t, unsigned int, 1, 1, -2, true);
      } else if (dstInfo.dims == 2 && srcInfo.dims == 2 && indContig) {
        if (indexIsMajor) {
          LARGE_INDEX(scalar_t, unsigned int, 2, 2, -2, true);
        } else {
          LARGE_INDEX(scalar_t, unsigned int, 2, 2, -2, false);
        }
      } else if (dstInfo.dims == 3 && srcInfo.dims == 3 && indContig) {
        if (indexIsMajor) {
          LARGE_INDEX(scalar_t, unsigned int, 3, 3, -2, true);
        } else {
          LARGE_INDEX(scalar_t, unsigned int, 3, 3, -2, false);
        }
      } else {
        LARGE_INDEX(scalar_t, unsigned int, -1, -1, -1, true);
      }
    }
  } else {
    TensorInfo<scalar_t, uint64_t> dstInfo =
      getTensorInfo<scalar_t, THCTensor, uint64_t>(state, dst);
    int dstCopyDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstCopyDim);

    TensorInfo<scalar_t, uint64_t> srcInfo =
      getTensorInfo<scalar_t, THCTensor, uint64_t>(state, src);
    int srcCopyDim = srcInfo.collapseDims(dim);
    srcInfo.reduceDim(srcCopyDim);

    TensorInfo<int64_t, uint64_t> indicesInfo =
      getTensorInfo<int64_t, THCudaLongTensor, uint64_t>(state, indices);
    indicesInfo.collapseDims();

    LARGE_INDEX(scalar_t, uint64_t, -1, -1, -1, true);
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
}

void THCTensor_(take)(THCState *state, THCTensor *dst, THCTensor *src, THCudaLongTensor *index)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, dst, src));
  THCAssertSameGPU(THCudaLongTensor_checkGPU(state, 1, index));

  THArgCheck(THCTensor_(nDimensionLegacyNoScalars)(state, src) <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  THArgCheck(THCTensor_(nDimensionLegacyNoScalars)(state, dst) <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  THArgCheck(THCudaLongTensor_nDimensionLegacyNoScalars(state, index) <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  THArgCheck(!(THCTensor_(numel)(state, src) == 0 && THCudaLongTensor_numel(state, index) != 0), 2,
             "tried to take from an empty tensor");

  THCTensor_(resizeNd)(state, dst, index->dim(), THTensor_getSizePtr(index), NULL);
  dispatchTakePut<scalar_t, TensorTakeOp>(state, src, dst, index);
}

static void THCTensor_(sort_indices)(THCState *state, THCudaLongTensor *index, THCTensor *src) {
  THCThrustAllocator thrustAlloc(state);

  auto index_iter = thrust::device_ptr<int64_t>(THCudaLongTensor_data(state, index));
  auto src_iter = thrust::device_ptr<scalar_t>(THCTensor_(data)(state, src));
  auto numel = THCTensor_(numel)(state, src);

  thrust::sort_by_key(
    thrust::cuda::par(thrustAlloc).on(c10::cuda::getCurrentCUDAStream()),
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

  THArgCheck(THCTensor_(nDimensionLegacyNoScalars)(state, dst) <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  THArgCheck(THCTensor_(nDimensionLegacyNoScalars)(state, src) <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  THArgCheck(THCudaLongTensor_nDimensionLegacyNoScalars(state, index) <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);

  if (numIndices == 0) {
    return;
  }

  if (accumulate) {
    // wrap indices so to replace negative indices
    THCudaLongTensor* sorted_index = THCudaLongTensor_new(state);
    THCudaLongTensor_resizeAs(state, sorted_index, index);
    THC_pointwiseApply2<int64_t, int64_t>(state, sorted_index, index, WrapIndexOp(dstSize));

    THCTensor* sorted_src = THCTensor_(newClone)(state, src);

    THCTensor_(sort_indices)(state, sorted_index, sorted_src);
    dispatchTakePut<scalar_t, TensorPutAccumulateOp>(state, dst, sorted_src, sorted_index);

    THCTensor_(free)(state, sorted_src);
    THCudaLongTensor_free(state, sorted_index);
  } else {
    dispatchTakePut<scalar_t, TensorPutOp>(state, dst, src, index);
  }
}

void THCTensor_(indexFill)(THCState *state, THCTensor *dst, int dim, THCudaLongTensor *indices, scalar_t val)
{
  at::NoNamesGuard guard;
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, dst));
  THCAssertSameGPU(THCudaLongTensor_checkGPU(state, 1, indices));
  dim = at::maybe_wrap_dim(dim, dst);
  int dims = THCTensor_(nDimensionLegacyNoScalars)(state, dst);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  dims = THCudaLongTensor_nDimensionLegacyNoScalars(state, indices);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 4, CUTORCH_DIM_WARNING);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  ptrdiff_t sliceSize =
    THCTensor_(getSliceSize)(state, dst, dim, indices, nullptr);
  ptrdiff_t dstTotalSize = THCTensor_(nElement)(state, dst);
  int64_t dstFillDimSize = THCTensor_(sizeLegacyNoScalars)(state, dst, dim);
  ptrdiff_t numIndices = THCudaLongTensor_nElement(state, indices);

  if (sliceSize == 0) {
    return;
  }
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  int indContig = THCudaLongTensor_isContiguous(state, indices);

  int mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

#define SMALL_INDEX(TENSOR_TYPE, TYPE, DST_DIM, IDX_DIM)  \
  indexFillSmallIndex<TENSOR_TYPE, TYPE, DST_DIM, IDX_DIM> \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(   \
      dstInfo, indicesInfo,                             \
      dstFillDim, sliceSize, dstFillDimSize, val);

#define LARGE_INDEX(TENSOR_TYPE, TYPE, DST_DIM, IDX_DIM, IDX_IS_MAJOR)   \
  indexFillLargeIndex<TENSOR_TYPE, TYPE, DST_DIM, IDX_DIM, IDX_IS_MAJOR> \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(                    \
      dstInfo, indicesInfo,                                              \
      dstFillDim, sliceSize * numIndices,                                \
      (IDX_IS_MAJOR) ? sliceSize : numIndices,                           \
      dstFillDimSize, val);

  dim3 smallIndexGrid(std::min(THCCeilDiv(sliceSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 smallIndexBlock(std::min(sliceSize, (ptrdiff_t)128));

  dim3 largeIndexGrid(std::min(THCCeilDiv(dstTotalSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 largeIndexBlock(std::min(dstTotalSize, (ptrdiff_t)128));

  if (THCTensor_canUse32BitIndexMath(state, dst) &&
      THCTensor_canUse32BitIndexMath(state, indices)) {
    TensorInfo<scalar_t, unsigned int> dstInfo =
      getTensorInfo<scalar_t, THCTensor, unsigned int>(state, dst);
    int dstFillDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstFillDim);

    TensorInfo<int64_t, unsigned int> indicesInfo =
      getTensorInfo<int64_t, THCudaLongTensor, unsigned int>(state, indices);
    indicesInfo.collapseDims();

    // A reasonable choice for when to have each thread iterate over
    // indices to choose
    if (numIndices <= 16) {
      if (dstInfo.dims == 1 && indContig) {
        SMALL_INDEX(scalar_t, unsigned int, 1, -2);
      } else if (dstInfo.dims == 2 && indContig) {
        SMALL_INDEX(scalar_t, unsigned int, 2, -2);
      } else if (dstInfo.dims == 3 && indContig) {
        SMALL_INDEX(scalar_t, unsigned int, 3, -2);
      } else {
        SMALL_INDEX(scalar_t, unsigned int, -1, -1);
      }
    } else {
      bool indexIsMajor = THCTensor_(indexShouldBeMajor)(dstInfo, dstFillDim);

      if (dstInfo.dims == 1 && indContig) {
        LARGE_INDEX(scalar_t, unsigned int, 1, -2, true);
      } else if (dstInfo.dims == 2 && indContig) {
        if (indexIsMajor) {
          LARGE_INDEX(scalar_t, unsigned int, 2, -2, true);
        } else {
          LARGE_INDEX(scalar_t, unsigned int, 2, -2, false);
        }
      } else if (dstInfo.dims == 3 && indContig) {
        if (indexIsMajor) {
          LARGE_INDEX(scalar_t, unsigned int, 3, -2, true);
        } else {
          LARGE_INDEX(scalar_t, unsigned int, 3, -2, false);
        }
      } else {
        LARGE_INDEX(scalar_t, unsigned int, -1, -1, true);
      }
    }
  } else {
    TensorInfo<scalar_t, uint64_t> dstInfo =
      getTensorInfo<scalar_t, THCTensor, uint64_t>(state, dst);
    int dstFillDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstFillDim);

    TensorInfo<int64_t, uint64_t> indicesInfo =
      getTensorInfo<int64_t, THCudaLongTensor, uint64_t>(state, indices);
    indicesInfo.collapseDims();

    LARGE_INDEX(scalar_t, uint64_t, -1, -1, true);
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
}

void THCTensor_(indexSelect)(THCState *state, THCTensor *dst, THCTensor *src, int dim, THCudaLongTensor *indices)
{
#if defined(THC_REAL_IS_BFLOAT16) && !defined(__HIP_PLATFORM_HCC__)
  TORCH_CHECK(false, "indexSelect not suppported with BFloat16");
#else
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, dst, src, indices));

  dim = at::maybe_wrap_dim(dim, src);
  int dims = THCTensor_(nDimensionLegacyNoScalars)(state, dst);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  dims = THCTensor_(nDimensionLegacyNoScalars)(state, src);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 3, CUTORCH_DIM_WARNING);
  dims = THCudaLongTensor_nDimensionLegacyNoScalars(state, indices);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 5, CUTORCH_DIM_WARNING);

  ptrdiff_t numIndices = THCudaLongTensor_nElement(state, indices);

  int srcDims = THCTensor_(nDimensionLegacyNoScalars)(state, src);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  THArgCheck(THCudaLongTensor_nDimensionLegacyNoScalars(state, indices) <= 1, 3,
             "Index is supposed to be an empty tensor or a vector");
  THArgCheck(dim < srcDims, 4, "Indexing dim is out of bounds");
  THArgCheck(srcDims > 0, 2, "Source tensor is empty");

  std::vector<int64_t> newSize = src->sizes().vec();
  if (src->dim() > 0) {
    newSize[dim] = numIndices;
  }
  THCTensor_(resize)(state, dst, newSize, {});

  ptrdiff_t dstTotalSize = THCTensor_(nElement)(state, dst);
  if (dstTotalSize == 0) {
    return;
  }

  int indContig = THCudaLongTensor_isContiguous(state, indices);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  int64_t srcSelectDimSize = THCTensor_(sizeLegacyNoScalars)(state, src, dim);
  ptrdiff_t sliceSize = dstTotalSize / numIndices;

  int mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

#define SMALL_INDEX(TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM) \
  indexSelectSmallIndex<TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM>     \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(           \
      dstInfo, srcInfo, indicesInfo,                            \
      dstSelectDim, srcSelectDim, static_cast<TYPE>(sliceSize), \
      srcSelectDimSize);

#define LARGE_INDEX(TENSOR_TYPE, TYPE,                           \
                    DST_DIM, SRC_DIM, IDX_DIM, IDX_IS_MAJOR)     \
  indexSelectLargeIndex<TENSOR_TYPE, TYPE,                       \
                        DST_DIM, SRC_DIM, IDX_DIM, IDX_IS_MAJOR> \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(            \
      dstInfo, srcInfo, indicesInfo,                             \
      dstSelectDim, srcSelectDim, static_cast<TYPE>(dstTotalSize), \
      static_cast<TYPE>((IDX_IS_MAJOR) ? sliceSize : numIndices),  \
      srcSelectDimSize);

  dim3 smallIndexGrid(std::min(THCCeilDiv(sliceSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 smallIndexBlock(std::min(sliceSize, (ptrdiff_t)128));

  dim3 largeIndexGrid(std::min(THCCeilDiv(dstTotalSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 largeIndexBlock(std::min(dstTotalSize, (ptrdiff_t)128));

  if (THCTensor_canUse32BitIndexMath(state, dst) &&
      THCTensor_canUse32BitIndexMath(state, src) &&
      THCTensor_canUse32BitIndexMath(state, indices)) {
    TensorInfo<scalar_t, unsigned int> dstInfo =
      getTensorInfo<scalar_t, THCTensor, unsigned int>(state, dst);
    int dstSelectDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstSelectDim);

    TensorInfo<scalar_t, unsigned int> srcInfo =
      getTensorInfo<scalar_t, THCTensor, unsigned int>(state, src);
    int srcSelectDim = srcInfo.collapseDims(dim);
    srcInfo.reduceDim(srcSelectDim);

    TensorInfo<int64_t, unsigned int> indicesInfo =
      getTensorInfo<int64_t, THCudaLongTensor, unsigned int>(state, indices);
    indicesInfo.collapseDims();

    // A reasonable choice for when to have each thread iterate over
    // indices to choose
    if (numIndices <= 16) {
      if (dstInfo.dims == 1 && srcInfo.dims == 1 && indContig) {
        SMALL_INDEX(scalar_t, unsigned int, 1, 1, -2);
      } else if (dstInfo.dims == 2 && srcInfo.dims == 2 && indContig) {
        SMALL_INDEX(scalar_t, unsigned int, 2, 2, -2);
      } else if (dstInfo.dims == 3 && srcInfo.dims == 3 && indContig) {
        SMALL_INDEX(scalar_t, unsigned int, 3, 3, -2);
      } else {
        SMALL_INDEX(scalar_t, unsigned int, -1, -1, -1);
      }
    } else {
      bool indexIsMajor = THCTensor_(indexShouldBeMajor)(dstInfo, dstSelectDim);

      if (dstInfo.dims == 1 && srcInfo.dims == 1 && indContig) {
        LARGE_INDEX(scalar_t, unsigned int, 1, 1, -2, true);
      } else if (dstInfo.dims == 2 && srcInfo.dims == 2 && indContig) {
        if (indexIsMajor) {
          LARGE_INDEX(scalar_t, unsigned int, 2, 2, -2, true);
        } else {
          LARGE_INDEX(scalar_t, unsigned int, 2, 2, -2, false);
        }
      } else if (dstInfo.dims == 3 && srcInfo.dims == 3 && indContig) {
        if (indexIsMajor) {
          LARGE_INDEX(scalar_t, unsigned int, 3, 3, -2, true);
        } else {
          LARGE_INDEX(scalar_t, unsigned int, 3, 3, -2, false);
        }
      } else {
        LARGE_INDEX(scalar_t, unsigned int, -1, -1, -1, true);
      }
    }
  } else {
    TensorInfo<scalar_t, uint64_t> dstInfo =
      getTensorInfo<scalar_t, THCTensor, uint64_t>(state, dst);
    int dstSelectDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstSelectDim);

    TensorInfo<scalar_t, uint64_t> srcInfo =
      getTensorInfo<scalar_t, THCTensor, uint64_t>(state, src);
    int srcSelectDim = srcInfo.collapseDims(dim);
    srcInfo.reduceDim(srcSelectDim);

    TensorInfo<int64_t, uint64_t> indicesInfo =
      getTensorInfo<int64_t, THCudaLongTensor, uint64_t>(state, indices);
    indicesInfo.collapseDims();

    LARGE_INDEX(scalar_t, uint64_t, -1, -1, -1, true);
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
#endif // THC_REAL_IS_BFLOAT16 && !__HIP_PLATFORM_HCC__
}


#endif
