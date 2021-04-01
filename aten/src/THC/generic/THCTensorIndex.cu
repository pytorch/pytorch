#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorIndex.cu"
#else

#include <ATen/cuda/CUDAContext.h>
#include <ATen/MemoryOverlap.h>
#include <c10/cuda/CUDAException.h>

// Check tensor dimensions for index operations, and return the slice size.
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

#endif
