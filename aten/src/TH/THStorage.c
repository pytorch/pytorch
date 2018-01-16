#include "THAtomic.h"
#include "THStorage.h"

#include "generic/THStorage.c"
#include "THGenerateAllTypes.h"

#include "generic/THStorage.c"
#include "THGenerateHalfType.h"

#include "generic/THStorageCopy.c"
#include "THGenerateAllTypes.h"

#include "generic/THStorageCopy.c"
#include "THGenerateHalfType.h"


THDescBuff THLongStorage_sizeDesc(const THLongStorage *size) {
  return _THSizeDesc(size->data, size->size);
}

THLongStorage *THLongStorage_newInferSize(THLongStorage *size, ptrdiff_t nElement)
{
  ptrdiff_t total_size = (size->size > 0 ? 1 : 0);
  ptrdiff_t dim_infer = -1;
  ptrdiff_t i;
  for (i = 0; i < size->size; i++) {
    if (size->data[i] == -1) {
      THArgCheck(dim_infer == -1, 1, "only one dimension can be inferred");
      dim_infer = i;
    } else {
      total_size *= size->data[i];
    }
  }
  if (dim_infer != -1) {
    THDescBuff buf = THLongStorage_sizeDesc(size);
    THArgCheck(total_size > 0 && nElement % total_size == 0, 2,
        "size '%s' is invalid for input with %td elements", buf.str, nElement);
  } else {
    THDescBuff buf = THLongStorage_sizeDesc(size);
    THArgCheck(nElement == total_size, 2,
        "size '%s' is invalid for input with %td elements", buf.str, nElement);
  }
  THLongStorage* copy = THLongStorage_newWithSize(size->size);
  THLongStorage_copy(copy, size);
  if (dim_infer != -1) {
    copy->data[dim_infer] = nElement / total_size;
  }
  return copy;
}

int THLongStorage_inferSize2(THLongStorage *output, int64_t *sizesA, int64_t dimsA, int64_t *sizesB, int64_t dimsB,
                             char *error_buffer, int buffer_len) {
  THArgCheck(sizesA != NULL, 1, "sizesA must not be null");
  THArgCheck(sizesB != NULL, 2, "sizesB must not be null");
  THArgCheck(dimsA, 1, "Can't expand empty tensor a");
  THArgCheck(dimsB, 1, "Can't expand empty tensor b");
  ptrdiff_t ndim = dimsA > dimsB ? dimsA : dimsB;

  int64_t *expandedSizes = THAlloc(sizeof(int64_t)*ndim);

  for (int64_t i = ndim - 1; i >= 0; --i) {
    int64_t offset = ndim - 1 - i;
    int64_t dimA = dimsA - 1 - offset;
    int64_t dimB = dimsB - 1 - offset;
    int64_t sizeA = (dimA >= 0) ? sizesA[dimA] : 1;
    int64_t sizeB = (dimB >= 0) ? sizesB[dimB] : 1;
    if (sizeA == sizeB || sizeA == 1 || sizeB == 1) {
      expandedSizes[i] = THMax(sizeA, sizeB);
    } else {
      THFree(expandedSizes);
      snprintf(error_buffer, buffer_len, "The size of tensor a (%" PRId64 ") must match the size of tensor b (%" PRId64 ") at "
               "non-singleton dimension %" PRId64 ".", sizeA, sizeB, i);
      return -1;
    }
  }
  THLongStorage_resize(output, ndim);
  memcpy(THLongStorage_data(output), expandedSizes, sizeof(int64_t)*ndim);
  THFree(expandedSizes);
  return 0;
}

int THLongStorage_inferSizeN(THLongStorage *output, int n, int64_t **sizes, int64_t *dims,
                             char *error_buffer, int buffer_len) {
  THArgCheck(n > 0, 2, "n must be greater than 0");
  THArgCheck(sizes != NULL, 1, "sizes must not be null");
  THArgCheck(dims != NULL, 1, "dims must not be null");

  ptrdiff_t ndim = 0;
  for (int j = 0; j < n; ++j) {
    THArgCheck(sizes[ j ] != NULL, 1, "size %d must not be null", j);
    THArgCheck(dims[ j ], 1, "Can't expand empty tensor %d", j);
    ndim = dims[ j ] > ndim ? dims[ j ] : ndim;
  }

  int64_t *expandedSizes = THAlloc(sizeof(int64_t)*ndim);

  for (int64_t i = ndim - 1; i >= 0; --i) {
    expandedSizes[ i ] = 1;
    int64_t offset = ndim - 1 - i;
    for (int j  = 0; j < n; ++j) {
      int64_t dim = dims[ j ] - 1 - offset;
      int64_t size = (dim >= 0) ? sizes[ j ][ dim ] : 1;
      if (size == expandedSizes[ i ] || size == 1 || expandedSizes[ i ] == 1) {
        expandedSizes[ i ] =  THMax(expandedSizes[ i ], size);
      } else {
        snprintf(error_buffer, buffer_len, "The size of tensor %i (%" PRId64 ") must match the expanded size"
                 "of tensor (%" PRId64 ") at non-singleton dimension %" PRId64 ".", j, size, expandedSizes[ i ], i);
        THFree(expandedSizes);
        return -1;
      }
    }
  }
  THLongStorage_resize(output, ndim);
  memcpy(THLongStorage_data(output), expandedSizes, sizeof(int64_t)*ndim);
  THFree(expandedSizes);
  return 0;
}

int THLongStorage_inferExpandGeometry(int64_t *tensorSizes, int64_t *tensorStrides, int64_t tensorDim,
                                        THLongStorage *sizes, int64_t **expandedSizes, int64_t **expandedStrides,
                                        char *error_buffer, int buffer_len) {
  ptrdiff_t ndim = THLongStorage_size(sizes);

  int64_t *expandedSizesCalc = THAlloc(sizeof(int64_t)*ndim);
  int64_t *expandedStridesCalc = THAlloc(sizeof(int64_t)*ndim);

  // create a new geometry for the tensors
  for (int64_t i = ndim - 1; i >= 0; --i) {
    int64_t offset = ndim - 1 - i;
    int64_t dim = tensorDim - 1 - offset;
    int64_t size = (dim >= 0) ? tensorSizes[dim] : 1;
    int64_t stride = (dim >= 0) ?
        tensorStrides[dim] : expandedSizesCalc[i + 1] * expandedStridesCalc[i+1];
    int64_t targetSize = THLongStorage_data(sizes)[i];
    if (targetSize == -1) {
      if (dim < 0) {
        THFree(expandedSizesCalc);
        THFree(expandedStridesCalc);
        snprintf(error_buffer, buffer_len, "The expanded size of the tensor (%" PRId64 ") isn't allowed in a leading, non-existing dimension %" PRId64 ".", targetSize, i);
        return -1;
      } else {
        targetSize = size;
      }
    }
    if (size != targetSize) {
      if (size == 1) {
        size = targetSize;
        stride = 0;
      } else {
        THFree(expandedSizesCalc);
        THFree(expandedStridesCalc);
        snprintf(error_buffer, buffer_len, "The expanded size of the tensor (%" PRId64 ") must match the existing size (%" PRId64 ") at "
                 "non-singleton dimension %" PRId64 ".", targetSize, size, i);
        return -1;
      }
    }
    expandedSizesCalc[i] = size;
    expandedStridesCalc[i] = stride;
  }
  *expandedSizes = expandedSizesCalc;
  *expandedStrides = expandedStridesCalc;
  return 0;
}
