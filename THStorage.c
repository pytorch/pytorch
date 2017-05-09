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
  const int L = TH_DESC_BUFF_LEN;
  THDescBuff buf;
  char *str = buf.str;
  int n = 0;
  n += snprintf(str, L-n, "[");
  int i;
  for(i = 0; i < size->size; i++) {
    if(n >= L) break;
    n += snprintf(str+n, L-n, "%ld", size->data[i]);
    if(i < size->size-1) {
      n += snprintf(str+n, L-n, " x ");
    }
  }
  if(n < L - 2) {
    snprintf(str+n, L-n, "]");
  } else {
    snprintf(str+L-5, 5, "...]");
  }
  return buf;
}

TH_API THLongStorage *THLongStorage_newInferSize(THLongStorage *size, ptrdiff_t nElement)
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
        "size '%s' is invalid for input of with %td elements", buf.str, nElement);
  } else {
    THDescBuff buf = THLongStorage_sizeDesc(size);
    THArgCheck(nElement == total_size, 2,
        "size '%s' is invalid for input of with %td elements", buf.str, nElement);
  }
  THLongStorage* copy = THLongStorage_newWithSize(size->size);
  THLongStorage_copy(copy, size);
  if (dim_infer != -1) {
    copy->data[dim_infer] = nElement / total_size;
  }
  return copy;
}

TH_API int THLongStorage_inferSize2(THLongStorage *output, long *sizesA, long dimsA, long *sizesB, long dimsB, int raiseErrors) {
  THArgCheck(sizesA != NULL, 1, "sizesA must not be null");
  THArgCheck(sizesB != NULL, 2, "sizesB must not be null");
  THArgCheck(dimsA, 1, "Can't expand empty tensor a");
  THArgCheck(dimsB, 1, "Can't expand empty tensor b");
  ptrdiff_t ndim = dimsA > dimsB ? dimsA : dimsB;

  long *expandedSizes = THAlloc(sizeof(long)*ndim);

  for (long i = ndim - 1; i >= 0; --i) {
    long offset = ndim - 1 - i;
    long dimA = dimsA - 1 - offset;
    long dimB = dimsB - 1 - offset;
    long sizeA = (dimA >= 0) ? sizesA[dimA] : 1;
    long sizeB = (dimB >= 0) ? sizesB[dimB] : 1;
    if (sizeA != sizeB) {
      if (sizeA == 1) {
        sizeA = sizeB;
      }
      else if (sizeB == 1) {
      }
      else {
        THFree(expandedSizes);
        if (raiseErrors) {
          THError("The size of tensor a (%ld) must match the size of tensor b (%ld) at "
                  "non-singleton dimension %ld.", sizeA, sizeB, i);
        }
        return -1;
      }
    }
    expandedSizes[ i ] = sizeA;
  }
  THLongStorage_resize(output, ndim);
  memcpy(THLongStorage_data(output), expandedSizes, sizeof(long)*ndim);
  THFree(expandedSizes);
  return 0;
}

TH_API int THLongStorage_inferSizeN(THLongStorage *output, int n, long **sizes, long *dims, int raiseErrors) {
  THArgCheck(n > 0, 2, "n must be greater than 0");
  THArgCheck(sizes != NULL, 1, "sizesA must not be null");
  THArgCheck(dims != NULL, 1, "dims must not be null");

  ptrdiff_t ndim = 0;
  for (int j = 0; j < n; ++j) {
    THArgCheck(sizes[ j ] != NULL, 1, "size %d must not be null", j);
    THArgCheck(dims[ j ], 1, "Can't expand empty tensor %d", j);
    ptrdiff_t ndim = dims[ j ] > ndim ? dims[ j ] : ndim;
  }

  long *expandedSizes = THAlloc(sizeof(long)*ndim);

  for (long i = ndim - 1; i >= 0; --i) {
    long max_dim_size = 1;
    long offset = ndim - 1 - i;
    for (int j  = 0; j < n; ++j) {
      long dim = dims[ j ] - 1 - offset;
      long size = (dim >= 0) ? sizes[ i ][ dim ] : 1;
      if (size != max_dim_size) {
        if (max_dim_size == 1){
          max_dim_size = size;
        } else if (size == 1) {
          // we'll expand, nothing to do
        } else {
          THFree(expandedSizes);
          if (raiseErrors) {
            THError("The size of tensor %i (%ld) must match the expanded size of tensor (%ld) at "
                    "non-singleton dimension %ld.", j, size, max_dim_size, i);
          }
          return -1;
        }
      }
    }
    expandedSizes[ i ] = max_dim_size;
  }
  THLongStorage_resize(output, ndim);
  memcpy(THLongStorage_data(output), expandedSizes, sizeof(long)*ndim);
  THFree(expandedSizes);
  return 0;
}

TH_API int THLongStorage_inferExpandGeometry(long *tensorSizes, long *tensorStrides, long tensorDim, THLongStorage *sizes, long **esz, long **est, int raiseErrors) {
  ptrdiff_t ndim = THLongStorage_size(sizes);

  long *expandedSizes = THAlloc(sizeof(long)*ndim);
  long *expandedStrides = THAlloc(sizeof(long)*ndim);

  // create a new geometry for the tensors
  for (long i = ndim - 1; i >= 0; --i) {
    long offset = ndim - 1 - i;
    long dim = tensorDim - 1 - offset;
    long size = (dim >= 0) ? tensorSizes[dim] : 1;
    long stride = (dim >= 0) ?
        tensorStrides[dim] : expandedSizes[i + 1] * expandedStrides[i+1];
    long targetSize = THLongStorage_data(sizes)[i];
    if (size != targetSize) {
      if (size == 1) {
        size = targetSize;
        stride = 0;
      } else {
        THFree(expandedSizes);
        THFree(expandedStrides);
        if (raiseErrors) {
          THError("The expanded size of the tensor (%d) must match the existing size (%d) at "
                  "non-singleton dimension %ld.", targetSize, size, i);
        }
        return -1;
      }
    }
    expandedSizes[i] = size;
    expandedStrides[i] = stride;
  }
  *esz = expandedSizes;
  *est = expandedStrides;
  return 0;
}
