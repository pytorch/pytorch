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
    n += snprintf(str+n, L-n, "%lld", size->data[i]);
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

TH_API void THLongStorage_calculateExpandGeometry(int64_t *tensorSizes, int64_t *tensorStrides, int64_t tensorDim, THLongStorage *sizes, int64_t **esz, int64_t **est) {
  ptrdiff_t ndim = THLongStorage_size(sizes);
  int64_t numUnsqueezed = ndim - tensorDim;

  int64_t *expandedSizes = THAlloc(sizeof(int64_t)*ndim);
  int64_t *expandedStrides = THAlloc(sizeof(int64_t)*ndim);

  for (int64_t i = numUnsqueezed; i < ndim; ++i) {
    expandedSizes[i] = tensorSizes[i - numUnsqueezed];
    expandedStrides[i] = tensorStrides[i - numUnsqueezed];
  }

  for (int64_t i = numUnsqueezed - 1; i > -1; --i) {
    expandedSizes[i] = 1;
    expandedStrides[i] = expandedSizes[i+1] * expandedStrides[i+1];
  }

  // create a new geometry for the tensor
  for (int64_t i = 0; i < ndim; ++i) {
    int64_t size = expandedSizes[i];
    int64_t targetSize = THLongStorage_data(sizes)[i];
    if (size == 1) {
      if (targetSize != 1) {
        expandedSizes[i] = targetSize;
        expandedStrides[i] = 0;
      }
    } else if (size != targetSize) {
      THFree(expandedSizes);
      THFree(expandedStrides);
      THError("The expanded size of the tensor (%d) must match the existing size (%d) at \
              non-singleton dimension %ld.", targetSize, size, i);
    }
  }
  *esz = expandedSizes;
  *est = expandedStrides;
}
