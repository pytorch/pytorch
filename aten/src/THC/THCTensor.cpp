#include "THCGeneral.h"
#include "THCTensor.hpp"
#include "THCTensorCopy.h"

#include <new>

#include "generic/THCTensor.cpp"
#include "THCGenerateAllTypes.h"

#include "THCTensorInfo.cuh"

int THCTensor_nDimension(THCState *state, const THCTensor *self) {
  return THTensor_nDimension(self);
}

int THCTensor_nDimensionLegacyNoScalars(THCState *state, const THCTensor *self) {
  return THTensor_nDimensionLegacyNoScalars(self);
}

int THCTensor_nDimensionLegacyAll(THCState *state, const THCTensor *self) {
  return THTensor_nDimensionLegacyAll(self);
}

int64_t THCTensor_size(THCState *state, const THCTensor *self, int dim) {
  THArgCheck((dim >= 0) && (dim < self->dim()), 2, "out of range");
  return self->size(dim);
}

int64_t THCTensor_sizeLegacyNoScalars(THCState *state, const THCTensor *self, int dim) {
  return THTensor_sizeLegacyNoScalars(self, dim);
}


int64_t THCTensor_stride(THCState *state, const THCTensor *self, int dim) {
  THArgCheck((dim >= 0) && (dim < self->dim()), 2, "out of range");
  return self->stride(dim);
}

int64_t THCTensor_strideLegacyNoScalars(THCState *state, const THCTensor *self, int dim) {
  return THTensor_strideLegacyNoScalars(self, dim);
}

THCTensor *THCTensor_new(THCState *state, at::ScalarType scalar_type) {
  switch(scalar_type) {
    case at::ScalarType::Byte:
      return THCudaByteTensor_new(state);
    case at::ScalarType::Char:
      return THCudaCharTensor_new(state);
    case at::ScalarType::Short:
      return THCudaShortTensor_new(state);
    case at::ScalarType::Int:
      return THCudaIntTensor_new(state);
    case at::ScalarType::Long:
      return THCudaLongTensor_new(state);
    case at::ScalarType::Half:
      return THCudaHalfTensor_new(state);
    case at::ScalarType::Float:
      return THCudaTensor_new(state);
    case at::ScalarType::Double:
      return THCudaDoubleTensor_new(state);
    default:
      AT_ERROR("unexpected ScalarType: ", at::toString(scalar_type));
  }
}

void THCTensor_resize(THCState *state, THCTensor *self, at::IntList size, at::IntList stride) {
  if(stride.data()) {
    THArgCheck(stride.size() == size.size(), 3, "invalid stride");
  }

#ifdef DEBUG
  THAssert(size.size() <= INT_MAX);
#endif
  THCTensor_resizeNd(state, self, size.size(), size.data(), stride.data());
}

void THCTensor_resizeAs(THCState *state, THCTensor *self, THCTensor *src) {
  int isSame = 0;
  int d;
  if(self->dim() == src->dim())
  {
    isSame = 1;
    for(d = 0; d < self->dim(); d++)
    {
      if(self->size(d) != src->size(d))
      {
        isSame = 0;
        break;
      }
    }
  }

  if(!isSame)
    THCTensor_resizeNd(state, self, src->dim(), THTensor_getSizePtr(src), NULL);
}

void THCTensor_resizeNd(THCState *state, THCTensor *self, int nDimension, const int64_t *size, const int64_t *stride)
{
  AT_CHECK(nDimension >= 0, "resizeNd nDimension must be non-negative");
  int d;
  ptrdiff_t totalSize;
  bool hascorrectsize = true;

  for(d = 0; d < nDimension; d++)
  {
    if((self->dim() > d) && (size[d] != self->size(d))) {
      hascorrectsize = false;
    }

    // NB: this used to test that stride[d] was >= 0
    if((self->dim() > d) && stride && (stride[d] != self->stride(d))) {
      hascorrectsize = false;
    }
  }

  if(nDimension != self->dim()) {
    hascorrectsize = false;
  }

  if(hascorrectsize) {
    return;
  }

  if(nDimension != self->dim())
  {
    THTensor_resizeDim(self, nDimension);
  }

  totalSize = 1;
  for(d = nDimension-1; d >= 0; d--)
  {
    THTensor_setSizeAtDim(self, d, size[d]);
    if(stride && (stride[d] >= 0) ) {
      THTensor_setStrideAtDim(self, d, stride[d]);
    } else {
      if(d == nDimension-1) {
        THTensor_setStrideAtDim(self, d, 1);
      } else {
        // Keep stride monotonically increasing to match NumPy.
        THTensor_setStrideAtDim(self, d, std::max<int64_t>(self->size(d+1),1)*self->stride(d+1));
      }
    }
    totalSize += (self->size(d)-1)*self->stride(d);
  }

  if(totalSize+self->storage_offset() > 0)
  {
    if(!THTensor_getStoragePtr(self)) {
      THError("Tensor: invalid null storage");
    }
    if(totalSize+self->storage_offset() > THTensor_getStoragePtr(self)->size()) {
      THCStorage_resize(state, THTensor_getStoragePtr(self), totalSize+self->storage_offset());
    }
  }
}

void THCTensor_set(THCState *state, THCTensor *self, THCTensor *src)
{
  if(self != src)
    THCTensor_setStorageNd(state,
                           self,
                           THTensor_getStoragePtr(src),
                           src->storage_offset(),
                           src->dim(),
                           THTensor_getSizePtr(src),
                           THTensor_getStridePtr(src));
}

void THCTensor_setStorage(THCState *state, THCTensor *self, THCStorage *storage_, ptrdiff_t storageOffset_, at::IntList size_, at::IntList stride_)
{
  if (stride_.data()) {
    THArgCheck(size_.size() == stride_.size(), 5, "inconsistent size/stride sizes");
  }

  THCTensor_setStorageNd(state,
                         self,
                         storage_,
                         storageOffset_,
                         size_.size(),
                         size_.data(),
                         stride_.data());
}

void THCTensor_setStorageNd(THCState *state, THCTensor *self, THCStorage *storage, ptrdiff_t storageOffset, int nDimension, const int64_t *size, const int64_t *stride)
{
  /* storage */
  if(THTensor_getStoragePtr(self) != storage)
  {
    if (!THTensor_getStoragePtr(self)) {
      THError("Tensor: invalid null storage");
    }
    auto scalar_type = THTensor_getStoragePtr(self)->scalar_type();
    THStorage_free(THTensor_getStoragePtr(self));

    if (storage) {
      THTensor_stealAndSetStoragePtr(self, storage);
      THStorage_retain(THTensor_getStoragePtr(self));
    } else {
      THTensor_stealAndSetStoragePtr(self, THCStorage_new(state, scalar_type));
    }
  }

  /* storageOffset */
  if (storageOffset < 0) {
    THError("Tensor: invalid storage offset");
  }
  THTensor_setStorageOffset(self, storageOffset);

  /* size and stride */
  THCTensor_resizeNd(state, self, nDimension, size, stride);
}

void THCTensor_squeeze1d(THCState *state, THCTensor *self, THCTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck(dimension < src->dim(), 3, "dimension out of range");

  THCTensor_set(state, self, src);

  if(src->size(dimension) == 1)
  {
    for(d = dimension; d < self->dim()-1; d++)
    {
      THTensor_setSizeAtDim(self, d, self->size(d+1));
      THTensor_setStrideAtDim(self, d, self->stride(d+1));
    }
    THTensor_resizeDim(self, self->dim() - 1);
  }
}

void THCTensor_unsqueeze1d(THCState *state, THCTensor *self, THCTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck((dimension >= 0) && (dimension <= src->dim()), 3, "dimension out of range");

  THCTensor_set(state, self, src);

  THTensor_resizeDim(self, self->dim() + 1);
  for (d = self->dim()-1; d > dimension; d--) {
    THTensor_setSizeAtDim(self, d, self->size(d-1));
    THTensor_setStrideAtDim(self, d, self->stride(d-1));
  }
  if (dimension+1 < self->dim()) {
    THTensor_setStrideAtDim(self, dimension, self->size(dimension+1) * self->stride(dimension+1));
  } else {
    THTensor_setStrideAtDim(self, dimension, 1);
  }
  THTensor_setSizeAtDim(self, dimension, 1);
}

bool THCTensor_isContiguous(THCState *state, const THCTensor *self) {
  if (self->is_empty()) return true;
  int64_t z = 1;
  int d;
  for(d = self->dim()-1; d >= 0; d--)
  {
    if(self->size(d) != 1)
    {
      if(self->stride(d) == z)
        z *= self->size(d);
      else
        return false;
    }
  }
  return true;
}

bool THCTensor_allContiguous(THCState *state, THCTensor **inputs, int numInputs) {
  THAssert(numInputs > 0);
  for (int i = 0; i < numInputs; ++i) {
    if (!THCTensor_isContiguous(state, inputs[i])) {
      return false;
    }
  }
  return true;
}

ptrdiff_t THCTensor_nElement(THCState *state, const THCTensor *self) {
  if(THTensor_nDimensionLegacyAll(self) == 0)
    return 0;
  else
  {
    ptrdiff_t nElement = 1;
    int d;
    for(d = 0; d < THTensor_nDimension(self); d++)
      nElement *= self->size(d);
    return nElement;
  }
}

void THCTensor_retain(THCState *state, THCTensor *self) {
  self->retain();
}

void THCTensor_free(THCState *state, THCTensor *self) {
  THTensor_free(self);
}

int THCTensor_getDevice(THCState* state, const THCTensor* tensor) {
  if (!THTensor_getStoragePtr(tensor)) return -1;
  return THCStorage_getDevice(state, THTensor_getStoragePtr(tensor));
}

bool THCTensor_allSameDevice(THCState* state, THCTensor ** inputs, int numInputs) {
  THAssert(numInputs > 0);
  int device = THCTensor_getDevice(state, inputs[0]);
  for (int i = 1; i < numInputs; ++i) {
    if (THCTensor_getDevice(state, inputs[i]) != device) {
      return false;
    }
  }
  return true;
}

bool THCTensor_canUse32BitIndexMath(THCState* state, const THCTensor* t, ptrdiff_t max_elem) {
  ptrdiff_t elements = THCTensor_nElement(state, t);
  if (elements >= max_elem) {
    return false;
  }
  if (t->dim() == 0) {
    return true;
  }

  ptrdiff_t offset = 0;
  ptrdiff_t linearId = elements - 1;

  for (int i = THCTensor_nDimensionLegacyAll(state, t) - 1; i >= 0; --i) {
    ptrdiff_t curDimIndex =
      linearId % THCTensor_size(state, t, i);
    ptrdiff_t curDimOffset = curDimIndex *
      THCTensor_stride(state, t, i);
    offset += curDimOffset;
    linearId /= THCTensor_size(state, t, i);
  }

  if (offset >= max_elem) {
    return false;
  }

  return true;
}

bool THCTensor_all32BitIndexable(THCState* state, THCTensor** inputs, int numInputs) {
  for (int i = 0; i < numInputs; ++i) {
    if (!THCTensor_canUse32BitIndexMath(state, inputs[i])) {
      return false;
    }
  }
  return true;
}

/* Due to the resize semantics of ops with `out=` keywords, if       */ \
/* the output `tensor` has the same shape as the output of the       */ \
/* reduction operation, then any noncontiguities in the output       */ \
/* `tensor` should be preserved. This needs to be special cased b/c  */ \
/* otherwise, when keepdim=False, the implementations of reduction   */ \
/* ops resize `tensor` to the reduced size with keepdim=True, and    */ \
/* then later squeeze `tensor` to the correct output size, breaking  */ \
/* the contiguity guarantees of the resize semantics.                */ \
void THCTensor_preserveReduceDimSemantics(THCState *state, THCTensor *tensor,
                                          int in_dims, int64_t dimension, int keepdim) {
  int out_dims = THCTensor_nDimensionLegacyAll(state, tensor);
  if (out_dims > 0 && !keepdim && out_dims == in_dims - 1) {
    THCTensor_unsqueeze1d(state, tensor, tensor, dimension);
  }
}

namespace {

struct SizeAndStride {
  int64_t size;
  int64_t stride;
};

/*
 A comparator that will sort SizeAndStride structs by stride,
 in ascending order.
 */
int compareSizeAndStride(const void* a, const void* b) {
  const SizeAndStride* aS = (const SizeAndStride*) a;
  const SizeAndStride* bS = (const SizeAndStride*) b;

  if (aS->stride < bS->stride) return -1;
  if (aS->stride == bS->stride) return 0;
  return 1;
}

}

/* Returns false if there is no possibility that the tensor    */
/* has "overlapping" indices and true otherwise.               */
/* "Overlapping" indices are two+ valid indices that specify   */
/* the same offset within the tensor.                          */
/* The function does this by checking for a sufficient but not */
/* necessary condition of no overlap. In particular, that      */
/* that there exists an ordering of the tensor's dimensions    */
/* that is nicely "nested," with each dimension contained      */
/* within the next one.                                        */
bool THCTensor_maybeOverlappingIndices(THCState* state, const THCTensor* t) {
  /* Extract size/stride arrays; only consider size >1 dims. */
  SizeAndStride info[MAX_CUTORCH_DIMS];

  int dims = THCTensor_nDimensionLegacyAll(state, t);
  int nonSize1Dims = 0;
  for (int i = 0; i < dims; ++i) {
    int64_t size = THCTensor_sizeLegacyNoScalars(state, t, i);

    if (size > 1) {
      info[nonSize1Dims].size = size;
      info[nonSize1Dims].stride =
        THCTensor_stride(state, t, i);

      if (info[nonSize1Dims].stride < 1) {
        return true;
      }

      ++nonSize1Dims;
    }
  }

  /* Short-circuits if tensor is a single element.             */
  if (nonSize1Dims == 0) {
    return false;
  }

  /* Ascending order (innermost dimension in sorted view is at [0]) */
  qsort(info, nonSize1Dims, sizeof(SizeAndStride), compareSizeAndStride);

  for (int i = 0; i < (nonSize1Dims - 1); ++i) {
    if (((info[i].size - 1) * info[i].stride) >= info[i + 1].stride) {
      return true;
    }
  }

  return false;
}
