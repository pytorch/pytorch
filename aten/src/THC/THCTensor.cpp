#include <THC/THCGeneral.h>
#include <THC/THCTensor.hpp>
#include <THC/THCTensorCopy.h>

#include <new>

#include <THC/generic/THCTensor.cpp>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensor.cpp>
#include <THC/THCGenerateBoolType.h>

#include <THC/generic/THCTensor.cpp>
#include <THC/THCGenerateBFloat16Type.h>

#include <THC/THCTensorInfo.cuh>

#include <ATen/native/cuda/Resize.cuh>

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

THCTensor *THCTensor_new(THCState *state, caffe2::TypeMeta type_meta) {
  auto scalar_type = at::typeMetaToScalarType(type_meta);
  switch (scalar_type) {
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
    case at::ScalarType::Bool:
      return THCudaBoolTensor_new(state);
    case at::ScalarType::BFloat16:
      return THCudaBFloat16Tensor_new(state);
    default:
      AT_ERROR("unexpected ScalarType: ", toString(scalar_type));
  }
}

void THCTensor_resize(THCState *state, THCTensor *self, at::IntArrayRef size, at::IntArrayRef stride) {
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
  TORCH_CHECK(nDimension >= 0, "resizeNd nDimension must be non-negative");
  at::IntArrayRef sizes(size, nDimension);
  at::optional<at::IntArrayRef> strides;
  if (stride) {
    strides = at::IntArrayRef(stride, nDimension);
  }
  at::native::resize_impl_cuda_(self, sizes, strides, /*device_guard=*/false);
}

void THCTensor_set(THCState *state, THCTensor *self, THCTensor *src)
{
  if(self != src)
    THCTensor_setStorage(state,
                         self,
                         THTensor_getStoragePtr(src),
                         src->storage_offset(),
                         src->sizes(),
                         src->strides());
}

void THCTensor_setStorage(THCState *state, THCTensor *self, THCStorage *storage_, ptrdiff_t storageOffset_, at::IntArrayRef size_, at::IntArrayRef stride_)
{
  c10::raw::intrusive_ptr::incref(storage_);
  THTensor_wrap(self).set_(at::Storage(c10::intrusive_ptr<at::StorageImpl>::reclaim(storage_)),
                           storageOffset_, size_, stride_);
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
    at::DimVector newSize(static_cast<size_t>(self->dim() - 1));
    at::DimVector newStride(static_cast<size_t>(self->dim() - 1));
    for (d = 0; d < dimension; d++)
    {
      newSize[d] = self->size(d);
      newStride[d] = self->stride(d);
    }

    for(d = dimension; d < self->dim()-1; d++)
    {
      newSize[d] = self->size(d+1);
      newStride[d] = self->stride(d+1);
    }
    self->set_sizes_and_strides(newSize, newStride);
  }
}

void THCTensor_unsqueeze1d(THCState *state, THCTensor *self, THCTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck((dimension >= 0) && (dimension <= src->dim()), 3, "dimension out of range");

  THCTensor_set(state, self, src);

  at::DimVector newSize(static_cast<size_t>(/* size */ self->dim()+1));
  at::DimVector newStride(static_cast<size_t>(/* size */ self->dim()+1));

  for(d = self->dim(); d > dimension; d--)
  {
    newSize[d] = self->size(d-1);
    newStride[d] = self->stride(d-1);
  }
  if (dimension < self->dim())
  {
    newStride[dimension] = self->size(dimension) * self->stride(dimension);
  }
  else
  {
    newStride[dimension] = 1;
  }
  newSize[dimension] = 1;
  for(d = dimension - 1; d >= 0; d--)
  {
    newSize[d] = self->size(d);
    newStride[d] = self->stride(d);
  }
  self->set_sizes_and_strides(newSize, newStride);
}

bool THCTensor_allContiguous(THCState *state, THCTensor **inputs, int numInputs) {
  THAssert(numInputs > 0);
  for (int i = 0; i < numInputs; ++i) {
    if (!inputs[i]->is_contiguous()) {
      return false;
    }
  }
  return true;
}

ptrdiff_t THCTensor_nElement(THCState *state, const THCTensor *self) {
  if(THTensor_nDimensionLegacyAll(self) == 0) {
    return 0;
  } else {
    return self->numel();
  }
}

// NB: It is INVALID to call this on an UndefinedTensor
void THCTensor_retain(THCState *state, THCTensor *self) {
  c10::raw::intrusive_ptr::incref(self);
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
