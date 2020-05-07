#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensor.cpp"
#else

#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>

/**** access methods ****/
THCStorage *THCTensor_(storage)(THCState *state, const THCTensor *self)
{
  return THTensor_getStoragePtr(self);
}

ptrdiff_t THCTensor_(storageOffset)(THCState *state, const THCTensor *self)
{
  return self->storage_offset();
}

int THCTensor_(nDimension)(THCState *state, const THCTensor *self)
{
  return THCTensor_nDimension(state, self);
}

int THCTensor_(nDimensionLegacyNoScalars)(THCState *state, const THCTensor *self)
{
  return THCTensor_nDimensionLegacyNoScalars(state, self);
}

int THCTensor_(nDimensionLegacyAll)(THCState *state, const THCTensor *self)
{
  return THCTensor_nDimensionLegacyAll(state, self);
}

int64_t THCTensor_(size)(THCState *state, const THCTensor *self, int dim)
{
  return THCTensor_size(state, self, dim);
}

int64_t THCTensor_(sizeLegacyNoScalars)(THCState *state, const THCTensor *self, int dim)
{
  return THTensor_sizeLegacyNoScalars(self, dim);
}

int64_t THCTensor_(stride)(THCState *state, const THCTensor *self, int dim)
{
  return THCTensor_stride(state, self, dim);
}

int64_t THCTensor_(strideLegacyNoScalars)(THCState *state, const THCTensor *self, int dim)
{
  return THTensor_strideLegacyNoScalars(self, dim);
}

scalar_t *THCTensor_(data)(THCState *state, const THCTensor *self)
{
  if(THTensor_getStoragePtr(self))
    return (THCStorage_(data)(state, THTensor_getStoragePtr(self))+self->storage_offset());
  else
    return NULL;
}

/**** creation methods ****/

/* Empty init */
THCTensor *THCTensor_(new)(THCState *state)
{
  return c10::make_intrusive<at::TensorImpl, at::UndefinedTensorImpl>(
    c10::intrusive_ptr<at::StorageImpl>::reclaim(THCStorage_(new)(state)),
    at::DispatchKey::CUDA
  ).release();
}

/* Pointer-copy init */
THCTensor *THCTensor_(newWithTensor)(THCState *state, THCTensor *tensor)
{
  return at::native::alias(THTensor_wrap(tensor)).unsafeReleaseTensorImpl();
}

THCTensor *THCTensor_(newWithStorage1d)(THCState *state, THCStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0)
{
  c10::raw::intrusive_ptr::incref(storage);
  THTensor *self = c10::make_intrusive<at::TensorImpl, at::UndefinedTensorImpl>(
    c10::intrusive_ptr<at::StorageImpl>::reclaim(storage),
    at::DispatchKey::CUDA
  ).release();
  THCTensor_(setStorage)(state, self, storage, storageOffset, {size0}, {stride0});

  return self;
}

THCTensor *THCTensor_(newWithSize)(THCState *state, at::IntArrayRef size, at::IntArrayRef stride)
{
  TORCH_INTERNAL_ASSERT(false, "this function should not be called and is in the process of being removed");
}

THCTensor *THCTensor_(newWithSize1d)(THCState *state, int64_t size0)
{
  THCStorage *new_storage = THCStorage_(new)(state);
  THCTensor *self = c10::make_intrusive<at::TensorImpl, at::UndefinedTensorImpl>(
    c10::intrusive_ptr<at::StorageImpl>::reclaim(new_storage),
    at::DispatchKey::CUDA
  ).release();
  THCTensor_(setStorage)(state, self, new_storage, 0, {size0}, {});

  return self;
}

THCTensor *THCTensor_(newClone)(THCState *state, THCTensor *self)
{
  // already available in Aten as at::clone()
  THCTensor *tensor = THCTensor_(new)(state);
  at::Tensor tensor_wrap = THTensor_wrap(tensor);
  at::Tensor self_wrap = THTensor_wrap(self);
  tensor_wrap.resize_as_(self_wrap);
  THCTensor_(copy)(state, tensor, self);
  return tensor;
}

THCTensor *THCTensor_(newContiguous)(THCState *state, THCTensor *self)
{
  if(!THCTensor_(isContiguous)(state, self)) {
    return THCTensor_(newClone)(state, self);
  } else {
    THCTensor_(retain)(state, self);
    return self;
  }
}

THCTensor *THCTensor_(newSelect)(THCState *state, THCTensor *tensor, int dimension_, int64_t sliceIndex_)
{
  THCTensor *self = THCTensor_(newWithTensor)(state, tensor);
  THCTensor_(select)(state, self, NULL, dimension_, sliceIndex_);
  return self;
}

THCTensor *THCTensor_(newNarrow)(THCState *state, THCTensor *tensor, int dimension_, int64_t firstIndex_, int64_t size_)
{
  THCTensor *self = THCTensor_(newWithTensor)(state, tensor);
  THCTensor_(narrow)(state, self, NULL, dimension_, firstIndex_, size_);
  return self;
}

THCTensor *THCTensor_(newTranspose)(THCState *state, THCTensor *tensor, int dimension1_, int dimension2_)
{
  THCTensor *self = THCTensor_(newWithTensor)(state, tensor);
  THCTensor_(transpose)(state, self, NULL, dimension1_, dimension2_);
  return self;
}

// Collapses the first two dimensions of a tensor.
// Assumes the input tensor is contiguous.
THCTensor *THCTensor_(newFoldBatchDim)(THCState *state, THCTensor *input) {
  int in_dims = THCTensor_(nDimensionLegacyAll)(state, input);
  THArgCheck(in_dims >= 2, 1, "Tensor needs to have at least two dimensions");
  THArgCheck(THCTensor_(isContiguous)(state, input), 1,
             "Tensor must be contiguous");
  std::vector<int64_t> new_size(in_dims - 1);
  new_size[0] = THCTensor_(size)(state, input, 0) * THCTensor_(size)(state, input, 1);
  for (int i = 2; i < in_dims; i++) {
    new_size[i - 1] = THCTensor_(size)(state, input, i);
  }
  THCTensor *output = at::native::view(THTensor_wrap(input), new_size).unsafeReleaseTensorImpl();
  return output;
}

/* Resize */
void THCTensor_(resize)(THCState *state, THCTensor *self, at::IntArrayRef size, at::IntArrayRef stride)
{
  THCTensor_resize(state, self, size, stride);
}

void THCTensor_(resizeAs)(THCState *state, THCTensor *self, THCTensor *src)
{
  // already available in Aten as at::resize_as_()
  THCTensor_resizeAs(state, self, src);
}

void THCTensor_(resize0d)(THCState *state, THCTensor *tensor)
{
  THCTensor_resizeNd(state, tensor, 0, {}, nullptr);
}

void THCTensor_(resize1d)(THCState *state, THCTensor *tensor, int64_t size0)
{
  int64_t size[1] = {size0};
  THCTensor_resizeNd(state, tensor, 1, size, nullptr);
}

void THCTensor_(resize2d)(THCState *state, THCTensor *tensor, int64_t size0, int64_t size1)
{
  int64_t size[2] = {size0, size1};
  THCTensor_resizeNd(state, tensor, 2, size, nullptr);
}

void THCTensor_(resize3d)(THCState *state, THCTensor *tensor, int64_t size0, int64_t size1, int64_t size2)
{
  int64_t size[3] = {size0, size1, size2};
  THCTensor_resizeNd(state, tensor, 3, size, nullptr);
}

void THCTensor_(resize4d)(THCState *state, THCTensor *self, int64_t size0, int64_t size1, int64_t size2, int64_t size3)
{
  int64_t size[4] = {size0, size1, size2, size3};
  THCTensor_resizeNd(state, self, 4, size, nullptr);
}

void THCTensor_(resize5d)(THCState *state, THCTensor *self, int64_t size0, int64_t size1, int64_t size2, int64_t size3, int64_t size4)
{
  int64_t size[5] = {size0, size1, size2, size3, size4};
  THCTensor_resizeNd(state, self, 5, size, nullptr);
}

void THCTensor_(set)(THCState *state, THCTensor *self, THCTensor *src)
{
  THCTensor_set(state, self, src);
}

void THCTensor_(setStorage)(THCState *state, THCTensor *self, THCStorage *storage_, ptrdiff_t storageOffset_, at::IntArrayRef size_, at::IntArrayRef stride_) {
  THCTensor_setStorage(state, self, storage_, storageOffset_, size_, stride_);
}

void THCTensor_(narrow)(THCState *state, THCTensor *self, THCTensor *src, int dimension, int64_t firstIndex, int64_t size)
{
  if(!src)
    src = self;

  THArgCheck( (dimension >= 0) && (dimension < src->dim()), 3, "out of range");
  THArgCheck( firstIndex >= 0, 4, "out of range");
  THArgCheck( size >= 0, 5, "out of range");
  THArgCheck(firstIndex+size <= src->size(dimension), 5, "out of range");

  THCTensor_(set)(state, self, src);

  if (firstIndex > 0) {
    self->set_storage_offset(self->storage_offset() + firstIndex*self->stride(dimension));
  }

  self->set_size(dimension, size);
}

void THCTensor_(select)(THCState *state, THCTensor *self, THCTensor *src, int dimension, int64_t sliceIndex)
{
  int d;

  if(!src)
    src = self;

  THArgCheck(src->dim() > 0, 1, "cannot select on a 0-dim tensor");
  THArgCheck((dimension >= 0) && (dimension < src->dim()), 3, "out of range");
  THArgCheck((sliceIndex >= 0) && (sliceIndex < src->size(dimension)), 4, "out of range");

  THCTensor_(set)(state, self, src);
  THCTensor_(narrow)(state, self, NULL, dimension, sliceIndex, 1);

  std::vector<int64_t> newSize(self->dim()-1);
  std::vector<int64_t> newStride(self->dim()-1);

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

void THCTensor_(transpose)(THCState *state, THCTensor *self, THCTensor *src, int dimension1, int dimension2)
{
  int64_t z;

  if(!src)
    src = self;

  THArgCheck( (dimension1 >= 0) && (dimension1 < THTensor_nDimensionLegacyNoScalars(src)), 1, "out of range");
  THArgCheck( (dimension2 >= 0) && (dimension2 < THTensor_nDimensionLegacyNoScalars(src)), 2, "out of range");

  THCTensor_(set)(state, self, src);

  if(dimension1 == dimension2)
    return;

  z = self->stride(dimension1);
  self->set_stride(dimension1, self->stride(dimension2));
  self->set_stride(dimension2, z);
  z = self->size(dimension1);
  self->set_size(dimension1, self->size(dimension2));
  self->set_size(dimension2, z);
}

void THCTensor_(squeeze1d)(THCState *state, THCTensor *self, THCTensor *src, int dimension)
{
  THCTensor_squeeze1d(state, self, src, dimension);
}

void THCTensor_(unsqueeze1d)(THCState *state, THCTensor *self, THCTensor *src, int dimension)
{
  THCTensor_unsqueeze1d(state, self, src, dimension);
}

int THCTensor_(isContiguous)(THCState *state, const THCTensor *self)
{
  return self->is_contiguous();
}

int THCTensor_(isSameSizeAs)(THCState *state, const THCTensor *self, const THCTensor* src)
{
  int d;
  if (self->dim() != src->dim())
    return 0;
  for(d = 0; d < self->dim(); ++d)
  {
    if(self->size(d) != src->size(d))
      return 0;
  }
  return 1;
}

ptrdiff_t THCTensor_(nElement)(THCState *state, const THCTensor *self)
{
  return THCTensor_nElement(state, self);
}

void THCTensor_(retain)(THCState *state, THCTensor *self)
{
  THCTensor_retain(state, self);
}

void THCTensor_(free)(THCState *state, THCTensor *self)
{
  THCTensor_free(state, self);
}

void THCTensor_(freeCopyTo)(THCState *state, THCTensor *self, THCTensor *dst)
{
  if(self != dst)
    THCTensor_(copy)(state, dst, self);

  THCTensor_(free)(state, self);
}

/*******************************************************************************/

void THCTensor_(resizeNd)(THCState *state, THCTensor *self, int nDimension, const int64_t *size, const int64_t *stride)
{
  THCTensor_resizeNd(state, self, nDimension, size, stride);
}

void THCTensor_(set0d)(THCState *state, THCTensor *tensor, scalar_t value)
{
  THArgCheck(THTensor_nDimension(tensor) == 0, 1, "tensor must have no dimensions");
  THCStorage_(set)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset(), value);
}


scalar_t THCTensor_(get0d)(THCState *state, const THCTensor *tensor)
{
  THArgCheck(THTensor_nDimension(tensor) == 0, 1, "tensor must have no dimensions dimension");
  return THCStorage_(get)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset());
}

void THCTensor_(set1d)(THCState *state, THCTensor *tensor, int64_t x0, scalar_t value)
{
  THArgCheck(THTensor_nDimensionLegacyNoScalars(tensor) == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < THTensor_sizeLegacyNoScalars(tensor, 0)), 2, "out of range");
  THCStorage_(set)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*THTensor_strideLegacyNoScalars(tensor, 0), value);
}

scalar_t THCTensor_(get1d)(THCState *state, const THCTensor *tensor, int64_t x0)
{
  THArgCheck(THTensor_nDimensionLegacyNoScalars(tensor) == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < THTensor_sizeLegacyNoScalars(tensor, 0)), 2, "out of range");
  return THCStorage_(get)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*THTensor_strideLegacyNoScalars(tensor, 0));
}

void THCTensor_(set2d)(THCState *state, THCTensor *tensor, int64_t x0, int64_t x1, scalar_t value)
{
  THArgCheck(tensor->dim() == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)), 2, "out of range");
  THCStorage_(set)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1), value);
}

scalar_t THCTensor_(get2d)(THCState *state, const THCTensor *tensor, int64_t x0, int64_t x1)
{
  THArgCheck(tensor->dim() == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)), 2, "out of range");
  return THCStorage_(get)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1));
}

void THCTensor_(set3d)(THCState *state, THCTensor *tensor, int64_t x0, int64_t x1, int64_t x2, scalar_t value)
{
  THArgCheck(tensor->dim() == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)) && (x2 >= 0) && (x2 < tensor->size(2)), 2, "out of range");
  THCStorage_(set)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1)+x2*tensor->stride(2), value);
}

scalar_t THCTensor_(get3d)(THCState *state, const THCTensor *tensor, int64_t x0, int64_t x1, int64_t x2)
{
  THArgCheck(tensor->dim() == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)) && (x2 >= 0) && (x2 < tensor->size(2)), 2, "out of range");
  return THCStorage_(get)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1)+x2*tensor->stride(2));
}

void THCTensor_(set4d)(THCState *state, THCTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3, scalar_t value)
{
  THArgCheck(tensor->dim() == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)) && (x2 >= 0) && (x2 < tensor->size(2)) && (x3 >= 0) && (x3 < tensor->size(3)), 2, "out of range");
  THCStorage_(set)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1)+x2*tensor->stride(2)+x3*tensor->stride(3), value);
}

scalar_t THCTensor_(get4d)(THCState *state, const THCTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3)
{
  THArgCheck(tensor->dim() == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)) && (x2 >= 0) && (x2 < tensor->size(2)) && (x3 >= 0) && (x3 < tensor->size(3)), 2, "out of range");
  return THCStorage_(get)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1)+x2*tensor->stride(2)+x3*tensor->stride(3));
}

int THCTensor_(checkGPU)(THCState *state, unsigned int nTensors, ...)
{
  int curDev = -1;
  THCudaCheck(cudaGetDevice(&curDev));
  va_list args;
  va_start(args, nTensors);
  int valid = 1;
  for (unsigned int i = 0; i < nTensors; i++) {
    THCTensor* tensor = va_arg(args, THCTensor*);
    if (tensor == NULL) {
      continue;
    }

    const int tensorDev = THCTensor_(getDevice)(state, tensor);

    // Skips CPU tensors
    if (tensorDev == -1) { continue; }

    // Checks all tensors are on the same device
    if (tensorDev != curDev) {
      valid = 0;
      break;
    }
  }

  va_end(args);
  return valid;
}

THCDescBuff THCTensor_(sizeDesc)(THCState *state, const THCTensor *tensor) {
  const int L = THC_DESC_BUFF_LEN;
  THCDescBuff buf;
  char *str = buf.str;
  int n = 0;
  n += snprintf(str, L-n, "[");
  int i;
  for(i = 0; i < tensor->dim(); i++) {
    if(n >= L) break;
    n += snprintf(str+n, L-n, "%" PRId64, tensor->size(i));
    if(i < tensor->dim()-1) {
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

#endif
