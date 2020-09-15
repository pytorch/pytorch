#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensor.cpp"
#else

#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>
#include <new>
#include <ATen/NamedTensorUtils.h>
#include <ATen/MemoryOverlap.h>

/**** access methods ****/
THStorage *THTensor_(storage)(const THTensor *self)
{
  return THTensor_getStoragePtr(self);
}

ptrdiff_t THTensor_(storageOffset)(const THTensor *self)
{
  return self->storage_offset();
}

int THTensor_(nDimension)(const THTensor *self)
{
  return THTensor_nDimension(self);
}

int THTensor_(nDimensionLegacyNoScalars)(const THTensor *self)
{
  return THTensor_nDimensionLegacyNoScalars(self);
}

int THTensor_(nDimensionLegacyAll)(const THTensor *self)
{
  return THTensor_nDimensionLegacyAll(self);
}

int64_t THTensor_(size)(const THTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->dim()), 2, "dimension %d out of range of %dD tensor",
      dim, THTensor_(nDimensionLegacyNoScalars)(self));
  return self->size(dim);
}

int64_t THTensor_(stride)(const THTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->dim()), 2, "dimension %d out of range of %dD tensor",
      dim, THTensor_(nDimensionLegacyNoScalars)(self));
  return self->stride(dim);
}

scalar_t *THTensor_(data)(const THTensor *self) {
  return self->data<scalar_t>();
}

/**** creation methods ****/

/* Empty init */
THTensor *THTensor_(new)(void)
{
  return c10::make_intrusive<at::TensorImpl, at::UndefinedTensorImpl>(
             c10::intrusive_ptr<at::StorageImpl>::reclaim(THStorage_(new)()),
             at::DispatchKey::CPU,
#ifdef THQUANTIZED
             caffe2::TypeMeta::Make<quantized_t>()
#else
             caffe2::TypeMeta::Make<scalar_t>()
#endif
                 )
      .release();
}

/* Pointer-copy init */
THTensor *THTensor_(newWithTensor)(THTensor *tensor)
{
  return at::native::alias(THTensor_wrap(tensor)).unsafeReleaseTensorImpl();
}

THTensor *THTensor_(newWithStorage1d)(THStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0)
{
  c10::raw::intrusive_ptr::incref(storage);
  THTensor* self = c10::make_intrusive<at::TensorImpl, at::UndefinedTensorImpl>(
                       c10::intrusive_ptr<at::StorageImpl>::reclaim(storage),
                       at::DispatchKey::CPU,
#ifdef THQUANTIZED
                       caffe2::TypeMeta::Make<quantized_t>()
#else
                       caffe2::TypeMeta::Make<scalar_t>()
#endif
                           )
                       .release();
  THTensor_(setStorage)(self, storage, storageOffset,  {size0}, {stride0});

  return self;
}

THTensor *THTensor_(newWithSize1d)(int64_t size0)
{
  THStorage *new_storage = THStorage_(new)();
  THTensor* self =
      c10::make_intrusive<at::TensorImpl, at::UndefinedTensorImpl>(
          c10::intrusive_ptr<at::StorageImpl>::reclaim(new_storage),
          at::DispatchKey::CPU,
          caffe2::TypeMeta::Make<scalar_t>())
          .release();
  THTensor_(setStorage)(self, new_storage, 0, {size0}, {});

  return self;
}

THTensor *THTensor_(newClone)(THTensor *self)
{
  // already available in Aten as at::clone()
  THTensor *tensor = THTensor_(new)();
  at::Tensor tensor_wrap = THTensor_wrap(tensor);
  at::Tensor self_wrap = THTensor_wrap(self);
  tensor_wrap.resize_as_(self_wrap);
  at::native::copy_(tensor_wrap, self_wrap, false);
  return tensor;
}

THTensor *THTensor_(newContiguous)(THTensor *self)
{
  if(!THTensor_(isContiguous)(self))
    return THTensor_(newClone)(self);
  else
  {
    THTensor_(retain)(self);
    return self;
  }
}

THTensor *THTensor_(newSelect)(THTensor *tensor, int dimension_, int64_t sliceIndex_)
{
  THTensor *self = THTensor_(newWithTensor)(tensor);
  THTensor_(select)(self, NULL, dimension_, sliceIndex_);
  return self;
}

THTensor *THTensor_(newNarrow)(THTensor *tensor, int dimension_, int64_t firstIndex_, int64_t size_)
{
  THTensor *self = THTensor_(newWithTensor)(tensor);
  THTensor_(narrow)(self, NULL, dimension_, firstIndex_, size_);
  return self;
}

THTensor *THTensor_(newTranspose)(THTensor *tensor, int dimension1_, int dimension2_)
{
  THTensor *self = THTensor_(newWithTensor)(tensor);
  THTensor_(transpose)(self, NULL, dimension1_, dimension2_);
  return self;
}

/* Resize */
void THTensor_(resize)(THTensor *self, at::IntArrayRef size, at::IntArrayRef stride)
{
  return THTensor_resize(self, size, stride);
}

void THTensor_(resizeAs)(THTensor *self, THTensor *src)
{
  // already available in Aten as at::resize_as_()
  if(!THTensor_(isSameSizeAs)(self, src))
    THTensor_(resizeNd)(self, src->dim(), THTensor_getSizePtr(src), NULL);
}

void THTensor_(resize0d)(THTensor *tensor)
{
  THTensor_(resizeNd)(tensor, 0, {}, nullptr);
}

void THTensor_(resize1d)(THTensor *tensor, int64_t size0)
{
  int64_t size[1] = {size0};
  THTensor_(resizeNd)(tensor, 1, size, nullptr);
}

void THTensor_(resize2d)(THTensor *tensor, int64_t size0, int64_t size1)
{
  int64_t size[2] = {size0, size1};
  THTensor_(resizeNd)(tensor, 2, size, nullptr);
}

void THTensor_(resize3d)(THTensor *tensor, int64_t size0, int64_t size1, int64_t size2)
{
  int64_t size[3] = {size0, size1, size2};
  THTensor_(resizeNd)(tensor, 3, size, nullptr);
}

void THTensor_(resize4d)(THTensor *self, int64_t size0, int64_t size1, int64_t size2, int64_t size3)
{
  int64_t size[4] = {size0, size1, size2, size3};
  THTensor_(resizeNd)(self, 4, size, nullptr);
}

void THTensor_(resize5d)(THTensor *self, int64_t size0, int64_t size1, int64_t size2, int64_t size3, int64_t size4)
{
  int64_t size[5] = {size0, size1, size2, size3, size4};
  THTensor_(resizeNd)(self, 5, size, nullptr);
}

void THTensor_(set)(THTensor *self, THTensor *src)
{
  if(self != src)
    THTensor_(setStorage)(self,
                            THTensor_getStoragePtr(src),
                            src->storage_offset(),
                            src->sizes(),
                            src->strides());
}

void THTensor_(setStorage)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_, at::IntArrayRef size_, at::IntArrayRef stride_)
{
  THTensor_setStorage(self, storage_, storageOffset_, size_, stride_);
}

void THTensor_(narrow)(THTensor *self, THTensor *src, int dimension, int64_t firstIndex, int64_t size)
{
  if(!src)
    src = self;

  THArgCheck( (dimension >= 0) && (dimension < src->dim()), 2, "out of range");
  THArgCheck( firstIndex >= 0, 3, "out of range");
  THArgCheck( size >= 0, 4, "out of range");
  THArgCheck(firstIndex <= src->size(dimension) - size, 4, "out of range");

  THTensor_(set)(self, src);

  if (firstIndex > 0) {
    self->set_storage_offset(self->storage_offset() + firstIndex*self->stride(dimension));
  }

  self->set_size(dimension, size);
}

void THTensor_(select)(THTensor *self, THTensor *src, int dimension, int64_t sliceIndex)
{
  int d;

  if(!src)
    src = self;

  THArgCheck(src->dim() > 0, 1, "cannot select on a 0-dim tensor");
  THArgCheck((dimension >= 0) && (dimension < src->dim()), 2, "out of range");
  THArgCheck((sliceIndex >= 0) && (sliceIndex < src->size(dimension)), 3, "out of range");

  THTensor_(set)(self, src);
  THTensor_(narrow)(self, NULL, dimension, sliceIndex, 1);

  at::DimVector newSize(static_cast<size_t>(self->dim()-1));
  at::DimVector newStride(static_cast<size_t>(self->dim()-1));
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

void THTensor_(transpose)(THTensor *self, THTensor *src, int dimension1, int dimension2)
{
  int64_t z;

  if(!src)
    src = self;

  THArgCheck( (dimension1 >= 0) && (dimension1 < THTensor_nDimensionLegacyNoScalars(src)), 1, "out of range");
  THArgCheck( (dimension2 >= 0) && (dimension2 < THTensor_nDimensionLegacyNoScalars(src)), 2, "out of range");

  THTensor_(set)(self, src);

  if(dimension1 == dimension2)
    return;

  z = self->stride(dimension1);
  self->set_stride(dimension1, self->stride(dimension2));
  self->set_stride(dimension2, z);
  z = self->size(dimension1);
  self->set_size(dimension1, self->size(dimension2));
  self->set_size(dimension2, z);
}

void THTensor_(squeeze1d)(THTensor *self, THTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck((dimension >= 0) && (dimension < src->dim()), 2, "dimension out of range");

  THTensor_(set)(self, src);

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

void THTensor_(unsqueeze1d)(THTensor *self, THTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck((dimension >= 0) && (dimension <= src->dim()), 2, "dimension out of range");

  THTensor_(set)(self, src);

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

int THTensor_(isTransposed)(const THTensor *self)
{
  if (THTensor_(isContiguous)(self)) {
    return 0;
  }
  int64_t max_stride = 1;
  int64_t size_max_stride = 1;
  int64_t z = 1;
  int d;
  for (d = 0; d < self->dim(); ++d) {
    if (self->stride(d) == 0 && self->size(d) != 1)
      return 0;
    if (self->stride(d) > max_stride) {
      max_stride = self->stride(d);
      size_max_stride = self->size(d);
    }
    z *= self->size(d);
  }
  if (z == max_stride * size_max_stride) {
    return 1;
  }
  return 0;
}

int THTensor_(isContiguous)(const THTensor *self)
{
  return self->is_contiguous();
}

int THTensor_(isSameSizeAs)(const THTensor *self, const THTensor* src)
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

ptrdiff_t THTensor_(nElement)(const THTensor *self)
{
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

// NB: It is INVALID to call this on an UndefinedTensorImpl
void THTensor_(retain)(THTensor *self)
{
  c10::raw::intrusive_ptr::incref(self);
}

void THTensor_(free)(THTensor *self)
{
  THTensor_free(self);
}

void THTensor_(freeCopyTo)(THTensor *self, THTensor *dst)
{
  if(self != dst) {
    at::Tensor dst_wrap = THTensor_wrap(dst);
    at::Tensor self_wrap = THTensor_wrap(self);
    at::native::copy_(dst_wrap, self_wrap, false);
  }

  THTensor_(free)(self);
}

/*******************************************************************************/

void THTensor_(resizeNd)(THTensor *self, int nDimension, const int64_t *size, const int64_t *stride)
{
  return THTensor_resizeNd(self, nDimension, size, stride);
}

void THTensor_(set0d)(THTensor *tensor, scalar_t value)
{
  THArgCheck(THTensor_nDimension(tensor) == 0, 1, "tensor must have no dimensions");
  THStorage_(set)(THTensor_getStoragePtr(tensor), tensor->storage_offset(), value);
}

scalar_t THTensor_(get0d)(const THTensor *tensor)
{
  THArgCheck(THTensor_nDimension(tensor) == 0, 1, "tensor must have no dimensions");
  return THStorage_(get)(THTensor_getStoragePtr(tensor), tensor->storage_offset());
}

void THTensor_(set1d)(THTensor *tensor, int64_t x0, scalar_t value)
{
  THArgCheck(THTensor_nDimensionLegacyNoScalars(tensor) == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < THTensor_sizeLegacyNoScalars(tensor, 0)), 2, "out of range");
  THStorage_(set)(THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*THTensor_strideLegacyNoScalars(tensor, 0), value);
}

scalar_t THTensor_(get1d)(const THTensor *tensor, int64_t x0)
{
  THArgCheck(THTensor_nDimensionLegacyNoScalars(tensor) == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < THTensor_sizeLegacyNoScalars(tensor, 0)), 2, "out of range");
  return THStorage_(get)(THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*THTensor_strideLegacyNoScalars(tensor, 0));
}

void THTensor_(set2d)(THTensor *tensor, int64_t x0, int64_t x1, scalar_t value)
{
  THArgCheck(THTensor_nDimensionLegacyAll(tensor) == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)), 2, "out of range");
  THStorage_(set)(THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1), value);
}

scalar_t THTensor_(get2d)(const THTensor *tensor, int64_t x0, int64_t x1)
{
  THArgCheck(THTensor_nDimensionLegacyAll(tensor) == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)), 2, "out of range");
  return THStorage_(get)(THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1));
}

void THTensor_(set3d)(THTensor *tensor, int64_t x0, int64_t x1, int64_t x2, scalar_t value)
{
  THArgCheck(THTensor_nDimensionLegacyAll(tensor) == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)) && (x2 >= 0) && (x2 < tensor->size(2)), 2, "out of range");
  THStorage_(set)(THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1)+x2*tensor->stride(2), value);
}

scalar_t THTensor_(get3d)(const THTensor *tensor, int64_t x0, int64_t x1, int64_t x2)
{
  THArgCheck(THTensor_nDimensionLegacyAll(tensor) == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)) && (x2 >= 0) && (x2 < tensor->size(2)), 2, "out of range");
  return THStorage_(get)(THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1)+x2*tensor->stride(2));
}

void THTensor_(set4d)(THTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3, scalar_t value)
{
  THArgCheck(THTensor_nDimensionLegacyAll(tensor) == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)) && (x2 >= 0) && (x2 < tensor->size(2)) && (x3 >= 0) && (x3 < tensor->size(3)), 2, "out of range");
  THStorage_(set)(THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1)+x2*tensor->stride(2)+x3*tensor->stride(3), value);
}

scalar_t THTensor_(get4d)(const THTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3)
{
  THArgCheck(THTensor_nDimensionLegacyAll(tensor) == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)) && (x2 >= 0) && (x2 < tensor->size(2)) && (x3 >= 0) && (x3 < tensor->size(3)), 2, "out of range");
  return THStorage_(get)(THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1)+x2*tensor->stride(2)+x3*tensor->stride(3));
}

THDescBuff THTensor_(desc)(const THTensor *tensor) {
  const int L = TH_DESC_BUFF_LEN;
  THDescBuff buf;
  char *str = buf.str;
  int n = 0;
#define _stringify(x) #x
  n += snprintf(str, L-n, "torch." _stringify(x) "Tensor of size ");
#undef _stringify
  int i;
  for(i = 0; i < THTensor_nDimension(tensor); i++) {
    if(n >= L) break;
    n += snprintf(str+n, L-n, "%" PRId64, tensor->size(i));
    if(i < THTensor_nDimension(tensor)-1) {
      n += snprintf(str+n, L-n, "x");
    }
  }
  if(n >= L) {
    snprintf(str+L-4, 4, "...");
  }
  return buf;
}

THDescBuff THTensor_(sizeDesc)(const THTensor *tensor) {
  THDescBuff buf = _THSizeDesc(tensor->sizes().data(), tensor->sizes().size());
  return buf;
}

#endif
