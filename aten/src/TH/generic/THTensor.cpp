#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensor.cpp"
#else

#include <ATen/InferSize.h>
#include <new>

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
      dim+TH_INDEX_BASE, THTensor_(nDimensionLegacyNoScalars)(self));
  return self->size(dim);
}

int64_t THTensor_(stride)(const THTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->dim()), 2, "dimension %d out of range of %dD tensor",
      dim+TH_INDEX_BASE, THTensor_(nDimensionLegacyNoScalars)(self));
  return self->stride(dim);
}

real *THTensor_(data)(const THTensor *self) {
  return self->data<real>();
}

/**** creation methods ****/

/* Empty init */
THTensor *THTensor_(new)(void)
{
  return new THTensor(THStorage_(new)(), at::CPUTensorId(), false);
}

/* Pointer-copy init */
THTensor *THTensor_(newWithTensor)(THTensor *tensor)
{
  THTensor *self = new THTensor(THStorage_(new)(), at::CPUTensorId(), false);
  THTensor_(setStorageNd)(self,
                          THTensor_getStoragePtr(tensor),
                          tensor->storage_offset(),
                          tensor->dim(),
                          THTensor_getSizePtr(tensor),
                          THTensor_getStridePtr(tensor));
  return self;
}

/* Storage init */
THTensor *THTensor_(newWithStorage)(THStorage *storage, ptrdiff_t storageOffset, at::IntList sizes, at::IntList strides) {
  if (strides.data()) {
    AT_CHECK(sizes.size() == strides.size(), "number of sizes and strides must match");
  }
  THTensor *self = new THTensor(THStorage_(new)(), at::CPUTensorId(), false);
  THTensor_(setStorageNd)(self, storage, storageOffset, sizes.size(),
                          const_cast<int64_t*>(sizes.data()), const_cast<int64_t*>(strides.data()));

  return self;
}

THTensor *THTensor_(newWithStorage1d)(THStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0)
{
  return THTensor_(newWithStorage)(storage, storageOffset, {size0}, {stride0});
}

THTensor *THTensor_(newWithStorage2d)(THStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0,
                               int64_t size1, int64_t stride1)
{
  return THTensor_(newWithStorage)(storage, storageOffset, {size0, size1}, {stride0, stride1});
}

THTensor *THTensor_(newWithStorage3d)(THStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0,
                               int64_t size1, int64_t stride1,
                               int64_t size2, int64_t stride2)
{
  return THTensor_(newWithStorage)(storage, storageOffset, {size0, size1, size2}, {stride0, stride1, stride2});
}

THTensor *THTensor_(newWithStorage4d)(THStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0,
                               int64_t size1, int64_t stride1,
                               int64_t size2, int64_t stride2,
                               int64_t size3, int64_t stride3)
{
  return THTensor_(newWithStorage)(storage, storageOffset,
                                          {size0, size1, size2, size3},
                                          {stride0, stride1, stride2, stride3});
}

THTensor *THTensor_(newWithSize)(at::IntList size, at::IntList stride)
{
  return THTensor_(newWithStorage)(NULL, 0, size, stride);
}

THTensor *THTensor_(newWithSize1d)(int64_t size0)
{
  return THTensor_(newWithSize)({size0}, {});
}

THTensor *THTensor_(newWithSize2d)(int64_t size0, int64_t size1)
{
  return THTensor_(newWithSize)({size0, size1}, {});
}

THTensor *THTensor_(newWithSize3d)(int64_t size0, int64_t size1, int64_t size2)
{
  return THTensor_(newWithSize)({size0, size1, size2}, {});
}

THTensor *THTensor_(newWithSize4d)(int64_t size0, int64_t size1, int64_t size2, int64_t size3)
{
  return THTensor_(newWithSize)({size0, size1, size2, size3}, {});
}

THTensor *THTensor_(newClone)(THTensor *self)
{
  THTensor *tensor = THTensor_(new)();
  THTensor_(resizeAs)(tensor, self);
  THTensor_(copy)(tensor, self);
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

THTensor *THTensor_(newUnfold)(THTensor *tensor, int dimension_, int64_t size_, int64_t step_)
{
  THTensor *self = THTensor_(newWithTensor)(tensor);
  THTensor_(unfold)(self, NULL, dimension_, size_, step_);
  return self;
}

THTensor *THTensor_(newView)(THTensor *tensor, at::IntList size)
{
  ptrdiff_t numel = THTensor_(nElement)(tensor);
  THTensor *self = THTensor_(new)();
  auto inferred_size = at::infer_size(size, numel);
  auto stride = THTensor_compute_stride(tensor->sizes(),
                                        tensor->strides(),
                                        inferred_size);
  THArgCheck(stride.has_value(), 2, "view size is "
    "not compatible with input tensor's size and stride (at least one dimension spans "
    "across two contiguous subspaces). Call .contiguous() before .view().");
  auto stride_value = *stride;
  THTensor_setStorage(self, THTensor_getStoragePtr(tensor), tensor->storage_offset(), inferred_size, stride_value);
  return self;
}

/* Resize */
void THTensor_(resize)(THTensor *self, at::IntList size, at::IntList stride)
{
  return THTensor_resize(self, size, stride);
}

void THTensor_(resizeAs)(THTensor *self, THTensor *src)
{
  if(!THTensor_(isSameSizeAs)(self, src))
    THTensor_(resizeNd)(self, src->dim(), THTensor_getSizePtr(src), NULL);
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
    THTensor_(setStorageNd)(self,
                            THTensor_getStoragePtr(src),
                            src->storage_offset(),
                            src->dim(),
                            THTensor_getSizePtr(src),
                            THTensor_getStridePtr(src));
}

void THTensor_(setStorage)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_, at::IntList size_, at::IntList stride_)
{
  THTensor_setStorage(self, storage_, storageOffset_, size_, stride_);
}

void THTensor_(setStorage1d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_)
{
  THTensor_(setStorage)(self, storage_, storageOffset_,
                       {size0_}, {stride0_});
}

void THTensor_(setStorage2d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_,
                             int64_t size1_, int64_t stride1_)
{
  THTensor_(setStorage)(self, storage_, storageOffset_,
                       {size0_, size1_},
                       {stride0_, stride1_});
}

void THTensor_(setStorage3d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_,
                             int64_t size1_, int64_t stride1_,
                             int64_t size2_, int64_t stride2_)
{
  THTensor_(setStorage)(self, storage_, storageOffset_,
                        {size0_, size1_, size2_},
                        {stride0_, stride1_, stride2_});
}

void THTensor_(setStorage4d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_,
                             int64_t size1_, int64_t stride1_,
                             int64_t size2_, int64_t stride2_,
                             int64_t size3_, int64_t stride3_)
{

  int64_t size[4] = {size0_, size1_, size2_, size3_};
  int64_t stride[4] = {stride0_, stride1_, stride2_, stride3_};

  THTensor_(setStorage)(self, storage_, storageOffset_, size, stride);
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
    THTensor_setStorageOffset(self, self->storage_offset() + firstIndex*self->stride(dimension));
  }

  THTensor_setSizeAtDim(self, dimension, size);
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
  for(d = dimension; d < self->dim()-1; d++)
  {
    THTensor_setSizeAtDim(self, d, self->size(d+1));
    THTensor_setStrideAtDim(self, d, self->stride(d+1));
  }
  THTensor_resizeDim(self, self->dim() - 1);
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
  THTensor_setStrideAtDim(self, dimension1, self->stride(dimension2));
  THTensor_setStrideAtDim(self, dimension2, z);
  z = self->size(dimension1);
  THTensor_setSizeAtDim(self, dimension1, self->size(dimension2));
  THTensor_setSizeAtDim(self, dimension2, z);
}

void THTensor_(unfold)(THTensor *self, THTensor *src, int dimension, int64_t size, int64_t step)
{
  int d;

  if(!src)
    src = self;

  THArgCheck((dimension >= 0) && (dimension < THTensor_nDimensionLegacyNoScalars(src)), 2, "out of range");
  THArgCheck(size <= THTensor_sizeLegacyNoScalars(src, dimension), 3, "out of range");
  THArgCheck(step > 0, 4, "invalid step");

  THTensor_(set)(self, src);

  std::vector<int64_t> newSize(/* size */ self->dim()+1);
  std::vector<int64_t> newStride(/* size */ self->dim()+1);

  newSize[self->dim()] = size;
  newStride[self->dim()] = THTensor_strideLegacyNoScalars(self, dimension);
  for(d = 0; d < self->dim(); d++)
  {
    auto self_size = THTensor_sizeLegacyNoScalars(self, d);
    auto self_stride = THTensor_strideLegacyNoScalars(self, d);
    if(d == dimension)
    {
      newSize[d] = (self_size - size) / step + 1;
      newStride[d] = step*self_stride;
    }
    else
    {
      newSize[d] = self_size;
      newStride[d] = self_stride;
    }
  }
  THTensor_setSizesAndStrides(self, std::move(newSize), std::move(newStride));
}

/* we have to handle the case where the result is a number */
void THTensor_(squeeze)(THTensor *self, THTensor *src)
{
  int ndim = 0;
  int d;

  if(!src)
    src = self;

  THTensor_(set)(self, src);

  for(d = 0; d < src->dim(); d++)
  {
    if(src->size(d) != 1)
    {
      if(d != ndim)
      {
        THTensor_setSizeAtDim(self, ndim, src->size(d));
        THTensor_setStrideAtDim(self, ndim, src->stride(d));
      }
      ndim++;
    }
  }

  THTensor_resizeDim(self, ndim);
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
    for(d = dimension; d < self->dim()-1; d++)
    {
      THTensor_setSizeAtDim(self, d, self->size(d+1));
      THTensor_setStrideAtDim(self, d, self->stride(d+1));
    }
    THTensor_resizeDim(self, self->dim() - 1);
  }
}

void THTensor_(unsqueeze1d)(THTensor *self, THTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck((dimension >= 0) && (dimension <= src->dim()), 2, "dimension out of range");

  THTensor_(set)(self, src);

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
  if (self->is_empty()) return 1;
  int64_t z = 1;
  int d;
  for(d = self->dim()-1; d >= 0; d--)
  {
    if(self->size(d) != 1)
    {
      if(self->stride(d) == z)
        z *= self->size(d);
      else
        return 0;
    }
  }
  return 1;
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

int THTensor_(isSetTo)(const THTensor *self, const THTensor* src)
{
  if (!THTensor_getStoragePtr(self))
    return 0;
  if (THTensor_getStoragePtr(self) == THTensor_getStoragePtr(src) &&
      self->storage_offset() == src->storage_offset() &&
      THTensor_nDimensionLegacyAll(self) == THTensor_nDimensionLegacyAll(src))
  {
    int d;
    for (d = 0; d < THTensor_nDimensionLegacyAll(self); ++d)
    {
      if (self->size(d) != src->size(d) || self->stride(d) != src->stride(d))
        return 0;
    }
    return 1;
  }
  return 0;
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

void THTensor_(retain)(THTensor *self)
{
  self->retain();
}

void THTensor_(free)(THTensor *self)
{
  THTensor_free(self);
}

void THTensor_(freeCopyTo)(THTensor *self, THTensor *dst)
{
  if(self != dst)
    THTensor_(copy)(dst, self);

  THTensor_(free)(self);
}

/*******************************************************************************/

void THTensor_(setStorageNd)(THTensor *self, THStorage *storage, ptrdiff_t storageOffset, int nDimension, int64_t *size, int64_t *stride)
{
  return THTensor_setStorageNd(self, storage, storageOffset, nDimension, size, stride);
}

void THTensor_(resizeNd)(THTensor *self, int nDimension, int64_t *size, int64_t *stride)
{
  return THTensor_resizeNd(self, nDimension, size, stride);
}

void THTensor_(set1d)(THTensor *tensor, int64_t x0, real value)
{
  THArgCheck(THTensor_nDimensionLegacyNoScalars(tensor) == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < THTensor_sizeLegacyNoScalars(tensor, 0)), 2, "out of range");
  THStorage_(set)(THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*THTensor_strideLegacyNoScalars(tensor, 0), value);
}

real THTensor_(get1d)(const THTensor *tensor, int64_t x0)
{
  THArgCheck(THTensor_nDimensionLegacyNoScalars(tensor) == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < THTensor_sizeLegacyNoScalars(tensor, 0)), 2, "out of range");
  return THStorage_(get)(THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*THTensor_strideLegacyNoScalars(tensor, 0));
}

void THTensor_(set2d)(THTensor *tensor, int64_t x0, int64_t x1, real value)
{
  THArgCheck(THTensor_nDimensionLegacyAll(tensor) == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)), 2, "out of range");
  THStorage_(set)(THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1), value);
}

real THTensor_(get2d)(const THTensor *tensor, int64_t x0, int64_t x1)
{
  THArgCheck(THTensor_nDimensionLegacyAll(tensor) == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)), 2, "out of range");
  return THStorage_(get)(THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1));
}

void THTensor_(set3d)(THTensor *tensor, int64_t x0, int64_t x1, int64_t x2, real value)
{
  THArgCheck(THTensor_nDimensionLegacyAll(tensor) == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)) && (x2 >= 0) && (x2 < tensor->size(2)), 2, "out of range");
  THStorage_(set)(THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1)+x2*tensor->stride(2), value);
}

real THTensor_(get3d)(const THTensor *tensor, int64_t x0, int64_t x1, int64_t x2)
{
  THArgCheck(THTensor_nDimensionLegacyAll(tensor) == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)) && (x2 >= 0) && (x2 < tensor->size(2)), 2, "out of range");
  return THStorage_(get)(THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1)+x2*tensor->stride(2));
}

void THTensor_(set4d)(THTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3, real value)
{
  THArgCheck(THTensor_nDimensionLegacyAll(tensor) == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)) && (x2 >= 0) && (x2 < tensor->size(2)) && (x3 >= 0) && (x3 < tensor->size(3)), 2, "out of range");
  THStorage_(set)(THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1)+x2*tensor->stride(2)+x3*tensor->stride(3), value);
}

real THTensor_(get4d)(const THTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3)
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
