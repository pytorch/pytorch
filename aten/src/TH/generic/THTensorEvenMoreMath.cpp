#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensorEvenMoreMath.cpp"
#else

#include <TH/generic/THTensorApply.hpp>

// Finds non-zero elements of a tensor and returns their subscripts
void THTensor_(nonzero)(THLongTensor *subscript, THTensor *tensor)
{
  ptrdiff_t numel = 0;
  int64_t *subscript_data;
  int64_t i = 0;
#ifdef TH_REAL_IS_HALF
#define IS_NONZERO(val) (c10::Half(0)!=val)
#elif defined(TH_REAL_IS_BFLOAT16)
#define IS_NONZERO(val) (c10::BFloat16(0)!=val)
#else
#define IS_NONZERO(val) ((val)!=0)
#endif

  /* First Pass to determine size of subscripts */
  TH_TENSOR_APPLY(scalar_t, tensor,
                  if IS_NONZERO(*tensor_data) {
                    ++numel;
                  });
#ifdef DEBUG
  THAssert(numel <= LONG_MAX);
#endif
  THLongTensor_resize2d(subscript, numel, tensor->dim());
  if (numel <= 0) {
    return;
  }
  int64_t dimensions = tensor->dim();
  // +1 faster than additional condition check inside loop
  int64_t *sizes = new int64_t[dimensions+1];
  int64_t *idx = new int64_t[dimensions+1];
  int64_t *ii;
  int64_t *ss;
  std::fill(idx, idx+dimensions+1, 0);
  for (i = 0; i < dimensions; ++i) {
    sizes[dimensions - i - 1] = THTensor_(size)(tensor, i); // reverse order important
  }
  sizes[dimensions] = 0;
  /* Second pass populates subscripts */
  subscript_data = THLongTensor_data(subscript);
  auto subscript_strides = THTensor_stridesLegacyNoScalars(subscript);
  subscript_strides[0] -= subscript_strides[1] * tensor->dim();
  TH_TENSOR_APPLY(scalar_t, tensor,
                  if IS_NONZERO(*tensor_data) {
                    ii = idx + dimensions;
                    for (int64_t dim = dimensions - 1; dim >= 0; dim--) {
                      --ii;
                      *subscript_data = *ii;
                      subscript_data += subscript_strides[1];
                    }
                    subscript_data += subscript_strides[0];
                  }
                  ii = idx;
                  ss = sizes;
                  ++(*ii);
                  while (*ii == *ss) {
                    *ii = 0;
                    ++ii;
                    ++ss;
                    ++(*ii);
                  }
                );
  delete [] sizes;
  delete [] idx;

#undef IS_NONZERO
}

#if !defined(TH_REAL_IS_HALF) /* non half part */

void THTensor_(maskedSelect)(THTensor *tensor, THTensor *src, THByteTensor *mask)
{
  ptrdiff_t numel = THByteTensor_sumall(mask);
  scalar_t *tensor_data;

#ifdef DEBUG
  THAssert(numel <= LONG_MAX);
#endif
  THTensor_(resize1d)(tensor,numel);
  tensor_data = tensor->data<scalar_t>();
  TH_TENSOR_APPLY2(scalar_t, src, unsigned char, mask,
                   if (*mask_data > 1)
                   {
                     THFree(mask_counter);
                     THFree(src_counter);
                     THError("Mask tensor can take 0 and 1 values only");
                   }
                   else if (*mask_data == 1)
                   {
                     *tensor_data = *src_data;
                     tensor_data++;
                   });
}

void THTensor_(maskedSelectBool)(THTensor *tensor, THTensor *src, THBoolTensor *mask)
{
  ptrdiff_t numel = THBoolTensor_sumall(mask);
  scalar_t *tensor_data;

#ifdef DEBUG
  THAssert(numel <= LONG_MAX);
#endif
  THTensor_(resize1d)(tensor,numel);
  tensor_data = tensor->data<scalar_t>();
  TH_TENSOR_APPLY2(scalar_t, src, bool, mask,
                   if (*mask_data)
                   {
                     *tensor_data = *src_data;
                     tensor_data++;
                   });
}

void THTensor_(scatterFill)(THTensor *tensor, int dim, THLongTensor *index, scalar_t val)
{
  int64_t elems_per_row, i, idx;
  int index_ndim_legacy_all = THLongTensor_nDimensionLegacyAll(index);

  THArgCheck(dim < THTensor_(nDimensionLegacyAll)(tensor), 2, "Index dimension is out of bounds");
  THArgCheck(index_ndim_legacy_all == 0 || index_ndim_legacy_all == THLongTensor_nDimensionLegacyAll(tensor), 3,
             "Index tensor must either be empty or have same dimensions as output tensor");

  // no-op if index is empty
  if (index_ndim_legacy_all == 0)
      return;

  elems_per_row = THTensor_sizeLegacyNoScalars(index, dim);

  TH_TENSOR_DIM_APPLY2(scalar_t, tensor, int64_t, index, dim,
                       for (i = 0; i < elems_per_row; ++i)
                       {
                         idx = *(index_data + i*index_stride);
                         if (idx < 0 || idx >= tensor_size)
                         {
                           THFree(TH_TENSOR_DIM_APPLY_counter);
                           THError("Invalid index in scatter");
                         }
                         tensor_data[idx * tensor_stride] = val;
                       })
}

void THTensor_(maskedFill)(THTensor *tensor, THByteTensor *mask, scalar_t value)
{
  int64_t tensor_size = THTensor_(nElement)(tensor);
  int tensor_contig = THTensor_(isContiguous)(tensor);
  int mask_contig = THTensor_(isContiguous)(mask);
  if (tensor_contig && mask_contig) {
    TH_TENSOR_APPLY2_PARALLEL(tensor_size, tensor_contig, mask_contig,
      scalar_t, tensor, unsigned char, mask,
      if (*mask_data > 1) {
        THError("Mask tensor can take 0 and 1 values only");
      } else if (*mask_data == 1) {
        *tensor_data = value;
      },
      TH_OMP_OVERHEAD_THRESHOLD);
  } else {
    TH_TENSOR_APPLY2(scalar_t, tensor, unsigned char, mask,
      if (*mask_data > 1) {
        THFree(mask_counter);
        THFree(tensor_counter);
        THError("Mask tensor can take 0 and 1 values only");
      } else if (*mask_data == 1) {
        *tensor_data = value;
      });
    }
}

void THTensor_(maskedFillBool)(THTensor *tensor, THBoolTensor *mask, scalar_t value)
{
  int64_t tensor_size = THTensor_(nElement)(tensor);
  int tensor_contig = THTensor_(isContiguous)(tensor);
  int mask_contig = THTensor_(isContiguous)(mask);
  if (tensor_contig && mask_contig) {
    TH_TENSOR_APPLY2_PARALLEL(tensor_size, tensor_contig, mask_contig,
      scalar_t, tensor, bool, mask,
      if (*mask_data) {
        *tensor_data = value;
      },
      TH_OMP_OVERHEAD_THRESHOLD);
  } else {
    TH_TENSOR_APPLY2(scalar_t, tensor, bool, mask,
      if (*mask_data) {
        *tensor_data = value;
      });
    }
}

void THTensor_(maskedCopy)(THTensor *tensor, THByteTensor *mask, THTensor* src )
{
  THTensor *srct = THTensor_(newContiguous)(src);
  scalar_t *src_data = srct->data<scalar_t>();
  ptrdiff_t cntr = 0;
  ptrdiff_t nelem = THTensor_(nElement)(srct);
  if (THTensor_(nElement)(tensor) != THByteTensor_nElement(mask))
  {
    c10::raw::intrusive_ptr::decref(srct);
    THError("Number of elements of destination tensor != Number of elements in mask");
  }
  TH_TENSOR_APPLY2(scalar_t, tensor, unsigned char, mask,
                   if (*mask_data > 1)
                   {
                     c10::raw::intrusive_ptr::decref(srct);
                     THFree(mask_counter);
                     THFree(tensor_counter);
                     THError("Mask tensor can take 0 and 1 values only");
                   }
                   else if (*mask_data == 1)
                   {
                     if (cntr == nelem)
                     {
                       c10::raw::intrusive_ptr::decref(srct);
                       THFree(mask_counter);
                       THFree(tensor_counter);
                       THError("Number of elements of src < number of ones in mask");
                     }
                     *tensor_data = *src_data;
                     src_data++;
                     cntr++;
                   });
  c10::raw::intrusive_ptr::decref(srct);
}

void THTensor_(maskedCopyBool)(THTensor *tensor, THBoolTensor *mask, THTensor* src )
{
  THTensor *srct = THTensor_(newContiguous)(src);
  scalar_t *src_data = srct->data<scalar_t>();
  ptrdiff_t cntr = 0;
  ptrdiff_t nelem = THTensor_(nElement)(srct);
  if (THTensor_(nElement)(tensor) != THBoolTensor_nElement(mask))
  {
    c10::raw::intrusive_ptr::decref(srct);
    THError("Number of elements of destination tensor != Number of elements in mask");
  }
  TH_TENSOR_APPLY2(scalar_t, tensor, bool, mask,
                   if (*mask_data)
                   {
                     if (cntr == nelem)
                     {
                       c10::raw::intrusive_ptr::decref(srct);
                       THFree(mask_counter);
                       THFree(tensor_counter);
                       THError("Number of elements of src < number of ones in mask");
                     }
                     *tensor_data = *src_data;
                     src_data++;
                     cntr++;
                   });
  c10::raw::intrusive_ptr::decref(srct);
}

#if !defined(TH_REAL_IS_BOOL)
void THTensor_(mul)(THTensor *r_, THTensor *t, scalar_t value)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  if (r_Contig && tContig) {
    TH_TENSOR_APPLY2_CONTIG(scalar_t, r_, scalar_t, t, THVector_(muls)(r__data, t_data, value, r__len););
  } else {
    TH_TENSOR_APPLY2_PARALLEL(r_Size, r_Contig, tContig, scalar_t, r_, scalar_t, t, *r__data = *t_data * value;, ORDIN_TH_OMP_OVERHEAD_THRESHOLD)
  }
}

void THTensor_(div)(THTensor *r_, THTensor *t, scalar_t value)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  if (r_Contig && tContig) {
    TH_TENSOR_APPLY2_CONTIG(scalar_t, r_, scalar_t, t, THVector_(divs)(r__data, t_data, value, r__len););
  } else {
    TH_TENSOR_APPLY2_PARALLEL(r_Size, r_Contig, tContig, scalar_t, r_, scalar_t, t, *r__data = *t_data / value;, ORDIN_TH_OMP_OVERHEAD_THRESHOLD)
  }
}
#endif

#if !defined(TH_REAL_IS_BFLOAT16) /* non bfloat16 part*/

accreal THTensor_(sumall)(THTensor *tensor)
{
  accreal sum = 0;
  TH_TENSOR_APPLY_REDUCTION_SUM_PARALLEL(
    scalar_t, tensor, *tensor_data, sum, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
  return sum;
}

void THTensor_(bitand)(THTensor *r_, THTensor *t, scalar_t value)
{
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_HALF) || defined(TH_REAL_IS_BFLOAT16)
  (void)r_;
  (void)t;
  (void)value;
  return THError("bitand is only supported for integer type tensors");
#else
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  if (r_Contig && tContig) {
    scalar_t *tp = t->data<scalar_t>();
    scalar_t *rp = r_->data<scalar_t>();
    at::parallel_for(0, r_Size, TH_OMP_OVERHEAD_THRESHOLD * 100,
        [&](int64_t start, int64_t end) {
      for (auto i = start; i < end; i++) {
        rp[i] = tp[i] & value;
      }
    });
  } else {
    TH_TENSOR_APPLY2_PARALLEL(r_Size, r_Contig, tContig, scalar_t, r_, scalar_t, t, *r__data = *t_data & value;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
  }
#endif
}

scalar_t THTensor_(minall)(THTensor *tensor)
{
  scalar_t theMin;
  scalar_t value;

  THArgCheck(
      THTensor_(nElement)(tensor) > 0,
      1,
      "cannot perform reduction function min "
      "on tensor with no elements because the "
      "operation does not have an identity"
  );

  theMin = tensor->data<scalar_t>()[0];
  TH_TENSOR_APPLY(scalar_t, tensor,
                  value = *tensor_data;
                  /* This is not the same as value<theMin in the case of NaNs */
                  if(!(value >= theMin))
                  {
                    theMin = value;
                    th_isnan_break(value)
                  });
  return theMin;
}

scalar_t THTensor_(maxall)(THTensor *tensor)
{
  scalar_t theMax;
  scalar_t value;

  THArgCheck(
      THTensor_(nElement)(tensor) > 0,
      1,
      "cannot perform reduction function max "
      "on tensor with no elements because the "
      "operation does not have an identity"
  );

  theMax = tensor->data<scalar_t>()[0];
  TH_TENSOR_APPLY(scalar_t, tensor,
                  value = *tensor_data;
                  /* This is not the same as value>theMax in the case of NaNs */
                  if(!(value <= theMax))
                  {
                    theMax = value;
                    th_isnan_break(value)
                  });
  return theMax;
}

void THTensor_(indexSelect)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index)
{
  ptrdiff_t i, numel;
  THTensor *tSlice, *sSlice;
  int64_t *index_data;
  scalar_t *tensor_data, *src_data;

  THArgCheck(THTensor_nDimensionLegacyNoScalars(index) == 1, 3, "Index is supposed to be 1-dimensional");
  THArgCheck(dim < THTensor_nDimensionLegacyNoScalars(src), 4, "Indexing dim %d is out of bounds of tensor", dim);

  numel = THLongTensor_nElement(index);

  std::vector<int64_t> newSize = THTensor_sizesLegacyNoScalars(src);
#ifdef DEBUG
  THAssert(numel <= LONG_MAX);
#endif
  newSize[dim] = numel;
  THTensor_(resize)(tensor,newSize,{});

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);

  if (dim == 0 && THTensor_(isContiguous)(src) && THTensor_(isContiguous)(tensor))
  {
    tensor_data = tensor->data<scalar_t>();
    src_data = src->data<scalar_t>();
    auto src_size0 = THTensor_sizeLegacyNoScalars(src, 0);
    ptrdiff_t rowsize = src_size0 == 0 ? 1 : THTensor_(nElement)(src) / src_size0;

    // check that the indices are within range
    int64_t max = src_size0 - 1;
    for (i=0; i<numel; i++) {
      if (index_data[i] < 0 || index_data[i] > max) {
        THLongTensor_free(index);
        THError("index out of range: Tried to access index %d out of table with %d rows.", index_data[i], max);
      }
    }

    // When src is empty, tensor_data maybe nullptr, and the memcpy will trigger
    // ubsan. So we skip copying at all when every slice to copy is empty.
    if (rowsize > 0) {
      if (src->dim() <= 1) {
        at::parallel_for(0, numel, TH_OMP_OVERHEAD_THRESHOLD,
            [&](int64_t start, int64_t end) {
          for (auto i = start; i < end; i++) {
            tensor_data[i] = src_data[index_data[i]];
          }
        });
      } else {
        at::parallel_for(0, numel, TH_OMP_OVERHEAD_THRESHOLD / rowsize,
            [&](int64_t start, int64_t end) {
          for (auto i = start; i < end; i++) {
            memcpy(
              tensor_data + i * rowsize,
              src_data + index_data[i] * rowsize,
              rowsize * sizeof(scalar_t));
          }
        });
      }
    }
  }
  else if (src->dim() <= 1)
  {
    for (i=0; i<numel; i++)
      THTensor_(set1d)(tensor,i,THTensor_(get1d)(src,index_data[i]));
  }
  else
  {
    for (i=0; i<numel; i++)
    {
      tSlice = THTensor_(new)();
      sSlice = THTensor_(new)();
      THTensor_(select)(tSlice, tensor, dim, i);
      THTensor_(select)(sSlice, src, dim, index_data[i]);
      at::Tensor tSlice_wrap = THTensor_wrap(tSlice);
      at::Tensor sSlice_wrap = THTensor_wrap(sSlice);
      at::native::copy_(tSlice_wrap, sSlice_wrap);
      c10::raw::intrusive_ptr::decref(tSlice);
      c10::raw::intrusive_ptr::decref(sSlice);
    }
  }

  THLongTensor_free(index);
}

void THTensor_(indexCopy)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src)
{
  ptrdiff_t i, numel;
  THTensor *tSlice, *sSlice;
  int64_t *index_data;

  // Error checking for this function has moved to ATen!!

  numel = THLongTensor_nElement(index);

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);

  if (tensor->dim() > 1 )
  {
    tSlice = THTensor_(new)();
    sSlice = THTensor_(new)();

    for (i=0; i<numel; i++)
    {
      THTensor_(select)(tSlice, tensor, dim, index_data[i]);
      THTensor_(select)(sSlice, src, dim, i);
      at::Tensor tSlice_wrap = THTensor_wrap(tSlice);
      at::Tensor sSlice_wrap = THTensor_wrap(sSlice);
      at::native::copy_(tSlice_wrap, sSlice_wrap);
    }

    c10::raw::intrusive_ptr::decref(tSlice);
    c10::raw::intrusive_ptr::decref(sSlice);
  }
  else
  {
    for (i=0; i<numel; i++)
    {
      THTensor_(set1d)(tensor, index_data[i], THTensor_(get1d)(src,i));
    }
  }
  THLongTensor_free(index);
}

static ptrdiff_t THTensor_(dataOffset)(THTensor* tensor, ptrdiff_t linearIndex) {
  auto size = THTensor_sizesLegacyNoScalars(tensor);
  auto stride = THTensor_stridesLegacyNoScalars(tensor);
  int nDim = THTensor_nDimensionLegacyAll(tensor);
  ptrdiff_t dataOffset = 0;
  for (int i = nDim - 1; i >= 0; i--) {
    dataOffset += (linearIndex % size[i]) * stride[i];
    linearIndex /= size[i];
  }
  return dataOffset;
}

static inline void THTensor_(checkLinearIndex)(int64_t linearIndex, int64_t numel) {
  THArgCheck(linearIndex < numel && linearIndex >= -numel, 2, "out of range: %d out of %d", (int)linearIndex, (int)numel);
}

static inline int64_t THTensor_(wrapLinearIndex)(int64_t linearIndex, int64_t numel) {
  return linearIndex < 0 ? linearIndex + numel : linearIndex;
}

void THTensor_(take)(THTensor *r_, THTensor *src, THLongTensor *index)
{
  THTensor_(resizeNd)(r_, index->dim(), THTensor_getSizePtr(index), NULL);
  THTensor* dst = THTensor_(newContiguous)(r_);

  index = THLongTensor_newContiguous(index);
  int64_t* index_data = THLongTensor_data(index);
  ptrdiff_t srcElements = THTensor_(nElement)(src);
  scalar_t* src_data = src->data<scalar_t>();
  scalar_t* dst_data = dst->data<scalar_t>();
  ptrdiff_t nIndices = THLongTensor_nElement(index);
  int isContiguous = THTensor_(isContiguous)(src);

  // Exceptions must not be thrown across parallel sections, so we
  // record the position of the invalid index and throw the exception after the
  // loop.
  std::atomic<int64_t> invalidIdxPos(-1);

  at::parallel_for(0, nIndices, TH_OMP_OVERHEAD_THRESHOLD,
      [&](int64_t start, int64_t end) {
    for (auto i = start; i < end; i++) {
      int64_t idx = index_data[i];
      if (idx < srcElements && idx >= -srcElements) {
        idx = THTensor_(wrapLinearIndex)(idx, srcElements);
        if (isContiguous) {
          dst_data[i] = src_data[idx];
        } else {
          dst_data[i] = src_data[THTensor_(dataOffset)(src, idx)];
        }
      } else {
        int64_t tmp = -1;
        invalidIdxPos.compare_exchange_strong(tmp, i);
      }
    }
  });

  if (invalidIdxPos >= 0) {
    THTensor_(checkLinearIndex)(index_data[invalidIdxPos], srcElements);
  }

  THLongTensor_free(index);
  THTensor_(freeCopyTo)(dst, r_);
}

void THTensor_(put)(THTensor *tensor, THLongTensor *index, THTensor *src, int accumulate)
{
  THArgCheck(THLongTensor_nElement(index) == THTensor_(nElement)(src), 3,
    "src should have the same number of elements as index");

  index = THLongTensor_newContiguous(index);
  src = THTensor_(newContiguous)(src);
  scalar_t* data = tensor->data<scalar_t>();
  ptrdiff_t numel = THTensor_(nElement)(tensor);
  int is_contiguous = THTensor_(isContiguous)(tensor);

  TH_TENSOR_APPLY2(int64_t, index, scalar_t, src,
    THTensor_(checkLinearIndex)(*index_data, numel);
    int64_t linearIndex = THTensor_(wrapLinearIndex)(*index_data, numel);
    int64_t dataOffset = is_contiguous ? linearIndex : THTensor_(dataOffset)(tensor, linearIndex);
    if (accumulate) {
      data[dataOffset] += *src_data;
    } else {
      data[dataOffset] = *src_data;
    }
  );

  c10::raw::intrusive_ptr::decref(src);
  THLongTensor_free(index);
}

void THTensor_(indexFill)(THTensor *tensor, int dim, THLongTensor *index, scalar_t val)
{
  ptrdiff_t i, numel;
  THTensor *tSlice;
  int64_t *index_data;

  numel = THLongTensor_nElement(index);
  THArgCheck(THTensor_nDimensionLegacyNoScalars(index) == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < THTensor_nDimensionLegacyNoScalars(tensor), 4,"Indexing dim %d is out of bounds of tensor", dim);

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);

  for (i=0; i<numel; i++)
  {
    if (tensor->dim() > 1)
    {
      tSlice = THTensor_(new)();
      THTensor_(select)(tSlice, tensor,dim,index_data[i]);
      THTensor_(fill)(tSlice, val);
      c10::raw::intrusive_ptr::decref(tSlice);
    }
    else
    {
      THTensor_(set1d)(tensor, index_data[i], val);
    }
  }
  THLongTensor_free(index);
}

void THTensor_(gather)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index)
{
  int64_t elems_per_row, i, idx;

  THArgCheck(THLongTensor_nDimensionLegacyNoScalars(index) == THTensor_(nDimensionLegacyNoScalars)(src), 4,
             "Index tensor must have same dimensions as input tensor");
  THArgCheck(dim >= 0 && dim < THTensor_(nDimensionLegacyNoScalars)(tensor), 3,
             "Index dimension is out of bounds");
  THArgCheck(THTensor_(nDimensionLegacyNoScalars)(src) == THTensor_(nDimensionLegacyNoScalars)(tensor), 2,
             "Input tensor must have same dimensions as output tensor");

  elems_per_row = THTensor_sizeLegacyNoScalars(index, dim);

  TH_TENSOR_DIM_APPLY3(scalar_t, tensor, scalar_t, src, int64_t, index, dim,
                       TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
                       for (i = 0; i < elems_per_row; ++i)
                       {
                         idx = *(index_data + i*index_stride);
                         if (idx < 0 || idx >= src_size)
                         {
                           THFree(TH_TENSOR_DIM_APPLY_counter);
                           THError("Invalid index in gather");
                         }
                         *(tensor_data + i*tensor_stride) = src_data[idx * src_stride];
                       })
}

void THTensor_(scatter)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src)
{
  int64_t elems_per_row, i, idx;
  int index_ndim_legacy_all = THTensor_nDimensionLegacyAll(index);

  THArgCheck(dim < THTensor_(nDimensionLegacyNoScalars)(tensor), 2, "Index dimension is out of bounds");
  THArgCheck(index_ndim_legacy_all == 0
             || THLongTensor_nDimensionLegacyNoScalars(index) == THTensor_(nDimensionLegacyNoScalars)(tensor), 3,
             "Index tensor must be either empty or have same dimensions as output tensor");
  THArgCheck(THTensor_(nDimensionLegacyNoScalars)(src) == THTensor_(nDimensionLegacyNoScalars)(tensor), 4,
             "Input tensor must have same dimensions as output tensor");

  // no-op if index is empty
  if (index_ndim_legacy_all == 0)
      return;

  elems_per_row = THTensor_sizeLegacyNoScalars(index, dim);

  TH_TENSOR_DIM_APPLY3(scalar_t, tensor, scalar_t, src, int64_t, index, dim,
                       TH_TENSOR_DIM_APPLY3_SIZE_SCATTER,
                       for (i = 0; i < elems_per_row; ++i)
                       {
                         idx = *(index_data + i*index_stride);
                         if (idx < 0 || idx >= tensor_size)
                         {
                           THFree(TH_TENSOR_DIM_APPLY_counter);
                           THError("Invalid index in scatter");
                         }
                         tensor_data[idx * tensor_stride] = *(src_data + i*src_stride);
                       })
}

void THTensor_(scatterAdd)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src)
{
  int64_t elems_per_row, i, idx;
  int index_ndim_legacy_all = THTensor_nDimensionLegacyAll(index);

  THArgCheck(dim < THTensor_(nDimensionLegacyNoScalars)(tensor), 2, "Index dimension is out of bounds");
  THArgCheck(index_ndim_legacy_all == 0
             || THLongTensor_nDimensionLegacyNoScalars(index) == THTensor_(nDimensionLegacyNoScalars)(tensor), 3,
             "Index tensor must have same dimensions as output tensor");
  THArgCheck(THTensor_(nDimensionLegacyNoScalars)(src) == THTensor_(nDimensionLegacyNoScalars)(tensor), 4,
             "Input tensor must have same dimensions as output tensor");

  // no-op if index is empty
  if (index_ndim_legacy_all == 0)
      return;

  elems_per_row = THTensor_sizeLegacyNoScalars(index, dim);

  TH_TENSOR_DIM_APPLY3(scalar_t, tensor, scalar_t, src, int64_t, index, dim,
                       TH_TENSOR_DIM_APPLY3_SIZE_SCATTER,
                       for (i = 0; i < elems_per_row; ++i)
                       {
                         idx = *(index_data + i*index_stride);
                         if (idx < 0 || idx >= tensor_size)
                         {
                           THFree(TH_TENSOR_DIM_APPLY_counter);
                           THError("Invalid index in scatterAdd");
                         }
                         tensor_data[idx * tensor_stride] += *(src_data + i*src_stride);
                       })
}

#if !defined(TH_REAL_IS_BOOL)

void THTensor_(indexAdd)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src)
{
  ptrdiff_t i, numel;
  THTensor *tSlice, *sSlice;
  int64_t *index_data;

  numel = THLongTensor_nElement(index);
  THArgCheck(THTensor_nDimensionLegacyNoScalars(index) == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < THTensor_nDimensionLegacyNoScalars(src), 4,"Indexing dim %d is out of bounds of tensor", dim);
  THArgCheck(numel == THTensor_sizeLegacyNoScalars(src, dim),4,"Number of indices should be equal to source:size(dim)");

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);

  if (tensor->dim() > 1)
  {
    tSlice = THTensor_(new)();
    sSlice = THTensor_(new)();

    for (i=0; i<numel; i++)
    {
      THTensor_(select)(tSlice, tensor, dim, index_data[i]);
      THTensor_(select)(sSlice, src, dim, i);
      THTensor_(cadd)(tSlice, tSlice, 1.0, sSlice);
    }

    c10::raw::intrusive_ptr::decref(tSlice);
    c10::raw::intrusive_ptr::decref(sSlice);
  }
  else
  {
    for (i=0; i<numel; i++)
    {
      THTensor_(set1d)(tensor,
              index_data[i],
              THTensor_(get1d)(src,i) + THTensor_(get1d)(tensor,index_data[i]));
    }
  }
  THLongTensor_free(index);
}

accreal THTensor_(dot)(THTensor *tensor, THTensor *src)
{
  if ( (THTensor_nDimension(tensor) != 1) || (THTensor_nDimension(src) != 1) ) {
    THError("1D tensors expected, got %dD, %dD tensors",
       THTensor_nDimension(tensor), THTensor_nDimension(src));
  }
  accreal sum = 0;
  /* we use a trick here. careful with that. */
  TH_TENSOR_APPLY2(scalar_t, tensor, scalar_t, src,
                   int64_t sz = (tensor_size-tensor_i < src_size-src_i ? tensor_size-tensor_i : src_size-src_i);
                   sum += THBlas_(dot)(sz, src_data, src_stride, tensor_data, tensor_stride);
                   tensor_i += sz;
                   src_i += sz;
                   tensor_data += sz*tensor_stride;
                   src_data += sz*src_stride;
                   break;);
  return sum;
}

void THTensor_(add)(THTensor *r_, THTensor *t, scalar_t value)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  if (r_Contig && tContig) {
    TH_TENSOR_APPLY2_CONTIG(scalar_t, r_, scalar_t, t, THVector_(adds)(r__data, t_data, value, r__len););
  } else {
    TH_TENSOR_APPLY2_PARALLEL(r_Size, r_Contig, tContig, scalar_t, r_, scalar_t, t, *r__data = *t_data + value;, ORDIN_TH_OMP_OVERHEAD_THRESHOLD)
  }
}

void THTensor_(sub)(THTensor *r_, THTensor *t, scalar_t value)
{
  THTensor_(add)(r_, t, -value);
}

void THTensor_(add_scaled)(THTensor *r_, THTensor *t, scalar_t value, scalar_t alpha)
{
  THTensor_(add)(r_, t, value * alpha);
}

void THTensor_(sub_scaled)(THTensor *r_, THTensor *t, scalar_t value, scalar_t alpha)
{
  THTensor_(add)(r_, t, -value * alpha);
}

void THTensor_(lshift)(THTensor *r_, THTensor *t, scalar_t value)
{
#if defined(TH_REAL_IS_FLOAT)
  return THTensor_(mul)(r_, t, powf(2, value));
#elif defined(TH_REAL_IS_DOUBLE)
  return THTensor_(mul)(r_, t, pow(2, value));
#elif defined(TH_REAL_IS_HALF)
  return THError("lshift is not supported for torch.HalfTensor");
#elif defined(TH_REAL_IS_BFLOAT16)
  return THError("lshift is not supported for torch.BFloat16Tensor");
#else
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  if (r_Contig && tContig) {
    scalar_t *tp = t->data<scalar_t>();
    scalar_t *rp = r_->data<scalar_t>();
    at::parallel_for(0, r_Size, TH_OMP_OVERHEAD_THRESHOLD * 100,
        [&](int64_t start, int64_t end) {
      for (auto i = start; i < end; i++) {
#if defined(TH_REAL_IS_BYTE)
        rp[i] = ((scalar_t) tp[i]) << value;
#else
        rp[i] = ((ureal) tp[i]) << value;
#endif
      }
    });
  } else {
#if defined(TH_REAL_IS_BYTE)
    TH_TENSOR_APPLY2_PARALLEL(r_Size, r_Contig, tContig, scalar_t, r_, scalar_t, t, *r__data = (((scalar_t) *t_data) << value);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#else
    TH_TENSOR_APPLY2_PARALLEL(r_Size, r_Contig, tContig, scalar_t, r_, scalar_t, t, *r__data = (((ureal) *t_data) << value);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#endif
  }
#endif
}

void THTensor_(rshift)(THTensor *r_, THTensor *t, scalar_t value)
{
#if defined(TH_REAL_IS_FLOAT)
  return THTensor_(div)(r_, t, powf(2, value));
#elif defined(TH_REAL_IS_DOUBLE)
  return THTensor_(div)(r_, t, pow(2, value));
#elif defined(TH_REAL_IS_HALF)
  return THError("rshift is not supported for torch.HalfTensor");
#elif defined(TH_REAL_IS_BFLOAT16)
  return THError("rshift is not supported for torch.BFloat16Tensor");
#else
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  if (r_Contig && tContig) {
    scalar_t *tp = t->data<scalar_t>();
    scalar_t *rp = r_->data<scalar_t>();
    at::parallel_for(0, r_Size, TH_OMP_OVERHEAD_THRESHOLD * 100,
        [&](int64_t start, int64_t end) {
      for (auto i = start; i < end; i++) {
#if defined(TH_REAL_IS_BYTE)
        rp[i] = ((scalar_t) tp[i]) >> value;
#else
        rp[i] = ((ureal) tp[i]) >> value;
#endif
      }
    });
  } else {
#if defined(TH_REAL_IS_BYTE)
    TH_TENSOR_APPLY2_PARALLEL(r_Size, r_Contig, tContig, scalar_t, r_, scalar_t, t, *r__data = (((scalar_t) *t_data) >> value);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#else
    TH_TENSOR_APPLY2_PARALLEL(r_Size, r_Contig, tContig, scalar_t, r_, scalar_t, t, *r__data = (((ureal) *t_data) >> value);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#endif
  }
#endif
}

void THTensor_(fmod)(THTensor *r_, THTensor *t, scalar_t value)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  if (r_Contig && tContig) {
    scalar_t *tp = t->data<scalar_t>();
    scalar_t *rp = r_->data<scalar_t>();
    at::parallel_for(0, r_Size, TH_OMP_OVERHEAD_THRESHOLD,
        [&](int64_t start, int64_t end) {
      for (auto i = start; i < end; i++) {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
        rp[i] = fmod(tp[i], value);
#else
        rp[i] = tp[i] % value;
#endif
      }
    });
  } else {

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
    TH_TENSOR_APPLY2_PARALLEL(r_Size, r_Contig, tContig, scalar_t, r_, scalar_t, t, *r__data = fmod(*t_data, value);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#else
    TH_TENSOR_APPLY2_PARALLEL(r_Size, r_Contig, tContig, scalar_t, r_, scalar_t, t, *r__data = (*t_data % value);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#endif
  }
}

// Should wrap if the value (a) has a different sign than the divisor (b), but is not 0.
static inline bool modulo_wrap(scalar_t a, scalar_t b) {
  return (a != 0) && (a < 0) != (b < 0);
}

void THTensor_(remainder)(THTensor *r_, THTensor *t, scalar_t value)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  if (r_Contig && tContig) {
    scalar_t *tp = t->data<scalar_t>();
    scalar_t *rp = r_->data<scalar_t>();
    at::parallel_for(0, r_Size, TH_OMP_OVERHEAD_THRESHOLD,
        [&](size_t start, size_t end) {
      for (auto i = start; i < end; i++) {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
        rp[i] = (value == 0)? NAN : tp[i] - value * floor(tp[i] / value);
#else
        // There is no NAN for integers
        rp[i] = tp[i] % value;
        if (modulo_wrap(rp[i], value))
          rp[i] += value;
#endif
      }
    });
  } else {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
    TH_TENSOR_APPLY2_PARALLEL(r_Size, r_Contig, tContig, scalar_t, r_, scalar_t, t, *r__data = (value == 0)? NAN : *t_data - value * floor(*t_data / value);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#else
    // There is no NAN for integers
    TH_TENSOR_APPLY2_PARALLEL(r_Size, r_Contig, tContig, scalar_t, r_, scalar_t, t, *r__data = *t_data % value;
        if (modulo_wrap(*r__data, value)) *r__data += value;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#endif
  }
}

#endif

#endif

#endif

#endif /* TH_GENERIC_FILE */
