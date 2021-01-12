#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensorEvenMoreMath.cpp"
#else

#include <TH/generic/THTensorApply.hpp>
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/MemoryOverlap.h>

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

#endif

#if !defined(TH_REAL_IS_BFLOAT16) /* non bfloat16 part*/

void THTensor_(indexCopy)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src)
{
  ptrdiff_t i, numel;
  THTensor *tSlice, *sSlice;
  int64_t *index_data;

  dim = at::maybe_wrap_dim(dim, tensor);
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
  at::NoNamesGuard guard;

  ptrdiff_t i, numel;
  THTensor *tSlice;
  int64_t *index_data;

  dim = at::maybe_wrap_dim(dim, tensor);
  numel = THLongTensor_nElement(index);
  THArgCheck(THTensor_nDimensionLegacyNoScalars(index) == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < THTensor_nDimensionLegacyNoScalars(tensor), 4,"Indexing dim %d is out of bounds of tensor", dim);
  at::assert_no_overlap(tensor, index);
  if (at::has_internal_overlap(tensor) == at::MemOverlap::YES) {
    TORCH_WARN(
      "Use of index_fill_ on expanded tensors is deprecated. "
      "Please clone() the tensor before performing this operation. "
      "This also applies to advanced indexing e.g. tensor[mask] = scalar");
  }

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);

  for (i=0; i<numel; i++)
  {
    if (tensor->dim() > 1)
    {
      tSlice = THTensor_(new)();
      THTensor_(select)(tSlice, tensor,dim,index_data[i]);
      THTensor_wrap(tSlice).fill_(val);
      c10::raw::intrusive_ptr::decref(tSlice);
    }
    else
    {
      THTensor_(set1d)(tensor, index_data[i], val);
    }
  }
  THLongTensor_free(index);
}

#endif

#endif

#endif /* TH_GENERIC_FILE */
