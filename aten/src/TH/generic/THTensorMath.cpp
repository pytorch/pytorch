#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorMath.cpp"
#else

#include <TH/generic/THTensorApply.hpp>

static inline real THTensor_(powOne)(real x, real y) {
#if defined(TH_REAL_IS_FLOAT)
  return powf(x, y);
#elif defined(TH_REAL_IS_DOUBLE)
  return pow(x, y);
#else
  THArgCheck(y >= 0, 1,
      "Integers to negative integer powers are not allowed");
  real result = 1;
  while (y) {
    if (y & 1) {
       result *= x;
    }
    y /= 2;
    x *= x;
  }
  return result;
#endif
}

void THTensor_(fill)(THTensor *r_, real value)
{
  if (THTensor_(isContiguous)(r_) || THTensor_(isTransposed)(r_)) {
    TH_TENSOR_APPLY_CONTIG(real, r_, THVector_(fill)(r__data, value, r__len););
  } else {
    TH_TENSOR_APPLY(real, r_,
      if (r__stride == 1) {
        THVector_(fill)(r__data, value, r__size);
	r__i = r__size;
	r__data += r__stride * r__size;
	break;
      } else {
        *r__data = value;
      }
      );
  }
}

void THTensor_(zero)(THTensor *r_)
{
  THTensor_(fill)(r_, 0);
}

void THTensor_(maskedFill)(THTensor *tensor, THByteTensor *mask, real value)
{
  TH_TENSOR_APPLY2(real, tensor, unsigned char, mask,
                   if (*mask_data > 1)
                   {
                     THFree(mask_counter);
                     THFree(tensor_counter);
                     THError("Mask tensor can take 0 and 1 values only");
                   }
                   else if (*mask_data == 1)
                   {
                     *tensor_data = value;
                   });
}

void THTensor_(maskedCopy)(THTensor *tensor, THByteTensor *mask, THTensor* src )
{
  THTensor *srct = THTensor_(newContiguous)(src);
  real *src_data = THTensor_(data)(srct);
  ptrdiff_t cntr = 0;
  ptrdiff_t nelem = THTensor_(nElement)(srct);
  if (THTensor_(nElement)(tensor) != THByteTensor_nElement(mask))
  {
    THTensor_(free)(srct);
    THError("Number of elements of destination tensor != Number of elements in mask");
  }
  TH_TENSOR_APPLY2(real, tensor, unsigned char, mask,
                   if (*mask_data > 1)
                   {
                     THTensor_(free)(srct);
                     THFree(mask_counter);
                     THFree(tensor_counter);
                     THError("Mask tensor can take 0 and 1 values only");
                   }
                   else if (*mask_data == 1)
                   {
                     if (cntr == nelem)
                     {
                       THTensor_(free)(srct);
                       THFree(mask_counter);
                       THFree(tensor_counter);
                       THError("Number of elements of src < number of ones in mask");
                     }
                     *tensor_data = *src_data;
                     src_data++;
                     cntr++;
                   });
  THTensor_(free)(srct);
}

void THTensor_(maskedSelect)(THTensor *tensor, THTensor *src, THByteTensor *mask)
{
  ptrdiff_t numel = THByteTensor_sumall(mask);
  real *tensor_data;

#ifdef DEBUG
  THAssert(numel <= LONG_MAX);
#endif
  THTensor_(resize1d)(tensor,numel);
  tensor_data = THTensor_(data)(tensor);
  TH_TENSOR_APPLY2(real, src, unsigned char, mask,
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

// Finds non-zero elements of a tensor and returns their subscripts
void THTensor_(nonzero)(THLongTensor *subscript, THTensor *tensor)
{
  ptrdiff_t numel = 0;
  int64_t *subscript_data;
  int64_t i = 0;
  int64_t dim;
  int64_t div = 1;
#ifdef TH_REAL_IS_HALF
#define IS_NONZERO(val) ((val.x & 0x7fff) != 0)
#else
#define IS_NONZERO(val) ((val)!=0)
#endif

  /* First Pass to determine size of subscripts */
  TH_TENSOR_APPLY(real, tensor,
                  if IS_NONZERO(*tensor_data) {
                    ++numel;
                  });
#ifdef DEBUG
  THAssert(numel <= LONG_MAX);
#endif
  THLongTensor_resize2d(subscript, numel, tensor->dim());

  /* Second pass populates subscripts */
  subscript_data = THLongTensor_data(subscript);
  TH_TENSOR_APPLY(real, tensor,
                  if IS_NONZERO(*tensor_data) {
                    div = 1;

                    for (dim = tensor->dim() - 1; dim >= 0; dim--) {
                      *(subscript_data + dim) = (i/div) % tensor->size(dim);
                      div *= tensor->size(dim);
                    }

                    subscript_data += tensor->dim();
                  }
                  ++i;);
}

void THTensor_(indexSelect)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index)
{
  ptrdiff_t i, numel;
  THLongStorage *newSize;
  THTensor *tSlice, *sSlice;
  int64_t *index_data;
  real *tensor_data, *src_data;

#ifndef USE_TH_SIZE_ZERO_DIM
  THArgCheck(index->_dim() <= 1, 3, "Index is supposed to be an empty tensor or a vector");
  THArgCheck(dim < src->_dim(), 4, "Indexing dim %d is out of bounds of tensor", dim + TH_INDEX_BASE);
  THArgCheck(src->_dim() > 0, 2, "Source tensor is empty");
#else
  THArgCheck(index->dim() == 1, 3, "Index is supposed to be 1-dimensional");
  THArgCheck(dim < src->dim(), 4, "Indexing dim %d is out of bounds of tensor", dim + TH_INDEX_BASE);
  //THArgCheck(src->dim() > 0, 2, "Source tensor is empty");
#endif

  numel = THLongTensor_nElement(index);

  newSize = THLongStorage_newWithSize(src->dim());
  THLongStorage_rawCopy(newSize, THTensor_getSizePtr(src));
#ifdef DEBUG
  THAssert(numel <= LONG_MAX);
#endif
  THLongStorage_data(newSize)[dim] = numel;
  THTensor_(resize)(tensor,newSize,NULL);
  THLongStorage_free(newSize);

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);

  if (dim == 0 && THTensor_(isContiguous)(src) && THTensor_(isContiguous)(tensor))
  {
    tensor_data = THTensor_(data)(tensor);
    src_data = THTensor_(data)(src);
    ptrdiff_t rowsize = src->size(0) == 0 ? 1: THTensor_(nElement)(src) / src->size(0);

    // check that the indices are within range
    int64_t max = src->size(0) - 1 + TH_INDEX_BASE;
    for (i=0; i<numel; i++) {
      if (index_data[i] < TH_INDEX_BASE || index_data[i] > max) {
        THLongTensor_free(index);
        THError("index out of range");
      }
    }

    if (src->dim() == 1) {
      #pragma omp parallel for if(numel > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<numel; i++)
        tensor_data[i] = src_data[index_data[i] - TH_INDEX_BASE];
    } else {
      #pragma omp parallel for if(numel*rowsize > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<numel; i++)
        memcpy(tensor_data + i*rowsize, src_data + (index_data[i] - TH_INDEX_BASE)*rowsize, rowsize*sizeof(real));
    }
  }
  else if (src->dim() == 1)
  {
    for (i=0; i<numel; i++)
      THTensor_(set1d)(tensor,i,THTensor_(get1d)(src,index_data[i] - TH_INDEX_BASE));
  }
  else
  {
    for (i=0; i<numel; i++)
    {
      tSlice = THTensor_(new)();
      sSlice = THTensor_(new)();
      THTensor_(select)(tSlice, tensor, dim, i);
      THTensor_(select)(sSlice, src, dim, index_data[i] - TH_INDEX_BASE);
      THTensor_(copy)(tSlice, sSlice);
      THTensor_(free)(tSlice);
      THTensor_(free)(sSlice);
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
      THTensor_(select)(tSlice, tensor, dim, index_data[i] - TH_INDEX_BASE);
      THTensor_(select)(sSlice, src, dim, i);
      THTensor_(copy)(tSlice, sSlice);
    }

    THTensor_(free)(tSlice);
    THTensor_(free)(sSlice);
  }
  else
  {
    for (i=0; i<numel; i++)
    {
      THTensor_(set1d)(tensor, index_data[i] - TH_INDEX_BASE, THTensor_(get1d)(src,i));
    }
  }
  THLongTensor_free(index);
}

static ptrdiff_t THTensor_(dataOffset)(THTensor* tensor, ptrdiff_t linearIndex) {
  auto size = tensor->sizes();
  auto stride = tensor->strides();
  int nDim = tensor->_dim();
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
  real* src_data = THTensor_(data)(src);
  real* dst_data = THTensor_(data)(dst);
  ptrdiff_t nIndices = THLongTensor_nElement(index);
  int isContiguous = THTensor_(isContiguous)(src);

  // Exceptions must not be thrown across OpenMP parallel sections, so we
  // record the position of the invalid index and throw the exception after the
  // loop.
  std::atomic<int64_t> invalidIdxPos(-1);

  ptrdiff_t i;
  #pragma omp parallel for if(nIndices > TH_OMP_OVERHEAD_THRESHOLD) private(i)
  for (i = 0; i < nIndices; i++) {
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
  real* data = THTensor_(data)(tensor);
  ptrdiff_t numel = THTensor_(nElement)(tensor);
  int is_contiguous = THTensor_(isContiguous)(tensor);

  TH_TENSOR_APPLY2(int64_t, index, real, src,
    THTensor_(checkLinearIndex)(*index_data, numel);
    int64_t linearIndex = THTensor_(wrapLinearIndex)(*index_data, numel);
    int64_t dataOffset = is_contiguous ? linearIndex : THTensor_(dataOffset)(tensor, linearIndex);
    if (accumulate) {
      data[dataOffset] += *src_data;
    } else {
      data[dataOffset] = *src_data;
    }
  );

  THTensor_(free)(src);
  THLongTensor_free(index);
}

void THTensor_(indexAdd)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src)
{
  ptrdiff_t i, numel;
  THTensor *tSlice, *sSlice;
  int64_t *index_data;

  numel = THLongTensor_nElement(index);
#ifndef USE_TH_SIZE_ZERO_DIM
  THArgCheck(index->_dim() == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < src->_dim(), 4,"Indexing dim %d is out of bounds of tensor", dim + TH_INDEX_BASE);
#else
  THArgCheck(index->dim() == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < src->dim(), 4,"Indexing dim %d is out of bounds of tensor", dim + TH_INDEX_BASE);
#endif
  THArgCheck(numel == src->size(dim),4,"Number of indices should be equal to source:size(dim)");

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);

  if (tensor->dim() > 1)
  {
    tSlice = THTensor_(new)();
    sSlice = THTensor_(new)();

    for (i=0; i<numel; i++)
    {
      THTensor_(select)(tSlice, tensor, dim, index_data[i] - TH_INDEX_BASE);
      THTensor_(select)(sSlice, src, dim, i);
      THTensor_(cadd)(tSlice, tSlice, 1.0, sSlice);
    }

    THTensor_(free)(tSlice);
    THTensor_(free)(sSlice);
  }
  else
  {
    for (i=0; i<numel; i++)
    {
      THTensor_(set1d)(tensor,
              index_data[i] - TH_INDEX_BASE,
              THTensor_(get1d)(src,i) + THTensor_(get1d)(tensor,index_data[i] - TH_INDEX_BASE));
    }
  }
  THLongTensor_free(index);
}

void THTensor_(indexFill)(THTensor *tensor, int dim, THLongTensor *index, real val)
{
  ptrdiff_t i, numel;
  THTensor *tSlice;
  int64_t *index_data;

  numel = THLongTensor_nElement(index);
#ifndef USE_TH_SIZE_ZERO_DIM
  THArgCheck(index->_dim() == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < tensor->_dim(), 4,"Indexing dim %d is out of bounds of tensor", dim + TH_INDEX_BASE);
#else
  THArgCheck(index->dim() == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < tensor->dim(), 4,"Indexing dim %d is out of bounds of tensor", dim + TH_INDEX_BASE);
#endif

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);

  for (i=0; i<numel; i++)
  {
    if (tensor->dim() > 1)
    {
      tSlice = THTensor_(new)();
      THTensor_(select)(tSlice, tensor,dim,index_data[i] - TH_INDEX_BASE);
      THTensor_(fill)(tSlice, val);
      THTensor_(free)(tSlice);
    }
    else
    {
      THTensor_(set1d)(tensor, index_data[i] - TH_INDEX_BASE, val);
    }
  }
  THLongTensor_free(index);
}

void THTensor_(gather)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index)
{
  int64_t elems_per_row, i, idx;

  THArgCheck(THLongTensor_nDimension(index) == THTensor_(nDimension)(src), 4,
             "Index tensor must have same dimensions as input tensor");
  THArgCheck(dim >= 0 && dim < THTensor_(nDimension)(tensor), 3,
             "Index dimension is out of bounds");
  THArgCheck(THTensor_(nDimension)(src) == THTensor_(nDimension)(tensor), 2,
             "Input tensor must have same dimensions as output tensor");

  elems_per_row = THLongTensor_size(index, dim);

  TH_TENSOR_DIM_APPLY3(real, tensor, real, src, int64_t, index, dim,
                       TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
                       for (i = 0; i < elems_per_row; ++i)
                       {
                         idx = *(index_data + i*index_stride);
                         if (idx < TH_INDEX_BASE || idx >= src_size + TH_INDEX_BASE)
                         {
                           THFree(TH_TENSOR_DIM_APPLY_counter);
                           THError("Invalid index in gather");
                         }
                         *(tensor_data + i*tensor_stride) = src_data[(idx - TH_INDEX_BASE) * src_stride];
                       })
}

void THTensor_(scatter)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src)
{
  int64_t elems_per_row, i, idx;

#ifndef USE_TH_SIZE_ZERO_DIM
  THArgCheck(dim < THTensor_(_nDimension)(tensor), 2, "Index dimension is out of bounds");
  THArgCheck(THLongTensor__nDimension(index) == THTensor_(_nDimension)(tensor), 3,
             "Index tensor must have same dimensions as output tensor");
  THArgCheck(THTensor_(_nDimension)(src) == THTensor_(_nDimension)(tensor), 4,
             "Input tensor must have same dimensions as output tensor");
#else
  THArgCheck(dim < THTensor_(nDimension)(tensor), 2, "Index dimension is out of bounds");
  THArgCheck(THLongTensor_nDimension(index) == THTensor_(nDimension)(tensor), 3,
             "Index tensor must have same dimensions as output tensor");
  THArgCheck(THTensor_(nDimension)(src) == THTensor_(nDimension)(tensor), 4,
             "Input tensor must have same dimensions as output tensor");
#endif

  elems_per_row = THLongTensor_size(index, dim);

  TH_TENSOR_DIM_APPLY3(real, tensor, real, src, int64_t, index, dim,
                       TH_TENSOR_DIM_APPLY3_SIZE_SCATTER,
                       for (i = 0; i < elems_per_row; ++i)
                       {
                         idx = *(index_data + i*index_stride);
                         if (idx < TH_INDEX_BASE || idx >= tensor_size + TH_INDEX_BASE)
                         {
                           THFree(TH_TENSOR_DIM_APPLY_counter);
                           THError("Invalid index in scatter");
                         }
                         tensor_data[(idx - TH_INDEX_BASE) * tensor_stride] = *(src_data + i*src_stride);
                       })
}

void THTensor_(scatterAdd)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src)
{
  int64_t elems_per_row, i, idx;

  THArgCheck(dim < THTensor_(nDimension)(tensor), 2, "Index dimension is out of bounds");
  THArgCheck(THLongTensor_nDimension(index) == THTensor_(nDimension)(tensor), 3,
             "Index tensor must have same dimensions as output tensor");
  THArgCheck(THTensor_(nDimension)(src) == THTensor_(nDimension)(tensor), 4,
             "Input tensor must have same dimensions as output tensor");

  elems_per_row = THLongTensor_size(index, dim);

  TH_TENSOR_DIM_APPLY3(real, tensor, real, src, int64_t, index, dim,
                       TH_TENSOR_DIM_APPLY3_SIZE_SCATTER,
                       for (i = 0; i < elems_per_row; ++i)
                       {
                         idx = *(index_data + i*index_stride);
                         if (idx < TH_INDEX_BASE || idx >= tensor_size + TH_INDEX_BASE)
                         {
                           THFree(TH_TENSOR_DIM_APPLY_counter);
                           THError("Invalid index in scatterAdd");
                         }
                         tensor_data[(idx - TH_INDEX_BASE) * tensor_stride] += *(src_data + i*src_stride);
                       })
}

void THTensor_(scatterFill)(THTensor *tensor, int dim, THLongTensor *index, real val)
{
  int64_t elems_per_row, i, idx;

  THArgCheck(dim < THTensor_(_nDimension)(tensor), 2, "Index dimension is out of bounds");
  THArgCheck(THLongTensor__nDimension(index) == THTensor_(_nDimension)(tensor), 3,
             "Index tensor must have same dimensions as output tensor");

  elems_per_row = THLongTensor_size(index, dim);

  TH_TENSOR_DIM_APPLY2(real, tensor, int64_t, index, dim,
                       for (i = 0; i < elems_per_row; ++i)
                       {
                         idx = *(index_data + i*index_stride);
                         if (idx < TH_INDEX_BASE || idx >= tensor_size + TH_INDEX_BASE)
                         {
                           THFree(TH_TENSOR_DIM_APPLY_counter);
                           THError("Invalid index in scatter");
                         }
                         tensor_data[(idx - TH_INDEX_BASE) * tensor_stride] = val;
                       })
}

accreal THTensor_(dot)(THTensor *tensor, THTensor *src)
{
  accreal sum = 0;
  /* we use a trick here. careful with that. */
  TH_TENSOR_APPLY2(real, tensor, real, src,
                   int64_t sz = (tensor_size-tensor_i < src_size-src_i ? tensor_size-tensor_i : src_size-src_i);
                   sum += THBlas_(dot)(sz, src_data, src_stride, tensor_data, tensor_stride);
                   tensor_i += sz;
                   src_i += sz;
                   tensor_data += sz*tensor_stride;
                   src_data += sz*src_stride;
                   break;);
  return sum;
}

real THTensor_(minall)(THTensor *tensor)
{
  real theMin;
  real value;

  THArgCheck(tensor->_dim() > 0, 1, "tensor must have one dimension");
  theMin = THTensor_(data)(tensor)[0];
  TH_TENSOR_APPLY(real, tensor,
                  value = *tensor_data;
                  /* This is not the same as value<theMin in the case of NaNs */
                  if(!(value >= theMin))
                  {
                    theMin = value;
                    th_isnan_break(value)
                  });
  return theMin;
}

real THTensor_(maxall)(THTensor *tensor)
{
  real theMax;
  real value;

  THArgCheck(tensor->_dim() > 0, 1, "tensor must have one dimension");
  theMax = THTensor_(data)(tensor)[0];
  TH_TENSOR_APPLY(real, tensor,
                  value = *tensor_data;
                  /* This is not the same as value>theMax in the case of NaNs */
                  if(!(value <= theMax))
                  {
                    theMax = value;
                    th_isnan_break(value)
                  });
  return theMax;
}

accreal THTensor_(sumall)(THTensor *tensor)
{
  accreal sum = 0;
  int serial_path = 0;
#ifdef _OPENMP
  int inOMP = omp_in_parallel();
  if(inOMP) {
    serial_path = 1;
  } else {
    TH_TENSOR_APPLY_REDUCTION_OMP(real, tensor, +:sum, sum += *tensor_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
  }
#else
    serial_path = 1;
#endif
  if (serial_path) {
    TH_TENSOR_APPLY(real, tensor, sum += *tensor_data;);
  }
  return sum;
}

accreal THTensor_(prodall)(THTensor *tensor)
{
  accreal prod = 1;
  int serial_path = 0;
#ifdef _OPENMP
  int inOMP = omp_in_parallel();
  if(inOMP) {
    serial_path = 1;
  } else {
    TH_TENSOR_APPLY_REDUCTION_OMP(real, tensor, *:prod, prod *= *tensor_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
  }
#else
    serial_path = 1;
#endif
  if (serial_path) {
    TH_TENSOR_APPLY(real, tensor, prod *= *tensor_data;);
  }
  return prod;
}

void THTensor_(add)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int serial_path = 0;
  if (r_Contig && tContig) {
    TH_TENSOR_APPLY2_CONTIG(real, r_, real, t, THVector_(adds)(r__data, t_data, value, r__len););
  } else {
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = *t_data + value;, ORDIN_TH_OMP_OVERHEAD_THRESHOLD)
    }
#else
    (void)r_Size;
    serial_path = 1;
#endif
  }
  if (serial_path) {
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data + value;);
  }
}

void THTensor_(sub)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(add)(r_, t, -value);
}

void THTensor_(add_scaled)(THTensor *r_, THTensor *t, real value, real alpha)
{
  THTensor_(add)(r_, t, value * alpha);
}

void THTensor_(sub_scaled)(THTensor *r_, THTensor *t, real value, real alpha)
{
  THTensor_(add)(r_, t, -value * alpha);
}

void THTensor_(mul)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int serial_path = 0;
  if (r_Contig && tContig) {
    TH_TENSOR_APPLY2_CONTIG(real, r_, real, t, THVector_(muls)(r__data, t_data, value, r__len););
  } else {
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = *t_data * value;, ORDIN_TH_OMP_OVERHEAD_THRESHOLD)
    }
#else
    (void)r_Size;
    serial_path = 1;
#endif
  }
  if (serial_path) {
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data * value;);
  }
}

void THTensor_(div)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int serial_path = 0;
  if (r_Contig && tContig) {
    TH_TENSOR_APPLY2_CONTIG(real, r_, real, t, THVector_(divs)(r__data, t_data, value, r__len););
  } else {
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = *t_data / value;, ORDIN_TH_OMP_OVERHEAD_THRESHOLD)
    }
#else
    (void)r_Size;
    serial_path = 1;
#endif
  }
  if (serial_path) {
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data / value;);
  }
}

void THTensor_(lshift)(THTensor *r_, THTensor *t, real value)
{
#if defined(TH_REAL_IS_FLOAT)
  return THTensor_(mul)(r_, t, powf(2, value));
#elif defined(TH_REAL_IS_DOUBLE)
  return THTensor_(mul)(r_, t, pow(2, value));
#elif defined(TH_REAL_IS_HALF)
  return THError("lshift is not supported for torch.HalfTensor");
#else
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int serial_path = 0;
  if (r_Contig && tContig) {
    real *tp = THTensor_(data)(t);
    real *rp = THTensor_(data)(r_);
    int64_t i;
    #pragma omp parallel for if(r_Size > TH_OMP_OVERHEAD_THRESHOLD * 100) private(i)
    for (i=0; i<r_Size; i++) {
#if defined(TH_REAL_IS_BYTE)
      rp[i] = ((real) tp[i]) << value;
#else
      rp[i] = ((ureal) tp[i]) << value;
#endif
    }
  } else {
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
#if defined(TH_REAL_IS_BYTE)
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = (((real) *t_data) << value);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#else
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = (((ureal) *t_data) << value);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#endif
    }
#else
    serial_path = 1;
#endif
  }
  if (serial_path) {
#if defined(TH_REAL_IS_BYTE)
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = (((real) *t_data) << value););
#else
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = (((ureal) *t_data) << value););
#endif
  }
#endif
}

void THTensor_(rshift)(THTensor *r_, THTensor *t, real value)
{
#if defined(TH_REAL_IS_FLOAT)
  return THTensor_(div)(r_, t, powf(2, value));
#elif defined(TH_REAL_IS_DOUBLE)
  return THTensor_(div)(r_, t, pow(2, value));
#elif defined(TH_REAL_IS_HALF)
  return THError("rshift is not supported for torch.HalfTensor");
#else
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int serial_path = 0;
  if (r_Contig && tContig) {
    real *tp = THTensor_(data)(t);
    real *rp = THTensor_(data)(r_);
    int64_t i;
    #pragma omp parallel for if(r_Size > TH_OMP_OVERHEAD_THRESHOLD * 100) private(i)
    for (i=0; i<r_Size; i++) {
#if defined(TH_REAL_IS_BYTE)
      rp[i] = ((real) tp[i]) >> value;
#else
      rp[i] = ((ureal) tp[i]) >> value;
#endif
    }
  } else {
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
#if defined(TH_REAL_IS_BYTE)
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = (((real) *t_data) >> value);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#else
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = (((ureal) *t_data) >> value);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#endif
    }
#else
    serial_path = 1;
#endif
  }
  if (serial_path) {
#if defined(TH_REAL_IS_BYTE)
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = (((real) *t_data) >> value););
#else
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = (((ureal) *t_data) >> value););
#endif
  }
#endif
}

void THTensor_(fmod)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int serial_path = 0;
  if (r_Contig && tContig) {
    real *tp = THTensor_(data)(t);
    real *rp = THTensor_(data)(r_);
    int64_t i;
    #pragma omp parallel for if(r_Size > TH_OMP_OVERHEAD_THRESHOLD) private(i)
    for (i=0; i<r_Size; i++) {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
      rp[i] = fmod(tp[i], value);
#else
      rp[i] = tp[i] % value;
#endif
    }
  } else {
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = fmod(*t_data, value);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#else
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = (*t_data % value);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#endif
    }
#else
    serial_path = 1;
#endif
  }
  if (serial_path) {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = fmod(*t_data, value););
#else
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = (*t_data % value););
#endif
  }
}

// Should wrap if the value (a) has a different sign than the divisor (b), but is not 0.
static inline bool modulo_wrap(real a, real b) {
  return (a != 0) && (a < 0) != (b < 0);
}

void THTensor_(remainder)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int serial_path = 0;
  if (r_Contig && tContig) {
    real *tp = THTensor_(data)(t);
    real *rp = THTensor_(data)(r_);
    int64_t i;
    #pragma omp parallel for if(r_Size > TH_OMP_OVERHEAD_THRESHOLD) private(i)
    for (i=0; i<r_Size; i++) {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
      rp[i] = (value == 0)? NAN : tp[i] - value * floor(tp[i] / value);
#else
      // There is no NAN for integers
      rp[i] = tp[i] % value;
      if (modulo_wrap(rp[i], value))
        rp[i] += value;
#endif
    }
  } else {
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = (value == 0)? NAN : *t_data - value * floor(*t_data / value);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#else
      // There is no NAN for integers
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = *t_data % value;
                                        if (modulo_wrap(*r__data, value)) *r__data += value;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#endif
    }
#else
    serial_path = 1;
#endif
  }
  if (serial_path) {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = (value == 0)? NAN : *t_data - value * floor(*t_data / value););
#else
    // There is no NAN for integers
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data % value;
                                          if (modulo_wrap(*r__data, value)) *r__data += value;);
#endif
  }
}

void THTensor_(bitand)(THTensor *r_, THTensor *t, real value)
{
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_HALF)
  (void)r_;
  (void)t;
  (void)value;
  return THError("bitand is only supported for integer type tensors");
#else
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int serial_path = 0;
  int tContig = THTensor_(isContiguous)(t);
  if (r_Contig && tContig) {
    real *tp = THTensor_(data)(t);
    real *rp = THTensor_(data)(r_);
    int64_t i;
    #pragma omp parallel for if(r_Size > TH_OMP_OVERHEAD_THRESHOLD * 100) private(i)
    for (i=0; i<r_Size; i++) {
      rp[i] = tp[i] & value;
    }
  } else {
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = *t_data & value;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
    }
#else
    serial_path = 1;
#endif
  }
  if (serial_path) {
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data & value;);
  }
#endif
}

void THTensor_(bitor)(THTensor *r_, THTensor *t, real value)
{
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_HALF)
  (void)r_;
  (void)t;
  (void)value;
  return THError("bitor is only supported for integer type tensors");
#else
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int serial_path = 0;
  if (r_Contig && tContig) {
    real *tp = THTensor_(data)(t);
    real *rp = THTensor_(data)(r_);
    int64_t i;
    #pragma omp parallel for if(r_Size > TH_OMP_OVERHEAD_THRESHOLD * 100) private(i)
    for (i=0; i<r_Size; i++) {
      rp[i] = tp[i] | value;
    }
  } else {
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = *t_data | value;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
    }
#else
    serial_path = 1;
#endif
  }
  if (serial_path) {
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data | value;);
  }
#endif
}

void THTensor_(bitxor)(THTensor *r_, THTensor *t, real value)
{
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_HALF)
  (void)r_;
  (void)t;
  (void)value;
  return THError("bitxor is only supported for integer type tensors");
#else
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int serial_path = 0;
  if (r_Contig && tContig) {
    real *tp = THTensor_(data)(t);
    real *rp = THTensor_(data)(r_);
    int64_t i;
    #pragma omp parallel for if(r_Size > TH_OMP_OVERHEAD_THRESHOLD * 100) private(i)
    for (i=0; i<r_Size; i++) {
      rp[i] = tp[i] ^ value;
    }
  } else {
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = *t_data ^ value;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
    }
#else
    serial_path = 1;
#endif
  }
  if (serial_path) {
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data ^ value;);
  }
#endif
}

void THTensor_(clamp)(THTensor *r_, THTensor *t, real min_value, real max_value)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int serial_path = 0;
  if (r_Contig && tContig) {
    real *tp = THTensor_(data)(t);
    real *rp = THTensor_(data)(r_);
    /* real t_val; */
    int64_t i;
    #pragma omp parallel for if(r_Size > TH_OMP_OVERHEAD_THRESHOLD) private(i)
    for (i=0; i<r_Size; i++)
      rp[i] = (tp[i] < min_value) ? min_value : (tp[i] > max_value ? max_value : tp[i]);
  } else {
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = (*t_data < min_value) ? min_value : (*t_data > max_value ? max_value : *t_data);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
    }
#else
    serial_path = 1;
#endif
  }
  if (serial_path) {
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = (*t_data < min_value) ? min_value : (*t_data > max_value ? max_value : *t_data););
  }
}

void THTensor_(cadd)(THTensor *r_, THTensor *t, real value, THTensor *src)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int64_t srcSize = THTensor_(nElement)(src);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int srcContig = THTensor_(isContiguous)(src);
  int serial_path = 0;
  if (srcSize == r_Size){
    if (r_Contig && tContig && srcContig) {
      if(r_ == t) {
        THBlas_(axpy)(THTensor_(nElement)(t), value, THTensor_(data)(src), 1, THTensor_(data)(r_), 1);
      } else {
        TH_TENSOR_APPLY3_CONTIG(real, r_, real, t, real, src, THVector_(cadd)(r__data, t_data, src_data, value, r__len););
      }
    } else {
#if _OPENMP
      int inOMP = omp_in_parallel();
      if (inOMP) {
        serial_path = 1;
      } else {
        TH_TENSOR_APPLY3_OMP(r_Size, r_Contig, tContig, srcContig, real, r_, real, t, real, src, *r__data = *t_data + value * *src_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
      }
#else
      serial_path = 1;
#endif
    }
  } else {
    serial_path = 1;
  }
  if (serial_path) {
    TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data + value * *src_data;);
  }
}

void THTensor_(csub)(THTensor *r_, THTensor *t, real value, THTensor *src)
{
  THTensor_(cadd)(r_, t, -value, src);
}

void THTensor_(cmul)(THTensor *r_, THTensor *t, THTensor *src)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int64_t srcSize = THTensor_(nElement)(src);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int srcContig = THTensor_(isContiguous)(src);
  int serial_path = 0;
  if (srcSize == r_Size){
    if (r_Contig && tContig && srcContig) {
      TH_TENSOR_APPLY3_CONTIG(real, r_, real, t, real, src, THVector_(cmul)(r__data, t_data, src_data, r__len););
    } else {
#if _OPENMP
      int inOMP = omp_in_parallel();
      if (inOMP) {
        serial_path = 1;
      } else {
        TH_TENSOR_APPLY3_OMP(r_Size, r_Contig, tContig, srcContig, real, r_, real, t, real, src, *r__data = *t_data * *src_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
      }
#else
      serial_path = 1;
#endif
    }
  } else {
    serial_path = 1;
  }
  if (serial_path) {
    TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data * *src_data;);
  }
}

void THTensor_(pow)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  if(value == 1){
    THTensor_(copy)(r_, t);
  }
  else if(value == 2){
    THTensor_(cmul)(r_, t, t);
  }
  else if(value == 3){
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data * *t_data * *t_data;);
  }
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
#if defined (TH_REAL_IS_FLOAT)
#define TH_MATH_NAME(fn) fn##f
#else
#define TH_MATH_NAME(fn) fn
#endif
  else if(value == 0.5){
    THTensor_(sqrt)(r_, t);
  }
  else if(value == -0.5){
    THTensor_(rsqrt)(r_, t);
  }
  else if(value == -1){
    THTensor_(cinv)(r_, t);
  }
  else if(value == -2){
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = TH_MATH_NAME(1.0) / (*t_data * *t_data););
  }
  else{
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = TH_MATH_NAME(pow)(*t_data, value););
  }
#undef TH_MATH_NAME
#else
  else {
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = THTensor_(powOne)(*t_data, value););
  }
#endif
}

void THTensor_(cpow)(THTensor *r_, THTensor *t, THTensor *src)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int64_t srcSize = THTensor_(nElement)(src);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int srcContig = THTensor_(isContiguous)(src);
  int serial_path = 0;
  if (srcSize == r_Size){
    if (r_Contig && tContig && srcContig) {
      real *tp = THTensor_(data)(t);
      real *sp = THTensor_(data)(src);
      real *rp = THTensor_(data)(r_);
      int64_t i;
      #pragma omp parallel for if(r_Size > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<r_Size; i++)
        rp[i] = THTensor_(powOne)(tp[i], sp[i]);
    } else {
#if _OPENMP
      int inOMP = omp_in_parallel();
      if (inOMP) {
        serial_path = 1;
      } else {
        TH_TENSOR_APPLY3_OMP(r_Size, r_Contig, tContig, srcContig, real, r_, real, t, real, src, *r__data = THTensor_(powOne)(*t_data, *src_data);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
      }
#else
      serial_path = 1;
#endif
    }
  } else {
    serial_path = 1;
  }
  if (serial_path) {
    TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = THTensor_(powOne)(*t_data, *src_data););
  }
}

void THTensor_(cdiv)(THTensor *r_, THTensor *t, THTensor *src)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int64_t srcSize = THTensor_(nElement)(src);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int srcContig = THTensor_(isContiguous)(src);
  int serial_path = 0;
  if (srcSize == r_Size){
    if (r_Contig && tContig && srcContig) {
      TH_TENSOR_APPLY3_CONTIG(real, r_, real, t, real, src, THVector_(cdiv)(r__data, t_data, src_data, r__len););
    } else {
#if _OPENMP
      int inOMP = omp_in_parallel();
      if (inOMP) {
        serial_path = 1;
      } else {
        TH_TENSOR_APPLY3_OMP(r_Size, r_Contig, tContig, srcContig, real, r_, real, t, real, src, *r__data = *t_data / *src_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
      }
#else
      serial_path = 1;
#endif
    }
  } else {
    serial_path = 1;
  }
  if (serial_path) {
    TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data / *src_data;);
  }
}

void THTensor_(clshift)(THTensor *r_, THTensor *t, THTensor *src)
{
#if defined(TH_REAL_IS_HALF)
  return THError("clshift is not supported for torch.HalfTensor");
#endif
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int64_t srcSize = THTensor_(nElement)(src);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int srcContig = THTensor_(isContiguous)(src);
  int serial_path = 0;
  if (srcSize == r_Size){
    if (r_Contig && tContig && srcContig) {
      real *tp = THTensor_(data)(t);
      real *sp = THTensor_(data)(src);
      real *rp = THTensor_(data)(r_);
      int64_t i;
      #pragma omp parallel for if(r_Size > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<r_Size; i++) {
#if defined(TH_REAL_IS_FLOAT)
        rp[i] = tp[i] * powf(2, sp[i]);
#elif defined(TH_REAL_IS_DOUBLE)
        rp[i] = tp[i] * pow(2, sp[i]);
#elif defined(TH_REAL_IS_BYTE)
        rp[i] = ((real) tp[i]) << sp[i];
#else
        rp[i] = ((ureal) tp[i]) << sp[i];
#endif
      }
    } else {
#if _OPENMP
      int inOMP = omp_in_parallel();
      if (inOMP) {
        serial_path = 1;
      } else {
#if defined(TH_REAL_IS_FLOAT)
        TH_TENSOR_APPLY3_OMP(r_Size, r_Contig, tContig, srcContig, real, r_, real, t, real, src, *r__data = *t_data * powf(2, *src_data);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#elif defined(TH_REAL_IS_DOUBLE)
        TH_TENSOR_APPLY3_OMP(r_Size, r_Contig, tContig, srcContig, real, r_, real, t, real, src, *r__data = *t_data * pow(2, *src_data);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#elif defined(TH_REAL_IS_BYTE)
        TH_TENSOR_APPLY3_OMP(r_Size, r_Contig, tContig, srcContig, real, r_, real, t, real, src, *r__data = ((real)*t_data) << *src_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#else
        TH_TENSOR_APPLY3_OMP(r_Size, r_Contig, tContig, srcContig, real, r_, real, t, real, src, *r__data = ((ureal)*t_data) << *src_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#endif
      }
#else
      serial_path = 1;
#endif
    }
  } else {
    serial_path = 1;
  }
  if (serial_path) {
#if defined(TH_REAL_IS_FLOAT)
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data * powf(2, *src_data););
#elif defined(TH_REAL_IS_DOUBLE)
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data * pow(2, *src_data););
#elif defined(TH_REAL_IS_BYTE)
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = ((real)*t_data) << *src_data;);
#else
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = ((ureal)*t_data) << *src_data;);
#endif
  }
}

void THTensor_(crshift)(THTensor *r_, THTensor *t, THTensor *src)
{
#if defined(TH_REAL_IS_HALF)
  return THError("crshift is not supported for torch.HalfTensor");
#endif
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int64_t srcSize = THTensor_(nElement)(src);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int srcContig = THTensor_(isContiguous)(src);
  int serial_path = 0;
  if (srcSize == r_Size){
    if (r_Contig && tContig && srcContig) {
      real *tp = THTensor_(data)(t);
      real *sp = THTensor_(data)(src);
      real *rp = THTensor_(data)(r_);
      int64_t i;
      #pragma omp parallel for if(r_Size > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<r_Size; i++) {
#if defined(TH_REAL_IS_FLOAT)
        rp[i] = tp[i] / powf(2, sp[i]);
#elif defined(TH_REAL_IS_DOUBLE)
        rp[i] = tp[i] / pow(2, sp[i]);
#elif defined(TH_REAL_IS_BYTE)
        rp[i] = ((real) tp[i]) >> sp[i];
#else
        rp[i] = ((ureal) tp[i]) >> sp[i];
#endif
      }
    } else {
#if _OPENMP
      int inOMP = omp_in_parallel();
      if (inOMP) {
        serial_path = 1;
      } else {
#if defined(TH_REAL_IS_FLOAT)
        TH_TENSOR_APPLY3_OMP(r_Size, r_Contig, tContig, srcContig, real, r_, real, t, real, src, *r__data = *t_data / powf(2, *src_data);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#elif defined(TH_REAL_IS_DOUBLE)
        TH_TENSOR_APPLY3_OMP(r_Size, r_Contig, tContig, srcContig, real, r_, real, t, real, src, *r__data = *t_data / pow(2, *src_data);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#elif defined(TH_REAL_IS_BYTE)
        TH_TENSOR_APPLY3_OMP(r_Size, r_Contig, tContig, srcContig, real, r_, real, t, real, src, *r__data = ((real)*t_data) >> *src_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#else
        TH_TENSOR_APPLY3_OMP(r_Size, r_Contig, tContig, srcContig, real, r_, real, t, real, src, *r__data = ((ureal)*t_data) >> *src_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#endif
      }
#else
      serial_path = 1;
#endif
    }
  } else {
    serial_path = 1;
  }
  if (serial_path) {
#if defined(TH_REAL_IS_FLOAT)
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data / powf(2, *src_data););
#elif defined(TH_REAL_IS_DOUBLE)
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data / pow(2, *src_data););
#elif defined(TH_REAL_IS_BYTE)
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = ((real)*t_data) >> *src_data;);
#else
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = ((ureal)*t_data) >> *src_data;);
#endif
  }
}

void THTensor_(cfmod)(THTensor *r_, THTensor *t, THTensor *src)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int64_t srcSize = THTensor_(nElement)(src);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int srcContig = THTensor_(isContiguous)(src);
  int serial_path = 0;
  if (srcSize == r_Size){
    if (r_Contig && tContig && srcContig) {
      real *tp = THTensor_(data)(t);
      real *sp = THTensor_(data)(src);
      real *rp = THTensor_(data)(r_);
      int64_t i;
      #pragma omp parallel for if(r_Size > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<r_Size; i++) {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
        rp[i] = fmod(tp[i], sp[i]);
#else
        rp[i] = tp[i] % sp[i];
#endif
      }
    } else {
#if _OPENMP
      int inOMP = omp_in_parallel();
      if (inOMP) {
        serial_path = 1;
      } else {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
        TH_TENSOR_APPLY3_OMP(r_Size, r_Contig, tContig, srcContig,real, r_, real, t, real, src, *r__data = fmod(*t_data, *src_data);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#else
        TH_TENSOR_APPLY3_OMP(r_Size, r_Contig, tContig, srcContig, real, r_, real, t, real, src, *r__data = (*t_data % *src_data);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#endif
      }
#else
      serial_path = 1;
#endif
    }
  } else {
    serial_path = 1;
  }
  if (serial_path) {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
    TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = fmod(*t_data, *src_data););
#else
    TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = (*t_data % *src_data););
#endif
  }
}

void THTensor_(cremainder)(THTensor *r_, THTensor *t, THTensor *src)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int64_t srcSize = THTensor_(nElement)(src);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int srcContig = THTensor_(isContiguous)(src);
  int serial_path = 0;
  if (srcSize == r_Size){
    if (r_Contig && tContig && srcContig) {
      real *tp = THTensor_(data)(t);
      real *sp = THTensor_(data)(src);
      real *rp = THTensor_(data)(r_);
      int64_t i;
      #pragma omp parallel for if(r_Size > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<r_Size; i++) {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
        rp[i] = (sp[i] == 0)? NAN : tp[i] - sp[i] * floor(tp[i] / sp[i]);
#else
        // There is no NAN for integers
        rp[i] = tp[i] % sp[i];
        if (modulo_wrap(rp[i], sp[i]))
          rp[i] += sp[i];
#endif
      }
    } else {
#if _OPENMP
      int inOMP = omp_in_parallel();
      if (inOMP) {
        serial_path = 1;
      } else {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
        TH_TENSOR_APPLY3_OMP(r_Size, r_Contig, tContig, srcContig, real, r_, real, t, real, src, *r__data = (*src_data == 0)? NAN : *t_data - *src_data * floor(*t_data / *src_data);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#else
        TH_TENSOR_APPLY3_OMP(r_Size, r_Contig, tContig, srcContig, real, r_, real, t, real, src, *r__data = *t_data % *src_data;
                                                     if (modulo_wrap(*r__data, *src_data)) *r__data += *src_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#endif
      }
#else
      serial_path = 1;
#endif
    }
  } else {
    serial_path = 1;
  }
  if (serial_path) {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
    TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = (*src_data == 0)? NAN : *t_data - *src_data * floor(*t_data / *src_data););
#else
    // There is no NAN for integers
    TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data % *src_data;
                                                     if (modulo_wrap(*r__data, *src_data)) *r__data += *src_data;);
#endif

  }
}

void THTensor_(cbitand)(THTensor *r_, THTensor *t, THTensor *src)
{
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_HALF)
  (void)r_;
  (void)t;
  (void)src;
  return THError("cbitand is only supported for integer type tensors");
#else
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int64_t srcSize = THTensor_(nElement)(src);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int srcContig = THTensor_(isContiguous)(src);
  int serial_path = 0;
  if (srcSize == r_Size){
    if (r_Contig && tContig && srcContig) {
      real *tp = THTensor_(data)(t);
      real *sp = THTensor_(data)(src);
      real *rp = THTensor_(data)(r_);
      int64_t i;
      #pragma omp parallel for if(r_Size > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<r_Size; i++) {
        rp[i] = tp[i] & sp[i];
      }
    } else {
#if _OPENMP
      int inOMP = omp_in_parallel();
      if (inOMP) {
        serial_path = 1;
      } else {
        TH_TENSOR_APPLY3_OMP(r_Size, r_Contig, tContig, srcContig, real, r_, real, t, real, src, *r__data = *t_data & *src_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
      }
#else
      serial_path = 1;
#endif
    }
  } else {
    serial_path = 1;
  }
  if (serial_path) {
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data & *src_data;);
  }
#endif
}

void THTensor_(cbitor)(THTensor *r_, THTensor *t, THTensor *src)
{
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_HALF)
  (void)r_;
  (void)t;
  (void)src;
  return THError("cbitor is only supported for integer type tensors");
#else
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int64_t srcSize = THTensor_(nElement)(src);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int srcContig = THTensor_(isContiguous)(src);
  int serial_path = 0;
  if (srcSize == r_Size){
    if (r_Contig && tContig && srcContig) {
      real *tp = THTensor_(data)(t);
      real *sp = THTensor_(data)(src);
      real *rp = THTensor_(data)(r_);
      int64_t i;
      #pragma omp parallel for if(r_Size > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<r_Size; i++) {
        rp[i] = tp[i] | sp[i];
      }
    } else {
#if _OPENMP
      int inOMP = omp_in_parallel();
      if (inOMP) {
        serial_path = 1;
      } else {
        TH_TENSOR_APPLY3_OMP(r_Size, r_Contig, tContig, srcContig, real, r_, real, t, real, src, *r__data = *t_data | *src_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
      }
#else
      serial_path = 1;
#endif
    }
  } else {
    serial_path = 1;
  }
  if (serial_path) {
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data | *src_data;);
  }
#endif
}

void THTensor_(cbitxor)(THTensor *r_, THTensor *t, THTensor *src)
{
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_HALF)
  (void)r_;
  (void)t;
  (void)src;
  return THError("cbitxor is only supported for integer type tensors");
#else
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int64_t srcSize = THTensor_(nElement)(src);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int srcContig = THTensor_(isContiguous)(src);
  int serial_path = 0;
  if (srcSize == r_Size){
    if (r_Contig && tContig && srcContig) {
      real *tp = THTensor_(data)(t);
      real *sp = THTensor_(data)(src);
      real *rp = THTensor_(data)(r_);
      int64_t i;
      #pragma omp parallel for if(r_Size > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<r_Size; i++) {
        rp[i] = tp[i] ^ sp[i];
      }
    } else {
#if _OPENMP
      int inOMP = omp_in_parallel();
      if (inOMP) {
        serial_path = 1;
      } else {
        TH_TENSOR_APPLY3_OMP(r_Size, r_Contig, tContig, srcContig, real, r_, real, t, real, src, *r__data = *t_data ^ *src_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
      }
#else
      serial_path = 1;
#endif
    }
  } else {
    serial_path = 1;
  }
  if (serial_path) {
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data ^ *src_data;);
  }
#endif
}

void THTensor_(tpow)(THTensor *r_, real value, THTensor *t)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int serial_path = 0;
  if (r_Contig && tContig) {
    real *tp = THTensor_(data)(t);
    real *rp = THTensor_(data)(r_);
    int64_t i;
    #pragma omp parallel for if(r_Size > TH_OMP_OVERHEAD_THRESHOLD) private(i)
    for (i=0; i<r_Size; i++)
      rp[i] = THTensor_(powOne)(value, tp[i]);
  } else {
#if _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = THTensor_(powOne)(value, *t_data);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
    }
#else
    serial_path = 1;
#endif
  }
  if (serial_path) {
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = THTensor_(powOne)(value, *t_data););
  }
}

void THTensor_(addcmul)(THTensor *r_, THTensor *t, real value, THTensor *src1, THTensor *src2)
{
  if(r_ != t)
  {
    THTensor_(resizeAs)(r_, t);
    THTensor_(copy)(r_, t);
  }
  int64_t r_Size = THTensor_(nElement)(r_);
  int64_t src1Size = THTensor_(nElement)(src1);
  int64_t src2Size = THTensor_(nElement)(src2);
  int r_Contig = THTensor_(isContiguous)(r_);
  int src1Contig = THTensor_(isContiguous)(src1);
  int src2Contig = THTensor_(isContiguous)(src2);
  int serial_path = 0;
  if( (src1Size == src2Size) && (src1Size == r_Size) ){
#if _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
      TH_TENSOR_APPLY3_OMP(r_Size, r_Contig, src1Contig, src2Contig, real, r_, real, src1, real, src2, *r__data += value * *src1_data * *src2_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
    }
#else
    (void)r_Contig;
    (void)src1Contig;
    (void)src2Contig;
    serial_path = 1;
#endif
  } else {
    serial_path = 1;
  }
  if (serial_path) {
    TH_TENSOR_APPLY3(real, r_, real, src1, real, src2, *r__data += value * *src1_data * *src2_data;);
  }
}

void THTensor_(addcdiv)(THTensor *r_, THTensor *t, real value, THTensor *src1, THTensor *src2)
{
  if(r_ != t)
  {
    THTensor_(resizeAs)(r_, t);
    THTensor_(copy)(r_, t);
  }
  int64_t r_Size = THTensor_(nElement)(r_);
  int64_t src1Size = THTensor_(nElement)(src1);
  int64_t src2Size = THTensor_(nElement)(src2);
  int r_Contig = THTensor_(isContiguous)(r_);
  int src1Contig = THTensor_(isContiguous)(src1);
  int src2Contig = THTensor_(isContiguous)(src2);
  int serial_path = 0;
  if( (src1Size == src2Size) && (src1Size == r_Size) ){
#if _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
      TH_TENSOR_APPLY3_OMP(r_Size, r_Contig, src1Contig, src2Contig, real, r_, real, src1, real, src2, *r__data += value * *src1_data / *src2_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
    }
#else
    (void)r_Contig;
    (void)src1Contig;
    (void)src2Contig;
    serial_path = 1;
#endif
  } else {
    serial_path = 1;
  }
  if (serial_path) {
    TH_TENSOR_APPLY3(real, r_, real, src1, real, src2, *r__data += value * *src1_data / *src2_data;);
  }
}

void THTensor_(addmv)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *mat, THTensor *vec)
{
  if( (mat->dim() != 2) || (vec->dim() != 1) )
    THError("matrix and vector expected, got %dD, %dD",
      mat->dim(), vec->dim());

  if( mat->size(1) != vec->size(0) ) {
    THDescBuff bm = THTensor_(sizeDesc)(mat);
    THDescBuff bv = THTensor_(sizeDesc)(vec);
    THError("size mismatch, %s, %s", bm.str, bv.str);
  }

  if(t->dim() != 1)
    THError("vector expected, got t: %dD", t->dim());

  if(t->size(0) != mat->size(0)) {
    THDescBuff bt = THTensor_(sizeDesc)(t);
    THDescBuff bm = THTensor_(sizeDesc)(mat);
    THError("size mismatch, t: %s, mat: %s", bt.str, bm.str);
  }

  if(r_ != t)
  {
    THTensor_(resizeAs)(r_, t);
    THTensor_(copy)(r_, t);
  }

  // n == 1 || lda >= max(1, m)
  #define LDA_COND(M, N, LDA) ((N) == 1 || (LDA) >= THMax(1, (M)))

  if(mat->stride(0) == 1 && LDA_COND(mat->size(0), mat->size(1), mat->stride(1)))
  {
    THBlas_(gemv)('n', mat->size(0), mat->size(1),
                  alpha, THTensor_(data)(mat), mat->stride(1),
                  THTensor_(data)(vec), vec->stride(0),
                  beta, THTensor_(data)(r_), r_->stride(0));
  }
  else if(mat->stride(1) == 1 && LDA_COND(mat->size(1), mat->size(0), mat->stride(0)))
  {
    THBlas_(gemv)('t',  mat->size(1), mat->size(0),
                  alpha, THTensor_(data)(mat), mat->stride(0),
                  THTensor_(data)(vec), vec->stride(0),
                  beta, THTensor_(data)(r_), r_->stride(0));
  }
  else
  {
    THTensor *cmat = THTensor_(newContiguous)(mat);

    THBlas_(gemv)('t',  mat->size(1), mat->size(0),
                  alpha, THTensor_(data)(cmat), cmat->stride(0),
                  THTensor_(data)(vec), vec->stride(0),
                  beta, THTensor_(data)(r_), r_->stride(0));

    THTensor_(free)(cmat);
  }

  #undef LDA_COND
}

void THTensor_(match)(THTensor *r_, THTensor *m1, THTensor *m2, real gain)
{
  int64_t N1 = m1->size(0);
  int64_t N2 = m2->size(0);
  int64_t dim;
  real *m1_p;
  real *m2_p;
  real *r_p;
  int64_t i;

  THTensor_(resize2d)(r_, N1, N2);

  m1 = THTensor_(newContiguous)(m1);
  m2 = THTensor_(newContiguous)(m2);

  THTensor_(resize2d)(m1, N1, THTensor_(nElement)(m1) / N1);
  THTensor_(resize2d)(m2, N2, THTensor_(nElement)(m2) / N2);

  dim = m1->size(1);
  THArgCheck(m1->size(1) == m2->size(1), 3, "m1 and m2 must have the same inner vector dim");

  m1_p = THTensor_(data)(m1);
  m2_p = THTensor_(data)(m2);
  r_p = THTensor_(data)(r_);

#pragma omp parallel for private(i)
  for (i=0; i<N1; i++) {
    int64_t j,k;
    for (j=0; j<N2; j++) {
      real sum = 0;
      for (k=0; k<dim; k++) {
        real term = m1_p[ i*dim + k ] - m2_p[ j*dim + k ];
        sum += term*term;
      }
      r_p[ i*N2 + j ] = gain * sum;
    }
  }

  THTensor_(free)(m1);
  THTensor_(free)(m2);
}

void THTensor_(addmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *m1, THTensor *m2)
{
  char transpose_r, transpose_m1, transpose_m2;
  THTensor *r__, *m1_, *m2_;
  int free_m1 = 0;
  int free_m2 = 0;

  if( (m1->dim() != 2) || (m2->dim() != 2))
    THError("matrices expected, got %dD, %dD tensors", m1->dim(), m2->dim());

  if(m1->size(1) != m2->size(0)) {
    THDescBuff bm1 = THTensor_(sizeDesc)(m1);
    THDescBuff bm2 = THTensor_(sizeDesc)(m2);
    THError("size mismatch, m1: %s, m2: %s", bm1.str, bm2.str);
  }

  if( t->dim() != 2 )
    THError("matrix expected, got %dD tensor for t", t->dim());

  if( (t->size(0) != m1->size(0)) || (t->size(1) != m2->size(1)) ) {
    THDescBuff bt  = THTensor_(sizeDesc)(t);
    THDescBuff bm1 = THTensor_(sizeDesc)(m1);
    THDescBuff bm2 = THTensor_(sizeDesc)(m2);
    THError("size mismatch, t: %s, m1: %s, m2: %s", bt.str, bm1.str, bm2.str);
  }

  if(t != r_)
  {
    THTensor_(resizeAs)(r_, t);
    if (beta != 0.0) {
      THTensor_(copy)(r_, t);
    }
  }

  // n == 1 || ldc >= max(1, m)
  #define LDC_COND(M, N, LDC) ((N) == 1 || (LDC) >= THMax(1, M))

  /* r_ */
  if(r_->stride(0) == 1 &&
     LDC_COND(r_->size(0), r_->size(1), r_->stride(1)))
  {
    transpose_r = 'n';
    r__ = r_;
  }
  else if(r_->stride(1) == 1 &&
          LDC_COND(r_->size(1), r_->size(0), r_->stride(0)))
  {
    THTensor *swap = m2;
    m2 = m1;
    m1 = swap;
    transpose_r = 't';
    r__ = r_;
  }
  else
  {
    transpose_r = 'n';
    // make r__ FORTRAN contiguous
    THTensor *transp_r_ = THTensor_(newTranspose)(r_, 0, 1);
    r__ = THTensor_(newClone)(transp_r_);
    THTensor_(free)(transp_r_);
    THTensor_(transpose)(r__, NULL, 0, 1);
  }

  #undef LDC_COND

  int64_t m = r__->size((transpose_r == 'n' ? 0 : 1));
  int64_t n = r__->size((transpose_r == 'n' ? 1 : 0));
  int64_t k = m1->size((transpose_r == 'n' ? 1 : 0));
  int64_t ldr__ = r__->stride((transpose_r == 'n' ? 1 : 0));

  /* m1 */
  /* Need ldm1_ >= max(1, (transpose_m1 == 'n' ? m : k)) */
  if(m1->stride((transpose_r == 'n' ? 0 : 1)) == 1 &&
     m1->stride((transpose_r == 'n' ? 1 : 0)) >= THMax(1, m))
  {
    transpose_m1 = 'n';
    m1_ = m1;
  }
  else if(m1->stride((transpose_r == 'n' ? 1 : 0)) == 1 &&
          m1->stride((transpose_r == 'n' ? 0 : 1)) >= THMax(1, k))
  {
    transpose_m1 = 't';
    m1_ = m1;
  }
  else
  {
    transpose_m1 = (transpose_r == 'n' ? 't' : 'n');
    m1_ = THTensor_(newContiguous)(m1);
    free_m1 = 1;
  }

  /* m2 */
  /* Need ldm2_ >= max(1, (transpose_m2 == 'n' ? k : n)) */
  if(m2->stride((transpose_r == 'n' ? 0 : 1)) == 1 &&
     m2->stride((transpose_r == 'n' ? 1 : 0)) >= THMax(1, k))
  {
    transpose_m2 = 'n';
    m2_ = m2;
  }
  else if(m2->stride((transpose_r == 'n' ? 1 : 0)) == 1 &&
          m2->stride((transpose_r == 'n' ? 0 : 1)) >= THMax(1, n))
  {
    transpose_m2 = 't';
    m2_ = m2;
  }
  else
  {
    transpose_m2 = (transpose_r == 'n' ? 't' : 'n');
    m2_ = THTensor_(newContiguous)(m2);
    free_m2 = 1;
  }

  int64_t ldm1_ = (transpose_m1 == 'n' ? m1_->stride((transpose_r == 'n' ? 1 : 0)) : m1_->stride((transpose_r == 'n' ? 0 : 1)));
  int64_t ldm2_ = (transpose_m2 == 'n' ? m2_->stride((transpose_r == 'n' ? 1 : 0)) : m2_->stride((transpose_r == 'n' ? 0 : 1)));

#pragma omp critical(blasgemm)
  /* do the operation */
  THBlas_(gemm)(transpose_m1,
                transpose_m2,
                m,
                n,
                k,
                alpha,
                THTensor_(data)(m1_),
                ldm1_,
                THTensor_(data)(m2_),
                ldm2_,
                beta,
                THTensor_(data)(r__),
                ldr__);

  /* free intermediate variables */
  if(free_m1)
    THTensor_(free)(m1_);

  if(free_m2)
    THTensor_(free)(m2_);

  if(r__ != r_)
    THTensor_(freeCopyTo)(r__, r_);
}

void THTensor_(addr)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *vec1, THTensor *vec2)
{
  if( (vec1->dim() != 1) || (vec2->dim() != 1) )
    THError("vector and vector expected, got %dD, %dD tensors",
        vec1->dim(), vec2->dim());

  if(t->dim() != 2)
    THError("expected matrix, got %dD tensor for t", t->dim());

  if( (t->size(0) != vec1->size(0)) || (t->size(1) != vec2->size(0)) ) {
    THDescBuff bt  = THTensor_(sizeDesc)(t);
    THDescBuff bv1 = THTensor_(sizeDesc)(vec1);
    THDescBuff bv2 = THTensor_(sizeDesc)(vec2);
    THError("size mismatch, t: %s, vec1: %s, vec2: %s", bt.str, bv1.str, bv2.str);
  }

  if(r_ != t)
  {
    THTensor_(resizeAs)(r_, t);
    THTensor_(copy)(r_, t);
  }

  if(beta == 0) {
    THTensor_(zero)(r_);
  }
  else if(beta != 1)
    THTensor_(mul)(r_, r_, beta);

  // n == 1 || lda >= max(1, m)
  #define LDA_COND(M, N, LDA) ((N) == 1 || (LDA) >= THMax(1, (M)))

  if(r_->stride(0) == 1 && LDA_COND(vec1->size(0), vec2->size(0), r_->stride(1)))
  {
    THBlas_(ger)(vec1->size(0), vec2->size(0),
                 alpha, THTensor_(data)(vec1), vec1->stride(0),
                 THTensor_(data)(vec2), vec2->stride(0),
                 THTensor_(data)(r_), r_->stride(1));
  }
  else if(r_->stride(1) == 1 && LDA_COND(vec2->size(0), vec1->size(0), r_->stride(0)))
  {
    THBlas_(ger)(vec2->size(0), vec1->size(0),
                 alpha, THTensor_(data)(vec2), vec2->stride(0),
                 THTensor_(data)(vec1), vec1->stride(0),
                 THTensor_(data)(r_), r_->stride(0));
  }
  else
  {
    THTensor *cr = THTensor_(newClone)(r_);

    THBlas_(ger)(vec2->size(0), vec1->size(0),
                 alpha, THTensor_(data)(vec2), vec2->stride(0),
                 THTensor_(data)(vec1), vec1->stride(0),
                 THTensor_(data)(cr), cr->stride(0));

    THTensor_(freeCopyTo)(cr, r_);
  }

  #undef LDA_COND
}

void THTensor_(addbmm)(THTensor *result, real beta, THTensor *t, real alpha, THTensor *batch1, THTensor *batch2)
{
  int64_t batch;

  THArgCheck(THTensor_(nDimension)(batch1) == 3, 1, "expected 3D tensor");
  THArgCheck(THTensor_(nDimension)(batch2) == 3, 2, "expected 3D tensor");
  THArgCheck(THTensor_(size)(batch1, 0) == THTensor_(size)(batch2, 0), 2,
             "equal number of batches expected, got %d, %d",
             THTensor_(size)(batch1, 0), THTensor_(size)(batch2, 0));
  THArgCheck(THTensor_(size)(batch1, 2) == THTensor_(size)(batch2, 1), 2,
             "wrong matrix size, batch1: %dx%d, batch2: %dx%d",
             THTensor_(size)(batch1, 1), THTensor_(size)(batch1,2),
             THTensor_(size)(batch2, 1), THTensor_(size)(batch2,2));

  int64_t dim1 = THTensor_(size)(batch1, 1);
  int64_t dim2 = THTensor_(size)(batch2, 2);
  THArgCheck(THTensor_(size)(t, 0) == dim1, 1, "output tensor of incorrect size");
  THArgCheck(THTensor_(size)(t, 1) == dim2, 1, "output tensor of incorrect size");

  if (t != result) {
    THTensor_(resizeAs)(result, t);
    if (beta != 0.0) {
      THTensor_(copy)(result, t);
    }
  }

  THTensor *matrix1 = THTensor_(new)();
  THTensor *matrix2 = THTensor_(new)();

  for (batch = 0; batch < THTensor_(size)(batch1, 0); ++batch) {
    THTensor_(select)(matrix1, batch1, 0, batch);
    THTensor_(select)(matrix2, batch2, 0, batch);

    THTensor_(addmm)(result, beta, result, alpha, matrix1, matrix2);
    beta = 1; // accumulate output once
  }

  THTensor_(free)(matrix1);
  THTensor_(free)(matrix2);
}

#endif /* TH_GENERIC_FILE */
