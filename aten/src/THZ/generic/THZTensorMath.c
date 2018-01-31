#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZTensorMath.c"
#else

#include "THZTypeMacros.h"

#ifndef NAN
  #define NAN (nan(NULL))
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define THZ_OMP_OVERHEAD_THRESHOLD 100000

#ifdef _OPENMP

#ifndef _WIN32
#define PRAGMA(P) _Pragma(#P)
#else
#define PRAGMA(P) __pragma(P)
#endif

#define THZ_TENSOR_APPLY_CONTIG(TYPE, TENSOR, CODE) \
{ \
  ptrdiff_t THZ_TENSOR_size = THZTensor_(nElement)(TENSOR); \
  PRAGMA(omp parallel if (THZ_TENSOR_size > THZ_OMP_OVERHEAD_THRESHOLD)) \
  { \
    size_t num_threads = omp_get_num_threads(); \
    size_t tid = omp_get_thread_num(); \
    ptrdiff_t THZ_TENSOR_offset = tid * (THZ_TENSOR_size / num_threads); \
    ptrdiff_t THZ_TENSOR_end = tid == num_threads - 1 ? THZ_TENSOR_size : \
      THZ_TENSOR_offset + THZ_TENSOR_size / num_threads; \
    ptrdiff_t TENSOR##_len = THZ_TENSOR_end - THZ_TENSOR_offset; \
    TYPE *TENSOR##_data = THZTensor_(data)(TENSOR) + THZ_TENSOR_offset; \
    CODE \
  } \
}

#define THZ_TENSOR_APPLY_CONTIG_PART(TYPE, TENSOR, CODE) \
{ \
  ptrdiff_t THZ_TENSOR_size = THZPartTensor_(nElement)(TENSOR); \
  PRAGMA(omp parallel if (THZ_TENSOR_size > THZ_OMP_OVERHEAD_THRESHOLD)) \
  { \
    size_t num_threads = omp_get_num_threads(); \
    size_t tid = omp_get_thread_num(); \
    ptrdiff_t THZ_TENSOR_offset = tid * (THZ_TENSOR_size / num_threads); \
    ptrdiff_t THZ_TENSOR_end = tid == num_threads - 1 ? THZ_TENSOR_size : \
      THZ_TENSOR_offset + THZ_TENSOR_size / num_threads; \
    ptrdiff_t TENSOR##_len = THZ_TENSOR_end - THZ_TENSOR_offset; \
    TYPE *TENSOR##_data = THZPartTensor_(data)(TENSOR) + THZ_TENSOR_offset; \
    CODE \
  } \
}
#else
#define THZ_TENSOR_APPLY_CONTIG(TYPE, TENSOR, CODE) \
{ \
  TYPE *TENSOR##_data = THZTensor_(data)(TENSOR); \
  ptrdiff_t TENSOR##_len = THZTensor_(nElement)(TENSOR); \
  CODE \
}

#define THZ_TENSOR_APPLY_CONTIG_PART(TYPE, TENSOR, CODE) \
{ \
  TYPE *TENSOR##_data = THZPartTensor_(data)(TENSOR); \
  ptrdiff_t TENSOR##_len = THZPartTensor_(nElement)(TENSOR); \
  CODE \
}
#endif

#ifdef _OPENMP
#define THZ_TENSOR_APPLY2_CONTIG(TYPE1, TENSOR1, TYPE2, TENSOR2, CODE) \
{ \
  ptrdiff_t THZ_TENSOR_size = THZTensor_(nElement)(TENSOR1); \
  PRAGMA(omp parallel if (THZ_TENSOR_size > THZ_OMP_OVERHEAD_THRESHOLD)) \
  { \
    size_t num_threads = omp_get_num_threads(); \
    size_t tid = omp_get_thread_num(); \
    ptrdiff_t THZ_TENSOR_offset = tid * (THZ_TENSOR_size / num_threads); \
    ptrdiff_t THZ_TENSOR_end = tid == num_threads - 1 ? THZ_TENSOR_size : \
      THZ_TENSOR_offset + THZ_TENSOR_size / num_threads; \
    ptrdiff_t TENSOR1##_len = THZ_TENSOR_end - THZ_TENSOR_offset; \
    TYPE1 *TENSOR1##_data = THZTensor_(data)(TENSOR1) + THZ_TENSOR_offset; \
    TYPE2 *TENSOR2##_data = THZTensor_(data)(TENSOR2) + THZ_TENSOR_offset; \
    CODE \
  } \
}

#define THZ_TENSOR_APPLY2_CONTIG_PART(TYPE1, TENSOR1, TYPE2, TENSOR2, CODE) \
{ \
  ptrdiff_t THZ_TENSOR_size = THZPartTensor_(nElement)(TENSOR1); \
  PRAGMA(omp parallel if (THZ_TENSOR_size > THZ_OMP_OVERHEAD_THRESHOLD)) \
  { \
    size_t num_threads = omp_get_num_threads(); \
    size_t tid = omp_get_thread_num(); \
    ptrdiff_t THZ_TENSOR_offset = tid * (THZ_TENSOR_size / num_threads); \
    ptrdiff_t THZ_TENSOR_end = tid == num_threads - 1 ? THZ_TENSOR_size : \
      THZ_TENSOR_offset + THZ_TENSOR_size / num_threads; \
    ptrdiff_t TENSOR1##_len = THZ_TENSOR_end - THZ_TENSOR_offset; \
    TYPE1 *TENSOR1##_data = THZPartTensor_(data)(TENSOR1) + THZ_TENSOR_offset; \
    TYPE2 *TENSOR2##_data = THZTensor_(data)(TENSOR2) + THZ_TENSOR_offset; \
    CODE \
  } \
}
#else
#define THZ_TENSOR_APPLY2_CONTIG(TYPE1, TENSOR1, TYPE2, TENSOR2, CODE) \
{ \
  TYPE1 *TENSOR1##_data = THZTensor_(data)(TENSOR1); \
  TYPE2 *TENSOR2##_data = THZTensor_(data)(TENSOR2); \
  ptrdiff_t TENSOR1##_len = THZTensor_(nElement)(TENSOR1); \
  CODE \
}

#define THZ_TENSOR_APPLY2_CONTIG_PART(TYPE1, TENSOR1, TYPE2, TENSOR2, CODE) \
{ \
  TYPE1 *TENSOR1##_data = THZPartTensor_(data)(TENSOR1); \
  TYPE2 *TENSOR2##_data = THZTensor_(data)(TENSOR2); \
  ptrdiff_t TENSOR1##_len = THZPartTensor_(nElement)(TENSOR1); \
  CODE \
}
#endif

#ifdef _OPENMP
#define THZ_TENSOR_APPLY3_CONTIG(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, CODE) \
{ \
  ptrdiff_t THZ_TENSOR_size = THZTensor_(nElement)(TENSOR1); \
  PRAGMA(omp parallel if (THZ_TENSOR_size > THZ_OMP_OVERHEAD_THRESHOLD)) \
  { \
    size_t num_threads = omp_get_num_threads(); \
    size_t tid = omp_get_thread_num(); \
    ptrdiff_t THZ_TENSOR_offset = tid * (THZ_TENSOR_size / num_threads); \
    ptrdiff_t THZ_TENSOR_end = tid == num_threads - 1 ? THZ_TENSOR_size : \
      THZ_TENSOR_offset + THZ_TENSOR_size / num_threads; \
    ptrdiff_t TENSOR1##_len = THZ_TENSOR_end - THZ_TENSOR_offset; \
    TYPE1 *TENSOR1##_data = THZTensor_(data)(TENSOR1) + THZ_TENSOR_offset; \
    TYPE2 *TENSOR2##_data = THZTensor_(data)(TENSOR2) + THZ_TENSOR_offset; \
    TYPE3 *TENSOR3##_data = THZTensor_(data)(TENSOR3) + THZ_TENSOR_offset; \
    CODE \
  } \
}
#else
#define THZ_TENSOR_APPLY3_CONTIG(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, CODE) \
{ \
  TYPE1 *TENSOR1##_data = THZTensor_(data)(TENSOR1); \
  TYPE2 *TENSOR2##_data = THZTensor_(data)(TENSOR2); \
  TYPE3 *TENSOR3##_data = THZTensor_(data)(TENSOR3); \
  ptrdiff_t TENSOR1##_len = THZTensor_(nElement)(TENSOR1); \
  CODE \
}
#endif

void THZTensor_(fill)(THZTensor *r_, ntype value)
{
  if (THZTensor_(isContiguous)(r_) || THZTensor_(isTransposed)(r_)) {
    THZ_TENSOR_APPLY_CONTIG(ntype, r_, THZVector_(fill)(r__data, value, r__len););
  } else {
    TH_TENSOR_APPLY(ntype, r_,
      if (r__stride == 1) {
        THZVector_(fill)(r__data, value, r__size);
  r__i = r__size;
  r__data += r__stride * r__size;
  break;
      } else {
        *r__data = value;
      }
      );
  }
}

void THZTensor_(zero)(THZTensor *r_)
{
  THZTensor_(fill)(r_, 0);
}

void THZTensor_(maskedFill)(THZTensor *tensor, THByteTensor *mask, ntype value)
{
  TH_TENSOR_APPLY2(ntype, tensor, unsigned char, mask,
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

void THZTensor_(maskedCopy)(THZTensor *tensor, THByteTensor *mask, THZTensor* src )
{
  THZTensor *srct = THZTensor_(newContiguous)(src);
  ntype *src_data = THZTensor_(data)(srct);
  ptrdiff_t cntr = 0;
  ptrdiff_t nelem = THZTensor_(nElement)(srct);
  if (THZTensor_(nElement)(tensor) != THByteTensor_nElement(mask))
  {
    THZTensor_(free)(srct);
    THError("Number of elements of destination tensor != Number of elements in mask");
  }
  TH_TENSOR_APPLY2(ntype, tensor, unsigned char, mask,
                   if (*mask_data > 1)
                   {
                     THZTensor_(free)(srct);
                     THFree(mask_counter);
                     THFree(tensor_counter);
                     THError("Mask tensor can take 0 and 1 values only");
                   }
                   else if (*mask_data == 1)
                   {
                     if (cntr == nelem)
                     {
                       THZTensor_(free)(srct);
                       THFree(mask_counter);
                       THFree(tensor_counter);
                       THError("Number of elements of src < number of ones in mask");
                     }
                     *tensor_data = *src_data;
                     src_data++;
                     cntr++;
                   });
  THZTensor_(free)(srct);
}

void THZTensor_(maskedSelect)(THZTensor *tensor, THZTensor *src, THByteTensor *mask)
{
  ptrdiff_t numel = THByteTensor_sumall(mask);
  ntype *tensor_data;

#ifdef DEBUG
  THAssert(numel <= LONG_MAX);
#endif
  THZTensor_(resize1d)(tensor,numel);
  tensor_data = THZTensor_(data)(tensor);
  TH_TENSOR_APPLY2(ntype, src, unsigned char, mask,
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
void THZTensor_(nonzero)(THLongTensor *subscript, THZTensor *tensor)
{
  ptrdiff_t numel = 0;
  int64_t *subscript_data;
  int64_t i = 0;
  int64_t dim;
  int64_t div = 1;
#define IS_NONZERO(val) ((val)!=0)

  /* First Pass to determine size of subscripts */
  TH_TENSOR_APPLY(ntype, tensor,
                  if IS_NONZERO(*tensor_data) {
                    ++numel;
                  });
#ifdef DEBUG
  THAssert(numel <= LONG_MAX);
#endif
  THLongTensor_resize2d(subscript, numel, tensor->nDimension);

  /* Second pass populates subscripts */
  subscript_data = THLongTensor_data(subscript);
  TH_TENSOR_APPLY(ntype, tensor,
                  if IS_NONZERO(*tensor_data) {
                    div = 1;

                    for (dim = tensor->nDimension - 1; dim >= 0; dim--) {
                      *(subscript_data + dim) = (i/div) % tensor->size[dim];
                      div *= tensor->size[dim];
                    }

                    subscript_data += tensor->nDimension;
                  }
                  ++i;);
}

void THZTensor_(indexSelect)(THZTensor *tensor, THZTensor *src, int dim, THLongTensor *index)
{
  ptrdiff_t i, numel;
  THLongStorage *newSize;
  THZTensor *tSlice, *sSlice;
  int64_t *index_data;
  ntype *tensor_data, *src_data;

  THArgCheck(index->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < src->nDimension, 4,"Indexing dim %d is out of bounds of tensor", dim + TH_INDEX_BASE);
  THArgCheck(src->nDimension > 0,2,"Source tensor is empty");

  numel = THLongTensor_nElement(index);

  newSize = THLongStorage_newWithSize(src->nDimension);
  THLongStorage_rawCopy(newSize,src->size);
#ifdef DEBUG
  THAssert(numel <= LONG_MAX);
#endif
  newSize->data[dim] = numel;
  THZTensor_(resize)(tensor,newSize,NULL);
  THLongStorage_free(newSize);

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);

  if (dim == 0 && THZTensor_(isContiguous)(src) && THZTensor_(isContiguous)(tensor))
  {
    tensor_data = THZTensor_(data)(tensor);
    src_data = THZTensor_(data)(src);
    ptrdiff_t rowsize = THZTensor_(nElement)(src) / src->size[0];

    // check that the indices are within range
    int64_t max = src->size[0] - 1 + TH_INDEX_BASE;
    for (i=0; i<numel; i++) {
      if (index_data[i] < TH_INDEX_BASE || index_data[i] > max) {
        THLongTensor_free(index);
        THError("index out of range");
      }
    }

    if (src->nDimension == 1) {
      #pragma omp parallel for if(numel > THZ_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<numel; i++)
        tensor_data[i] = src_data[index_data[i] - TH_INDEX_BASE];
    } else {
      #pragma omp parallel for if(numel*rowsize > THZ_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<numel; i++)
        memcpy(tensor_data + i*rowsize, src_data + (index_data[i] - TH_INDEX_BASE)*rowsize, rowsize*sizeof(ntype));
    }
  }
  else if (src->nDimension == 1)
  {
    for (i=0; i<numel; i++)
      THZTensor_(set1d)(tensor,i,THZTensor_(get1d)(src,index_data[i] - TH_INDEX_BASE));
  }
  else
  {
    for (i=0; i<numel; i++)
    {
      tSlice = THZTensor_(new)();
      sSlice = THZTensor_(new)();
      THZTensor_(select)(tSlice, tensor, dim, i);
      THZTensor_(select)(sSlice, src, dim, index_data[i] - TH_INDEX_BASE);
      THZTensor_(copy)(tSlice, sSlice);
      THZTensor_(free)(tSlice);
      THZTensor_(free)(sSlice);
    }
  }

  THLongTensor_free(index);
}

void THZTensor_(indexCopy)(THZTensor *tensor, int dim, THLongTensor *index, THZTensor *src)
{
  ptrdiff_t i, numel;
  THZTensor *tSlice, *sSlice;
  int64_t *index_data;

  numel = THLongTensor_nElement(index);
  THArgCheck(index->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < src->nDimension, 4, "Indexing dim %d is out of bounds of tensor", dim + TH_INDEX_BASE);
  THArgCheck(numel == src->size[dim],4,"Number of indices should be equal to source:size(dim)");

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);

  if (tensor->nDimension > 1 )
  {
    tSlice = THZTensor_(new)();
    sSlice = THZTensor_(new)();

    for (i=0; i<numel; i++)
    {
      THZTensor_(select)(tSlice, tensor, dim, index_data[i] - TH_INDEX_BASE);
      THZTensor_(select)(sSlice, src, dim, i);
      THZTensor_(copy)(tSlice, sSlice);
    }

    THZTensor_(free)(tSlice);
    THZTensor_(free)(sSlice);
  }
  else
  {
    for (i=0; i<numel; i++)
    {
      THZTensor_(set1d)(tensor, index_data[i] - TH_INDEX_BASE, THZTensor_(get1d)(src,i));
    }
  }
  THLongTensor_free(index);
}

static ptrdiff_t THZTensor_(dataOffset)(THZTensor* tensor, ptrdiff_t linearIndex) {
  int64_t *size = tensor->size;
  int64_t *stride = tensor->stride;
  int nDim = tensor->nDimension;
  ptrdiff_t dataOffset = 0;
  for (int i = nDim - 1; i >= 0; i--) {
    dataOffset += (linearIndex % size[i]) * stride[i];
    linearIndex /= size[i];
  }
  return dataOffset;
}

static int64_t THZTensor_(wrapLinearIndex)(int64_t linearIndex, int64_t numel) {
  THArgCheck(linearIndex < numel && linearIndex >= -numel, 2, "out of range: %d out of %d", (int)linearIndex, (int)numel);
  return linearIndex < 0 ? linearIndex + numel : linearIndex;
}

void THZTensor_(take)(THZTensor *r_, THZTensor *src, THLongTensor *index)
{
  THZTensor_(resizeNd)(r_, index->nDimension, index->size, NULL);
  THZTensor* dst = THZTensor_(newContiguous)(r_);

  index = THLongTensor_newContiguous(index);
  int64_t* index_data = THLongTensor_data(index);
  ptrdiff_t srcElements = THZTensor_(nElement)(src);
  ntype* src_data = THZTensor_(data)(src);
  ntype* dst_data = THZTensor_(data)(dst);

  ptrdiff_t nIndices = THLongTensor_nElement(index);
  if (THZTensor_(isContiguous)(src)) {
    ptrdiff_t i;
    #pragma omp parallel for if(nIndices > THZ_OMP_OVERHEAD_THRESHOLD) private(i)
    for (i = 0; i < nIndices; i++) {
      int64_t linearIndex = THZTensor_(wrapLinearIndex)(index_data[i], srcElements);
      dst_data[i] = src_data[linearIndex];
    }
  } else {
    ptrdiff_t i;
    #pragma omp parallel for if(nIndices > THZ_OMP_OVERHEAD_THRESHOLD) private(i)
    for (i = 0; i < nIndices; i++) {
      int64_t linearIndex = THZTensor_(wrapLinearIndex)(index_data[i], srcElements);
      int64_t dataOffset = THZTensor_(dataOffset)(src, linearIndex);
      dst_data[i] = src_data[dataOffset];
    }
  }

  THLongTensor_free(index);
  THZTensor_(freeCopyTo)(dst, r_);
}

void THZTensor_(put)(THZTensor *tensor, THLongTensor *index, THZTensor *src, int accumulate)
{
  THArgCheck(THLongTensor_nElement(index) == THZTensor_(nElement)(src), 3,
    "src should have the same number of elements as index");

  index = THLongTensor_newContiguous(index);
  src = THZTensor_(newContiguous)(src);
  ntype* data = THZTensor_(data)(tensor);
  ptrdiff_t numel = THZTensor_(nElement)(tensor);
  int is_contiguous = THZTensor_(isContiguous)(tensor);

  TH_TENSOR_APPLY2(int64_t, index, ntype, src,
    int64_t linearIndex = THZTensor_(wrapLinearIndex)(*index_data, numel);
    int64_t dataOffset = is_contiguous ? linearIndex : THZTensor_(dataOffset)(tensor, linearIndex);
    if (accumulate) {
      data[dataOffset] += *src_data;
    } else {
      data[dataOffset] = *src_data;
    }
  );

  THZTensor_(free)(src);
  THLongTensor_free(index);
}

void THZTensor_(indexAdd)(THZTensor *tensor, int dim, THLongTensor *index, THZTensor *src)
{
  ptrdiff_t i, numel;
  THZTensor *tSlice, *sSlice;
  int64_t *index_data;

  numel = THLongTensor_nElement(index);
  THArgCheck(index->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < src->nDimension, 4,"Indexing dim %d is out of bounds of tensor", dim + TH_INDEX_BASE);
  THArgCheck(numel == src->size[dim],4,"Number of indices should be equal to source:size(dim)");

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);

  if (tensor->nDimension > 1)
  {
    tSlice = THZTensor_(new)();
    sSlice = THZTensor_(new)();

    for (i=0; i<numel; i++)
    {
      THZTensor_(select)(tSlice, tensor, dim, index_data[i] - TH_INDEX_BASE);
      THZTensor_(select)(sSlice, src, dim, i);
      THZTensor_(cadd)(tSlice, tSlice, 1.0, sSlice);
    }

    THZTensor_(free)(tSlice);
    THZTensor_(free)(sSlice);
  }
  else
  {
    for (i=0; i<numel; i++)
    {
      THZTensor_(set1d)(tensor,
              index_data[i] - TH_INDEX_BASE,
              THZTensor_(get1d)(src,i) + THZTensor_(get1d)(tensor,index_data[i] - TH_INDEX_BASE));
    }
  }
  THLongTensor_free(index);
}

void THZTensor_(indexFill)(THZTensor *tensor, int dim, THLongTensor *index, ntype val)
{
  ptrdiff_t i, numel;
  THZTensor *tSlice;
  int64_t *index_data;

  numel = THLongTensor_nElement(index);
  THArgCheck(index->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < tensor->nDimension, 4,"Indexing dim %d is out of bounds of tensor", dim + TH_INDEX_BASE);

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);

  for (i=0; i<numel; i++)
  {
    if (tensor->nDimension > 1)
    {
      tSlice = THZTensor_(new)();
      THZTensor_(select)(tSlice, tensor,dim,index_data[i] - TH_INDEX_BASE);
      THZTensor_(fill)(tSlice, val);
      THZTensor_(free)(tSlice);
    }
    else
    {
      THZTensor_(set1d)(tensor, index_data[i] - TH_INDEX_BASE, val);
    }
  }
  THLongTensor_free(index);
}

void THZTensor_(gather)(THZTensor *tensor, THZTensor *src, int dim, THLongTensor *index)
{
  int64_t elems_per_row, i, idx;

  THArgCheck(THLongTensor_nDimension(index) == THZTensor_(nDimension)(src), 4,
             "Index tensor must have same dimensions as input tensor");
  THArgCheck(dim >= 0 && dim < THZTensor_(nDimension)(tensor), 3,
             "Index dimension is out of bounds");
  THArgCheck(THZTensor_(nDimension)(src) == THZTensor_(nDimension)(tensor), 2,
             "Input tensor must have same dimensions as output tensor");

  elems_per_row = THLongTensor_size(index, dim);

  TH_TENSOR_DIM_APPLY3(ntype, tensor, ntype, src, int64_t, index, dim,
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

void THZTensor_(scatter)(THZTensor *tensor, int dim, THLongTensor *index, THZTensor *src)
{
  int64_t elems_per_row, i, idx;

  THArgCheck(dim < THZTensor_(nDimension)(tensor), 2, "Index dimension is out of bounds");
  THArgCheck(THLongTensor_nDimension(index) == THZTensor_(nDimension)(tensor), 3,
             "Index tensor must have same dimensions as output tensor");
  THArgCheck(THZTensor_(nDimension)(src) == THZTensor_(nDimension)(tensor), 4,
             "Input tensor must have same dimensions as output tensor");

  elems_per_row = THLongTensor_size(index, dim);

  // Assumes TENSOR1 is ntype
  //         TENSOR2 is src
  //         TENSOR3 is index
  // Tests:
  //   1. index->size[d] <= src->size[d] for all d
  //   2. index->size[d] <= real->size[d] for all d != dim
  #define TH_TENSOR_DIM_APPLY3_SIZE_SCATTER(TENSOR1, TENSOR2, TENSOR3, DIMENSION) \
  { \
    int shape_check_flag = 0; \
    for(TH_TENSOR_DIM_APPLY_i = 0; TH_TENSOR_DIM_APPLY_i < TENSOR1->nDimension; TH_TENSOR_DIM_APPLY_i++) \
    { \
      int64_t TENSOR3##_dim_size = TENSOR3->size[TH_TENSOR_DIM_APPLY_i]; \
      if (TH_TENSOR_DIM_APPLY_i != DIMENSION) { \
        if (TENSOR3##_dim_size > TENSOR1->size[TH_TENSOR_DIM_APPLY_i]) { \
          shape_check_flag = 1; \
          break; \
        } \
      } \
      if (TENSOR3##_dim_size > TENSOR2->size[TH_TENSOR_DIM_APPLY_i]) { \
        shape_check_flag = 1; \
        break; \
      } \
    } \
    if (shape_check_flag == 1) { \
      THDescBuff T1buff = _THSizeDesc(TENSOR1->size, TENSOR1->nDimension); \
      THDescBuff T2buff = _THSizeDesc(TENSOR2->size, TENSOR2->nDimension); \
      THDescBuff T3buff = _THSizeDesc(TENSOR3->size, TENSOR3->nDimension); \
      THError("Expected %s %s to be smaller size than %s %s and to be smaller than %s %s apart from dimension %d", \
              #TENSOR3, T3buff.str, #TENSOR2, T2buff.str, #TENSOR1, T1buff.str, DIMENSION); \
    } \
  }

  TH_TENSOR_DIM_APPLY3(ntype, tensor, ntype, src, int64_t, index, dim,
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

void THZTensor_(scatterAdd)(THZTensor *tensor, int dim, THLongTensor *index, THZTensor *src)
{
  int64_t elems_per_row, i, idx;

  THArgCheck(dim < THZTensor_(nDimension)(tensor), 2, "Index dimension is out of bounds");
  THArgCheck(THLongTensor_nDimension(index) == THZTensor_(nDimension)(tensor), 3,
             "Index tensor must have same dimensions as output tensor");
  THArgCheck(THZTensor_(nDimension)(src) == THZTensor_(nDimension)(tensor), 4,
             "Input tensor must have same dimensions as output tensor");

  elems_per_row = THLongTensor_size(index, dim);

  TH_TENSOR_DIM_APPLY3(ntype, tensor, ntype, src, int64_t, index, dim,
                       TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
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

void THZTensor_(scatterFill)(THZTensor *tensor, int dim, THLongTensor *index, ntype val)
{
  int64_t elems_per_row, i, idx;

  THArgCheck(dim < THZTensor_(nDimension)(tensor), 2, "Index dimension is out of bounds");
  THArgCheck(THLongTensor_nDimension(index) == THZTensor_(nDimension)(tensor), 3,
             "Index tensor must have same dimensions as output tensor");

  elems_per_row = THLongTensor_size(index, dim);

  TH_TENSOR_DIM_APPLY2(ntype, tensor, int64_t, index, dim,
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

accntype THZTensor_(dot)(THZTensor *tensor, THZTensor *src)
{
  accntype sum = 0;
  /* we use a trick here. careful with that. */
  TH_TENSOR_APPLY2(ntype, tensor, ntype, src,
                   int64_t sz = (tensor_size-tensor_i < src_size-src_i ? tensor_size-tensor_i : src_size-src_i);
                   sum += THZBlas_(dot)(sz, src_data, src_stride, tensor_data, tensor_stride);
                   tensor_i += sz;
                   src_i += sz;
                   tensor_data += sz*tensor_stride;
                   src_data += sz*src_stride;
                   break;);
  return sum;
}

ntype THZTensor_(minall)(THZTensor *tensor)
{
  THError("minall is not supported for complex type");
}

ntype THZTensor_(maxall)(THZTensor *tensor)
{
  THError("maxall is not supported for complex type");
}

ntype THZTensor_(medianall)(THZTensor *tensor)
{
  THError("medianall is not supported for complex type");
}

accntype THZTensor_(sumall)(THZTensor *tensor)
{
  accntype sum = 0;
  TH_TENSOR_APPLY(ntype, tensor, sum += *tensor_data;);
  return sum;
}

accntype THZTensor_(prodall)(THZTensor *tensor)
{
  accntype prod = 1;
  TH_TENSOR_APPLY(ntype, tensor, prod *= *tensor_data;);
  return prod;
}

void THZTensor_(add)(THZTensor *r_, THZTensor *t, ntype value)
{
  THZTensor_(resizeAs)(r_, t);
  if (THZTensor_(isContiguous)(r_) && THZTensor_(isContiguous)(t) && THZTensor_(nElement)(r_) == THZTensor_(nElement)(t)) {
    THZ_TENSOR_APPLY2_CONTIG(ntype, r_, ntype, t, THZVector_(adds)(r__data, t_data, value, r__len););
  } else {
    TH_TENSOR_APPLY2(ntype, r_, ntype, t, *r__data = *t_data + value;);
  }
}

void THZTensor_(sub)(THZTensor *r_, THZTensor *t, ntype value)
{
  THZTensor_(add)(r_, t, -value);
}

void THZTensor_(add_scaled)(THZTensor *r_, THZTensor *t, ntype value, ntype alpha)
{
  THZTensor_(add)(r_, t, value * alpha);
}

void THZTensor_(sub_scaled)(THZTensor *r_, THZTensor *t, ntype value, ntype alpha)
{
  THZTensor_(add)(r_, t, -value * alpha);
}

void THZTensor_(mul)(THZTensor *r_, THZTensor *t, ntype value)
{
  THZTensor_(resizeAs)(r_, t);
  if (THZTensor_(isContiguous)(r_) && THZTensor_(isContiguous)(t) && THZTensor_(nElement)(r_) == THZTensor_(nElement)(t)) {
    THZ_TENSOR_APPLY2_CONTIG(ntype, r_, ntype, t, THZVector_(muls)(r__data, t_data, value, r__len););
  } else {
    TH_TENSOR_APPLY2(ntype, r_, ntype, t, *r__data = *t_data * value;);
  }
}

void THZTensor_(div)(THZTensor *r_, THZTensor *t, ntype value)
{
  THZTensor_(resizeAs)(r_, t);
  if (THZTensor_(isContiguous)(r_) && THZTensor_(isContiguous)(t) && THZTensor_(nElement)(r_) == THZTensor_(nElement)(t)) {
    THZ_TENSOR_APPLY2_CONTIG(ntype, r_, ntype, t, THZVector_(divs)(r__data, t_data, value, r__len););
  } else {
    TH_TENSOR_APPLY2(ntype, r_, ntype, t, *r__data = *t_data / value;);
  }
}

void THZTensor_(lshift)(THZTensor *r_, THZTensor *t, ntype value)
{
  return THError("lshift is not supported for complex type");
}

void THZTensor_(rshift)(THZTensor *r_, THZTensor *t, ntype value)
{
  return THError("rshift is not supported for complex type");
}

void THZTensor_(fmod)(THZTensor *r_, THZTensor *t, ntype value)
{
  return THError("fmod is not supported for complex type");
}

void THZTensor_(remainder)(THZTensor *r_, THZTensor *t, ntype value)
{
  return THError("remainder is not supported for complex type");
}

void THZTensor_(bitand)(THZTensor *r_, THZTensor *t, ntype value)
{
  return THError("bitand is only supported for integer type tensors");
}

void THZTensor_(bitor)(THZTensor *r_, THZTensor *t, ntype value)
{
  return THError("bitor is only supported for integer type tensors");
}

void THZTensor_(bitxor)(THZTensor *r_, THZTensor *t, ntype value)
{
  return THError("bitxor is only supported for integer type tensors");
}

void THZTensor_(clamp)(THZTensor *r_, THZTensor *t, ntype min_value, ntype max_value)
{
  return THError("clamp is not supported for complex types");
}

void THZTensor_(cadd)(THZTensor *r_, THZTensor *t, ntype value, THZTensor *src)
{
  THZTensor_(resizeAs)(r_, t);
  if (THZTensor_(isContiguous)(r_) && THZTensor_(isContiguous)(t) && THZTensor_(isContiguous)(src) && THZTensor_(nElement)(r_) == THZTensor_(nElement)(src)) {
    if(r_ == t) {
      THZBlas_(axpy)(THZTensor_(nElement)(t), value, THZTensor_(data)(src), 1, THZTensor_(data)(r_), 1);
    } else {
      THZ_TENSOR_APPLY3_CONTIG(ntype, r_, ntype, t, ntype, src, THZVector_(cadd)(r__data, t_data, src_data, value, r__len););
    }
  } else {
    TH_TENSOR_APPLY3(ntype, r_, ntype, t, ntype, src, *r__data = *t_data + value * *src_data;);
  }
}

void THZTensor_(csub)(THZTensor *r_, THZTensor *t, ntype value, THZTensor *src)
{
  THZTensor_(cadd)(r_, t, -value, src);
}

void THZTensor_(cmul)(THZTensor *r_, THZTensor *t, THZTensor *src)
{
  THZTensor_(resizeAs)(r_, t);
  if (THZTensor_(isContiguous)(r_) && THZTensor_(isContiguous)(t) && THZTensor_(isContiguous)(src) && THZTensor_(nElement)(r_) == THZTensor_(nElement)(src)) {
    THZ_TENSOR_APPLY3_CONTIG(ntype, r_, ntype, t, ntype, src, THZVector_(cmul)(r__data, t_data, src_data, r__len););
  } else {
    TH_TENSOR_APPLY3(ntype, r_, ntype, t, ntype, src, *r__data = *t_data * *src_data;);
  }
}

void THZTensor_(cpow)(THZTensor *r_, THZTensor *t, THZTensor *src)
{
  THZTensor_(resizeAs)(r_, t);
  if (THZTensor_(isContiguous)(r_) && THZTensor_(isContiguous)(t) && THZTensor_(isContiguous)(src) && THZTensor_(nElement)(r_) == THZTensor_(nElement)(src)) {
    ntype *tp = THZTensor_(data)(t);
    ntype *sp = THZTensor_(data)(src);
    ntype *rp = THZTensor_(data)(r_);
    ptrdiff_t sz = THZTensor_(nElement)(t);
    ptrdiff_t i;
    #pragma omp parallel for if(sz > THZ_OMP_OVERHEAD_THRESHOLD) private(i)
    for (i=0; i<sz; i++)
      rp[i] = THZ_MATH_NAME(pow)(tp[i], sp[i]);
  } else {
    TH_TENSOR_APPLY3(ntype, r_, ntype, t, ntype, src, *r__data = THZ_MATH_NAME(pow)(*t_data, *src_data););
  }
}

void THZTensor_(cdiv)(THZTensor *r_, THZTensor *t, THZTensor *src)
{
  THZTensor_(resizeAs)(r_, t);
  if (THZTensor_(isContiguous)(r_) && THZTensor_(isContiguous)(t) && THZTensor_(isContiguous)(src) && THZTensor_(nElement)(r_) == THZTensor_(nElement)(src)) {
    THZ_TENSOR_APPLY3_CONTIG(ntype, r_, ntype, t, ntype, src, THZVector_(cdiv)(r__data, t_data, src_data, r__len););
  } else {
    TH_TENSOR_APPLY3(ntype, r_, ntype, t, ntype, src, *r__data = *t_data / *src_data;);
  }
}

void THZTensor_(clshift)(THZTensor *r_, THZTensor *t, THZTensor *src)
{
  return THError("clshift is not supported for complex type");
}

void THZTensor_(crshift)(THZTensor *r_, THZTensor *t, THZTensor *src)
{
  return THError("crshift is not supported for complex types");
}

void THZTensor_(cfmod)(THZTensor *r_, THZTensor *t, THZTensor *src)
{
  return THError("cfmod is not supported for complex types");
}

void THZTensor_(cremainder)(THZTensor *r_, THZTensor *t, THZTensor *src)
{
  return THError("cremainder is not supported for complex types");
}

void THZTensor_(cbitand)(THZTensor *r_, THZTensor *t, THZTensor *src)
{
  return THError("cbitand is only supported for integer type tensors");
}

void THZTensor_(cbitor)(THZTensor *r_, THZTensor *t, THZTensor *src)
{
  return THError("cbitor is only supported for integer type tensors");
}

void THZTensor_(cbitxor)(THZTensor *r_, THZTensor *t, THZTensor *src)
{
  return THError("cbitxor is only supported for integer type tensors");
}

void THZTensor_(tpow)(THZTensor *r_, ntype value, THZTensor *t)
{
  THZTensor_(resizeAs)(r_, t);
  if (THZTensor_(isContiguous)(r_) && THZTensor_(isContiguous)(t) && THZTensor_(nElement)(r_) == THZTensor_(nElement)(t)) {
    ntype *tp = THZTensor_(data)(t);
    ntype *rp = THZTensor_(data)(r_);
    ptrdiff_t sz = THZTensor_(nElement)(t);
    ptrdiff_t i;
    #pragma omp parallel for if(sz > THZ_OMP_OVERHEAD_THRESHOLD) private(i)
    for (i=0; i<sz; i++)
      rp[i] = THZ_MATH_NAME(pow)(value, tp[i]);
  } else {
    TH_TENSOR_APPLY2(ntype, r_, ntype, t, *r__data = THZ_MATH_NAME(pow)(value, *t_data););
  }
}

void THZTensor_(addcmul)(THZTensor *r_, THZTensor *t, ntype value, THZTensor *src1, THZTensor *src2)
{
  if(r_ != t)
  {
    THZTensor_(resizeAs)(r_, t);
    THZTensor_(copy)(r_, t);
  }

  TH_TENSOR_APPLY3(ntype, r_, ntype, src1, ntype, src2, *r__data += value * *src1_data * *src2_data;);
}


void THZTensor_(addcdiv)(THZTensor *r_, THZTensor *t, ntype value, THZTensor *src1, THZTensor *src2)
{
  if(r_ != t)
  {
    THZTensor_(resizeAs)(r_, t);
    THZTensor_(copy)(r_, t);
  }

  TH_TENSOR_APPLY3(ntype, r_, ntype, src1, ntype, src2, *r__data += value * *src1_data / *src2_data;);
}

void THZTensor_(addmv)(THZTensor *r_, ntype beta, THZTensor *t, ntype alpha, THZTensor *mat, THZTensor *vec)
{
  if( (mat->nDimension != 2) || (vec->nDimension != 1) )
    THError("matrix and vector expected, got %dD, %dD",
      mat->nDimension, vec->nDimension);

  if( mat->size[1] != vec->size[0] ) {
    THDescBuff bm = THZTensor_(sizeDesc)(mat);
    THDescBuff bv = THZTensor_(sizeDesc)(vec);
    THError("size mismatch, %s, %s", bm.str, bv.str);
  }

  if(t->nDimension != 1)
    THError("vector expected, got t: %dD", t->nDimension);

  if(t->size[0] != mat->size[0]) {
    THDescBuff bt = THZTensor_(sizeDesc)(t);
    THDescBuff bm = THZTensor_(sizeDesc)(mat);
    THError("size mismatch, t: %s, mat: %s", bt.str, bm.str);
  }

  if(r_ != t)
  {
    THZTensor_(resizeAs)(r_, t);
    THZTensor_(copy)(r_, t);
  }

  if(mat->stride[0] == 1)
  {
    THZBlas_(gemv)('n', mat->size[0], mat->size[1],
                  alpha, THZTensor_(data)(mat), mat->stride[1],
                  THZTensor_(data)(vec), vec->stride[0],
                  beta, THZTensor_(data)(r_), r_->stride[0]);
  }
  else if(mat->stride[1] == 1)
  {
    THZBlas_(gemv)('t',  mat->size[1], mat->size[0],
                  alpha, THZTensor_(data)(mat), mat->stride[0],
                  THZTensor_(data)(vec), vec->stride[0],
                  beta, THZTensor_(data)(r_), r_->stride[0]);
  }
  else
  {
    THZTensor *cmat = THZTensor_(newContiguous)(mat);

    THZBlas_(gemv)('t',  mat->size[1], mat->size[0],
                  alpha, THZTensor_(data)(cmat), cmat->stride[0],
                  THZTensor_(data)(vec), vec->stride[0],
                  beta, THZTensor_(data)(r_), r_->stride[0]);

    THZTensor_(free)(cmat);
  }
}

void THZTensor_(match)(THZTensor *r_, THZTensor *m1, THZTensor *m2, ntype gain)
{
  int64_t N1 = m1->size[0];
  int64_t N2 = m2->size[0];
  int64_t dim;
  ntype *m1_p;
  ntype *m2_p;
  ntype *r_p;
  int64_t i;

  THZTensor_(resize2d)(r_, N1, N2);

  m1 = THZTensor_(newContiguous)(m1);
  m2 = THZTensor_(newContiguous)(m2);

  THZTensor_(resize2d)(m1, N1, THZTensor_(nElement)(m1) / N1);
  THZTensor_(resize2d)(m2, N2, THZTensor_(nElement)(m2) / N2);

  dim = m1->size[1];
  THArgCheck(m1->size[1] == m2->size[1], 3, "m1 and m2 must have the same inner vector dim");

  m1_p = THZTensor_(data)(m1);
  m2_p = THZTensor_(data)(m2);
  r_p = THZTensor_(data)(r_);

#pragma omp parallel for private(i)
  for (i=0; i<N1; i++) {
    int64_t j,k;
    for (j=0; j<N2; j++) {
      ntype sum = 0;
      for (k=0; k<dim; k++) {
        ntype term = m1_p[ i*dim + k ] - m2_p[ j*dim + k ];
        sum += term*term;
      }
      r_p[ i*N2 + j ] = gain * sum;
    }
  }

  THZTensor_(free)(m1);
  THZTensor_(free)(m2);
}

void THZTensor_(addmm)(THZTensor *r_, ntype beta, THZTensor *t, ntype alpha, THZTensor *m1, THZTensor *m2)
{
  char transpose_r, transpose_m1, transpose_m2;
  THZTensor *r__, *m1_, *m2_;
  int free_m1 = 0;
  int free_m2 = 0;

  if( (m1->nDimension != 2) || (m2->nDimension != 2))
    THError("matrices expected, got %dD, %dD tensors", m1->nDimension, m2->nDimension);

  if(m1->size[1] != m2->size[0]) {
    THDescBuff bm1 = THZTensor_(sizeDesc)(m1);
    THDescBuff bm2 = THZTensor_(sizeDesc)(m2);
    THError("size mismatch, m1: %s, m2: %s", bm1.str, bm2.str);
  }

  if( t->nDimension != 2 )
    THError("matrix expected, got %dD tensor for t", t->nDimension);

  if( (t->size[0] != m1->size[0]) || (t->size[1] != m2->size[1]) ) {
    THDescBuff bt  = THZTensor_(sizeDesc)(t);
    THDescBuff bm1 = THZTensor_(sizeDesc)(m1);
    THDescBuff bm2 = THZTensor_(sizeDesc)(m2);
    THError("size mismatch, t: %s, m1: %s, m2: %s", bt.str, bm1.str, bm2.str);
  }

  if(t != r_)
  {
    THZTensor_(resizeAs)(r_, t);
    if (beta != 0.0) {
      THZTensor_(copy)(r_, t);
    }
  }

  /* r_ */
  if(r_->stride[0] == 1 &&
     r_->stride[1] != 0)
  {
    transpose_r = 'n';
    r__ = r_;
  }
  else if(r_->stride[1] == 1 &&
          r_->stride[0] != 0)
  {
    THZTensor *swap = m2;
    m2 = m1;
    m1 = swap;
    transpose_r = 't';
    r__ = r_;
  }
  else
  {
    transpose_r = 'n';

    THZTensor *transp_r_ = THZTensor_(newTranspose)(r_, 0, 1);
    r__ = THZTensor_(newClone)(transp_r_);
    THZTensor_(free)(transp_r_);
    THZTensor_(transpose)(r__, NULL, 0, 1);
  }

  /* m1 */
  if(m1->stride[(transpose_r == 'n' ? 0 : 1)] == 1 &&
     m1->stride[(transpose_r == 'n' ? 1 : 0)] != 0)
  {
    transpose_m1 = 'n';
    m1_ = m1;
  }
  else if(m1->stride[(transpose_r == 'n' ? 1 : 0)] == 1 &&
          m1->stride[(transpose_r == 'n' ? 0 : 1)] != 0)
  {
    transpose_m1 = 't';
    m1_ = m1;
  }
  else
  {
    transpose_m1 = (transpose_r == 'n' ? 't' : 'n');
    m1_ = THZTensor_(newContiguous)(m1);
    free_m1 = 1;
  }

  /* m2 */
  if(m2->stride[(transpose_r == 'n' ? 0 : 1)] == 1 &&
     m2->stride[(transpose_r == 'n' ? 1 : 0)] != 0)
  {
    transpose_m2 = 'n';
    m2_ = m2;
  }
  else if(m2->stride[(transpose_r == 'n' ? 1 : 0)] == 1 &&
          m2->stride[(transpose_r == 'n' ? 0 : 1)] != 0)
  {
    transpose_m2 = 't';
    m2_ = m2;
  }
  else
  {
    transpose_m2 = (transpose_r == 'n' ? 't' : 'n');
    m2_ = THZTensor_(newContiguous)(m2);
    free_m2 = 1;
  }

#pragma omp critical(blasgemm)
  /* do the operation */
  THZBlas_(gemm)(transpose_m1,
                transpose_m2,
                r__->size[(transpose_r == 'n' ? 0 : 1)],
                r__->size[(transpose_r == 'n' ? 1 : 0)],
                m1_->size[(transpose_r == 'n' ? 1 : 0)],
                alpha,
                THZTensor_(data)(m1_),
                (transpose_m1 == 'n' ? m1_->stride[(transpose_r == 'n' ? 1 : 0)] : m1_->stride[(transpose_r == 'n' ? 0 : 1)]),
                THZTensor_(data)(m2_),
                (transpose_m2 == 'n' ? m2_->stride[(transpose_r == 'n' ? 1 : 0)] : m2_->stride[(transpose_r == 'n' ? 0 : 1)]),
                beta,
                THZTensor_(data)(r__),
                r__->stride[(transpose_r == 'n' ? 1 : 0)]);

  /* free intermediate variables */
  if(free_m1)
    THZTensor_(free)(m1_);

  if(free_m2)
    THZTensor_(free)(m2_);

  if(r__ != r_)
    THZTensor_(freeCopyTo)(r__, r_);
}

void THZTensor_(addr)(THZTensor *r_, ntype beta, THZTensor *t, ntype alpha, THZTensor *vec1, THZTensor *vec2)
{
  if( (vec1->nDimension != 1) || (vec2->nDimension != 1) )
    THError("vector and vector expected, got %dD, %dD tensors",
        vec1->nDimension, vec2->nDimension);

  if(t->nDimension != 2)
    THError("expected matrix, got %dD tensor for t", t->nDimension);

  if( (t->size[0] != vec1->size[0]) || (t->size[1] != vec2->size[0]) ) {
    THDescBuff bt  = THZTensor_(sizeDesc)(t);
    THDescBuff bv1 = THZTensor_(sizeDesc)(vec1);
    THDescBuff bv2 = THZTensor_(sizeDesc)(vec2);
    THError("size mismatch, t: %s, vec1: %s, vec2: %s", bt.str, bv1.str, bv2.str);
  }

  if(r_ != t)
  {
    THZTensor_(resizeAs)(r_, t);
    THZTensor_(copy)(r_, t);
  }

  if(beta == 0) {
    THZTensor_(zero)(r_);
  }
  else if(beta != 1)
    THZTensor_(mul)(r_, r_, beta);

  if(r_->stride[0] == 1)
  {
    THZBlas_(ger)(vec1->size[0], vec2->size[0],
                 alpha, THZTensor_(data)(vec1), vec1->stride[0],
                 THZTensor_(data)(vec2), vec2->stride[0],
                 THZTensor_(data)(r_), r_->stride[1]);
  }
  else if(r_->stride[1] == 1)
  {
    THZBlas_(ger)(vec2->size[0], vec1->size[0],
                 alpha, THZTensor_(data)(vec2), vec2->stride[0],
                 THZTensor_(data)(vec1), vec1->stride[0],
                 THZTensor_(data)(r_), r_->stride[0]);
  }
  else
  {
    THZTensor *cr = THZTensor_(newClone)(r_);

    THZBlas_(ger)(vec2->size[0], vec1->size[0],
                 alpha, THZTensor_(data)(vec2), vec2->stride[0],
                 THZTensor_(data)(vec1), vec1->stride[0],
                 THZTensor_(data)(cr), cr->stride[0]);

    THZTensor_(freeCopyTo)(cr, r_);
  }
}

void THZTensor_(addbmm)(THZTensor *result, ntype beta, THZTensor *t, ntype alpha, THZTensor *batch1, THZTensor *batch2)
{
  int64_t batch;

  THArgCheck(THZTensor_(nDimension)(batch1) == 3, 1, "expected 3D tensor");
  THArgCheck(THZTensor_(nDimension)(batch2) == 3, 2, "expected 3D tensor");
  THArgCheck(THZTensor_(size)(batch1, 0) == THZTensor_(size)(batch2, 0), 2,
             "equal number of batches expected, got %d, %d",
             THZTensor_(size)(batch1, 0), THZTensor_(size)(batch2, 0));
  THArgCheck(THZTensor_(size)(batch1, 2) == THZTensor_(size)(batch2, 1), 2,
             "wrong matrix size, batch1: %dx%d, batch2: %dx%d",
             THZTensor_(size)(batch1, 1), THZTensor_(size)(batch1,2),
             THZTensor_(size)(batch2, 1), THZTensor_(size)(batch2,2));

  int64_t dim1 = THZTensor_(size)(batch1, 1);
  int64_t dim2 = THZTensor_(size)(batch2, 2);
  THArgCheck(THZTensor_(size)(t, 0) == dim1, 1, "output tensor of incorrect size");
  THArgCheck(THZTensor_(size)(t, 1) == dim2, 1, "output tensor of incorrect size");

  if (t != result) {
    THZTensor_(resizeAs)(result, t);
    if (beta != 0.0) {
      THZTensor_(copy)(result, t);
    }
  }

  THZTensor *matrix1 = THZTensor_(new)();
  THZTensor *matrix2 = THZTensor_(new)();

  for (batch = 0; batch < THZTensor_(size)(batch1, 0); ++batch) {
    THZTensor_(select)(matrix1, batch1, 0, batch);
    THZTensor_(select)(matrix2, batch2, 0, batch);

    THZTensor_(addmm)(result, beta, result, alpha, matrix1, matrix2);
    beta = 1; // accumulate output once
  }

  THZTensor_(free)(matrix1);
  THZTensor_(free)(matrix2);
}

void THZTensor_(baddbmm)(THZTensor *result, ntype beta, THZTensor *t, ntype alpha, THZTensor *batch1, THZTensor *batch2)
{
  int64_t batch;

  THArgCheck(THZTensor_(nDimension)(batch1) == 3, 1, "expected 3D tensor, got %dD", THZTensor_(nDimension)(batch1));
  THArgCheck(THZTensor_(nDimension)(batch2) == 3, 2, "expected 3D tensor, got %dD", THZTensor_(nDimension)(batch2));
  THArgCheck(THZTensor_(size)(batch1, 0) == THZTensor_(size)(batch2, 0), 2,
             "equal number of batches expected, got %d, %d",
             THZTensor_(size)(batch1, 0), THZTensor_(size)(batch2, 0));
  THArgCheck(THZTensor_(size)(batch1, 2) == THZTensor_(size)(batch2, 1), 2,
             "wrong matrix size, batch1: %dx%d, batch2: %dx%d",
             THZTensor_(size)(batch1, 1), THZTensor_(size)(batch1, 2),
             THZTensor_(size)(batch2, 1), THZTensor_(size)(batch2, 2));

  int64_t bs = THZTensor_(size)(batch1, 0);
  int64_t dim1 = THZTensor_(size)(batch1, 1);
  int64_t dim2 = THZTensor_(size)(batch2, 2);
  THArgCheck(THZTensor_(size)(t, 0) == bs, 1,   "output tensor of incorrect size");
  THArgCheck(THZTensor_(size)(t, 1) == dim1, 1, "output tensor of incorrect size");
  THArgCheck(THZTensor_(size)(t, 2) == dim2, 1, "output tensor of incorrect size");

  if (t != result) {
    THZTensor_(resizeAs)(result, t);
    if (beta != 0.0) {
      THZTensor_(copy)(result, t);
    }
  }

  THZTensor *matrix1 = THZTensor_(new)();
  THZTensor *matrix2 = THZTensor_(new)();
  THZTensor *result_matrix = THZTensor_(new)();

  for (batch = 0; batch < THZTensor_(size)(batch1, 0); ++batch) {
    THZTensor_(select)(matrix1, batch1, 0, batch);
    THZTensor_(select)(matrix2, batch2, 0, batch);
    THZTensor_(select)(result_matrix, result, 0, batch);

    THZTensor_(addmm)(result_matrix, beta, result_matrix, alpha, matrix1, matrix2);
  }

  THZTensor_(free)(matrix1);
  THZTensor_(free)(matrix2);
  THZTensor_(free)(result_matrix);
}

ptrdiff_t THZTensor_(numel)(THZTensor *t)
{
  return THZTensor_(nElement)(t);
}

void THZTensor_(max)(THZTensor *values_, THLongTensor *indices_, THZTensor *t, int dimension, int keepdim)
{
  return THError("max is not supported for complex type");
}

void THZTensor_(min)(THZTensor *values_, THLongTensor *indices_, THZTensor *t, int dimension, int keepdim)
{
  return THError("min is supported for complex type");
}


void THZTensor_(sum)(THZTensor *r_, THZTensor *t, int dimension, int keepdim)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THZTensor_(nDimension)(t), 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  dim = THZTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THZTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  // two implementations optimized for data locality
  if (t->stride[dimension] == 1) {
    TH_TENSOR_DIM_APPLY2(ntype, t, ntype, r_, dimension,
                         accntype sum = 0;
                         int64_t i;
                         for(i = 0; i < t_size; i++)
                           sum += t_data[i*t_stride];
                         *r__data = (ntype)sum;);
  } else {
    THZTensor_(zero)(r_);
    THZTensor *temp_ = THZTensor_(newWithTensor)(r_);
    // r_.expand_as(t)
    temp_->size[dimension] = t->size[dimension];
    temp_->stride[dimension] = 0;

    TH_TENSOR_APPLY2(ntype, temp_, ntype, t, *temp__data = *temp__data + *t_data;);
    THZTensor_(free)(temp_);
  }

  if (!keepdim) {
    THZTensor_(squeeze1d)(r_, r_, dimension);
  }
}

void THZTensor_(prod)(THZTensor *r_, THZTensor *t, int dimension, int keepdim)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THZTensor_(nDimension)(t), 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  dim = THZTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THZTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  // two implementations optimized for data locality
  if (t->stride[dimension] == 1) {
    TH_TENSOR_DIM_APPLY2(ntype, t, ntype, r_, dimension,
                         accntype prod = 1;
                         int64_t i;
                         for(i = 0; i < t_size; i++)
                           prod *= t_data[i*t_stride];
                         *r__data = (ntype)prod;);
  } else {
    THZTensor_(fill)(r_, 1);
    THZTensor *temp_ = THZTensor_(newWithTensor)(r_);
    // r_.expand_as(t)
    temp_->size[dimension] = t->size[dimension];
    temp_->stride[dimension] = 0;

    TH_TENSOR_APPLY2(ntype, temp_, ntype, t, *temp__data = *temp__data * *t_data;);
    THZTensor_(free)(temp_);
  }

  if (!keepdim) {
    THZTensor_(squeeze1d)(r_, r_, dimension);
  }
}

void THZTensor_(cumsum)(THZTensor *r_, THZTensor *t, int dimension)
{
  THArgCheck(dimension >= 0 && dimension < THZTensor_(nDimension)(t), 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  THZTensor_(resizeAs)(r_, t);

  TH_TENSOR_DIM_APPLY2(ntype, t, ntype, r_, dimension,
                       accntype cumsum = 0;
                       int64_t i;
                       for(i = 0; i < t_size; i++)
                       {
                         cumsum += t_data[i*t_stride];
                         r__data[i*r__stride] = (ntype)cumsum;
                       });
}

void THZTensor_(cumprod)(THZTensor *r_, THZTensor *t, int dimension)
{
  THArgCheck(dimension >= 0 && dimension < THZTensor_(nDimension)(t), 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  THZTensor_(resizeAs)(r_, t);

  TH_TENSOR_DIM_APPLY2(ntype, t, ntype, r_, dimension,
                       accntype cumprod = 1;
                       int64_t i;
                       for(i = 0; i < t_size; i++)
                       {
                         cumprod *= t_data[i*t_stride];
                         r__data[i*r__stride] = (ntype)cumprod;
                       });
}


void THZTensor_(sign)(THZTensor *r_, THZTensor *t)
{
  THZTensor_(resizeAs)(r_, t);

  TH_TENSOR_APPLY2(ntype, r_, ntype, t,
    *r__data = *t_data / THZ_MATH_NAME(abs)(*t_data););
}


accntype THZTensor_(trace)(THZTensor *t)
{
  ntype *t_data = THZTensor_(data)(t);
  accntype sum = 0;
  int64_t i = 0;
  int64_t t_stride_0, t_stride_1, t_diag_size;

  THArgCheck(THZTensor_(nDimension)(t) == 2, 1, "expected a matrix");

  t_stride_0 = THZTensor_(stride)(t, 0);
  t_stride_1 = THZTensor_(stride)(t, 1);
  t_diag_size = THMin(THZTensor_(size)(t, 0), THZTensor_(size)(t, 1));
  while(i < t_diag_size)
  {
    sum += t_data[i*(t_stride_0+t_stride_1)];
    i++;
  }

  return sum;
}

void THZTensor_(cross)(THZTensor *r_, THZTensor *a, THZTensor *b, int dimension)
{
  int i;

  if(THZTensor_(nDimension)(a) != THZTensor_(nDimension)(b))
    THError("inconsistent tensor dimension %dD, %dD",
        THZTensor_(nDimension)(a), THZTensor_(nDimension)(b));

  for(i = 0; i < THZTensor_(nDimension)(a); i++)
  {
    if(THZTensor_(size)(a, i) != THZTensor_(size)(b, i)) {
        THDescBuff ba = THZTensor_(sizeDesc)(a);
        THDescBuff bb = THZTensor_(sizeDesc)(b);
        THError("inconsistent tensor sizes %s, %s", ba.str, bb.str);
    }
  }

  if(dimension < 0)
  {
    for(i = 0; i < THZTensor_(nDimension)(a); i++)
    {
      if(THZTensor_(size)(a, i) == 3)
      {
        dimension = i;
        break;
      }
    }
    if(dimension < 0) {
      THDescBuff ba = THZTensor_(sizeDesc)(a);
      THError("no dimension of size 3 in a: %s", ba.str);
    }
  }

  THArgCheck(dimension >= 0 && dimension < THZTensor_(nDimension)(a), 3, "dimension %d out of range",
      dimension + TH_INDEX_BASE);
  THArgCheck(THZTensor_(size)(a, dimension) == 3, 3, "dimension %d does not have size 3",
      dimension + TH_INDEX_BASE);

  THZTensor_(resizeAs)(r_, a);

  TH_TENSOR_DIM_APPLY3(ntype, a, ntype, b, ntype, r_, dimension,
                       TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
                       r__data[0*r__stride] = a_data[1*a_stride]*b_data[2*b_stride] - a_data[2*a_stride]*b_data[1*b_stride];
                       r__data[1*r__stride] = a_data[2*a_stride]*b_data[0*b_stride] - a_data[0*a_stride]*b_data[2*b_stride];
                       r__data[2*r__stride] = a_data[0*a_stride]*b_data[1*b_stride] - a_data[1*a_stride]*b_data[0*b_stride];);
}

void THZTensor_(cmax)(THZTensor *r, THZTensor *t, THZTensor *src) {
  return THError("cmax is not supported for complex type");
}

void THZTensor_(cmin)(THZTensor *r, THZTensor *t, THZTensor *src) {
  return THError("cmin is not supported for complex type");
}

void THZTensor_(cmaxValue)(THZTensor *r, THZTensor *t, ntype value) {
  return THError("cmaxValue is not supported for complex type");
}

void THZTensor_(cminValue)(THZTensor *r, THZTensor *t, ntype value) {
  return THError("cminValue is not supported for complex type");
}

void THZTensor_(zeros)(THZTensor *r_, THLongStorage *size)
{
  THZTensor_(resize)(r_, size, NULL);
  THZTensor_(zero)(r_);
}

void THZTensor_(zerosLike)(THZTensor *r_, THZTensor *input)
{
  THZTensor_(resizeAs)(r_, input);
  THZTensor_(zero)(r_);
}

void THZTensor_(onesLike)(THZTensor *r_, THZTensor *input)
{
  THZTensor_(resizeAs)(r_, input);
  THZTensor_(fill)(r_, 1);
}

void THZTensor_(ones)(THZTensor *r_, THLongStorage *size)
{
  THZTensor_(resize)(r_, size, NULL);
  THZTensor_(fill)(r_, 1);
}

void THZTensor_(diag)(THZTensor *r_, THZTensor *t, int k)
{
  THArgCheck(THZTensor_(nDimension)(t) == 1 || THZTensor_(nDimension)(t) == 2, 1, "matrix or a vector expected");

  if(THZTensor_(nDimension)(t) == 1)
  {
    ntype *t_data = THZTensor_(data)(t);
    int64_t t_stride_0 = THZTensor_(stride)(t, 0);
    int64_t t_size = THZTensor_(size)(t, 0);
    int64_t sz = t_size + (k >= 0 ? k : -k);
    ntype *r__data;
    int64_t r__stride_0;
    int64_t r__stride_1;
    int64_t i;

    THZTensor_(resize2d)(r_, sz, sz);
    THZTensor_(zero)(r_);
    r__data = THZTensor_(data)(r_);
    r__stride_0 = THZTensor_(stride)(r_, 0);
    r__stride_1 = THZTensor_(stride)(r_, 1);
    r__data += (k >= 0 ? k*r__stride_1 : -k*r__stride_0);

    for(i = 0; i < t_size; i++)
      r__data[i*(r__stride_0+r__stride_1)] = t_data[i*t_stride_0];
  }
  else
  {
    ntype *t_data = THZTensor_(data)(t);
    int64_t t_stride_0 = THZTensor_(stride)(t, 0);
    int64_t t_stride_1 = THZTensor_(stride)(t, 1);
    int64_t sz;
    ntype *r__data;
    int64_t r__stride_0;
    int64_t i;

    if(k >= 0)
      sz = THMin(THZTensor_(size)(t, 0), THZTensor_(size)(t, 1)-k);
    else
      sz = THMin(THZTensor_(size)(t, 0)+k, THZTensor_(size)(t, 1));
    THZTensor_(resize1d)(r_, sz);
    r__data = THZTensor_(data)(r_);
    r__stride_0 = THZTensor_(stride)(r_, 0);

    t_data += (k >= 0 ? k*t_stride_1 : -k*t_stride_0);
    for(i = 0; i < sz; i++)
      r__data[i*r__stride_0] = t_data[i*(t_stride_0+t_stride_1)];
  }
}

void THZTensor_(eye)(THZTensor *r_, int64_t n, int64_t m)
{
  ntype *r__data;
  int64_t i, sz;

  THArgCheck(n > 0, 1, "invalid argument");

  if(m <= 0)
    m = n;

  THZTensor_(resize2d)(r_, n, m);
  THZTensor_(zero)(r_);

  i = 0;
  r__data = THZTensor_(data)(r_);
  sz = THMin(THZTensor_(size)(r_, 0), THZTensor_(size)(r_, 1));
  for(i = 0; i < sz; i++)
    r__data[i*(r_->stride[0]+r_->stride[1])] = 1;
}


void THZTensor_(range)(THZTensor *r_, accntype xmin, accntype xmax, accntype step)
{
  return THError("range is not supported for complex type");
}

void THZTensor_(arange)(THZTensor *r_, accntype xmin, accntype xmax, accntype step) {
  return THError("arange is not supported for complex type");
}

void THZTensor_(randperm)(THZTensor *r_, THGenerator *_generator, int64_t n)
{
  ntype *r__data;
  int64_t r__stride_0;
  int64_t i;

  THArgCheck(n > 0, 1, "must be strictly positive");

  THZTensor_(resize1d)(r_, n);
  r__data = THZTensor_(data)(r_);
  r__stride_0 = THZTensor_(stride)(r_,0);

  for(i = 0; i < n; i++)
    r__data[i*r__stride_0] = (ntype)(i);

  for(i = 0; i < n-1; i++)
  {
    int64_t z = THRandom_random(_generator) % (n-i);
    ntype sav = r__data[i*r__stride_0];
    r__data[i*r__stride_0] = r__data[(z+i)*r__stride_0];
    r__data[(z+i)*r__stride_0] = sav;
  }
}

void THZTensor_(reshape)(THZTensor *r_, THZTensor *t, THLongStorage *size)
{
  THZTensor_(resize)(r_, size, NULL);
  THZTensor_(copy)(r_, t);
}

void THZTensor_(sort)(THZTensor *rt_, THLongTensor *ri_, THZTensor *t, int dimension, int descendingOrder)
{
  return THError("sort is not supported for complex type");
}

void THZTensor_(mode)(THZTensor *values_, THLongTensor *indices_, THZTensor *t, int dimension, int keepdim)
{
  return THError("mode is not supported for complex type");
}

void THZTensor_(kthvalue)(THZTensor *values_, THLongTensor *indices_, THZTensor *t, int64_t k, int dimension, int keepdim)
{
  return THError("kthvalue is not supported for complex");
}

void THZTensor_(median)(THZTensor *values_, THLongTensor *indices_, THZTensor *t, int dimension, int keepdim)
{
  return THError("median is not supported for complex type");
}

void THZTensor_(topk)(THZTensor *rt_, THLongTensor *ri_, THZTensor *t, int64_t k, int dim, int dir, int sorted)
{
  return THError("topk is not supported for complex type");
}

void THZTensor_(tril)(THZTensor *r_, THZTensor *t, int64_t k)
{
  int64_t t_size_0, t_size_1;
  int64_t t_stride_0, t_stride_1;
  int64_t r__stride_0, r__stride_1;
  ntype *t_data, *r__data;
  int64_t r, c;

  THArgCheck(THZTensor_(nDimension)(t) == 2, 1, "expected a matrix");

  THZTensor_(resizeAs)(r_, t);

  t_size_0 = THZTensor_(size)(t, 0);
  t_size_1 = THZTensor_(size)(t, 1);
  t_stride_0 = THZTensor_(stride)(t, 0);
  t_stride_1 = THZTensor_(stride)(t, 1);
  r__stride_0 = THZTensor_(stride)(r_, 0);
  r__stride_1 = THZTensor_(stride)(r_, 1);
  r__data = THZTensor_(data)(r_);
  t_data = THZTensor_(data)(t);

  for(r = 0; r < t_size_0; r++)
  {
    int64_t sz = THMin(r+k+1, t_size_1);
    for(c = THMax(0, r+k+1); c < t_size_1; c++)
      r__data[r*r__stride_0+c*r__stride_1] = 0;
    for(c = 0; c < sz; c++)
      r__data[r*r__stride_0+c*r__stride_1] = t_data[r*t_stride_0+c*t_stride_1];
  }
}

void THZTensor_(triu)(THZTensor *r_, THZTensor *t, int64_t k)
{
  int64_t t_size_0, t_size_1;
  int64_t t_stride_0, t_stride_1;
  int64_t r__stride_0, r__stride_1;
  ntype *t_data, *r__data;
  int64_t r, c;

  THArgCheck(THZTensor_(nDimension)(t) == 2, 1, "expected a matrix");

  THZTensor_(resizeAs)(r_, t);

  t_size_0 = THZTensor_(size)(t, 0);
  t_size_1 = THZTensor_(size)(t, 1);
  t_stride_0 = THZTensor_(stride)(t, 0);
  t_stride_1 = THZTensor_(stride)(t, 1);
  r__stride_0 = THZTensor_(stride)(r_, 0);
  r__stride_1 = THZTensor_(stride)(r_, 1);
  r__data = THZTensor_(data)(r_);
  t_data = THZTensor_(data)(t);

  for(r = 0; r < t_size_0; r++)
  {
    int64_t sz = THMin(r+k, t_size_1);
    for(c = THMax(0, r+k); c < t_size_1; c++)
      r__data[r*r__stride_0+c*r__stride_1] = t_data[r*t_stride_0+c*t_stride_1];
    for(c = 0; c < sz; c++)
      r__data[r*r__stride_0+c*r__stride_1] = 0;
  }
}

void THZTensor_(cat)(THZTensor *r_, THZTensor *ta, THZTensor *tb, int dimension)
{
  THZTensor* inputs[2];
  inputs[0] = ta;
  inputs[1] = tb;
  THZTensor_(catArray)(r_, inputs, 2, dimension);
}

void THZTensor_(catArray)(THZTensor *result, THZTensor **inputs, int numInputs, int dimension)
{
  THLongStorage *size;
  int i, j;
  int64_t offset;
  int maxDim = dimension + 1;
  int allEmpty = 1;
  int allContiguous = 1;

  // cat_dimension is the actual dimension we cat along
  int cat_dimension = dimension;

  for (i = 0; i < numInputs; i++)
  {
    maxDim = THMax(maxDim, inputs[i]->nDimension);
  }

  // When the user input dimension is -1 (i.e. -2 in C)
  // Then we pick the maximum last dimension across all tensors.
  if ( dimension + TH_INDEX_BASE == -1 )
  {
    cat_dimension = maxDim?(maxDim-1):0;
  }

  THArgCheck(numInputs > 0, 3, "invalid number of inputs %d", numInputs);
  THArgCheck(cat_dimension >= 0, 4, "invalid dimension %d", dimension + TH_INDEX_BASE);

  size = THLongStorage_newWithSize(maxDim);

  for(i = 0; i < maxDim; i++)
  {
    // dimSize is either the size of the dim if it exists, either 1 if #dim > 0, otherwise 0
    int64_t dimSize = i < inputs[0]->nDimension ? inputs[0]->size[i] : THMin(inputs[0]->nDimension, 1);
    if (i == cat_dimension)
    {
      for (j = 1; j < numInputs; j++)
      {
        // accumulate the size over the dimension we want to cat on.
        // Empty tensors are allowed
        dimSize += i < inputs[j]->nDimension ? inputs[j]->size[i] : THMin(inputs[j]->nDimension, 1);
      }
    }
    else
    {
      for (j = 1; j < numInputs; j++)
      {
        int64_t sz = (i < inputs[j]->nDimension ? inputs[j]->size[i] : THMin(inputs[j]->nDimension, 1));
        // If it's a dimension we're not catting on
        // Then fail if sizes are different AND > 0
        if (dimSize != sz && dimSize && sz)
        {
          THLongStorage_free(size);
          THError("inconsistent tensor sizes");
        }
        else if(!dimSize)
        {
          dimSize = sz;
        }
      }
    }
    allEmpty = allEmpty && !dimSize;
    size->data[i] = dimSize;
  }

  // Initiate catting and resizing
  // If at least one of the input is not empty
  if (!allEmpty)
  {
    THZTensor_(resize)(result, size, NULL);

    // Check contiguity of all inputs and result
    for (i = 0; i < numInputs; i++) {
      if(inputs[i]->nDimension) {
        allContiguous = allContiguous && THZTensor_(isContiguous)(inputs[i]);
      }
    }
    allContiguous = allContiguous && THZTensor_(isContiguous)(result);

    // First path is for contiguous inputs along dim 1
    // Second path for non-contiguous
    if (cat_dimension == 0 && allContiguous)
    {
      ntype* result_data = result->storage->data + result->storageOffset;
      offset = 0;
      for (j = 0; j < numInputs; j++)
      {
        if (inputs[j]->nDimension)
        {
          THZTensor* input0 = inputs[j];
          ntype* input0_data = input0->storage->data + input0->storageOffset;
          int64_t input0_size = THZTensor_(nElement)(input0);
          memcpy(result_data + offset, input0_data, input0_size*sizeof(ntype));
          offset += input0_size;
        }
      }
    }
    else
    {
      offset = 0;
      for (j = 0; j < numInputs; j++)
      {
        if (inputs[j]->nDimension)
        {
          int64_t dimSize = cat_dimension < inputs[j]->nDimension ? inputs[j]->size[cat_dimension] : 1;
          THZTensor *nt = THZTensor_(newWithTensor)(result);
          THZTensor_(narrow)(nt, NULL, cat_dimension, offset, dimSize);
          THZTensor_(copy)(nt, inputs[j]);
          THZTensor_(free)(nt);
          offset += dimSize;
        }
      }
    }
  }
  THLongStorage_free(size);
}

int THZTensor_(equal)(THZTensor *ta, THZTensor* tb)
{
  int equal = 1;
  if(!THZTensor_(isSameSizeAs)(ta, tb))
    return 0;

  if (THZTensor_(isContiguous)(ta) && THZTensor_(isContiguous)(tb)) {
    ntype *tap = THZTensor_(data)(ta);
    ntype *tbp = THZTensor_(data)(tb);
    ptrdiff_t sz = THZTensor_(nElement)(ta);
    ptrdiff_t i;
    for (i=0; i<sz; ++i){
      if(tap[i] != tbp[i]) return 0;
    }
  } else {
    // Short-circuit the apply function on inequality
    TH_TENSOR_APPLY2(ntype, ta, ntype, tb,
                     if (equal && *ta_data != *tb_data) {
                        equal = 0;
                        TH_TENSOR_APPLY_hasFinished = 1; break;
                     })
  }
  return equal;
}

#define TENSOR_IMPLEMENT_LOGICAL(NAME,OP)       \
  void THZTensor_(NAME##Value)(THByteTensor *r_, THZTensor* t, ntype value) \
  {                 \
    THByteTensor_resizeNd(r_, t->nDimension, t->size, NULL);    \
    TH_TENSOR_APPLY2(unsigned char, r_, ntype, t,     \
         *r__data = (*t_data OP value) ? 1 : 0;); \
  }                 \
  void THZTensor_(NAME##ValueT)(THZTensor* r_, THZTensor* t, ntype value)  \
  {                 \
    THZTensor_(resizeNd)(r_, t->nDimension, t->size, NULL);    \
    TH_TENSOR_APPLY2(ntype, r_, ntype, t,         \
         *r__data = (*t_data OP value) ? 1 : 0;); \
  }                 \
  void THZTensor_(NAME##Tensor)(THByteTensor *r_, THZTensor *ta, THZTensor *tb) \
  {                 \
    THByteTensor_resizeNd(r_, ta->nDimension, ta->size, NULL);    \
    TH_TENSOR_APPLY3(unsigned char, r_, ntype, ta, ntype, tb,   \
         *r__data = (*ta_data OP *tb_data) ? 1 : 0;); \
  }                 \
  void THZTensor_(NAME##TensorT)(THZTensor *r_, THZTensor *ta, THZTensor *tb) \
  {                 \
    THZTensor_(resizeNd)(r_, ta->nDimension, ta->size, NULL);    \
    TH_TENSOR_APPLY3(ntype, r_, ntype, ta, ntype, tb,     \
         *r__data = (*ta_data OP *tb_data) ? 1 : 0;); \
  }                 \


#define TENSOR_IMPLEMENT_COMPLEX_LOGICAL(NAME,OP)       \
  void THZTensor_(NAME##Value)(THByteTensor *r_, THZTensor* t, ntype value) \
  {                 \
    return THError(#OP " is not supported for complex type"); \
  }                 \
  void THZTensor_(NAME##ValueT)(THZTensor* r_, THZTensor* t, ntype value)  \
  {                 \
    return THError(#OP " is not supported for complex type"); \
  }                 \
  void THZTensor_(NAME##Tensor)(THByteTensor *r_, THZTensor *ta, THZTensor *tb) \
  {                 \
    return THError(#OP " is not supported for complex type"); \
  }                 \
  void THZTensor_(NAME##TensorT)(THZTensor *r_, THZTensor *ta, THZTensor *tb) \
  {                 \
    return THError(#OP " is not supported for complex type"); \
  }                 \

TENSOR_IMPLEMENT_COMPLEX_LOGICAL(lt, <)
TENSOR_IMPLEMENT_COMPLEX_LOGICAL(gt, >)
TENSOR_IMPLEMENT_COMPLEX_LOGICAL(le, <=)
TENSOR_IMPLEMENT_COMPLEX_LOGICAL(ge, >=)

TENSOR_IMPLEMENT_LOGICAL(eq,==)
TENSOR_IMPLEMENT_LOGICAL(ne,!=)

#define LAB_IMPLEMENT_BASIC_FUNCTION(NAME, CFUNC)             \
  void THZTensor_(NAME)(THZTensor *r_, THZTensor *t)                \
  {                                                           \
    THZTensor_(resizeAs)(r_, t);                               \
    if (THZTensor_(isContiguous)(r_) && THZTensor_(isContiguous)(t)) { \
      THZ_TENSOR_APPLY2_CONTIG(ntype, r_, ntype, t, THZVector_(NAME)(r__data, t_data, r__len););  \
    } else {  \
      TH_TENSOR_APPLY2(ntype, r_, ntype, t, *r__data = CFUNC(*t_data);); \
    } \
  }

#define LAB_IMPLEMENT_BASIC_FUNCTION_PART(NAME, CFUNC)             \
  void THZTensor_(NAME)(THZPartTensor *r_, THZTensor *t)                \
  {                                                           \
    THZPartTensor_(resizeNd)(r_, t->nDimension, t->size, NULL); \
    if (THZPartTensor_(isContiguous)(r_) && THZTensor_(isContiguous)(t)) { \
      THZ_TENSOR_APPLY2_CONTIG_PART(part, r_, ntype, t, THZVector_(NAME)(r__data, t_data, r__len););  \
    } else {  \
      TH_TENSOR_APPLY2(part, r_, ntype, t, *r__data = CFUNC(*t_data);); \
    } \
  }



/* floating point only now */

LAB_IMPLEMENT_BASIC_FUNCTION(log,THZ_MATH_NAME(log))
LAB_IMPLEMENT_BASIC_FUNCTION(exp,THZ_MATH_NAME(exp))
LAB_IMPLEMENT_BASIC_FUNCTION(cos,THZ_MATH_NAME(cos))
LAB_IMPLEMENT_BASIC_FUNCTION(acos,THZ_MATH_NAME(acos))
LAB_IMPLEMENT_BASIC_FUNCTION(cosh,THZ_MATH_NAME(cosh))
LAB_IMPLEMENT_BASIC_FUNCTION(sin,THZ_MATH_NAME(sin))
LAB_IMPLEMENT_BASIC_FUNCTION(asin,THZ_MATH_NAME(asin))
LAB_IMPLEMENT_BASIC_FUNCTION(sinh,THZ_MATH_NAME(sinh))
LAB_IMPLEMENT_BASIC_FUNCTION(tan,THZ_MATH_NAME(tan))
LAB_IMPLEMENT_BASIC_FUNCTION(atan,THZ_MATH_NAME(atan))
LAB_IMPLEMENT_BASIC_FUNCTION(tanh,THZ_MATH_NAME(tanh))
LAB_IMPLEMENT_BASIC_FUNCTION(sqrt,THZ_MATH_NAME(sqrt))
LAB_IMPLEMENT_BASIC_FUNCTION(neg,-)


// additional math functions
LAB_IMPLEMENT_BASIC_FUNCTION(sigmoid,THZMath_(sigmoid))
LAB_IMPLEMENT_BASIC_FUNCTION(rsqrt,THZMath_(rsqrt))

// complex only
LAB_IMPLEMENT_BASIC_FUNCTION_PART(abs,THZ_MATH_NAME(abs))
LAB_IMPLEMENT_BASIC_FUNCTION_PART(real,THZ_MATH_NAME(real))
LAB_IMPLEMENT_BASIC_FUNCTION_PART(imag,THZ_MATH_NAME(imag))
LAB_IMPLEMENT_BASIC_FUNCTION_PART(arg,THZ_MATH_NAME(arg))
LAB_IMPLEMENT_BASIC_FUNCTION_PART(proj,THZ_MATH_NAME(proj))
// c conjugate is the only function
// that does not use c as name prefix
LAB_IMPLEMENT_BASIC_FUNCTION(conj, THZ_MATH_NAME(onj))
LAB_IMPLEMENT_BASIC_FUNCTION(log1p,THZMath_(log1p))
LAB_IMPLEMENT_BASIC_FUNCTION(cinv, 1.0 / )

// other functions ...

void THZTensor_(pow)(THZTensor *r_, THZTensor *t, ntype value)
{
  THZTensor_(resizeAs)(r_, t);
  if(value == 1){
    THZTensor_(copy)(r_, t);
  }
  else if(value == 2){
    THZTensor_(cmul)(r_, t, t);
  }
  else if(value == 3){
    TH_TENSOR_APPLY2(ntype, r_, ntype, t, *r__data = *t_data * *t_data * *t_data;);
  }
  else if(value == 0.5){
    THZTensor_(sqrt)(r_, t);
  }
  else if(value == -0.5){
    THZTensor_(rsqrt)(r_, t);
  }
  else if(value == -1){
    THZTensor_(cinv)(r_, t);
  }
  else if(value == -2){
    TH_TENSOR_APPLY2(ntype, r_, ntype, t, *r__data = 1.0 / (*t_data * *t_data););
  }
  else{
    TH_TENSOR_APPLY2(ntype, r_, ntype, t, *r__data = THZ_MATH_NAME(pow)(*t_data, value););
  }
}

void THZTensor_(atan2)(THZTensor *r_, THZTensor *tx, THZTensor *ty)
{
  return THError("atan2 is not supported for complex type");
}

void THZTensor_(lerp)(THZTensor *r_, THZTensor *a, THZTensor *b, ntype weight)
{
  THArgCheck(THZTensor_(nElement)(a) == THZTensor_(nElement)(b), 2, "sizes do not match");
  THZTensor_(resizeAs)(r_, a);
  TH_TENSOR_APPLY3(ntype, r_, ntype, a, ntype, b, *r__data = THZMath_(lerp)(*a_data, *b_data, weight););
}

void THZTensor_(mean)(THZTensor *r_, THZTensor *t, int dimension, int keepdim)
{
  THArgCheck(dimension >= 0 && dimension < THZTensor_(nDimension)(t), 2, "invalid dimension %d",
      dimension + TH_INDEX_BASE);

  THZTensor_(sum)(r_, t, dimension, keepdim);
  THZTensor_(div)(r_, r_, t->size[dimension]);
}

void THZTensor_(std)(THZTensor *r_, THZTensor *t, int dimension, int biased, int keepdim)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THZTensor_(nDimension)(t), 3, "invalid dimension %d",
      dimension + TH_INDEX_BASE);

  dim = THZTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THZTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY2(ntype, t, ntype, r_, dimension,
                       // Uses Welford's algorithm for numeric stability
                       accntype mean = 0;
                       accntype M2 = 0;

                       int64_t i;
                       for (i = 0; i < t_size; i++)
                       {
                         ntype z = t_data[i*t_stride];
                         ntype delta = z - mean;
                         mean += delta / (i + 1);
                         ntype delta2 = z - mean;
                         M2 += THZ_MULC(delta, delta2);
                       }

                       if (biased && t_size >= 2)
                       {
                         *r__data = THZ_MATH_NAME(sqrt)(M2 / t_size);
                       } else if (!biased && t_size >= 2) {
                         *r__data = THZ_MATH_NAME(sqrt)(M2 / (t_size - 1));
                       } else if (biased && t_size == 1) {
                         *r__data = 0;
                       } else {
                         *r__data = NAN;
                       });

  if (!keepdim) {
    THZTensor_(squeeze1d)(r_, r_, dimension);
  }
}

void THZTensor_(var)(THZTensor *r_, THZTensor *t, int dimension, int biased, int keepdim)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THZTensor_(nDimension)(t), 3, "invalid dimension %d",
      dimension + TH_INDEX_BASE);

  dim = THZTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THZTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY2(ntype, t, ntype, r_, dimension,
                       // Uses Welford's algorithm for numeric stability
                       accntype mean = 0;
                       accntype M2 = 0;

                       int64_t i;
                       for (i = 0; i < t_size; i++)
                       {
                         ntype z = t_data[i*t_stride];
                         ntype delta = z - mean;
                         mean += delta / (i + 1);
                         ntype delta2 = z - mean;
                         M2 += THZ_MULC(delta, delta2);
                       }

                       if (biased && t_size >= 2)
                       {
                         *r__data = M2 / t_size;
                       } else if (!biased && t_size >= 2) {
                         *r__data = M2 / (t_size - 1);
                       } else if (biased && t_size == 1) {
                         *r__data = 0;
                       } else {
                         *r__data = NAN;
                       });

  if (!keepdim) {
    THZTensor_(squeeze1d)(r_, r_, dimension);
  }
}

void THZTensor_(norm)(THZTensor *r_, THZTensor *t, ntype value, int dimension, int keepdim)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THZTensor_(nDimension)(t), 3, "invalid dimension %d",
      dimension + TH_INDEX_BASE);

  dim = THZTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THZTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  if(value == 0) {
    TH_TENSOR_DIM_APPLY2(ntype, t, ntype, r_, dimension,
                         accntype sum = 0;
                         int64_t i;
                         for(i = 0; i < t_size; i++)
                           sum += t_data[i*t_stride] != 0.0;
                         *r__data = sum;)
  } else {
    TH_TENSOR_DIM_APPLY2(ntype, t, ntype, r_, dimension,
                         accntype sum = 0;
                         int64_t i;
                         for(i = 0; i < t_size; i++) {
                           sum += THZ_MATH_NAME(pow)(
                            THZ_ABS(t_data[i*t_stride]), value);
                         }
                         *r__data = THZ_MATH_NAME(pow)(sum, 1.0/value);)
  }

  if (!keepdim) {
    THZTensor_(squeeze1d)(r_, r_, dimension);
  }
}

accntype THZTensor_(normall)(THZTensor *tensor, ntype value)
{
  accntype sum = 0;
  if(value == 0) {
    TH_TENSOR_APPLY(ntype, tensor, sum += *tensor_data != 0.0;);
    return sum;
  } else if(value == 1) {
    TH_TENSOR_APPLY(ntype, tensor, sum += THZ_ABS(*tensor_data););
    return sum;
  } else if(value == 2) {
    TH_TENSOR_APPLY(ntype, tensor, accntype z = *tensor_data; sum += THZ_MULC(z, z););
    return sqrt(sum);
  } else {
    TH_TENSOR_APPLY(ntype, tensor, sum += THZ_MATH_NAME(pow)(THZ_ABS(*tensor_data), value););
    return THZ_MATH_NAME(pow)(sum, 1.0/value);
  }
}


void THZTensor_(renorm)(THZTensor *res, THZTensor *src, ntype value, int dimension, ntype maxnorm)
{
  return THError("renorm is not supported for complex types");
}

accntype THZTensor_(dist)(THZTensor *tensor, THZTensor *src, ntype value)
{
  ntype sum = 0;
  TH_TENSOR_APPLY2(ntype, tensor, ntype, src,
                   sum += THZ_MATH_NAME(pow)(
                     THZ_ABS(*tensor_data - *src_data), value););
  return THZ_MATH_NAME(pow)(sum, 1.0/value);
}

accntype THZTensor_(meanall)(THZTensor *tensor)
{
  THArgCheck(tensor->nDimension > 0, 1, "empty Tensor");
  return THZTensor_(sumall)(tensor)/THZTensor_(nElement)(tensor);
}

accntype THZTensor_(varall)(THZTensor *tensor, int biased)
{
  accntype mean = THZTensor_(meanall)(tensor);
  accntype sum = 0;
  TH_TENSOR_APPLY(ntype, tensor, sum += THZ_MULC(*tensor_data - mean, *tensor_data - mean););
  sum /= THZTensor_(nElement)(tensor) - (biased ? 0 : 1);
  return sum;
}

accntype THZTensor_(stdall)(THZTensor *tensor, int biased)
{
  return sqrt(THZTensor_(varall)(tensor, biased));
}

void THZTensor_(linspace)(THZTensor *r_, ntype a, ntype b, int64_t n)
{
  ntype i = 0;

  THArgCheck(n > 1 || (n == 1 && (a == b)), 3, "invalid number of points");

  if (THZTensor_(nElement)(r_) != n) {
    THZTensor_(resize1d)(r_, n);
  }

  if(n == 1) {
    THZTensor_(set1d)(r_, 0, a);
  } else {
     TH_TENSOR_APPLY(ntype, r_,
             *r__data = a + i*(b-a)/((ntype)(n-1));
             i++;
           );
  }
}

void THZTensor_(logspace)(THZTensor *r_, ntype a, ntype b, int64_t n)
{
  ntype i = 0;

  THArgCheck(n > 1 || (n == 1 && (a == b)), 3, "invalid number of points");

  if (THZTensor_(nElement)(r_) != n) {
    THZTensor_(resize1d)(r_, n);
  }

  if(n == 1) {
    THZTensor_(set1d)(r_, 0, THZ_MATH_NAME(pow)(10.0, a));
  } else {
    TH_TENSOR_APPLY(ntype, r_,
        *r__data = THZ_MATH_NAME(pow)(10.0, a + i*(b-a)/((ntype)(n-1)));
        i++;
        );
  }
}

void THZTensor_(rand)(THZTensor *r_, THGenerator *_generator, THLongStorage *size)
{
  THZTensor_(resize)(r_, size, NULL);
  THZTensor_(uniform)(r_, _generator, 0, 1);
}

void THZTensor_(randn)(THZTensor *r_, THGenerator *_generator, THLongStorage *size)
{
  return THError("randn is not supported for complex type");
}

void THZTensor_(histc)(THZTensor *hist, THZTensor *tensor, int64_t nbins, ntype minvalue, ntype maxvalue)
{
  return THError("histc is not supported for complex type");
}

void THZTensor_(bhistc)(THZTensor *hist, THZTensor *tensor, int64_t nbins, ntype minvalue, ntype maxvalue)
{
  return THError("bhistc is not supported for complex types");
}

#undef IS_NONZERO

#endif
