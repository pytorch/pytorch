#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensorMoreMath.cpp"
#else

#include <TH/generic/THTensorApply.hpp>
#include <TH/THGenerator.hpp>

void THTensor_(baddbmm)(THTensor *result, scalar_t beta, THTensor *t, scalar_t alpha, THTensor *batch1, THTensor *batch2)
{
  int64_t batch;

  THArgCheck(THTensor_(nDimensionLegacyNoScalars)(batch1) == 3, 1, "expected 3D tensor, got %dD", THTensor_(nDimensionLegacyNoScalars)(batch1));
  THArgCheck(THTensor_(nDimensionLegacyNoScalars)(batch2) == 3, 2, "expected 3D tensor, got %dD", THTensor_(nDimensionLegacyNoScalars)(batch2));
  THArgCheck(THTensor_(size)(batch1, 0) == THTensor_(size)(batch2, 0), 2,
             "equal number of batches expected, got %d, %d",
             THTensor_(size)(batch1, 0), THTensor_(size)(batch2, 0));
  THArgCheck(THTensor_(size)(batch1, 2) == THTensor_(size)(batch2, 1), 2,
             "wrong matrix size, batch1: %dx%d, batch2: %dx%d",
             THTensor_(size)(batch1, 1), THTensor_(size)(batch1, 2),
             THTensor_(size)(batch2, 1), THTensor_(size)(batch2, 2));

  int64_t bs = THTensor_(size)(batch1, 0);
  int64_t dim1 = THTensor_(size)(batch1, 1);
  int64_t dim2 = THTensor_(size)(batch2, 2);
  THArgCheck(THTensor_(size)(t, 0) == bs, 1,   "output tensor of incorrect size");
  THArgCheck(THTensor_(size)(t, 1) == dim1, 1, "output tensor of incorrect size");
  THArgCheck(THTensor_(size)(t, 2) == dim2, 1, "output tensor of incorrect size");

  if (t != result) {
    THTensor_(resizeAs)(result, t);
    if (beta != 0.0) {
      at::Tensor result_wrap = THTensor_wrap(result);
      at::Tensor t_wrap = THTensor_wrap(t);
      at::_copy_same_type_(result_wrap, t_wrap);
    }
  }

  THTensor *matrix1 = THTensor_(new)();
  THTensor *matrix2 = THTensor_(new)();
  THTensor *result_matrix = THTensor_(new)();

  for (batch = 0; batch < THTensor_(size)(batch1, 0); ++batch) {
    THTensor_(select)(matrix1, batch1, 0, batch);
    THTensor_(select)(matrix2, batch2, 0, batch);
    THTensor_(select)(result_matrix, result, 0, batch);

    THTensor_(addmm)(result_matrix, beta, result_matrix, alpha, matrix1, matrix2);
  }

  c10::raw::intrusive_ptr::decref(matrix1);
  c10::raw::intrusive_ptr::decref(matrix2);
  c10::raw::intrusive_ptr::decref(result_matrix);
}

ptrdiff_t THTensor_(numel)(THTensor *t)
{
  return THTensor_(nElement)(t);
}


// Helper function to be used in a reduction operation.
// Due to resize semantics of outputs, if the specified output tensor r_ has
// same size as the output of the reduction operation, then any noncontiguities
// in r_ should be preserved.
// The reduction operation, however, needs to act on r_ with an extra dimension
// (the reduced dimension), so this function "resizes" r_ and preserves its
// noncontiguities if necessary.
void THTensor_(preserveReduceDimSemantics)(
    THTensor *r_, int in_dims, int reduce_dimension, int keepdim) {
  if (r_ && !keepdim &&
      THTensor_(nDimensionLegacyAll)(r_) == in_dims - 1 &&
      THTensor_(nDimensionLegacyAll)(r_) != 0) {
    THTensor_(unsqueeze1d)(r_, r_, reduce_dimension);
  }
}

void THTensor_(max)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim)
{
  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimensionLegacyAll)(t), 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  int in_dims = THTensor_(nDimensionLegacyAll)(t);
  THTensor_(preserveReduceDimSemantics)(values_, in_dims, dimension, keepdim);
  THLongTensor_preserveReduceDimSemantics(indices_, in_dims, dimension, keepdim);
  std::vector<int64_t> dim = THTensor_sizesLegacyNoScalars(t);
  dim[dimension] = 1;
  THTensor_(resize)(values_, dim, {});
  THLongTensor_resize(indices_, dim, {});

  // two implementations optimized for data locality
  if (THTensor_strideLegacyNoScalars(t, dimension) == 1) {
    scalar_t theMax;
    scalar_t value;
    int64_t theIndex;
    int64_t i;
    TH_TENSOR_DIM_APPLY3(scalar_t, t, scalar_t, values_, int64_t, indices_, dimension,
                         TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
                         theMax = t_data[0];
                         theIndex = 0;

                         for(i = 0; i < t_size; i++)
                         {
                           value = t_data[i*t_stride];
                           /* This is not the same as value>theMax in the case of NaNs */
                           if(!(value <= theMax))
                           {
                             theIndex = i;
                             theMax = value;
                             th_isnan_break(value)
                           }
                         }
                         *indices__data = theIndex;
                         *values__data = theMax;);
  } else {
    if (THTensor_(nDimensionLegacyAll)(t) > 1) {
      THTensor *t0 = THTensor_(newSelect)(t, dimension, 0);
      at::Tensor values__wrap = THTensor_wrap(values_);
      at::Tensor t0_wrap = THTensor_wrap(t0);
      at::_copy_same_type_(values__wrap, t0_wrap);
      c10::raw::intrusive_ptr::decref(t0);
    } else {
      THTensor_(fill)(values_, THTensor_(get1d)(t, 0));
    }
    THLongTensor_zero(indices_);

    if(THTensor_sizeLegacyNoScalars(t, dimension) == 1) {
      if (!keepdim) {
        THTensor_(squeeze1d)(values_, values_, dimension);
        THLongTensor_squeeze1d(indices_, indices_, dimension);
      }
      return;
    }

    THTensor *tempValues_ = THTensor_(newWithTensor)(values_);
    // tempValues_.expand_as(t)
    tempValues_->set_size(dimension,THTensor_sizeLegacyNoScalars(t, dimension));
    tempValues_->set_stride(dimension, 0);

    THLongTensor *tempIndices_ = THLongTensor_newWithTensor(indices_);
    // tempIndices_.expand_as(t)
    tempIndices_->set_size(dimension,THTensor_sizeLegacyNoScalars(t, dimension));
    tempIndices_->set_stride(dimension, 0);

    TH_TENSOR_APPLY3_D(scalar_t, t, scalar_t, tempValues_, int64_t, tempIndices_, dimension,
                          if(!(*t_data <= *tempValues__data) && !th_isnan(*tempValues__data)) {
                            *tempValues__data = *t_data;
                            *tempIndices__data = *tempIndices__dimOffset;
                          });

    c10::raw::intrusive_ptr::decref(tempValues_);
    THLongTensor_free(tempIndices_);
  }

  if (!keepdim) {
    THTensor_(squeeze1d)(values_, values_, dimension);
    THLongTensor_squeeze1d(indices_, indices_, dimension);
  }
}

void THTensor_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim)
{
  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimensionLegacyAll)(t), 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  int in_dims = THTensor_(nDimensionLegacyAll)(t);
  THTensor_(preserveReduceDimSemantics)(values_, in_dims, dimension, keepdim);
  THLongTensor_preserveReduceDimSemantics(indices_, in_dims, dimension, keepdim);
  std::vector<int64_t> dim = THTensor_sizesLegacyNoScalars(t);
  dim[dimension] = 1;
  THTensor_(resize)(values_, dim, {});
  THLongTensor_resize(indices_, dim, {});

  // two implementations optimized for data locality
  if (THTensor_strideLegacyNoScalars(t, dimension) == 1) {
    scalar_t theMax;
    scalar_t value;
    int64_t theIndex;
    int64_t i;
    TH_TENSOR_DIM_APPLY3(scalar_t, t, scalar_t, values_, int64_t, indices_, dimension,
                         TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
                         theMax = t_data[0];
                         theIndex = 0;

                         for(i = 0; i < t_size; i++)
                         {
                           value = t_data[i*t_stride];
                           /* This is not the same as value>theMax in the case of NaNs */
                           if(!(value >= theMax))
                           {
                             theIndex = i;
                             theMax = value;
                             th_isnan_break(value)
                           }
                         }
                         *indices__data = theIndex;
                         *values__data = theMax;);
  } else {
    if (THTensor_(nDimensionLegacyAll)(t) > 1) {
      THTensor *t0 = THTensor_(newSelect)(t, dimension, 0);
      at::Tensor values__wrap = THTensor_wrap(values_);
      at::Tensor t0_wrap = THTensor_wrap(t0);
      at::_copy_same_type_(values__wrap, t0_wrap);
      c10::raw::intrusive_ptr::decref(t0);
    } else {
      THTensor_(fill)(values_, THTensor_(get1d)(t, 0));
    }
    THLongTensor_zero(indices_);

    if(THTensor_sizeLegacyNoScalars(t, dimension) == 1) {
      if (!keepdim) {
        THTensor_(squeeze1d)(values_, values_, dimension);
        THLongTensor_squeeze1d(indices_, indices_, dimension);
      }
      return;
    }

    THTensor *tempValues_ = THTensor_(newWithTensor)(values_);
    // tempValues_.expand_as(t)
    tempValues_->set_size(dimension,THTensor_sizeLegacyNoScalars(t, dimension));
    tempValues_->set_stride(dimension, 0);

    THLongTensor *tempIndices_ = THLongTensor_newWithTensor(indices_);
    // tempIndices_.expand_as(t)
    tempIndices_->set_size(dimension,THTensor_sizeLegacyNoScalars(t, dimension));
    tempIndices_->set_stride(dimension, 0);

    TH_TENSOR_APPLY3_D(scalar_t, t, scalar_t, tempValues_, int64_t, tempIndices_, dimension,
                          if(!(*t_data >= *tempValues__data) && !th_isnan(*tempValues__data)) {
                            *tempValues__data = *t_data;
                            *tempIndices__data = *tempIndices__dimOffset;
                          });

    c10::raw::intrusive_ptr::decref(tempValues_);
    THLongTensor_free(tempIndices_);
  }

  if (!keepdim) {
    THTensor_(squeeze1d)(values_, values_, dimension);
    THLongTensor_squeeze1d(indices_, indices_, dimension);
  }
}

void THTensor_(sum)(THTensor *r_, THTensor *t, int dimension, int keepdim)
{
  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimensionLegacyAll)(t), 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  THTensor_(preserveReduceDimSemantics)(r_, THTensor_(nDimensionLegacyAll)(t), dimension, keepdim);
  std::vector<int64_t> dim = THTensor_sizesLegacyNoScalars(t);
  dim[dimension] = 1;
  THTensor_(resize)(r_, dim, {});

  int serial_path = 0;
#ifdef _OPENMP
  int inOMP = omp_in_parallel();
  if (inOMP) {
    serial_path = 1;
  } else {
    int r_Contig = THTensor_(isContiguous)(r_);
    scalar_t *tp = t->data<scalar_t>();
    scalar_t *rp = r_->data<scalar_t>();
    if(r_Contig && (tp != rp)){
      ptrdiff_t iter = 0;
      ptrdiff_t r_Size = THTensor_(nElement)(r_);
      int r_Dim = THTensor_nDimensionLegacyAll(r_);
      #pragma omp parallel for if ( r_Size > HYPER_TH_OMP_OVERHEAD_THRESHOLD)
      for (iter = 0; iter < r_Size; iter++) {
        int j;
        int64_t quot;
        int64_t rem = iter;
        ptrdiff_t tBasicIndex = 0;

        for(j = 0; j < r_Dim; ++j) {
          if(j != dimension){
            quot = rem/r_->stride(j);
            rem = rem%r_->stride(j);
            tBasicIndex += quot*t->stride(j);
          }
        }
        scalar_t *t_data = tp+tBasicIndex;
        scalar_t *r__data = rp+iter;
        *r__data = 0;
        for(j=0; j < THTensor_sizeLegacyNoScalars(t, dimension); ++j) {
          *r__data += *(t_data + j*THTensor_strideLegacyNoScalars(t, dimension));
        }
      }
    } else {
      serial_path = 1;
    }
  }
#else
  serial_path = 1;
#endif
  if (serial_path) {
    // two implementations optimized for data locality
    if (THTensor_strideLegacyNoScalars(t, dimension) == 1) {
      TH_TENSOR_DIM_APPLY2(scalar_t, t, scalar_t, r_, dimension,
                           accreal sum = 0;
                           int64_t i;
                           for(i = 0; i < t_size; i++)
                             sum += t_data[i*t_stride];
                           *r__data = (scalar_t)sum;);
    } else {
      THTensor_(zero)(r_);
      THTensor *temp_ = THTensor_(newWithTensor)(r_);
      // r_.expand_as(t)
      temp_->set_size(dimension,THTensor_sizeLegacyNoScalars(t, dimension));
      temp_->set_stride(dimension, 0);

      TH_TENSOR_APPLY2(scalar_t, temp_, scalar_t, t, *temp__data = *temp__data + *t_data;);
      c10::raw::intrusive_ptr::decref(temp_);
    }
  }

  if (!keepdim) {
    THTensor_(squeeze1d)(r_, r_, dimension);
  }
}

void THTensor_(prod)(THTensor *r_, THTensor *t, int dimension, int keepdim)
{
  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimensionLegacyAll)(t), 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  THTensor_(preserveReduceDimSemantics)(r_, THTensor_(nDimensionLegacyAll)(t), dimension, keepdim);
  std::vector<int64_t> dim = THTensor_sizesLegacyNoScalars(t);
  dim[dimension] = 1;
  THTensor_(resize)(r_, dim, {});

  int serial_path = 0;
#ifdef _OPENMP
  int inOMP = omp_in_parallel();
  if (inOMP) {
    serial_path = 1;
  } else {
    int r_Contig = THTensor_(isContiguous)(r_);
    scalar_t *tp = t->data<scalar_t>();
    scalar_t *rp = r_->data<scalar_t>();
    if(r_Contig && (tp != rp)){
      ptrdiff_t iter = 0;
      ptrdiff_t r_Size = THTensor_(nElement)(r_);
      int r_Dim = THTensor_nDimensionLegacyAll(r_);
      #pragma omp parallel for if ( r_Size > HYPER_TH_OMP_OVERHEAD_THRESHOLD)
      for (iter = 0; iter < r_Size; iter++) {
        int j;
        int64_t quot;
        int64_t rem = iter;
        ptrdiff_t tBasicIndex = 0;

        for(j = 0; j < r_Dim; ++j) {
          if(j != dimension){
            quot = rem/r_->stride(j);
            rem = rem%r_->stride(j);
            tBasicIndex += quot*t->stride(j);
          }
        }
        scalar_t *t_data = tp+tBasicIndex;
        scalar_t *r__data = rp+iter;
        *r__data = 1;
        for(j=0; j < THTensor_sizeLegacyNoScalars(t, dimension); ++j) {
          *r__data *= *(t_data + j*THTensor_strideLegacyNoScalars(t, dimension));
        }
      }
    } else {
      serial_path = 1;
    }
  }
#else
  serial_path = 1;
#endif

  if(serial_path) {
    // two implementations optimized for data locality
    if (THTensor_strideLegacyNoScalars(t, dimension) == 1) {
      TH_TENSOR_DIM_APPLY2(scalar_t, t, scalar_t, r_, dimension,
                           accreal prod = 1;
                           int64_t i;
                           for(i = 0; i < t_size; i++)
                             prod *= t_data[i*t_stride];
                           *r__data = (scalar_t)prod;);
    } else {
      THTensor_(fill)(r_, 1);
      THTensor *temp_ = THTensor_(newWithTensor)(r_);
      // r_.expand_as(t)
      temp_->set_size(dimension,THTensor_sizeLegacyNoScalars(t, dimension));
      temp_->set_stride(dimension, 0);

      TH_TENSOR_APPLY2(scalar_t, temp_, scalar_t, t, *temp__data = *temp__data * *t_data;);
      c10::raw::intrusive_ptr::decref(temp_);
    }
  }
  if (!keepdim) {
    THTensor_(squeeze1d)(r_, r_, dimension);
  }
}

void THTensor_(cumsum)(THTensor *r_, THTensor *t, int dimension)
{
  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimensionLegacyNoScalars)(t), 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  THTensor_(resizeAs)(r_, t);

  TH_TENSOR_DIM_APPLY2(scalar_t, t, scalar_t, r_, dimension,
                       accreal cumsum = 0;
                       int64_t i;
                       for(i = 0; i < t_size; i++)
                       {
                         cumsum += t_data[i*t_stride];
                         r__data[i*r__stride] = (scalar_t)cumsum;
                       });
}

void THTensor_(cumprod)(THTensor *r_, THTensor *t, int dimension)
{
  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimensionLegacyNoScalars)(t), 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  THTensor_(resizeAs)(r_, t);

  TH_TENSOR_DIM_APPLY2(scalar_t, t, scalar_t, r_, dimension,
                       accreal cumprod = 1;
                       int64_t i;
                       for(i = 0; i < t_size; i++)
                       {
                         cumprod *= t_data[i*t_stride];
                         r__data[i*r__stride] = (scalar_t)cumprod;
                       });
}


void THTensor_(sign)(THTensor *r_, THTensor *t)
{
  THTensor_(resizeAs)(r_, t);

#if defined (TH_REAL_IS_BYTE)
  TH_TENSOR_APPLY2(scalar_t, r_, scalar_t, t,
    if (*t_data > 0) *r__data = 1;
    else *r__data = 0;);
#else
  TH_TENSOR_APPLY2(scalar_t, r_, scalar_t, t,
    if (*t_data > 0) *r__data = 1;
    else if (*t_data < 0) *r__data = -1;
    else *r__data = 0;);
#endif
}


accreal THTensor_(trace)(THTensor *t)
{
  scalar_t *t_data = t->data<scalar_t>();
  accreal sum = 0;
  int64_t i = 0;
  int64_t t_stride_0, t_stride_1, t_diag_size;

  THArgCheck(THTensor_(nDimensionLegacyAll)(t) == 2, 1, "expected a matrix");

  t_stride_0 = THTensor_(stride)(t, 0);
  t_stride_1 = THTensor_(stride)(t, 1);
  t_diag_size = THMin(THTensor_(size)(t, 0), THTensor_(size)(t, 1));
  while(i < t_diag_size)
  {
    sum += t_data[i*(t_stride_0+t_stride_1)];
    i++;
  }

  return sum;
}

void THTensor_(cross)(THTensor *r_, THTensor *a, THTensor *b, int dimension)
{
  int i;

  if(THTensor_(nDimensionLegacyNoScalars)(a) != THTensor_(nDimensionLegacyNoScalars)(b))
    THError("inconsistent tensor dimension %dD, %dD",
        THTensor_(nDimensionLegacyNoScalars)(a), THTensor_(nDimensionLegacyNoScalars)(b));

  for(i = 0; i < a->dim(); i++)
  {
    if(THTensor_(size)(a, i) != THTensor_(size)(b, i)) {
        THDescBuff ba = THTensor_(sizeDesc)(a);
        THDescBuff bb = THTensor_(sizeDesc)(b);
        THError("inconsistent tensor sizes %s, %s", ba.str, bb.str);
    }
  }

  if(dimension < 0)
  {
    for(i = 0; i < THTensor_(nDimensionLegacyNoScalars)(a); i++)
    {
      if(THTensor_sizeLegacyNoScalars(a, i) == 3)
      {
        dimension = i;
        break;
      }
    }
    if(dimension < 0) {
      THDescBuff ba = THTensor_(sizeDesc)(a);
      THError("no dimension of size 3 in a: %s", ba.str);
    }
  }

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimensionLegacyNoScalars)(a), 3, "dimension %d out of range",
      dimension + TH_INDEX_BASE);
  THArgCheck(THTensor_sizeLegacyNoScalars(a, dimension) == 3, 3, "dimension %d does not have size 3",
      dimension + TH_INDEX_BASE);

  THTensor_(resizeAs)(r_, a);

  TH_TENSOR_DIM_APPLY3(scalar_t, a, scalar_t, b, scalar_t, r_, dimension,
                       TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
                       r__data[0*r__stride] = a_data[1*a_stride]*b_data[2*b_stride] - a_data[2*a_stride]*b_data[1*b_stride];
                       r__data[1*r__stride] = a_data[2*a_stride]*b_data[0*b_stride] - a_data[0*a_stride]*b_data[2*b_stride];
                       r__data[2*r__stride] = a_data[0*a_stride]*b_data[1*b_stride] - a_data[1*a_stride]*b_data[0*b_stride];);
}

void THTensor_(cmax)(THTensor *r, THTensor *t, THTensor *src) {
  THTensor_(resizeAs)(r, t);
  TH_TENSOR_APPLY3(scalar_t, r, scalar_t, t, scalar_t, src,
                   *r_data = *t_data > *src_data ? *t_data : *src_data;);
}

void THTensor_(cmin)(THTensor *r, THTensor *t, THTensor *src) {
  THTensor_(resizeAs)(r, t);
  TH_TENSOR_APPLY3(scalar_t, r, scalar_t, t, scalar_t, src,
                   *r_data = *t_data < *src_data ? *t_data : *src_data;);
}

void THTensor_(cmaxValue)(THTensor *r, THTensor *t, scalar_t value) {
  THTensor_(resizeAs)(r, t);
  TH_TENSOR_APPLY2(scalar_t, r, scalar_t, t,
                   *r_data = *t_data < value ? value : *t_data;);  // this order propagates NaN
}

void THTensor_(cminValue)(THTensor *r, THTensor *t, scalar_t value) {
  THTensor_(resizeAs)(r, t);
  TH_TENSOR_APPLY2(scalar_t, r, scalar_t, t,
                   *r_data = *t_data > value ? value : *t_data;);  // this order propagates NaN
}

void THTensor_(zerosLike)(THTensor *r_, THTensor *input)
{
  THTensor_(resizeAs)(r_, input);
  THTensor_(zero)(r_);
}

void THTensor_(onesLike)(THTensor *r_, THTensor *input)
{
  THTensor_(resizeAs)(r_, input);
  THTensor_(fill)(r_, 1);
}

void THTensor_(diag)(THTensor *r_, THTensor *t, int k)
{
  THArgCheck(THTensor_(nDimensionLegacyNoScalars)(t) == 1 || THTensor_(nDimensionLegacyNoScalars)(t) == 2, 1, "matrix or a vector expected");

  if(THTensor_(nDimensionLegacyNoScalars)(t) == 1)
  {
    scalar_t *t_data = t->data<scalar_t>();
    int64_t t_stride_0 = THTensor_strideLegacyNoScalars(t, 0);
    int64_t t_size = THTensor_sizeLegacyNoScalars(t, 0);
    int64_t sz = t_size + (k >= 0 ? k : -k);
    scalar_t *r__data;
    int64_t r__stride_0;
    int64_t r__stride_1;
    int64_t i;

    THTensor_(resize2d)(r_, sz, sz);
    THTensor_(zero)(r_);
    r__data = r_->data<scalar_t>();
    r__stride_0 = THTensor_(stride)(r_, 0);
    r__stride_1 = THTensor_(stride)(r_, 1);
    r__data += (k >= 0 ? k*r__stride_1 : -k*r__stride_0);

    for(i = 0; i < t_size; i++)
      r__data[i*(r__stride_0+r__stride_1)] = t_data[i*t_stride_0];
  }
  else
  {
    scalar_t *t_data = t->data<scalar_t>();
    int64_t t_stride_0 = THTensor_(stride)(t, 0);
    int64_t t_stride_1 = THTensor_(stride)(t, 1);
    int64_t sz;
    scalar_t *r__data;
    int64_t r__stride_0;
    int64_t i;

    if(k >= 0)
      sz = THMin(THTensor_(size)(t, 0), THTensor_(size)(t, 1)-k);
    else
      sz = THMin(THTensor_(size)(t, 0)+k, THTensor_(size)(t, 1));
    THTensor_(resize1d)(r_, sz);
    r__data = r_->data<scalar_t>();
    r__stride_0 = THTensor_(stride)(r_, 0);

    t_data += (k >= 0 ? k*t_stride_1 : -k*t_stride_0);
    for(i = 0; i < sz; i++)
      r__data[i*r__stride_0] = t_data[i*(t_stride_0+t_stride_1)];
  }
}

void THTensor_(eye)(THTensor *r_, int64_t n, int64_t m)
{
  scalar_t *r__data;
  int64_t i, sz;

  THArgCheck(n > 0, 1, "invalid argument");

  if(m <= 0)
    m = n;

  THTensor_(resize2d)(r_, n, m);
  THTensor_(zero)(r_);

  i = 0;
  r__data = r_->data<scalar_t>();
  sz = THMin(THTensor_(size)(r_, 0), THTensor_(size)(r_, 1));
  for(i = 0; i < sz; i++)
    r__data[i*(r_->stride(0)+r_->stride(1))] = 1;
}

void THTensor_(randperm)(THTensor *r_, THGenerator *_generator, int64_t n)
{
  std::lock_guard<std::mutex> lock(_generator->mutex);
  scalar_t *r__data;
  int64_t r__stride_0;
  int64_t i;

  THArgCheck(n > 0, 1, "must be strictly positive");

  THTensor_(resize1d)(r_, n);
  r__data = r_->data<scalar_t>();
  r__stride_0 = THTensor_(stride)(r_,0);

  for(i = 0; i < n; i++)
    r__data[i*r__stride_0] = (scalar_t)(i);

  for(i = 0; i < n-1; i++)
  {
    int64_t z = THRandom_random(_generator) % (n-i);
    scalar_t sav = r__data[i*r__stride_0];
    r__data[i*r__stride_0] = r__data[(z+i)*r__stride_0];
    r__data[(z+i)*r__stride_0] = sav;
  }
}

/* I cut and pasted (slightly adapted) the quicksort code from
   Sedgewick's 1978 "Implementing Quicksort Programs" article
   http://www.csie.ntu.edu.tw/~b93076/p847-sedgewick.pdf

   It is the state of the art existing implementation. The macros
   are here to make as close a match as possible to the pseudocode of
   Program 2 p.851

   Note that other partition schemes exist, and are typically presented
   in textbook, but those are less efficient. See e.g.
   http://cs.stackexchange.com/questions/11458/quicksort-partitioning-hoare-vs-lomuto

   Julien, November 12th 2013
*/
#define MAX_LEVELS  300
#define M_SMALL 10 /* Limit for small subfiles */

#define ARR(III) arr[(III)*stride]
#define IDX(III) idx[(III)*stride]

#define LONG_SWAP(AAA, BBB) swap = AAA; AAA = BBB; BBB = swap
#define REAL_SWAP(AAA, BBB) rswap = AAA; AAA = BBB; BBB = rswap

#define ARR_SWAP(III, JJJ) \
  REAL_SWAP(ARR(III), ARR(JJJ));

#define BOTH_SWAP(III, JJJ) \
  REAL_SWAP(ARR(III), ARR(JJJ)); \
  LONG_SWAP(IDX(III), IDX(JJJ))

/* Emulate NumPy behavior of putting NaNs
 * at the end of an ascending list. */
#define GT_OR_NAN(x, y) \
  ((x != x && y == y) || (x > y))

static void THTensor_(quicksortascend)(scalar_t *arr, int64_t *idx, int64_t elements, int64_t stride)
{
  int64_t beg[MAX_LEVELS], end[MAX_LEVELS], i, j, L, R, P, swap, pid, stack = 0, sz_right, sz_left;
  scalar_t rswap, piv;
  unsigned char done = 0;

  /* beg[0]=0; end[0]=elements; */
  stack = 0;
  L = 0; R = elements-1;
  done = elements-1 <= M_SMALL;

  while(!done) {
      /* Use median of three for pivot choice */
    P=(L+R)>>1;
    BOTH_SWAP(P, L+1);
    if (GT_OR_NAN(ARR(L+1), ARR(R))) { BOTH_SWAP(L+1, R); }
    if (GT_OR_NAN(ARR(L), ARR(R))) { BOTH_SWAP(L, R); }
    if (GT_OR_NAN(ARR(L+1), ARR(L))) { BOTH_SWAP(L+1, L); }

    i = L+1; j = R; piv = ARR(L); pid = IDX(L);

    do {
      do { i = i+1; } while(GT_OR_NAN(piv, ARR(i)));
      do { j = j-1; } while(GT_OR_NAN(ARR(j), piv));
      if (j < i)
          break;
      BOTH_SWAP(i, j);
    } while(1);
    BOTH_SWAP(L, j);
    /* Left subfile is (L, j-1) */
    /* Right subfile is (i, R) */
    sz_left = j-L;
    sz_right = R-i+1;
    if (sz_left <= M_SMALL && sz_right <= M_SMALL) {
      /* both subfiles are small */
      /* if stack empty */
      if (stack == 0) {
        done = 1;
      } else {
        stack--;
        L = beg[stack];
        R = end[stack];
      }
    } else if (sz_left <= M_SMALL || sz_right <= M_SMALL) {
      /* exactly one of the subfiles is small */
      /* (L,R) = large subfile */
      if (sz_left > sz_right) {
        /* Implicit: L = L; */
        R = j-1;
      } else {
        L = i;
        /* Implicit: R = R; */
      }
    } else {
      /* none of the subfiles is small */
      /* push large subfile */
      /* (L,R) = small subfile */
      if (sz_left > sz_right) {
        beg[stack] = L;
        end[stack] = j-1;
        stack++;
        L = i;
        /* Implicit: R = R */
      } else {
        beg[stack] = i;
        end[stack] = R;
        stack++;
        /* Implicit: L = L; */
        R = j-1;
      }
    }
  } /* while not done */
  /* Now insertion sort on the concatenation of subfiles */
  for(i=elements-2; i>=0; i--) {
    if (GT_OR_NAN(ARR(i),ARR(i+1))) {
      piv = ARR(i);
      pid = IDX(i);
      j = i+1;
      do {
        ARR(j-1) = ARR(j);
        IDX(j-1) = IDX(j);
        j = j+1;
      } while(j < elements && GT_OR_NAN(piv, ARR(j)));
      ARR(j-1) = piv;
      IDX(j-1) = pid;
     }
  }
}

static void THTensor_(quicksortdescend)(scalar_t *arr, int64_t *idx, int64_t elements, int64_t stride)
{
  int64_t beg[MAX_LEVELS], end[MAX_LEVELS], i, j, L, R, P, swap, pid, stack = 0, sz_right, sz_left;
  scalar_t rswap, piv;
  unsigned char done = 0;

  /* beg[0]=0; end[0]=elements; */
  stack = 0;
  L = 0; R = elements-1;
  done = elements-1 <= M_SMALL;

  while(!done) {
      /* Use median of three for pivot choice */
    P=(L+R)>>1;
    BOTH_SWAP(P, L+1);
    if (GT_OR_NAN(ARR(R), ARR(L+1))) { BOTH_SWAP(L+1, R); }
    if (GT_OR_NAN(ARR(R), ARR(L))) { BOTH_SWAP(L, R); }
    if (GT_OR_NAN(ARR(L), ARR(L+1))) { BOTH_SWAP(L+1, L); }

    i = L+1; j = R; piv = ARR(L); pid = IDX(L);

    do {
      do { i = i+1; } while(GT_OR_NAN(ARR(i), piv));
      do { j = j-1; } while(GT_OR_NAN(piv, ARR(j)));
      if (j < i)
          break;
      BOTH_SWAP(i, j);
    } while(1);
    BOTH_SWAP(L, j);
    /* Left subfile is (L, j-1) */
    /* Right subfile is (i, R) */
    sz_left = j-L;
    sz_right = R-i+1;
    if (sz_left <= M_SMALL && sz_right <= M_SMALL) {
      /* both subfiles are small */
      /* if stack empty */
      if (stack == 0) {
        done = 1;
      } else {
        stack--;
        L = beg[stack];
        R = end[stack];
      }
    } else if (sz_left <= M_SMALL || sz_right <= M_SMALL) {
      /* exactly one of the subfiles is small */
      /* (L,R) = large subfile */
      if (sz_left > sz_right) {
        /* Implicit: L = L; */
        R = j-1;
      } else {
        L = i;
        /* Implicit: R = R; */
      }
    } else {
      /* none of the subfiles is small */
      /* push large subfile */
      /* (L,R) = small subfile */
      if (sz_left > sz_right) {
        beg[stack] = L;
        end[stack] = j-1;
        stack++;
        L = i;
        /* Implicit: R = R */
      } else {
        beg[stack] = i;
        end[stack] = R;
        stack++;
        /* Implicit: L = L; */
        R = j-1;
      }
    }
  } /* while not done */
  /* Now insertion sort on the concatenation of subfiles */
  for(i=elements-2; i>=0; i--) {
    if (GT_OR_NAN(ARR(i+1), ARR(i))) {
      piv = ARR(i);
      pid = IDX(i);
      j = i+1;
      do {
        ARR(j-1) = ARR(j);
        IDX(j-1) = IDX(j);
        j = j+1;
      } while(j < elements && GT_OR_NAN(ARR(j), piv));
      ARR(j-1) = piv;
      IDX(j-1) = pid;
     }
  }
}

#undef MAX_LEVELS
#undef M_SMALL

void THTensor_(sort)(THTensor *rt_, THLongTensor *ri_, THTensor *t, int dimension, int descendingOrder)
{
  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimensionLegacyNoScalars)(t), 2, "invalid dimension %d",
      dimension + TH_INDEX_BASE);

  THTensor_(resizeAs)(rt_, t);
  at::Tensor rt__wrap = THTensor_wrap(rt_);
  at::Tensor t_wrap = THTensor_wrap(t);
  at::_copy_same_type_(rt__wrap, t_wrap);
  THLongTensor_resize(ri_, t->sizes(), {});

  if(descendingOrder)
  {
    TH_TENSOR_DIM_APPLY2(scalar_t, rt_, int64_t, ri_, dimension,
                         int64_t i;
                         for(i = 0; i < ri__size; i++)
                           ri__data[i*ri__stride] = i;
                         THTensor_(quicksortdescend)(rt__data, ri__data, rt__size, rt__stride);)
      }
  else
  {
    TH_TENSOR_DIM_APPLY2(scalar_t, rt_, int64_t, ri_, dimension,
                         int64_t i;
                         for(i = 0; i < ri__size; i++)
                           ri__data[i*ri__stride] = i;
                         THTensor_(quicksortascend)(rt__data, ri__data, rt__size, rt__stride);)
      }
}

/* Implementation of the Quickselect algorithm, based on Nicolas Devillard's
public domain implementation at http://ndevilla.free.fr/median/median/
Adapted similarly to the above Quicksort algorithm.
This version does not produce indices along with values. */
static void THTensor_(quickselectnoidx)(scalar_t *arr, int64_t k, int64_t elements, int64_t stride)
{
  int64_t P, L, R, i, j;
  scalar_t rswap, piv;
  L = 0;
  R = elements-1;

  do {
    if (R <= L) /* One element only */
      return;

    if (R == L+1) {  /* Two elements only */
      if (ARR(L) > ARR(R)) {
        ARR_SWAP(L, R);
      }
      return;
    }

    /* Use median of three for pivot choice */
    P=(L+R)>>1;
    ARR_SWAP(P, L+1);
    if (ARR(L+1) > ARR(R)) { ARR_SWAP(L+1, R); }
    if (ARR(L) > ARR(R)) { ARR_SWAP(L, R); }
    if (ARR(L+1) > ARR(L)) { ARR_SWAP(L+1, L); }

    i = L+1;
    j = R;
    piv = ARR(L);
    do {
      do i++; while(ARR(i) < piv);
      do j--; while(ARR(j) > piv);
      if (j < i)
        break;
      ARR_SWAP(i, j);
    } while(1);
    ARR_SWAP(L, j);

    /* Re-set active partition */
    if (j <= k) L=i;
    if (j >= k) R=j-1;
  } while(1);
}

/* Implementation of the Quickselect algorithm, based on Nicolas Devillard's
public domain implementation at http://ndevilla.free.fr/median/median/
Adapted similarly to the above Quicksort algorithm. */
static void THTensor_(quickselect)(scalar_t *arr, int64_t *idx, int64_t k, int64_t elements, int64_t stride)
{
  int64_t P, L, R, i, j, swap;
  scalar_t rswap, piv;
  L = 0;
  R = elements-1;

  do {
    if (R <= L) /* One element only */
      return;

    if (R == L+1) {  /* Two elements only */
      if (ARR(L) > ARR(R)) {
        BOTH_SWAP(L, R);
      }
      return;
    }

    /* Use median of three for pivot choice */
    P=(L+R)>>1;
    BOTH_SWAP(P, L+1);
    if (ARR(L+1) > ARR(R)) { BOTH_SWAP(L+1, R); }
    if (ARR(L) > ARR(R)) { BOTH_SWAP(L, R); }
    if (ARR(L+1) > ARR(L)) { BOTH_SWAP(L+1, L); }

    i = L+1;
    j = R;
    piv = ARR(L);
    do {
      do i++; while(ARR(i) < piv);
      do j--; while(ARR(j) > piv);
      if (j < i)
        break;
      BOTH_SWAP(i, j);
    } while(1);
    BOTH_SWAP(L, j);

    /* Re-set active partition */
    if (j <= k) L=i;
    if (j >= k) R=j-1;
  } while(1);
}

#undef ARR
#undef IDX
#undef LONG_SWAP
#undef REAL_SWAP
#undef BOTH_SWAP

scalar_t THTensor_(medianall)(THTensor *tensor)
{
  THArgCheck(THTensor_nDimensionLegacyAll(tensor) > 0, 1, "tensor must have one dimension");

  scalar_t theMedian;
  ptrdiff_t numel;
  int64_t k;
  THTensor *temp_;
  scalar_t *temp__data;

  numel = THTensor_(nElement)(tensor);
  k = (numel-1) >> 1;

  temp_ = THTensor_(newClone)(tensor);
  temp__data = temp_->data<scalar_t>();

  THTensor_(quickselectnoidx)(temp__data, k, numel, 1);

  theMedian = temp__data[k];

  c10::raw::intrusive_ptr::decref(temp_);

  return theMedian;
}

void THTensor_(mode)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim)
{
  THTensor *temp_;
  THLongTensor *tempi_;
  scalar_t *temp__data;
  int64_t *tempi__data;
  int64_t t_size_dim;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimensionLegacyAll)(t), 3, "dimension out of range");

  int in_dims = THTensor_(nDimensionLegacyAll)(t);
  THTensor_(preserveReduceDimSemantics)(values_, in_dims, dimension, keepdim);
  THLongTensor_preserveReduceDimSemantics(indices_, in_dims, dimension, keepdim);
  std::vector<int64_t> dim = THTensor_sizesLegacyNoScalars(t);
  dim[dimension] = 1;
  THTensor_(resize)(values_, dim, {});
  THLongTensor_resize(indices_, dim, {});

  t_size_dim = THTensor_sizeLegacyNoScalars(t, dimension);

  temp_ = THTensor_(new)();
  THTensor_(resize1d)(temp_, t_size_dim);
  temp__data = temp_->data<scalar_t>();

  tempi_ = THLongTensor_new();
  THLongTensor_resize1d(tempi_, t_size_dim);
  tempi__data = THLongTensor_data(tempi_);

  TH_TENSOR_DIM_APPLY3(scalar_t, t, scalar_t, values_, int64_t, indices_, dimension,
                       TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
                       int64_t i;
                       scalar_t mode = 0;
                       int64_t modei = 0;
                       int64_t temp_freq = 0;
                       int64_t max_freq = 0;
                       for(i = 0; i < t_size_dim; i++)
                          temp__data[i] = t_data[i*t_stride];
                       for(i = 0; i < t_size_dim; i++)
                          tempi__data[i] = i;
                       THTensor_(quicksortascend)(temp__data, tempi__data, t_size_dim, 1);

                       for(i = 0; i < t_size_dim; i++)
                       {
                          temp_freq++;
                          if ((i == t_size_dim - 1) || (temp__data[i] != temp__data[i+1]))
                          {
                              if (temp_freq > max_freq)
                              {
                                 mode = temp__data[i];
                                 modei = tempi__data[i];
                                 max_freq = temp_freq;
                              }
                              temp_freq = 0;
                          }
                       }
                       *values__data = mode;
                       *indices__data = modei;);

  c10::raw::intrusive_ptr::decref(temp_);
  THLongTensor_free(tempi_);
  if (!keepdim) {
    THTensor_(squeeze1d)(values_, values_, dimension);
    THLongTensor_squeeze1d(indices_, indices_, dimension);
  }
}

void THTensor_(kthvalue)(THTensor *values_, THLongTensor *indices_, THTensor *t, int64_t k, int dimension, int keepdim)
{
  THTensor *temp_;
  THLongTensor *tempi_;
  scalar_t *temp__data;
  int64_t *tempi__data;
  int64_t t_size_dim;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimensionLegacyAll)(t), 3, "dimension out of range");
  THArgCheck(k > 0 && k <= THTensor_sizeLegacyNoScalars(t, dimension), 2, "selected index out of range");

  int in_dims = THTensor_(nDimensionLegacyAll)(t);
  THTensor_(preserveReduceDimSemantics)(values_, in_dims, dimension, keepdim);
  THLongTensor_preserveReduceDimSemantics(indices_, in_dims, dimension, keepdim);
  std::vector<int64_t> dim = THTensor_sizesLegacyNoScalars(t);
  dim[dimension] = 1;
  THTensor_(resize)(values_, dim, {});
  THLongTensor_resize(indices_, dim, {});

  t_size_dim = THTensor_sizeLegacyNoScalars(t, dimension);

  temp_ = THTensor_(new)();
  THTensor_(resize1d)(temp_, t_size_dim);
  temp__data = temp_->data<scalar_t>();

  tempi_ = THLongTensor_new();
  THLongTensor_resize1d(tempi_, t_size_dim);
  tempi__data = THLongTensor_data(tempi_);

  TH_TENSOR_DIM_APPLY3(scalar_t, t, scalar_t, values_, int64_t, indices_, dimension,
                       TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
                       int64_t i;
                       for(i = 0; i < t_size_dim; i++)
                          temp__data[i] = t_data[i*t_stride];
                       for(i = 0; i < t_size_dim; i++)
                          tempi__data[i] = i;
                       THTensor_(quickselect)(temp__data, tempi__data, k - 1, t_size_dim, 1);
                       *values__data = temp__data[k-1];
                       *indices__data = tempi__data[k-1];);

  c10::raw::intrusive_ptr::decref(temp_);
  THLongTensor_free(tempi_);
  if (!keepdim) {
    THTensor_(squeeze1d)(values_, values_, dimension);
    THLongTensor_squeeze1d(indices_, indices_, dimension);
  }
}

void THTensor_(median)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim)
{
  int64_t t_size_dim, k;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimensionLegacyAll)(t), 3, "dimension out of range");

  t_size_dim = THTensor_sizeLegacyNoScalars(t, dimension);
  k = (t_size_dim-1) >> 1; /* take middle or one-before-middle element */

  THTensor_(kthvalue)(values_, indices_, t, k+1, dimension, keepdim);
}

void THTensor_(topk)(THTensor *rt_, THLongTensor *ri_, THTensor *t, int64_t k, int dim, int dir, int sorted)
{
  int numDims = THTensor_(nDimensionLegacyNoScalars)(t);
  THArgCheck(dim >= 0 && dim < numDims, 3, "dim not in range");

  int64_t sliceSize = THTensor_sizeLegacyNoScalars(t, dim);
  THArgCheck(k >= 0 && k <= sliceSize, 2, "k not in range for dimension");

  THTensor *tmpResults = THTensor_(new)();
  THTensor_(resize1d)(tmpResults, sliceSize);
  scalar_t *tmp__data = tmpResults->data<scalar_t>();

  THLongTensor *tmpIndices = THLongTensor_new();
  THLongTensor_resize1d(tmpIndices, sliceSize);
  int64_t *tmpi__data = THLongTensor_data(tmpIndices);

  std::vector<int64_t> topKSize = t->sizes().vec();
  if (topKSize.size() > 0) { // handle 0-dim vs 1-dim differences.
    topKSize[dim] = k;
  }
  THTensor_(resize)(rt_, topKSize, {});
  THLongTensor_resize(ri_, topKSize, {});

  if (dir) {
    /* k largest elements, descending order (optional: see sorted) */
    int64_t K = sliceSize - k;
    TH_TENSOR_DIM_APPLY3(scalar_t, t, scalar_t, rt_, int64_t, ri_, dim,
                         TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
                         int64_t i;
                         for(i = 0; i < sliceSize; i++)
                         {
                           tmp__data[i] = t_data[i*t_stride];
                           tmpi__data[i] = i;
                         }
                         if (K > 0)
                           THTensor_(quickselect)(tmp__data, tmpi__data, K - 1, sliceSize, 1);
                         if (sorted)
                           THTensor_(quicksortdescend)(tmp__data + K, tmpi__data + K, k, 1);
                         for(i = 0; i < k; i++)
                         {
                           rt__data[i*rt__stride] = tmp__data[i + K];
                           ri__data[i*ri__stride] = tmpi__data[i + K];
                         })
  }
  else {
    /* k smallest elements, ascending order (optional: see sorted) */
    TH_TENSOR_DIM_APPLY3(scalar_t, t, scalar_t, rt_, int64_t, ri_, dim,
                         TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
                         int64_t i;
                         for(i = 0; i < sliceSize; i++)
                         {
                           tmp__data[i] = t_data[i*t_stride];
                           tmpi__data[i] = i;
                         }
                         THTensor_(quickselect)(tmp__data, tmpi__data, k - 1, sliceSize, 1);
                         if (sorted)
                           THTensor_(quicksortascend)(tmp__data, tmpi__data, k - 1, 1);
                         for(i = 0; i < k; i++)
                         {
                           rt__data[i*rt__stride] = tmp__data[i];
                           ri__data[i*ri__stride] = tmpi__data[i];
                         })
  }

  c10::raw::intrusive_ptr::decref(tmpResults);
  THLongTensor_free(tmpIndices);
}

void THTensor_(triu)(THTensor *r_, THTensor *t, int64_t k)
{
  int64_t t_size_0, t_size_1;
  int64_t t_stride_0, t_stride_1;
  int64_t r__stride_0, r__stride_1;
  scalar_t *t_data, *r__data;
  int64_t r, c;

  THArgCheck(THTensor_(nDimensionLegacyAll)(t) == 2, 1, "expected a matrix");

  THTensor_(resizeAs)(r_, t);

  t_size_0 = THTensor_(size)(t, 0);
  t_size_1 = THTensor_(size)(t, 1);
  t_stride_0 = THTensor_(stride)(t, 0);
  t_stride_1 = THTensor_(stride)(t, 1);
  r__stride_0 = THTensor_(stride)(r_, 0);
  r__stride_1 = THTensor_(stride)(r_, 1);
  r__data = r_->data<scalar_t>();
  t_data = t->data<scalar_t>();

  for(r = 0; r < t_size_0; r++)
  {
    int64_t sz = THMin(r+k, t_size_1);
    for(c = THMax(0, r+k); c < t_size_1; c++)
      r__data[r*r__stride_0+c*r__stride_1] = t_data[r*t_stride_0+c*t_stride_1];
    for(c = 0; c < sz; c++)
      r__data[r*r__stride_0+c*r__stride_1] = 0;
  }
}

int THTensor_(equal)(THTensor *ta, THTensor* tb)
{
  int equal = 1;
  if(!THTensor_(isSameSizeAs)(ta, tb))
    return 0;

  if (THTensor_(isContiguous)(ta) && THTensor_(isContiguous)(tb)) {
    scalar_t *tap = ta->data<scalar_t>();
    scalar_t *tbp = tb->data<scalar_t>();
    ptrdiff_t sz = THTensor_(nElement)(ta);
    ptrdiff_t i;
    for (i=0; i<sz; ++i){
      if(tap[i] != tbp[i]) return 0;
    }
  } else {
    // Short-circuit the apply function on inequality
    TH_TENSOR_APPLY2(scalar_t, ta, scalar_t, tb,
                     if (equal && *ta_data != *tb_data) {
                        equal = 0;
                        TH_TENSOR_APPLY_hasFinished = 1; break;
                     })
  }
  return equal;
}

#define TENSOR_IMPLEMENT_LOGICAL(NAME,OP)				\
  void THTensor_(NAME##Value)(THByteTensor *r_, THTensor* t, scalar_t value)	\
  {									\
    THByteTensor_resizeNd(r_, t->dim(), THTensor_getSizePtr(t), NULL);		\
    TH_TENSOR_APPLY2(unsigned char, r_, scalar_t, t,			\
		     *r__data = (*t_data OP value) ? 1 : 0;); \
  }									\
  void THTensor_(NAME##ValueT)(THTensor* r_, THTensor* t, scalar_t value)	\
  {									\
    THTensor_(resizeNd)(r_, t->dim(), THTensor_getSizePtr(t), NULL);		\
    TH_TENSOR_APPLY2(scalar_t, r_, scalar_t, t,					\
		     *r__data = (*t_data OP value) ? 1 : 0;); \
  }									\
  void THTensor_(NAME##Tensor)(THByteTensor *r_, THTensor *ta, THTensor *tb) \
  {									\
    THByteTensor_resizeNd(r_, ta->dim(), THTensor_getSizePtr(ta), NULL);		\
    TH_TENSOR_APPLY3(unsigned char, r_, scalar_t, ta, scalar_t, tb,		\
		     *r__data = (*ta_data OP *tb_data) ? 1 : 0;); \
  }									\
  void THTensor_(NAME##TensorT)(THTensor *r_, THTensor *ta, THTensor *tb) \
  {									\
    THTensor_(resizeNd)(r_, ta->dim(), THTensor_getSizePtr(ta), NULL);		\
    TH_TENSOR_APPLY3(scalar_t, r_, scalar_t, ta, scalar_t, tb,			\
		     *r__data = (*ta_data OP *tb_data) ? 1 : 0;); \
  }									\


TENSOR_IMPLEMENT_LOGICAL(lt,<)
TENSOR_IMPLEMENT_LOGICAL(gt,>)
TENSOR_IMPLEMENT_LOGICAL(le,<=)
TENSOR_IMPLEMENT_LOGICAL(ge,>=)
TENSOR_IMPLEMENT_LOGICAL(eq,==)
TENSOR_IMPLEMENT_LOGICAL(ne,!=)


#ifdef _OPENMP

#define LAB_IMPLEMENT_BASIC_FUNCTION_3_ARGS(NAME, CFUNC, OMP_THRESHOLD)             \
  void THTensor_(NAME)(THTensor *r_, THTensor *t)             \
  {                                                           \
    THTensor_(resizeAs)(r_, t);                               \
    ptrdiff_t r_Size = THTensor_(nElement)(r_);               \
    int r_Contig = THTensor_(isContiguous)(r_);               \
    int tContig = THTensor_(isContiguous)(t);                 \
    int inOMP = omp_in_parallel();                            \
    if( !inOMP ){   \
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, scalar_t, r_, scalar_t, t, *r__data = CFUNC(*t_data);, OMP_THRESHOLD);        \
    } else {                                                                                                   \
      TH_TENSOR_APPLY2(scalar_t, r_, scalar_t, t, *r__data = CFUNC(*t_data););                                       \
    }                                                                                                        \
  }

#define LAB_IMPLEMENT_BASIC_FUNCTION_2_ARGS(NAME, CFUNC)      \
  LAB_IMPLEMENT_BASIC_FUNCTION_3_ARGS(NAME, CFUNC, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD)

#define LAB_IMPLEMENT_VECTORIZED_FUNCTION_3_ARGS(NAME, CFUNC, OMP_THRESHOLD)             \
  void THTensor_(NAME)(THTensor *r_, THTensor *t)             \
  {                                                           \
    THTensor_(resizeAs)(r_, t);                               \
    ptrdiff_t r_Size = THTensor_(nElement)(r_);               \
    int r_Contig = THTensor_(isContiguous)(r_);               \
    int tContig = THTensor_(isContiguous)(t);                 \
    if (r_Contig && tContig) {                                \
      TH_TENSOR_APPLY2_CONTIG(scalar_t, r_, scalar_t, t, THVector_(NAME)(r__data, t_data, r__len););                   \
    } else {                                                                                                   \
      int inOMP = omp_in_parallel();                            \
      if( !inOMP ){   \
        TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, scalar_t, r_, scalar_t, t, *r__data = CFUNC(*t_data);, OMP_THRESHOLD);        \
      }                                                                                                        \
      else {                                                                                                   \
        TH_TENSOR_APPLY2(scalar_t, r_, scalar_t, t, *r__data = CFUNC(*t_data););                                       \
      }                                                                                                        \
    }                                                                                                          \
  }

#define LAB_IMPLEMENT_VECTORIZED_FUNCTION_2_ARGS(NAME, CFUNC) \
  LAB_IMPLEMENT_VECTORIZED_FUNCTION_3_ARGS(NAME, CFUNC, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD)

#else

#define LAB_IMPLEMENT_BASIC_FUNCTION_2_ARGS(NAME, CFUNC)             \
  void THTensor_(NAME)(THTensor *r_, THTensor *t)                \
  {                                                           \
    THTensor_(resizeAs)(r_, t);                               \
    TH_TENSOR_APPLY2(scalar_t, t, scalar_t, r_, *r__data = CFUNC(*t_data);); \
  }                                                           \

#define LAB_IMPLEMENT_BASIC_FUNCTION_3_ARGS(NAME, CFUNC, PSEUDO_OMP_THRESHOLD) \
  LAB_IMPLEMENT_BASIC_FUNCTION_2_ARGS(NAME, CFUNC)

#define LAB_IMPLEMENT_VECTORIZED_FUNCTION_2_ARGS(NAME, CFUNC)             \
  void THTensor_(NAME)(THTensor *r_, THTensor *t)                \
  {                                                           \
    THTensor_(resizeAs)(r_, t);                               \
    int r_Contig = THTensor_(isContiguous)(r_);               \
    int tContig = THTensor_(isContiguous)(t);                 \
    if (r_Contig && tContig) {                                \
      TH_TENSOR_APPLY2_CONTIG(scalar_t, r_, scalar_t, t, THVector_(NAME)(r__data, t_data, r__len);); \
    } else {                                                           \
      TH_TENSOR_APPLY2(scalar_t, t, scalar_t, r_, *r__data = CFUNC(*t_data);); \
    }                                                           \
  }                                                             \

#define LAB_IMPLEMENT_VECTORIZED_FUNCTION_3_ARGS(NAME, CFUNC, PSEUDO_OMP_THRESHOLD) \
  LAB_IMPLEMENT_VECTORIZED_FUNCTION_2_ARGS(NAME, CFUNC)

#endif

#define EXPAND(...) __VA_ARGS__

#define GET_4TH_ARG(ARG0, ARG1, ARG2, ARG3, ...) ARG3

#define LAB_IMPLEMENT_BASIC_FUNCTION_CHOOSE(...) \
  EXPAND(GET_4TH_ARG(__VA_ARGS__, LAB_IMPLEMENT_BASIC_FUNCTION_3_ARGS, LAB_IMPLEMENT_BASIC_FUNCTION_2_ARGS, ))

#define LAB_IMPLEMENT_VECTORIZED_FUNCTION_CHOOSE(...) \
  EXPAND(GET_4TH_ARG(__VA_ARGS__, LAB_IMPLEMENT_VECTORIZED_FUNCTION_3_ARGS, LAB_IMPLEMENT_VECTORIZED_FUNCTION_2_ARGS, ))

#define LAB_IMPLEMENT_BASIC_FUNCTION(...) EXPAND(LAB_IMPLEMENT_BASIC_FUNCTION_CHOOSE(__VA_ARGS__)(__VA_ARGS__))

#define LAB_IMPLEMENT_VECTORIZED_FUNCTION(...) EXPAND(LAB_IMPLEMENT_VECTORIZED_FUNCTION_CHOOSE(__VA_ARGS__)(__VA_ARGS__))

/*
 * LAB_IMPLEMENT_BASIC_FUNCTION is a macro with optional parameters, you can use it flexibly.
 * The macro will discard the invalid openmp threshold if openmp is unavailable. The macro will give a default threshold even if you forget to pass one.
 * In other word,
 * (A), If openmp is UNavailable, the two usage below is both right.
 *      (1) LAB_IMPLEMENT_BASIC_FUNCTION(type_func, func_entity, OMP_OVERHEAD_THRESHOLD) // discard the invalid openmp threshold
 *      (2) LAB_IMPLEMENT_BASIC_FUNCTION(type_func, func_entity)
 * (B), If openmp is available, the two usage below is also both right.
 *      (1) LAB_IMPLEMENT_BASIC_FUNCTION(type_func, func_entity, OMP_OVERHEAD_THRESHOLD)
 *      (2) LAB_IMPLEMENT_BASIC_FUNCTION(type_func, func_entity) // pass the default openmp threshold
 * So do LAB_IMPLEMENT_VECTORIZED_FUNCTION.
*/

LAB_IMPLEMENT_BASIC_FUNCTION(neg,-)

#if defined(TH_REAL_IS_LONG)
LAB_IMPLEMENT_BASIC_FUNCTION(abs,labs)
#endif /* int64_t only part */

#if defined(TH_REAL_IS_SHORT) || defined(TH_REAL_IS_INT) || defined(TH_REAL_IS_CHAR)
LAB_IMPLEMENT_BASIC_FUNCTION(abs,abs)
#endif /* int only part */

#if defined(TH_REAL_IS_BYTE)
LAB_IMPLEMENT_BASIC_FUNCTION(abs,)
#endif /* for byte, identity due to it being unsigned */

/* floating point only now */
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

#if defined (TH_REAL_IS_FLOAT)
#define TH_MATH_NAME(fn) fn##f
#else
#define TH_MATH_NAME(fn) fn
#endif

LAB_IMPLEMENT_BASIC_FUNCTION(log,TH_MATH_NAME(log))
LAB_IMPLEMENT_BASIC_FUNCTION(lgamma,TH_MATH_NAME(lgamma))
LAB_IMPLEMENT_BASIC_FUNCTION(digamma,TH_MATH_NAME(TH_digamma))
LAB_IMPLEMENT_BASIC_FUNCTION(trigamma,TH_MATH_NAME(TH_trigamma))
LAB_IMPLEMENT_BASIC_FUNCTION(log10,TH_MATH_NAME(log10))
LAB_IMPLEMENT_BASIC_FUNCTION(log1p,TH_MATH_NAME(log1p))
LAB_IMPLEMENT_BASIC_FUNCTION(log2,TH_MATH_NAME(log2))
LAB_IMPLEMENT_BASIC_FUNCTION(erf,TH_MATH_NAME(erf))
LAB_IMPLEMENT_BASIC_FUNCTION(erfc,TH_MATH_NAME(erfc))
LAB_IMPLEMENT_BASIC_FUNCTION(erfinv,TH_erfinv)
LAB_IMPLEMENT_BASIC_FUNCTION(ceil,TH_MATH_NAME(ceil))
LAB_IMPLEMENT_BASIC_FUNCTION(floor,TH_MATH_NAME(floor))
LAB_IMPLEMENT_BASIC_FUNCTION(round,TH_MATH_NAME(round))
LAB_IMPLEMENT_BASIC_FUNCTION(abs,TH_MATH_NAME(fabs))
LAB_IMPLEMENT_BASIC_FUNCTION(trunc,TH_MATH_NAME(trunc))
LAB_IMPLEMENT_BASIC_FUNCTION(frac,TH_MATH_NAME(TH_frac))
LAB_IMPLEMENT_BASIC_FUNCTION(cinv, TH_MATH_NAME(1.0) / )

LAB_IMPLEMENT_BASIC_FUNCTION(exp,TH_MATH_NAME(exp),HYPER_TH_OMP_OVERHEAD_THRESHOLD)
LAB_IMPLEMENT_BASIC_FUNCTION(expm1,TH_MATH_NAME(expm1),HYPER_TH_OMP_OVERHEAD_THRESHOLD)
LAB_IMPLEMENT_BASIC_FUNCTION(cos,TH_MATH_NAME(cos),HYPER_TH_OMP_OVERHEAD_THRESHOLD)
LAB_IMPLEMENT_BASIC_FUNCTION(acos,TH_MATH_NAME(acos),HYPER_TH_OMP_OVERHEAD_THRESHOLD)
LAB_IMPLEMENT_BASIC_FUNCTION(cosh,TH_MATH_NAME(cosh),HYPER_TH_OMP_OVERHEAD_THRESHOLD)
LAB_IMPLEMENT_BASIC_FUNCTION(sin,TH_MATH_NAME(sin),HYPER_TH_OMP_OVERHEAD_THRESHOLD)
LAB_IMPLEMENT_BASIC_FUNCTION(asin,TH_MATH_NAME(asin),HYPER_TH_OMP_OVERHEAD_THRESHOLD)
LAB_IMPLEMENT_BASIC_FUNCTION(sinh,TH_MATH_NAME(sinh),HYPER_TH_OMP_OVERHEAD_THRESHOLD)
LAB_IMPLEMENT_BASIC_FUNCTION(tan,TH_MATH_NAME(tan),HYPER_TH_OMP_OVERHEAD_THRESHOLD)
LAB_IMPLEMENT_BASIC_FUNCTION(atan,TH_MATH_NAME(atan),HYPER_TH_OMP_OVERHEAD_THRESHOLD)
LAB_IMPLEMENT_BASIC_FUNCTION(tanh,TH_MATH_NAME(tanh),HYPER_TH_OMP_OVERHEAD_THRESHOLD)
LAB_IMPLEMENT_BASIC_FUNCTION(sqrt,TH_MATH_NAME(sqrt),HYPER_TH_OMP_OVERHEAD_THRESHOLD)
LAB_IMPLEMENT_BASIC_FUNCTION(rsqrt,TH_MATH_NAME(TH_rsqrt),HYPER_TH_OMP_OVERHEAD_THRESHOLD)

LAB_IMPLEMENT_VECTORIZED_FUNCTION(sigmoid,TH_MATH_NAME(TH_sigmoid),HYPER_TH_OMP_OVERHEAD_THRESHOLD)

void THTensor_(atan2)(THTensor *r_, THTensor *tx, THTensor *ty)
{
  THTensor_(resizeAs)(r_, tx);
  TH_TENSOR_APPLY3(scalar_t, r_, scalar_t, tx, scalar_t, ty, *r__data = TH_MATH_NAME(atan2)(*tx_data,*ty_data););
}

void THTensor_(polygamma)(THTensor *r_, int64_t n, THTensor *t) {
  switch (n) {
    case 0: THTensor_(digamma)(r_, t); return;
    case 1: THTensor_(trigamma)(r_, t); return;
    default: THError("polygamma(n,x) is not implemented for n>=2");
  }
}

void THTensor_(lerp)(THTensor *r_, THTensor *a, THTensor *b, scalar_t weight)
{
  THArgCheck(THTensor_(nElement)(a) == THTensor_(nElement)(b), 2, "sizes do not match");
  THTensor_(resizeAs)(r_, a);
  TH_TENSOR_APPLY3(scalar_t, r_, scalar_t, a, scalar_t, b, *r__data = TH_MATH_NAME(TH_lerp)(*a_data, *b_data, weight););
}

void THTensor_(mean)(THTensor *r_, THTensor *t, int dimension, int keepdim)
{
  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimensionLegacyAll)(t), 2, "invalid dimension %d",
      dimension + TH_INDEX_BASE);

  THTensor_(sum)(r_, t, dimension, keepdim);
  THTensor_(div)(r_, r_, THTensor_sizeLegacyNoScalars(t, dimension));
}

void THTensor_(std)(THTensor *r_, THTensor *t, int dimension, int biased, int keepdim)
{
  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimensionLegacyAll)(t), 3, "invalid dimension %d",
      dimension + TH_INDEX_BASE);

  THTensor_(preserveReduceDimSemantics)(r_, THTensor_(nDimensionLegacyAll)(t), dimension, keepdim);
  std::vector<int64_t> dim = THTensor_sizesLegacyNoScalars(t);
  dim[dimension] = 1;
  THTensor_(resize)(r_, dim, {});

  TH_TENSOR_DIM_APPLY2(scalar_t, t, scalar_t, r_, dimension,
                       // Uses Welford's algorithm for numeric stability
                       accreal mean = 0;
                       accreal M2 = 0;

                       int64_t i;
                       for (i = 0; i < t_size; i++)
                       {
                         scalar_t z = t_data[i*t_stride];
                         scalar_t delta = z - mean;
                         mean += delta / (i + 1);
                         scalar_t delta2 = z - mean;
                         M2 += delta * delta2;
                       }

                       if (biased && t_size >= 2)
                       {
                         *r__data = TH_MATH_NAME(sqrt)(M2 / t_size);
                       } else if (!biased && t_size >= 2) {
                         *r__data = TH_MATH_NAME(sqrt)(M2 / (t_size - 1));
                       } else if (biased && t_size == 1) {
                         *r__data = 0;
                       } else {
                         *r__data = NAN;
                       });

  if (!keepdim) {
    THTensor_(squeeze1d)(r_, r_, dimension);
  }
}

void THTensor_(var)(THTensor *r_, THTensor *t, int dimension, int biased, int keepdim)
{
  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimensionLegacyAll)(t), 3, "invalid dimension %d",
      dimension + TH_INDEX_BASE);

  THTensor_(preserveReduceDimSemantics)(r_, THTensor_(nDimensionLegacyAll)(t), dimension, keepdim);
  std::vector<int64_t> dim = THTensor_sizesLegacyNoScalars(t);
  dim[dimension] = 1;
  THTensor_(resize)(r_, dim, {});

  TH_TENSOR_DIM_APPLY2(scalar_t, t, scalar_t, r_, dimension,
                       // Uses Welford's algorithm for numeric stability
                       accreal mean = 0;
                       accreal M2 = 0;

                       int64_t i;
                       for (i = 0; i < t_size; i++)
                       {
                         scalar_t z = t_data[i*t_stride];
                         scalar_t delta = z - mean;
                         mean += delta / (i + 1);
                         scalar_t delta2 = z - mean;
                         M2 += delta * delta2;
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
    THTensor_(squeeze1d)(r_, r_, dimension);
  }
}

void THTensor_(norm)(THTensor *r_, THTensor *t, scalar_t value, int dimension, int keepdim)
{
  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimensionLegacyAll)(t), 3, "invalid dimension %d",
      dimension + TH_INDEX_BASE);

  THTensor_(preserveReduceDimSemantics)(r_, THTensor_(nDimensionLegacyAll)(t), dimension, keepdim);
  std::vector<int64_t> dim = THTensor_sizesLegacyNoScalars(t);
  dim[dimension] = 1;
  THTensor_(resize)(r_, dim, {});

  #define DIM_REDUCE(reduce, transform, init) \
    TH_TENSOR_DIM_APPLY2(scalar_t, t, scalar_t, r_, dimension,      \
                         accreal sum = init;                \
                         int64_t i;                         \
                         for(i = 0; i < t_size; i++) {      \
                           (reduce);                        \
                         }                                  \
                         (transform);)                      \

  if(value == 0) {
    DIM_REDUCE(sum += t_data[i*t_stride] != 0.0,
               *r__data = sum, 0);
  } else if (value == 1) {
    DIM_REDUCE(sum += TH_MATH_NAME(fabs)(t_data[i*t_stride]),
               *r__data = sum, 0);
  } else if (value == 2) {
    DIM_REDUCE(sum += t_data[i*t_stride] * t_data[i*t_stride],
               *r__data = TH_MATH_NAME(sqrt)(sum), 0);
  } else if (value == 3) {
    DIM_REDUCE(sum += TH_MATH_NAME(fabs)(t_data[i*t_stride] * t_data[i*t_stride] * t_data[i*t_stride]),
               *r__data = TH_MATH_NAME(pow)(sum, 1.0/3), 0);
  } else if (value == INFINITY) {
    DIM_REDUCE(sum = THMax(sum, TH_MATH_NAME(fabs)(t_data[i*t_stride])),
	       *r__data = sum, 0);
  } else if (value == -INFINITY) {
    DIM_REDUCE(sum = THMin(sum, TH_MATH_NAME(fabs)(t_data[i*t_stride])),
	       *r__data = sum, INFINITY);
  } else {
    DIM_REDUCE(sum += TH_MATH_NAME(pow)(TH_MATH_NAME(fabs)(t_data[i*t_stride]), value),
               *r__data = TH_MATH_NAME(pow)(sum, 1.0/value), 0);
  }

  if (!keepdim) {
    THTensor_(squeeze1d)(r_, r_, dimension);
  }
  #undef DIM_REDUCE
}

accreal THTensor_(normall)(THTensor *tensor, scalar_t value)
{
  accreal sum = 0;
  if(value == 0) {
    TH_TENSOR_APPLY(scalar_t, tensor, sum += *tensor_data != 0.0;);
    return sum;
  } else if(value == 1) {
    TH_TENSOR_APPLY(scalar_t, tensor, sum += TH_MATH_NAME(fabs)(*tensor_data););
    return sum;
  } else if(value == 2) {
    TH_TENSOR_APPLY(scalar_t, tensor, accreal z = *tensor_data; sum += z*z;);
    return sqrt(sum);
  } else if(value == 3) {
    TH_TENSOR_APPLY(scalar_t, tensor, accreal z = *tensor_data; sum += std::abs(z*z*z););
    return TH_MATH_NAME(pow)(sum, 1.0/3);
  } else if(value == INFINITY) {
    TH_TENSOR_APPLY(scalar_t, tensor, sum = THMax(sum, TH_MATH_NAME(fabs)(*tensor_data)););
    return sum;
  } else if(value == -INFINITY) {
    sum = INFINITY;
    TH_TENSOR_APPLY(scalar_t, tensor, sum = THMin(sum, TH_MATH_NAME(fabs)(*tensor_data)););
    return sum;
  } else {
    TH_TENSOR_APPLY(scalar_t, tensor, sum += TH_MATH_NAME(pow)(TH_MATH_NAME(fabs)(*tensor_data), value););
    return TH_MATH_NAME(pow)(sum, 1.0/value);
  }
}

void THTensor_(renorm)(THTensor *res, THTensor *src, scalar_t value, int dimension, scalar_t maxnorm)
{
  THTensor *rowR, *rowS;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimensionLegacyNoScalars)(src), 3, "invalid dimension %d",
      dimension + TH_INDEX_BASE);
  THArgCheck(value > 0, 2, "non-positive-norm not supported");
  THArgCheck(THTensor_(nDimensionLegacyNoScalars)(src) > 1, 1, "need at least 2 dimensions, got %d dimensions",
      THTensor_(nDimensionLegacyNoScalars)(src));

  rowR = THTensor_(new)();
  rowS = THTensor_(new)();

  THTensor_(resizeAs)(res, src);

  for (int64_t i = 0; i < THTensor_sizeLegacyNoScalars(src, dimension); i++)
  {
    scalar_t norm = 0;
    scalar_t new_norm;

    THTensor_(select)(rowS, src, dimension, i);
    THTensor_(select)(rowR, res, dimension, i);
    if (value == 1) {
      TH_TENSOR_APPLY(scalar_t, rowS, norm += fabs(*rowS_data););
    } else if (value == 2) {
      TH_TENSOR_APPLY(scalar_t, rowS, accreal z = *rowS_data; norm += z*z;);
    } else if (value == INFINITY) {
      TH_TENSOR_APPLY(scalar_t, rowS, norm = THMax(norm, TH_MATH_NAME(fabs)(*rowS_data)););
    } else {
      TH_TENSOR_APPLY(scalar_t, rowS, norm += TH_MATH_NAME(pow)(TH_MATH_NAME(fabs)(*rowS_data), value););
    }

    if (value != INFINITY) {
      norm = pow(norm, 1/value);
    }

    if (norm > maxnorm)
    {
      new_norm = maxnorm / (norm + 1e-7);

      TH_TENSOR_APPLY2(
        scalar_t, rowR, scalar_t, rowS,
        *rowR_data = (*rowS_data) * new_norm;
      )
    }
    else
    {
      at::Tensor rowR_wrap = THTensor_wrap(rowR);
      at::Tensor rowS_wrap = THTensor_wrap(rowS);
      at::_copy_same_type_(rowR_wrap, rowS_wrap);
    }
  }

  c10::raw::intrusive_ptr::decref(rowR);
  c10::raw::intrusive_ptr::decref(rowS);
}

accreal THTensor_(dist)(THTensor *tensor, THTensor *src, scalar_t value)
{
  scalar_t sum;
  if (value == INFINITY) {
    sum = -1.0;
    TH_TENSOR_APPLY2(scalar_t, tensor, scalar_t, src,
                     sum = THMax(sum, TH_MATH_NAME(fabs)(*tensor_data - *src_data)););
    return sum;
  } else if (value == -INFINITY) {
    sum = INFINITY;
    TH_TENSOR_APPLY2(scalar_t, tensor, scalar_t, src,
                     sum = THMin(sum, TH_MATH_NAME(fabs)(*tensor_data - *src_data)););
    return sum;
  } else if (value == 0.0) {
    sum = 0.0;
    TH_TENSOR_APPLY2(scalar_t, tensor, scalar_t, src,
                     sum += (*tensor_data - *src_data != 0.0););
    return sum;
  } else {
    sum = 0.0;
    TH_TENSOR_APPLY2(scalar_t, tensor, scalar_t, src,
                     sum += TH_MATH_NAME(pow)(
                       TH_MATH_NAME(fabs)(*tensor_data - *src_data), value););
    return TH_MATH_NAME(pow)(sum, 1.0/value);
  }
}

accreal THTensor_(meanall)(THTensor *tensor)
{
  return THTensor_(sumall)(tensor)/THTensor_(nElement)(tensor);
}

accreal THTensor_(varall)(THTensor *tensor, int biased)
{
  accreal mean = THTensor_(meanall)(tensor);
  accreal sum = 0;
  TH_TENSOR_APPLY(scalar_t, tensor, sum += (*tensor_data - mean)*(*tensor_data - mean););
  sum /= std::max<int64_t>(0, THTensor_(nElement)(tensor) - (biased ? 0 : 1));
  return sum;
}

accreal THTensor_(stdall)(THTensor *tensor, int biased)
{
  return sqrt(THTensor_(varall)(tensor, biased));
}

void THTensor_(histc)(THTensor *hist, THTensor *tensor, int64_t nbins, scalar_t minvalue, scalar_t maxvalue)
{
  if (nbins <= 0) {
      THError("bins must be > 0");
  }
  scalar_t minval;
  scalar_t maxval;
  scalar_t *h_data;

  THTensor_(resize1d)(hist, nbins);
  THTensor_(zero)(hist);
  minval = minvalue;
  maxval = maxvalue;
  if (minval == maxval)
  {
    minval = THTensor_(minall)(tensor);
    maxval = THTensor_(maxall)(tensor);
  }
  if (minval == maxval)
  {
    minval = minval - 1;
    maxval = maxval + 1;
  }

  h_data = hist->data<scalar_t>();

  TH_TENSOR_APPLY(scalar_t, tensor,
    if (*tensor_data >= minval && *tensor_data <= maxval) {
      const int bin = (int)((*tensor_data-minval) / (maxval-minval) * nbins);
      h_data[THMin(bin, nbins-1)] += 1;
    }
  );
}

void THTensor_(bhistc)(THTensor *hist, THTensor *tensor, int64_t nbins, scalar_t minvalue, scalar_t maxvalue)
{
  THArgCheck(THTensor_(nDimensionLegacyAll)(tensor) < 3, 2, "invalid dimension %d, the input must be a 2d tensor", THTensor_(nDimensionLegacyAll)(tensor));

  int dimension = 1;
  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimensionLegacyAll)(tensor), 2, "invalid dimension %d",
      dimension + TH_INDEX_BASE);

  scalar_t minval;
  scalar_t maxval;

  THTensor_(resize2d)(hist, THTensor_sizeLegacyNoScalars(tensor, 0), nbins);
  THTensor_(zero)(hist);

  minval = minvalue;
  maxval = maxvalue;
  if (minval == maxval)
  {
    minval = THTensor_(minall)(tensor);
    maxval = THTensor_(maxall)(tensor);
  }
  if (minval == maxval)
  {
    minval = minval - 1;
    maxval = maxval + 1;
  }

  TH_TENSOR_DIM_APPLY2(scalar_t, tensor, scalar_t, hist, dimension, int64_t i;
                        for(i = 0; i < tensor_size; i++)
                        {
                          if(tensor_data[i*tensor_stride] >= minval && tensor_data[i*tensor_stride] <= maxval) {
                            const int bin = (int)((tensor_data[i*tensor_stride]-minval) / (maxval-minval) * nbins);
                            hist_data[THMin(bin, nbins-1)] += 1;
                          }
                        }
  );
}

// Approximate reparameterized gradient of Beta(x,alpha,beta) wrt alpha.
// Assumes x is close to zero and uses a Taylor expansion.
static inline scalar_t THTensor_(beta_grad_alpha_small)(scalar_t x, scalar_t alpha, scalar_t beta) {
  const scalar_t factor = TH_MATH_NAME(TH_digamma)(alpha) - TH_MATH_NAME(TH_digamma)(alpha + beta) - TH_MATH_NAME(log)(x);
  scalar_t numer = 1;
  scalar_t series = numer / alpha * (factor + 1 / alpha);
  for (int i = 1; i <= 10; ++i) {
    numer *= (i - beta) * x / i;
    const scalar_t denom = alpha + i;
    series += numer / denom * (factor + 1 / denom);
  }
  const scalar_t result = x * TH_MATH_NAME(pow)(1 - x, -beta) * series;
  return th_isnan(result) ? 0.0 : result;
}

// Approximate reparameterized gradient of Beta(x,alpha,beta) wrt beta.
// Assumes x is close to zero and uses a Taylor expansion.
static inline scalar_t THTensor_(beta_grad_beta_small)(scalar_t x, scalar_t alpha, scalar_t beta) {
  const scalar_t factor = TH_MATH_NAME(TH_digamma)(alpha+beta) - TH_MATH_NAME(TH_digamma)(beta);
  scalar_t numer = 1;
  scalar_t betas = 1;
  scalar_t dbetas = 0;
  scalar_t series = factor / alpha;
  for (int i = 1; i <= 8; ++i) {
    numer *= -x / i;
    dbetas = dbetas * (beta - i) + betas;
    betas = betas * (beta - i);
    series += numer / (alpha + i) * (dbetas + factor * betas);
  }
  const scalar_t result = -TH_MATH_NAME(pow)(1 - x, 1 - beta) * series;
  return th_isnan(result) ? 0.0 : result;
}

// Approximate reparameterized gradient of Beta(x,alpha,beta) wrt alpha.
// Assumes alpha and beta are both large and uses a Rice saddle point expansion.
// To ensure numerical stability, this computation is performed at higher precision.
static inline scalar_t THTensor_(beta_grad_alpha_mid)(double x, double alpha, double beta) {
  const double total = alpha + beta;
  const double mean = alpha / total;
  const double std = sqrt(alpha * beta / (total + 1)) / total;
  if (mean - 0.1 * std <= x && x <= mean + 0.1 * std) {
    // Avoid the singularity at x = mean.
    const double poly = 47 * x * (beta*beta)*(beta*beta) + alpha * (
                      (43 + 20 * (16 + 27 * beta) * x) * (beta*beta)*beta + alpha * (
                      3 * (59 + 180 * beta - 90 * x) * (beta*beta) + alpha * (
                      (453 + 1620 * beta * (1 - x) - 455 * x) * beta + alpha * (
                      8 * (1 - x) * (135 * beta - 11)))));
    const double prefactor_num = (1 + 12 * alpha) * (1 + 12 * beta) / (total * total);
    const double prefactor_den = 12960 * alpha * alpha * alpha * beta * beta * (1 + 12 * total);
    return prefactor_num / (1 - x) * poly / prefactor_den;
  }
  const double prefactor = -x / sqrt(2 * alpha * beta / total);
  const double stirling = (1 + 1 / (12 * alpha) + 1 / (288 * alpha*alpha))
                        * (1 + 1 / (12 * beta) + 1 / (288 * beta*beta))
                        / (1 + 1 / (12 * total) + 1 / (288 * total*total));
  const double term1_num = 2 * (alpha*alpha) * (x - 1) + alpha * beta * (x - 1) - x * (beta*beta);
  const double axbx = alpha * (x-1) + beta * x;
  const double term1_den = sqrt(2 * alpha / beta) * pow(total, 1.5f) * axbx*axbx;
  const double term1 = term1_num / term1_den;
  const double term2 = 0.5f * log(alpha / (total * x));
  const double term3_num = sqrt(8 * alpha * beta / total);
  const double term3_den = beta * x + alpha * (x - 1);
  const double term3 = term3_num / term3_den;
  const double term4_base = beta * log(beta / (total * (1 - x))) +
                          alpha * log(alpha / (total * x));
  const double term4 = pow(term4_base, -1.5f);
  const double term1234 = term1 + term2 * (term3 + (x < mean ? term4 : -term4));
  return stirling * prefactor * term1234;
}

// Computes a scaled reparameterized gradient
//   -(d/dalpha cdf(x;alpha,beta)) / pdf(x;alpha,beta) / (1-x)
// for random number x drawn from a Beta distribution Beta(alpha,beta).
// This function inputs total=alpha+beta to make it easy to implement
// Dirichlet reparameterized gradients in terms of Betas.
static inline scalar_t THTensor_(dirichlet_grad_one)(scalar_t x, scalar_t alpha, scalar_t total) {
  const scalar_t beta = total - alpha;
  const scalar_t boundary = total * x * (1 - x);

  // Use an asymptotic approximation for x close to 0.
  if (x <= 0.5f && boundary < 2.5f) {
    return THTensor_(beta_grad_alpha_small)(x, alpha, beta);
  }

  // Use an asymptotic approximation for x close to 1.
  if (x >= 0.5f && boundary < 0.75f) {
    return -THTensor_(beta_grad_beta_small)(1 - x, beta, alpha);
  }

  // Use an asymptotic approximation when alpha and (total - alpha) are both large.
  if (alpha > 6 && beta > 6) {
    return THTensor_(beta_grad_alpha_mid)(x, alpha, beta);
  }

  // Use a rational correction to an analytic approximation.
  static const scalar_t c[2][3][3][4] = {
    {{{1.003668233, -0.01061107488, -0.0657888334, 0.01201642863},
      {0.6336835991, -0.3557432599, 0.05486251648, -0.001465281033},
      {-0.03276231906, 0.004474107445, 0.002429354597, -0.0001557569013}},
     {{0.221950385, -0.3187676331, 0.01799915743, 0.01074823814},
      {-0.2951249643, 0.06219954479, 0.01535556598, 0.001550077057},
      {0.02155310298, 0.004170831599, 0.001292462449, 6.976601077e-05}},
     {{-0.05980841433, 0.008441916499, 0.01085618172, 0.002319392565},
      {0.02911413504, 0.01400243777, -0.002721828457, 0.000751041181},
      {0.005900514878, -0.001936558688, -9.495446725e-06, 5.385558597e-05}}},
    {{{1, -0.02924021934, -0.04438342661, 0.007285809825},
      {0.6357567472, -0.3473456711, 0.05454656494, -0.002407477521},
      {-0.03301322327, 0.004845219414, 0.00231480583, -0.0002307248149}},
     {{0.5925320577, -0.1757678135, 0.01505928619, 0.000564515273},
      {0.1014815858, -0.06589186703, 0.01272886114, -0.0007316646956},
      {-0.007258481865, 0.001096195486, 0.0003934994223, -4.12701925e-05}},
     {{0.06469649321, -0.0236701437, 0.002902096474, -5.896963079e-05},
      {0.001925008108, -0.002869809258, 0.0008000589141, -6.063713228e-05},
      {-0.0003477407336, 6.959756487e-05, 1.097287507e-05, -1.650964693e-06}}},
  };
  const scalar_t u = TH_MATH_NAME(log)(x);
  const scalar_t a = TH_MATH_NAME(log)(alpha) - u;
  const scalar_t b = TH_MATH_NAME(log)(total) - a;
  const scalar_t pow_u[3] = {1, u, u * u};
  const scalar_t pow_a[3] = {1, a, a * a};
  scalar_t p = 0.0;
  scalar_t q = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const scalar_t ua = pow_u[i] * pow_a[j];
      p += ua * (c[0][i][j][0] + b * (c[0][i][j][1] + b * (c[0][i][j][2] + b * c[0][i][j][3])));
      q += ua * (c[1][i][j][0] + b * (c[1][i][j][1] + b * (c[1][i][j][2] + b * c[1][i][j][3])));
    }
  }
  const scalar_t approx = x * (TH_MATH_NAME(TH_digamma)(total) - TH_MATH_NAME(TH_digamma)(alpha)) / beta;
  return p / q * approx;
}

void THTensor_(dirichlet_grad)(THTensor *self, THTensor *x, THTensor *alpha, THTensor *total)
{
  x = THTensor_(newContiguous)(x);
  alpha = THTensor_(newContiguous)(alpha);
  total = THTensor_(newContiguous)(total);
  TH_CHECK_SAME_SIZE(alpha, x);
  TH_CHECK_SAME_SIZE(total, x);
  THTensor_(resizeAs)(self, x);
  THTensor* grad = THTensor_(newContiguous)(self);

  scalar_t*const grad_data = grad->data<scalar_t>();
  scalar_t*const x_data = x->data<scalar_t>();
  scalar_t*const alpha_data = alpha->data<scalar_t>();
  scalar_t*const total_data = total->data<scalar_t>();
  const int64_t numel = THTensor_(nElement)(x);
  int64_t i;
  #pragma omp parallel for if(numel > TH_OMP_OVERHEAD_THRESHOLD) private(i)
  for(i = 0; i < numel; ++i) {
    grad_data[i] = THTensor_(dirichlet_grad_one)(x_data[i], alpha_data[i], total_data[i]);
  }

  THTensor_(freeCopyTo)(grad, self);
}

#undef TH_MATH_NAME
#endif /* floating point only part */
#undef IS_NONZERO

#endif /* TH_GENERIC_FILE */
