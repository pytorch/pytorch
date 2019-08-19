#include "ATen/ATen.h"
#include "ATen/Parallel.h"

#define TH_TENSOR_DIM_APPLY3(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, DIMENSION, SIZE_CHECK, CODE) \
{ \
  TYPE1 *TENSOR1##_data = NULL; \
  TH_UNUSED int64_t TENSOR1##_stride = 0, TENSOR1##_size = 0; \
  TYPE2 *TENSOR2##_data = NULL; \
  TH_UNUSED int64_t TENSOR2##_stride = 0, TENSOR2##_size = 0; \
  TYPE3 *TENSOR3##_data = NULL; \
  TH_UNUSED int64_t TENSOR3##_stride = 0, TENSOR3##_size = 0; \
  int64_t *TH_TENSOR_DIM_APPLY_counter = NULL; \
  int TH_TENSOR_DIM_APPLY_hasFinished = THTensor_(numel)(TENSOR1) == 0; \
  int TH_TENSOR_DIM_APPLY_i; \
\
  if( (DIMENSION < 0) || (DIMENSION >= THTensor_nDimensionLegacyNoScalars(TENSOR1)) ) \
    THError("invalid dimension %d (expected to be 0 <= dim < %d)", DIMENSION, THTensor_nDimensionLegacyNoScalars(TENSOR1)); \
  int same_dims = 1;                                                    \
  if( THTensor_nDimensionLegacyNoScalars(TENSOR1) != THTensor_nDimensionLegacyNoScalars(TENSOR2) ) { \
    same_dims = 0;                                                      \
  } \
  if( THTensor_nDimensionLegacyNoScalars(TENSOR1) != THTensor_nDimensionLegacyNoScalars(TENSOR3) ) { \
    same_dims = 0;                                   \
  } \
  if (same_dims == 0) { \
    AT_ERROR("inconsistent tensor size, expected ", #TENSOR1, " ", TENSOR1->sizes(), ", ", #TENSOR2, " ", TENSOR2->sizes(), " and ", #TENSOR3, " ",TENSOR3->sizes() , " to have the same number of dimensions"); \
  }                                                                     \
  SIZE_CHECK(TENSOR1, TENSOR2, TENSOR3, DIMENSION)                      \
\
  if (TH_TENSOR_DIM_APPLY_hasFinished) { \
    return; \
  } \
  TH_TENSOR_DIM_APPLY_counter = (int64_t*)THAlloc(sizeof(int64_t)*(THTensor_nDimensionLegacyNoScalars(TENSOR1))); \
  for(TH_TENSOR_DIM_APPLY_i = 0; TH_TENSOR_DIM_APPLY_i < THTensor_nDimensionLegacyNoScalars(TENSOR1); TH_TENSOR_DIM_APPLY_i++) \
    TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i] = 0; \
\
  TENSOR1##_data = THTensor_getStoragePtr(TENSOR1)->data<TYPE1>()+(TENSOR1)->storage_offset(); \
  TENSOR1##_stride = THTensor_strideLegacyNoScalars((TENSOR1), DIMENSION); \
  TENSOR1##_size = THTensor_sizeLegacyNoScalars((TENSOR1), DIMENSION); \
\
  TENSOR2##_data = THTensor_getStoragePtr(TENSOR2)->data<TYPE2>()+(TENSOR2)->storage_offset(); \
  TENSOR2##_stride = THTensor_strideLegacyNoScalars((TENSOR2), DIMENSION); \
  TENSOR2##_size = THTensor_sizeLegacyNoScalars((TENSOR2), DIMENSION);  \
\
  TENSOR3##_data = THTensor_getStoragePtr(TENSOR3)->data<TYPE3>()+(TENSOR3)->storage_offset(); \
  TENSOR3##_stride = THTensor_strideLegacyNoScalars((TENSOR3), DIMENSION); \
  TENSOR3##_size = THTensor_sizeLegacyNoScalars((TENSOR3), DIMENSION); \
\
  while(!TH_TENSOR_DIM_APPLY_hasFinished) \
  { \
    CODE \
\
    if(THTensor_nDimensionLegacyNoScalars(TENSOR1) == 1) \
       break; \
 \
    for(TH_TENSOR_DIM_APPLY_i = 0; TH_TENSOR_DIM_APPLY_i < THTensor_nDimensionLegacyNoScalars(TENSOR1); TH_TENSOR_DIM_APPLY_i++) \
    { \
      if(TH_TENSOR_DIM_APPLY_i == DIMENSION) \
      { \
        if(TH_TENSOR_DIM_APPLY_i == THTensor_nDimensionLegacyNoScalars(TENSOR1)-1) \
        { \
          TH_TENSOR_DIM_APPLY_hasFinished = 1; \
          break; \
        } \
        continue; \
      } \
\
      TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i]++; \
      TENSOR1##_data += THTensor_strideLegacyNoScalars(TENSOR1, TH_TENSOR_DIM_APPLY_i); \
      TENSOR2##_data += THTensor_strideLegacyNoScalars(TENSOR2, TH_TENSOR_DIM_APPLY_i); \
      TENSOR3##_data += THTensor_strideLegacyNoScalars(TENSOR3, TH_TENSOR_DIM_APPLY_i); \
\
      if(TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i] == THTensor_sizeLegacyNoScalars(TENSOR1, TH_TENSOR_DIM_APPLY_i)) \
      { \
        if(TH_TENSOR_DIM_APPLY_i == THTensor_nDimensionLegacyNoScalars(TENSOR1)-1) \
        { \
          TH_TENSOR_DIM_APPLY_hasFinished = 1; \
          break; \
        } \
        else \
        { \
          TENSOR1##_data -= TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i]*THTensor_strideLegacyNoScalars(TENSOR1, TH_TENSOR_DIM_APPLY_i); \
          TENSOR2##_data -= TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i]*THTensor_strideLegacyNoScalars(TENSOR2, TH_TENSOR_DIM_APPLY_i); \
          TENSOR3##_data -= TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i]*THTensor_strideLegacyNoScalars(TENSOR3, TH_TENSOR_DIM_APPLY_i); \
          TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i] = 0; \
        } \
      } \
      else \
        break; \
    } \
  } \
  THFree(TH_TENSOR_DIM_APPLY_counter); \
}

namespace at { namespace native {

Tensor & gather_out_cpu(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index, bool) {
  int64_t num_dims = std::max<int64_t>(self.dim(), 1);
  TORCH_CHECK(std::max<int64_t>(index.dim(), 1) == num_dims, "Index tensor must have same dimensions as input tensor");
  TORCH_CHECK(dim >= 0 && dim < num_dims, "Index dimension is out of bounds");
  // TORCH_CHECK(std::max<int64_t>(result.dim(), 1) == num_dims, "Input tensor must have same dimensions as output tensor");

  int64_t elems_per_row = (index.dim() == 0 ? 1 : index.size(dim));
  int64_t self_dim_size = self.size(dim);
  int64_t outer_size = 1;
  for(int64_t i = 0; i < num_dims; i++) {
    if(i != dim) {
      AT_CHECK(index.size(i) == self.size(i), "Size does not match at dimension ", i, " get ", self.size(i), " vs ", index.size(i));
      outer_size *= index.size(i);
    }
  }
  result.resize_as_(index);

  AT_DISPATCH_ALL_TYPES_AND(ScalarType::Bool, self.scalar_type(), "gather_out_cpu", [&](){
    scalar_t *result_data = result.data<scalar_t>();
    scalar_t *self_data = self.data<scalar_t>();
    int64_t *index_data = index.data<int64_t>();
    int64_t result_dim_stride = result.stride(dim);
    int64_t index_dim_stride = index.stride(dim);
    int64_t self_dim_stride = self.stride(dim);

    at::parallel_for(0, outer_size, at::internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
      for(int64_t i = begin; i < end; i++) {
        scalar_t *result_base = result_data;
        int64_t *index_base = index_data;
        scalar_t *self_base = self_data;
        int64_t global_index = i;
        for(int64_t k = 0; k < num_dims; k++) {
          if(dim != k) {
            int64_t index_at_k = global_index % result.size(k);
            result_base += result.stride(k) * index_at_k;
            index_base += index.stride(k) * index_at_k;
            self_base += self.stride(k) * index_at_k;
            global_index /= result.size(k);
          }
        }
        for(int64_t j = 0; j < elems_per_row; j++) {
          int64_t index = *(index_base + j * index_dim_stride);
          AT_CHECK(index >= 0 && index < self_dim_size, "Invalid index in gather: out of range");
          *(result_base + j * result_dim_stride) = *(self_base + index * self_dim_stride);
        }
      }
    });
  });
  return result;
}

Tensor gather(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  Tensor result = at::empty({}, self.options());
  return at::gather_out(result, self, dim, index, sparse_grad);
}

Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  return self.clone().scatter_(dim, index, source);
}

Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, Scalar source) {
  return self.clone().scatter_(dim, index, source);
}

Tensor scatter_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  return self.clone().scatter_add_(dim, index, source);
}

}}  // namespace at::native
