#ifndef NAN
  #define NAN (nan(NULL))
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define HYPER_TH_OMP_OVERHEAD_THRESHOLD 2000
#define ORDIN_TH_OMP_OVERHEAD_THRESHOLD 20000
#define UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD 50000
#define TH_OMP_OVERHEAD_THRESHOLD 100000

#ifdef _OPENMP

#ifdef _WIN32
// MSVC doesing support loop pragmas, but does support others. Create a new macro to account for those differences.
#define PRAGMA_LOOP(P)    // Noop
#define PRAGMA(P)         __pragma(P)
#else
#define PRAGMA_LOOP(P)    _Pragma(#P)
#define PRAGMA(P)         _Pragma(#P)
#endif

#define TH_TENSOR_APPLY_CONTIG(TYPE, TENSOR, CODE) \
{ \
  int inOmp = omp_in_parallel(); \
  ptrdiff_t TH_TENSOR_size = THTensor_(nElement)(TENSOR); \
  PRAGMA(omp parallel if ((TH_TENSOR_size > TH_OMP_OVERHEAD_THRESHOLD) && (!inOmp))) \
  { \
    size_t num_threads = omp_get_num_threads(); \
    size_t tid = omp_get_thread_num(); \
    ptrdiff_t TH_TENSOR_offset = tid * (TH_TENSOR_size / num_threads); \
    ptrdiff_t TH_TENSOR_end = tid == num_threads - 1 ? TH_TENSOR_size : \
      TH_TENSOR_offset + TH_TENSOR_size / num_threads; \
    ptrdiff_t TENSOR##_len = TH_TENSOR_end - TH_TENSOR_offset; \
    TYPE *TENSOR##_data = TENSOR->data<scalar_t>() + TH_TENSOR_offset; \
    CODE \
  } \
}
#else
#define TH_TENSOR_APPLY_CONTIG(TYPE, TENSOR, CODE) \
{ \
  TYPE *TENSOR##_data = TENSOR->data<scalar_t>(); \
  ptrdiff_t TENSOR##_len = THTensor_(nElement)(TENSOR); \
  CODE \
}
#endif

#ifdef _OPENMP
#define TH_TENSOR_APPLY2_CONTIG(TYPE1, TENSOR1, TYPE2, TENSOR2, CODE) \
{ \
  int inOmp = omp_in_parallel(); \
  ptrdiff_t TH_TENSOR_size = THTensor_(nElement)(TENSOR1); \
  PRAGMA(omp parallel if ((TH_TENSOR_size > TH_OMP_OVERHEAD_THRESHOLD) && (!inOmp))) \
  { \
    size_t num_threads = omp_get_num_threads(); \
    size_t tid = omp_get_thread_num(); \
    ptrdiff_t TH_TENSOR_offset = tid * (TH_TENSOR_size / num_threads); \
    ptrdiff_t TH_TENSOR_end = tid == num_threads - 1 ? TH_TENSOR_size : \
      TH_TENSOR_offset + TH_TENSOR_size / num_threads; \
    ptrdiff_t TENSOR1##_len = TH_TENSOR_end - TH_TENSOR_offset; \
    TYPE1 *TENSOR1##_data = TENSOR1->data<scalar_t>() + TH_TENSOR_offset; \
    TYPE2 *TENSOR2##_data = TENSOR2->data<scalar_t>() + TH_TENSOR_offset; \
    CODE \
  } \
}
#else
#define TH_TENSOR_APPLY2_CONTIG(TYPE1, TENSOR1, TYPE2, TENSOR2, CODE) \
{ \
  TYPE1 *TENSOR1##_data = TENSOR1->data<scalar_t>(); \
  TYPE2 *TENSOR2##_data = TENSOR2->data<scalar_t>(); \
  ptrdiff_t TENSOR1##_len = THTensor_(nElement)(TENSOR1); \
  CODE \
}
#endif

#ifdef _OPENMP
#define TH_TENSOR_APPLY3_CONTIG(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, CODE) \
{ \
  int inOmp = omp_in_parallel(); \
  ptrdiff_t TH_TENSOR_size = THTensor_(nElement)(TENSOR1); \
  PRAGMA(omp parallel if ((TH_TENSOR_size > TH_OMP_OVERHEAD_THRESHOLD) && (!inOmp))) \
  { \
    size_t num_threads = omp_get_num_threads(); \
    size_t tid = omp_get_thread_num(); \
    ptrdiff_t TH_TENSOR_offset = tid * (TH_TENSOR_size / num_threads); \
    ptrdiff_t TH_TENSOR_end = tid == num_threads - 1 ? TH_TENSOR_size : \
      TH_TENSOR_offset + TH_TENSOR_size / num_threads; \
    ptrdiff_t TENSOR1##_len = TH_TENSOR_end - TH_TENSOR_offset; \
    TYPE1 *TENSOR1##_data = TENSOR1->data<scalar_t>() + TH_TENSOR_offset; \
    TYPE2 *TENSOR2##_data = TENSOR2->data<scalar_t>() + TH_TENSOR_offset; \
    TYPE3 *TENSOR3##_data = TENSOR3->data<scalar_t>() + TH_TENSOR_offset; \
    CODE \
  } \
}
#else
#define TH_TENSOR_APPLY3_CONTIG(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, CODE) \
{ \
  TYPE1 *TENSOR1##_data = TENSOR1->data<scalar_t>(); \
  TYPE2 *TENSOR2##_data = TENSOR2->data<scalar_t>(); \
  TYPE3 *TENSOR3##_data = TENSOR3->data<scalar_t>(); \
  ptrdiff_t TENSOR1##_len = THTensor_(nElement)(TENSOR1); \
  CODE \
}
#endif

#define TH_CHECK_SAME_SIZE(TENSOR1, TENSOR2) \
{ \
  if(!THTensor_(isSameSizeAs)(TENSOR1, TENSOR2)) { \
    AT_ERROR("inconsistent tensor size, expected ", #TENSOR1, " ", TENSOR1->sizes(), " and ", #TENSOR2, " ", TENSOR2->sizes(), " to have the same size"); \
  } \
}

// Used for `scatter` and `scatterAdd`
// Assumes TENSOR1 is real
//         TENSOR2 is src
//         TENSOR3 is index
// Tests:
//   1. index->size(d) <= src->size(d) for all d
//   2. index->size(d) <= real->size(d) for all d != dim
#define TH_TENSOR_DIM_APPLY3_SIZE_SCATTER(TENSOR1, TENSOR2, TENSOR3, DIMENSION) \
{ \
  int shape_check_flag = 0; \
  for(TH_TENSOR_DIM_APPLY_i = 0; TH_TENSOR_DIM_APPLY_i < THTensor_nDimensionLegacyAll(TENSOR1); TH_TENSOR_DIM_APPLY_i++) \
  { \
    int64_t TENSOR3##_dim_size = THTensor_sizeLegacyNoScalars(TENSOR3, TH_TENSOR_DIM_APPLY_i); \
    if (TH_TENSOR_DIM_APPLY_i != DIMENSION) { \
      if (TENSOR3##_dim_size > THTensor_sizeLegacyNoScalars(TENSOR1, TH_TENSOR_DIM_APPLY_i)) { \
        shape_check_flag = 1; \
        break; \
      } \
    } \
    if (TENSOR3##_dim_size > THTensor_sizeLegacyNoScalars(TENSOR2, TH_TENSOR_DIM_APPLY_i)) { \
      shape_check_flag = 1; \
      break; \
    } \
  } \
  if (shape_check_flag == 1) { \
    AT_ERROR("Expected ", #TENSOR3, " ", TENSOR3->sizes(), " to be smaller size than ", #TENSOR2, " ", TENSOR2->sizes(), " and to be smaller than ", #TENSOR1, " ", TENSOR1->sizes(), " apart from dimension ", DIMENSION); \
  } \
}

#undef th_isnan
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
#define th_isnan(val) \
(std::isnan(val))
#else
#define th_isnan(val) (0)
#endif

#undef th_isnan_break
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
#define th_isnan_break(val) \
if (std::isnan(val)) break;
#else
#define th_isnan_break(val)
#endif
