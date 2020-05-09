#include <TH/THTensorApply.h>

#ifndef NAN
  #define NAN (nan(NULL))
#endif

#define HYPER_TH_OMP_OVERHEAD_THRESHOLD (at::internal::GRAIN_SIZE / 16)
#define ORDIN_TH_OMP_OVERHEAD_THRESHOLD (at::internal::GRAIN_SIZE / 4)
#define UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD (at::internal::GRAIN_SIZE / 2)
#define TH_OMP_OVERHEAD_THRESHOLD (at::internal::GRAIN_SIZE)

#define TH_CHECK_SAME_SIZE(TENSOR1, TENSOR2) \
{ \
  if (!THTensor_(isSameSizeAs)(TENSOR1, TENSOR2)) { \
    AT_ERROR("inconsistent tensor size, expected ", #TENSOR1, " ", TENSOR1->sizes(), " and ", #TENSOR2, " ", TENSOR2->sizes(), " to have the same size"); \
  } \
}

// Used for `scatter` and `scatterAdd`
// Assumes TENSOR1 is index
//         TENSOR2 is real
//         TENSOR3 is src
// Tests:
//   1. index->size(d) <= src->size(d) for all d
//   2. index->size(d) <= real->size(d) for all d != dim
#define TH_TENSOR_DIM_APPLY3_SIZE_SCATTER(TENSOR1, TENSOR2, TENSOR3, DIMENSION) \
{ \
  int shape_check_flag = 0; \
  for (TH_TENSOR_DIM_APPLY_i = 0; TH_TENSOR_DIM_APPLY_i < THTensor_nDimensionLegacyAll(TENSOR2); TH_TENSOR_DIM_APPLY_i++) \
  { \
    int64_t TENSOR1##_dim_size = THTensor_sizeLegacyNoScalars(TENSOR1, TH_TENSOR_DIM_APPLY_i); \
    if (TH_TENSOR_DIM_APPLY_i != DIMENSION) { \
      if (TENSOR1##_dim_size > THTensor_sizeLegacyNoScalars(TENSOR2, TH_TENSOR_DIM_APPLY_i)) { \
        shape_check_flag = 1; \
        break; \
      } \
    } \
    if (TENSOR1##_dim_size > THTensor_sizeLegacyNoScalars(TENSOR3, TH_TENSOR_DIM_APPLY_i)) { \
      shape_check_flag = 1; \
      break; \
    } \
  } \
  if (shape_check_flag == 1) { \
    AT_ERROR("Expected ", #TENSOR1, " ", TENSOR1->sizes(), " to be smaller size than ", #TENSOR3, " ", TENSOR3->sizes(), " and to be smaller than ", #TENSOR2, " ", TENSOR2->sizes(), " apart from dimension ", DIMENSION); \
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

#if defined(__clang__)
#define PRAGMA(P) _Pragma(#P)
#define PRAGMA_IVDEP      // Noop
#define PRAGMA_SIMD       // Noop
#elif defined(_MSC_VER)
#define PRAGMA(P)         __pragma(P)
# if _MSC_VER < 1920
// MSVC < 2019 doesn't support loop pragmas.
#  define PRAGMA_IVDEP    // Noop
#  define PRAGMA_SIMD     // Noop
# else
#  define PRAGMA_IVDEP    PRAGMA(loop(ivdep))
#  define PRAGMA_SIMD     PRAGMA(omp simd)
# endif
#else
#define PRAGMA(P)         _Pragma(#P)
#define PRAGMA_IVDEP      PRAGMA(ivdep)
#define PRAGMA_SIMD       PRAGMA(simd)
#endif

#define TH_TENSOR_APPLY2_PARALLEL(SIZE, CONTIG1, CONTIG2, TYPE1, TENSOR1, TYPE2, TENSOR2, CODE, THRESHOLD) \
{ \
  /* for advanced searching index*/ \
  if (CONTIG1 && CONTIG2) { \
    TYPE1 *rp = THTensor_getStoragePtr(TENSOR1)->data<TYPE1>()+TENSOR1->storage_offset(); \
    TYPE2 *tp = THTensor_getStoragePtr(TENSOR2)->data<TYPE2>()+TENSOR2->storage_offset(); \
    if (tp != (TYPE2*)rp) { \
      at::parallel_for(0, SIZE, (THRESHOLD * 10), [&](int64_t begin, int64_t end) { \
        PRAGMA_IVDEP \
        for (auto iter = begin; iter < end; iter++) { \
          TYPE2 *TENSOR2##_data = tp+iter; \
          TYPE1 *TENSOR1##_data = rp+iter; \
          CODE \
        } \
      }); \
    } else { \
      at::parallel_for(0, SIZE, (THRESHOLD * 10), [&](int64_t begin, int64_t end) { \
        PRAGMA_SIMD \
        for (auto iter = begin; iter < end; iter++) { \
          TYPE2* TENSOR2##_data = tp+iter; \
          TYPE1* TENSOR1##_data = rp+iter; \
          CODE \
        } \
      }); \
    } \
  } else { \
    /* The following strategy is not easy to understand.
     * 1. Collapse the dimension of the tensors in order to decrease the number of nested loops.
     * 2. Calculate the numbers of elements allocated in each thread and the line index of the first one.
     * 3. Calculate the memory offset of the first element and the indexes in each dimension of the
     *    first one.
     * 4. iterate all elements in each thread. update the indexes in each dimension of the rest.
    */ \
    int TH_TENSOR_APPLY_hasFinished = 0; \
    int64_t TH_TENSOR_dim_index = 0; \
    /*step 1*/ \
    __TH_TENSOR_APPLYX_PREAMBLE(TYPE2, TENSOR2, -1, 1) \
    __TH_TENSOR_APPLYX_PREAMBLE(TYPE1, TENSOR1, -1, 1) \
    if (0 == TH_TENSOR_APPLY_hasFinished) { \
      auto TENSOR1##_i_local = TENSOR1##_i; \
      auto TENSOR2##_i_local = TENSOR2##_i; \
      auto TENSOR1##_data_local = TENSOR1##_data; \
      auto TENSOR2##_data_local = TENSOR2##_data; \
      at::parallel_for(0, SIZE, THRESHOLD, [&](int64_t begin, int64_t end) { \
        auto TENSOR1##_i = TENSOR1##_i_local; \
        auto TENSOR2##_i = TENSOR2##_i_local; \
        auto TENSOR1##_data = TENSOR1##_data_local; \
        auto TENSOR2##_data = TENSOR2##_data_local; \
        /*step 2*/ \
        ptrdiff_t line_index_start = begin; \
        ptrdiff_t line_seg_length = (end - begin); \
        /* step 3*/ \
        __TH_TENSOR_APPLYX_CAL_MEMORY_OFFSET(TENSOR2); \
        __TH_TENSOR_APPLYX_CAL_MEMORY_OFFSET(TENSOR1); \
        TENSOR2##_data += TENSOR2##_memory_offset; \
        TENSOR1##_data += TENSOR1##_memory_offset; \
        ptrdiff_t count = 0; \
        ptrdiff_t TENSOR2##_start =  TENSOR2##_counter_tmp[TENSOR2##_dim-1]; \
        ptrdiff_t TENSOR1##_start =  TENSOR1##_counter_tmp[TENSOR1##_dim-1]; \
        /* step 4*/ \
        while (count < line_seg_length) { \
          for (TENSOR2##_i=TENSOR2##_start, TENSOR1##_i = TENSOR1##_start; ((count < line_seg_length) && (TENSOR2##_i < TENSOR2##_size) && (TENSOR1##_i < TENSOR1##_size)); ++TENSOR2##_i, ++TENSOR1##_i, ++count) { \
            CODE \
            TENSOR2##_data += TENSOR2##_stride; \
            TENSOR1##_data += TENSOR1##_stride; \
          } \
          if (count < line_seg_length) { \
            __TH_TENSOR_APPLYX_UPDATE_COUNTERS_PARALLEL(TENSOR2); \
            __TH_TENSOR_APPLYX_UPDATE_COUNTERS_PARALLEL(TENSOR1); \
          } \
        } \
        if (TENSOR1##_counter_tmp != NULL) { \
          THFree(TENSOR1##_counter_tmp); \
        } \
        if (TENSOR2##_counter_tmp != NULL) { \
          THFree(TENSOR2##_counter_tmp); \
        } \
      }); \
    } \
    if (TENSOR2##_counter != NULL) { \
      THFree(TENSOR2##_counter); \
    } \
    if (TENSOR1##_counter != NULL) { \
      THFree(TENSOR1##_counter); \
    } \
  } \
}

#define TH_TENSOR_APPLY3_PARALLEL(SIZE, CONTIG1, CONTIG2, CONTIG3, TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, CODE, THRESHOLD) \
{ \
  /* for adveanced searching index*/ \
  if (CONTIG1 && CONTIG2 && CONTIG3) { \
    TYPE1 *rp = THTensor_getStoragePtr(TENSOR1)->data<TYPE1>()+TENSOR1->storage_offset(); \
    TYPE2 *tp = THTensor_getStoragePtr(TENSOR2)->data<TYPE2>()+TENSOR2->storage_offset(); \
    TYPE3 *srcp = THTensor_getStoragePtr(TENSOR3)->data<TYPE3>()+TENSOR3->storage_offset(); \
    if (tp != (TYPE2*)rp) { \
      at::parallel_for(0, SIZE, (THRESHOLD * 10), [&](int64_t begin, int64_t end) { \
        PRAGMA_IVDEP \
        for (auto iter = begin; iter < end; iter++) { \
          TYPE1 *TENSOR1##_data = rp+iter; \
          TYPE2 *TENSOR2##_data = tp+iter; \
          TYPE3 *TENSOR3##_data = srcp+iter; \
          CODE \
        } \
      }); \
    } else { \
      at::parallel_for(0, SIZE, (THRESHOLD * 10), [&](int64_t begin, int64_t end) { \
        PRAGMA_SIMD \
        for (auto iter = begin; iter < end; iter++) { \
          TYPE1 *TENSOR1##_data = rp+iter; \
          TYPE2 *TENSOR2##_data = tp+iter; \
          TYPE3 *TENSOR3##_data = srcp+iter; \
          CODE \
        } \
      }); \
    } \
  } else { \
    int TH_TENSOR_APPLY_hasFinished = 0; \
    int64_t TH_TENSOR_dim_index = 0; \
    __TH_TENSOR_APPLYX_PREAMBLE(TYPE1, TENSOR1, -1, 1) \
    __TH_TENSOR_APPLYX_PREAMBLE(TYPE2, TENSOR2, -1, 1) \
    __TH_TENSOR_APPLYX_PREAMBLE(TYPE3, TENSOR3, -1, 1) \
    if (0 == TH_TENSOR_APPLY_hasFinished) { \
      auto TENSOR1##_i_local = TENSOR1##_i; \
      auto TENSOR2##_i_local = TENSOR2##_i; \
      auto TENSOR3##_i_local = TENSOR3##_i; \
      auto TENSOR1##_data_local = TENSOR1##_data; \
      auto TENSOR2##_data_local = TENSOR2##_data; \
      auto TENSOR3##_data_local = TENSOR3##_data; \
      at::parallel_for(0, SIZE, THRESHOLD, [&](int64_t begin, int64_t end) { \
        auto TENSOR1##_i = TENSOR1##_i_local; \
        auto TENSOR2##_i = TENSOR2##_i_local; \
        auto TENSOR3##_i = TENSOR3##_i_local; \
        auto TENSOR1##_data = TENSOR1##_data_local; \
        auto TENSOR2##_data = TENSOR2##_data_local; \
        auto TENSOR3##_data = TENSOR3##_data_local; \
        ptrdiff_t line_index_start = begin; \
        ptrdiff_t line_seg_length = (end - begin); \
        __TH_TENSOR_APPLYX_CAL_MEMORY_OFFSET(TENSOR1); \
        __TH_TENSOR_APPLYX_CAL_MEMORY_OFFSET(TENSOR2); \
        __TH_TENSOR_APPLYX_CAL_MEMORY_OFFSET(TENSOR3); \
        TENSOR1##_data += TENSOR1##_memory_offset; \
        TENSOR2##_data += TENSOR2##_memory_offset; \
        TENSOR3##_data += TENSOR3##_memory_offset; \
        ptrdiff_t count = 0; \
        ptrdiff_t TENSOR1##_start = TENSOR1##_counter_tmp[TENSOR1##_dim - 1]; \
        ptrdiff_t TENSOR2##_start = TENSOR2##_counter_tmp[TENSOR2##_dim - 1]; \
        ptrdiff_t TENSOR3##_start = TENSOR3##_counter_tmp[TENSOR3##_dim - 1]; \
        while (count < line_seg_length) { \
          for (TENSOR1##_i=TENSOR1##_start, TENSOR2##_i=TENSOR2##_start,TENSOR3##_i=TENSOR3##_start; (count<line_seg_length)&&(TENSOR1##_i<TENSOR1##_size)&&(TENSOR2##_i<TENSOR2##_size)&&(TENSOR3##_i<TENSOR3##_size); ++TENSOR1##_i,++TENSOR2##_i,++TENSOR3##_i,++count) { \
            CODE \
            TENSOR1##_data += TENSOR1##_stride; \
            TENSOR2##_data += TENSOR2##_stride; \
            TENSOR3##_data += TENSOR3##_stride; \
          } \
          if (count < line_seg_length) { \
            __TH_TENSOR_APPLYX_UPDATE_COUNTERS_PARALLEL(TENSOR1); \
            __TH_TENSOR_APPLYX_UPDATE_COUNTERS_PARALLEL(TENSOR2); \
            __TH_TENSOR_APPLYX_UPDATE_COUNTERS_PARALLEL(TENSOR3); \
          } \
        } \
        if (TENSOR1##_counter_tmp != NULL) { \
          THFree(TENSOR1##_counter_tmp); \
        } \
        if (TENSOR2##_counter_tmp != NULL) { \
          THFree(TENSOR2##_counter_tmp); \
        } \
        if (TENSOR3##_counter_tmp != NULL) { \
          THFree(TENSOR3##_counter_tmp); \
        } \
      }); \
    } \
    if (TENSOR1##_counter != NULL) { \
      THFree(TENSOR1##_counter); \
    } \
    if (TENSOR2##_counter != NULL) { \
      THFree(TENSOR2##_counter); \
    } \
    if (TENSOR3##_counter != NULL) { \
      THFree(TENSOR3##_counter); \
    } \
  } \
}

#define TH_TENSOR_APPLY_REDUCTION_SUM_PARALLEL(TYPE, TENSOR, EXPR, OUTPUT, THRESHOLD) \
{ \
  int TENSOR##Contig = THTensor_(isContiguous)(TENSOR); \
  ptrdiff_t TENSOR##Size = THTensor_(nElement)(TENSOR); \
  if (TENSOR##Contig) { \
    TYPE *rp = THTensor_getStoragePtr(TENSOR)->data<TYPE>()+TENSOR->storage_offset(); \
    OUTPUT = at::parallel_reduce(0, TENSOR##Size, (THRESHOLD * 10), (accreal)0, [&](int64_t begin, int64_t end, accreal ident)->accreal { \
      accreal r = ident; \
      for (auto iter = begin; iter < end; iter++) { \
        TYPE *TENSOR##_data = rp+iter; \
        r += (EXPR); \
      } \
      return r; \
    }, std::plus<accreal>()); \
  } else { \
    int TH_TENSOR_APPLY_hasFinished = 0; \
    int64_t TH_TENSOR_dim_index = 0; \
    __TH_TENSOR_APPLYX_PREAMBLE(TYPE, TENSOR, -1, 1); \
    if (0 == TH_TENSOR_APPLY_hasFinished) { \
      auto TENSOR##_data_local = TENSOR##_data; \
      auto TENSOR##_i_local = TENSOR##_i; \
      OUTPUT = at::parallel_reduce(0, TENSOR##Size, THRESHOLD, (accreal)0, [&](int64_t begin, int64_t end, accreal ident)->accreal { \
        auto TENSOR##_data = TENSOR##_data_local; \
        auto TENSOR##_i = TENSOR##_i_local; \
        ptrdiff_t line_index_start = begin; \
        ptrdiff_t line_seg_length = (end - begin); \
        __TH_TENSOR_APPLYX_CAL_MEMORY_OFFSET(TENSOR); \
        TENSOR##_data += TENSOR##_memory_offset; \
        ptrdiff_t count = 0; \
        ptrdiff_t TENSOR##_start = TENSOR##_counter_tmp[TENSOR##_dim - 1]; \
        accreal r = ident; \
        while (count < line_seg_length) { \
          for (TENSOR##_i=TENSOR##_start; (count < line_seg_length)&&(TENSOR##_i < TENSOR##_size); ++TENSOR##_i, ++count) { \
            r += (EXPR); \
            TENSOR##_data += TENSOR##_stride; \
          } \
          if (count < line_seg_length) { \
            __TH_TENSOR_APPLYX_UPDATE_COUNTERS_PARALLEL(TENSOR); \
          } \
        } \
        if (TENSOR##_counter_tmp != NULL) { \
          THFree(TENSOR##_counter_tmp); \
        } \
        return r; \
      }, std::plus<accreal>()); \
    } \
    if (TENSOR##_counter != NULL) { \
      THFree(TENSOR##_counter); \
    } \
  } \
}

#define TH_TENSOR_APPLY_CONTIG(TYPE, TENSOR, CODE) \
{ \
  auto code_fn = [&](int64_t begin, int64_t end) { \
    ptrdiff_t TENSOR##_len = end - begin; \
    TYPE *TENSOR##_data = TENSOR->data<scalar_t>() + begin; \
    CODE \
  }; \
  int in_parallel = at::in_parallel_region(); \
  ptrdiff_t TH_TENSOR_size = THTensor_(nElement)(TENSOR); \
  if (!in_parallel) { \
    at::parallel_for(0, TH_TENSOR_size, TH_OMP_OVERHEAD_THRESHOLD, code_fn); \
  } else { \
    code_fn(0, TH_TENSOR_size); \
  } \
}

#define TH_TENSOR_APPLY2_CONTIG(TYPE1, TENSOR1, TYPE2, TENSOR2, CODE) \
{ \
  auto code_fn = [&](int64_t begin, int64_t end) { \
    ptrdiff_t TENSOR1##_len = end - begin; \
    TYPE1 *TENSOR1##_data = TENSOR1->data<scalar_t>() + begin; \
    TYPE2 *TENSOR2##_data = TENSOR2->data<scalar_t>() + begin; \
    CODE \
  }; \
  int in_parallel = at::in_parallel_region(); \
  ptrdiff_t TH_TENSOR_size = THTensor_(nElement)(TENSOR1); \
  if (!in_parallel) { \
    at::parallel_for(0, TH_TENSOR_size, TH_OMP_OVERHEAD_THRESHOLD, code_fn); \
  } else { \
    code_fn(0, TH_TENSOR_size); \
  } \
}

#define TH_TENSOR_APPLY3_CONTIG(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, CODE) \
{ \
  auto code_fn = [&](int64_t begin, int64_t end) { \
    ptrdiff_t TENSOR1##_len = end - begin; \
    TYPE1 *TENSOR1##_data = TENSOR1->data<scalar_t>() + begin; \
    TYPE2 *TENSOR2##_data = TENSOR2->data<scalar_t>() + begin; \
    TYPE3 *TENSOR3##_data = TENSOR3->data<scalar_t>() + begin; \
    CODE \
  }; \
  int in_parallel = at::in_parallel_region(); \
  ptrdiff_t TH_TENSOR_size = THTensor_(nElement)(TENSOR1); \
  if (!in_parallel) { \
    at::parallel_for(0, TH_TENSOR_size, TH_OMP_OVERHEAD_THRESHOLD, code_fn); \
  } else { \
    code_fn(0, TH_TENSOR_size); \
  } \
}
