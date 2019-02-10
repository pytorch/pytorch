#ifndef TH_TENSOR_APPLY_INC
#define TH_TENSOR_APPLY_INC

/*
 * The basic strategy for apply is as follows:
 *
 * 1. Starting with the outermost index, loop until we reach a dimension where the
 * data is no longer contiguous, i.e. the stride at that dimension is not equal to
 * the size of the tensor defined by the outer dimensions. Let's call this outer
 * (contiguous) tensor A. Note that if the Tensor is contiguous, then A is equal
 * to the entire Tensor. Let's call the inner tensor B.
 *
 * 2. We loop through the indices in B, starting at its outermost dimension. For
 * example, if B is a 2x2 matrix, then we do:
 *
 * B[0][0]
 * B[0][1]
 * B[1][0]
 * B[1][1]
 *
 * We set the offset into the underlying storage as (storageOffset + stride_B * index_B),
 * i.e. basically we compute the offset into the storage as we would normally for a
 * Tensor. But because we are guaranteed the subsequent data is contiguous in memory, we
 * can simply loop for sizeof(A) iterations and perform the operation, without having to
 * follow the order described by the strides of A.
 *
 * 3. As an optimization, we merge dimensions of A that are contiguous in memory. For
 * example, if A is a 3x3x3x3 tensor narrowed from a 3x3x4x3 tensor, then the first two
 * dimensions can be merged for the purposes of APPLY, reducing the number of nested
 * loops.
 */

#define __TH_TENSOR_APPLYX_PREAMBLE(TYPE, TENSOR, DIM, ALLOW_CONTIGUOUS) \
  TYPE *TENSOR##_data = NULL; \
  int64_t *TENSOR##_counter = NULL, *TENSOR##_sizes = NULL, *TENSOR##_strides = NULL, *TENSOR##_dimOffset = NULL; \
  int64_t TENSOR##_stride = 0, TENSOR##_size = 0, TENSOR##_dim = 0, TENSOR##_i, TENSOR##_n; \
  int TENSOR##_contiguous = ALLOW_CONTIGUOUS && DIM < 0; \
  TENSOR##_n = 1; \
  for(TENSOR##_i = 0; TENSOR##_i < TENSOR->dim(); TENSOR##_i++) \
    TENSOR##_n *= TENSOR->size(TENSOR##_i); \
\
  if(TENSOR->is_empty()) \
    TH_TENSOR_APPLY_hasFinished = 1; \
  else \
  { \
    TENSOR##_data = THTensor_getStoragePtr(TENSOR)->data<TYPE>()+TENSOR->storage_offset(); \
    TENSOR##_size = 1; \
    TENSOR##_stride = 1; \
    for(TENSOR##_i = THTensor_nDimensionLegacyAll(TENSOR)-1; TENSOR##_i >= 0; TENSOR##_i--) { \
      if(THTensor_sizeLegacyNoScalars(TENSOR, TENSOR##_i) != 1) { \
        if(THTensor_strideLegacyNoScalars(TENSOR, TENSOR##_i) == TENSOR##_size && TENSOR##_i != DIM) \
          TENSOR##_size *= THTensor_sizeLegacyNoScalars(TENSOR, TENSOR##_i); \
        else{ \
          TENSOR##_contiguous = 0; \
          break; \
        } \
      } \
    } \
    if (!TENSOR##_contiguous) { \
      /* Find the dimension of contiguous sections */ \
      TENSOR##_dim = 1; \
      for(TENSOR##_i = THTensor_nDimensionLegacyAll(TENSOR)-2; TENSOR##_i >= 0; TENSOR##_i--) \
      { \
        if(TENSOR->stride(TENSOR##_i) != TENSOR->stride(TENSOR##_i+1) * TENSOR->size(TENSOR##_i+1) || TENSOR##_i == DIM || TENSOR##_i+1 == DIM) \
          TENSOR##_dim++; \
      } \
      /* Allocate an array of 3*dim elements, where dim is the number of contiguous sections */ \
      TENSOR##_counter = (int64_t*)THAlloc(sizeof(int64_t)*(3*TENSOR##_dim)); \
      TENSOR##_sizes = TENSOR##_counter + TENSOR##_dim; \
      TENSOR##_strides = TENSOR##_counter + 2*TENSOR##_dim; \
      TH_TENSOR_dim_index = TENSOR##_dim-1; \
      TENSOR##_dimOffset = (DIM == THTensor_nDimensionLegacyAll(TENSOR)-1) ? &TENSOR##_i : &TENSOR##_counter[DIM]; \
      TENSOR##_sizes[TH_TENSOR_dim_index] = THTensor_sizeLegacyNoScalars(TENSOR, THTensor_nDimensionLegacyAll(TENSOR)-1); \
      TENSOR##_strides[TH_TENSOR_dim_index] = THTensor_strideLegacyNoScalars(TENSOR, THTensor_nDimensionLegacyAll(TENSOR)-1); \
      /* TENSOR##_counter tracks where we are in the storage. The offset into the */ \
      /* storage is given by storage_offset + (i * j), where i is the stride */ \
      /* vector and j is tensor_counter vector. This sets the starting position for the loop. */ \
      for(TENSOR##_i = TENSOR##_dim-1; TENSOR##_i >= 0; --TENSOR##_i) { \
        TENSOR##_counter[TENSOR##_i] = 0; \
      } \
      for(TENSOR##_i = THTensor_nDimensionLegacyAll(TENSOR)-2; TENSOR##_i >= 0; --TENSOR##_i) { \
        if (TENSOR->stride(TENSOR##_i) == TENSOR->stride(TENSOR##_i+1) * TENSOR->size(TENSOR##_i+1) && TENSOR##_i != DIM && TENSOR##_i+1 != DIM) { \
          TENSOR##_sizes[TH_TENSOR_dim_index] = TENSOR->size(TENSOR##_i) * TENSOR##_sizes[TH_TENSOR_dim_index]; \
          if (DIM != THTensor_nDimensionLegacyAll(TENSOR)-1 && TENSOR##_i < DIM) \
            TENSOR##_dimOffset--; \
        } else { \
          --TH_TENSOR_dim_index; \
          TENSOR##_sizes[TH_TENSOR_dim_index] = TENSOR->size(TENSOR##_i); \
          TENSOR##_strides[TH_TENSOR_dim_index] = TENSOR->stride(TENSOR##_i); \
        } \
      } \
      /* Size of the inner most section */ \
      TENSOR##_size = TENSOR##_sizes[TENSOR##_dim-1]; \
      /* Stride of the inner most section */ \
      TENSOR##_stride = TENSOR##_strides[TENSOR##_dim-1]; \
    } \
    else{\
      TENSOR##_dim = 1;\
      TENSOR##_counter = (int64_t*)THAlloc(sizeof(int64_t)*3);\
      TENSOR##_sizes = TENSOR##_counter + 1;\
      TENSOR##_strides = TENSOR##_counter + 2;\
      TENSOR##_sizes[0] = TENSOR##_n;\
      TENSOR##_strides[0] = 1;\
      TENSOR##_size = TENSOR##_sizes[0];\
      TENSOR##_stride = TENSOR##_strides[0];\
    }\
  } \
  TENSOR##_i = 0;

#define  __TH_TENSOR_APPLYX_UPDATE_COUNTERS(TENSOR, ALWAYS_UPDATE) \
  if(TENSOR##_i == TENSOR##_size || ALWAYS_UPDATE) \
  { \
    if(TENSOR##_contiguous) \
      break; \
\
    if(TENSOR##_dim == 1) \
       break; \
\
    /* Reset pointer to beginning of loop */ \
    TENSOR##_data -= TENSOR##_size*TENSOR##_stride; \
    for(TENSOR##_i = TENSOR##_dim-2; TENSOR##_i >= 0; TENSOR##_i--) \
    { \
      TENSOR##_counter[TENSOR##_i]++; \
      /* Jump ahread by the stride of this dimension */ \
      TENSOR##_data += TENSOR##_strides[TENSOR##_i]; \
\
      if(TENSOR##_counter[TENSOR##_i]  == TENSOR##_sizes[TENSOR##_i]) \
      { \
        if(TENSOR##_i == 0) \
        { \
          TH_TENSOR_APPLY_hasFinished = 1; \
          break; \
        } \
          else \
        { \
          /* Reset the pointer to the beginning of the chunk defined by this dimension */ \
          TENSOR##_data -= TENSOR##_counter[TENSOR##_i]*TENSOR##_strides[TENSOR##_i]; \
          TENSOR##_counter[TENSOR##_i] = 0; \
        } \
      } \
      else \
        break; \
    } \
    TENSOR##_i = 0; \
  } \

#define TH_TENSOR_APPLY3_D(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, DIM, CODE) \
{ \
  int TH_TENSOR_APPLY_hasFinished = 0; \
  int64_t TH_TENSOR_dim_index = 0; \
  __TH_TENSOR_APPLYX_PREAMBLE(TYPE1, TENSOR1, DIM, 1) \
  __TH_TENSOR_APPLYX_PREAMBLE(TYPE2, TENSOR2, DIM, 1) \
  __TH_TENSOR_APPLYX_PREAMBLE(TYPE3, TENSOR3, DIM, 1) \
                                                                        \
  int elements_equal = 1;                                               \
  if(TENSOR1##_n != TENSOR2##_n) {                                      \
    elements_equal = 0;                                                 \
  }                                                                     \
  else if(TENSOR1##_n != TENSOR3##_n) {                                 \
    elements_equal = 0;                                                 \
  }                                                                     \
  if (elements_equal == 0) {                                            \
    AT_ERROR("inconsistent tensor size, expected ",                     \
            #TENSOR1, " ", TENSOR1->sizes(), ", ",                      \
            #TENSOR2, " ", TENSOR2->sizes(), " and ",                   \
            #TENSOR3, " ", TENSOR3->sizes(), " to have the same "       \
            "number of elements, but got ", TENSOR1##_n, ", ",          \
            TENSOR2##_n, " and ", TENSOR3##_n, " elements respectively"); \
  }                                                                     \
                                                                        \
  while(!TH_TENSOR_APPLY_hasFinished) \
  { \
    /* Loop through the inner most region of the Tensor */ \
    for(; TENSOR1##_i < TENSOR1##_size && TENSOR2##_i < TENSOR2##_size && TENSOR3##_i < TENSOR3##_size; TENSOR1##_i++, TENSOR2##_i++, TENSOR3##_i++, TENSOR1##_data += TENSOR1##_stride, TENSOR2##_data += TENSOR2##_stride, TENSOR3##_data += TENSOR3##_stride) /* 0 et pas TENSOR##_dim! */ \
    { \
      CODE \
    } \
    __TH_TENSOR_APPLYX_UPDATE_COUNTERS(TENSOR1, 0) \
    __TH_TENSOR_APPLYX_UPDATE_COUNTERS(TENSOR2, 0) \
    __TH_TENSOR_APPLYX_UPDATE_COUNTERS(TENSOR3, 0) \
  } \
  if(TENSOR1##_counter != NULL) \
    THFree(TENSOR1##_counter); \
  if(TENSOR2##_counter != NULL) \
    THFree(TENSOR2##_counter); \
  if(TENSOR3##_counter != NULL) \
    THFree(TENSOR3##_counter); \
}

#define TH_TENSOR_APPLY3(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, CODE) \
  TH_TENSOR_APPLY3_D(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, -1, CODE)

#define TH_TENSOR_APPLY2_D(TYPE1, TENSOR1, TYPE2, TENSOR2, DIM, CODE) \
{ \
  int TH_TENSOR_APPLY_hasFinished = 0; \
  int64_t TH_TENSOR_dim_index = 0; \
  __TH_TENSOR_APPLYX_PREAMBLE(TYPE1, TENSOR1, DIM, 1) \
  __TH_TENSOR_APPLYX_PREAMBLE(TYPE2, TENSOR2, DIM, 1) \
\
    if(TENSOR1##_n != TENSOR2##_n) {                                    \
      AT_ERROR("inconsistent tensor size, expected ",                   \
      #TENSOR1, " ", TENSOR1->sizes(), " and ",                         \
      #TENSOR2, " ", TENSOR2->sizes(),                                  \
      " to have the same number of elements, but got ",                 \
      TENSOR1##_n, " and ", TENSOR2##_n, " elements respectively");     \
    }                                                                   \
  while(!TH_TENSOR_APPLY_hasFinished) \
  { \
    /* Loop through the inner most region of the Tensor */ \
    for(; TENSOR1##_i < TENSOR1##_size && TENSOR2##_i < TENSOR2##_size; TENSOR1##_i++, TENSOR2##_i++, TENSOR1##_data += TENSOR1##_stride, TENSOR2##_data += TENSOR2##_stride) /* 0 et pas TENSOR##_dim! */ \
    { \
      CODE \
    } \
    __TH_TENSOR_APPLYX_UPDATE_COUNTERS(TENSOR1, 0) \
    __TH_TENSOR_APPLYX_UPDATE_COUNTERS(TENSOR2, 0) \
  } \
  if(TENSOR1##_counter != NULL) \
    THFree(TENSOR1##_counter); \
  if(TENSOR2##_counter != NULL) \
    THFree(TENSOR2##_counter); \
}

#define TH_TENSOR_APPLY2(TYPE1, TENSOR1, TYPE2, TENSOR2, CODE) \
  TH_TENSOR_APPLY2_D(TYPE1, TENSOR1, TYPE2, TENSOR2, -1, CODE)

#define TH_TENSOR_APPLY_D(TYPE, TENSOR, DIM, CODE) \
{ \
  int TH_TENSOR_APPLY_hasFinished = 0; \
  int64_t TH_TENSOR_dim_index = 0; \
  __TH_TENSOR_APPLYX_PREAMBLE(TYPE, TENSOR, DIM, 0) \
\
  while(!TH_TENSOR_APPLY_hasFinished) \
  { \
    /* Loop through the inner most region of the Tensor */ \
    for(; TENSOR##_i < TENSOR##_size; TENSOR##_i++, TENSOR##_data += TENSOR##_stride) /* 0 et pas TENSOR##_dim! */ \
    { \
      CODE \
    } \
    __TH_TENSOR_APPLYX_UPDATE_COUNTERS(TENSOR, 1) \
  } \
  THFree(TENSOR##_counter); \
}

#define TH_TENSOR_APPLY(TYPE, TENSOR, CODE) \
  TH_TENSOR_APPLY_D(TYPE, TENSOR, -1, CODE)


#ifdef _OPENMP

#ifdef _WIN32  
// MSVC doesn't support loop pragmas, but does support others. Create a new macro to account for those differences.  
#define PRAGMA_LOOP(P)    // Noop  
#define PRAGMA(P)         __pragma(P)
#else
#define PRAGMA_LOOP(P)    _Pragma(#P)  
#define PRAGMA(P)         _Pragma(#P)
#endif

#include <omp.h>

/*
 * Calcuate the memory offset of an element in a tensor. The strategy is below:
 *
 * 1. convert the line index(the index of the element) to the indexs(coordinates) in the tensor.
 *    It can hinted by a classical problem: Getting each individual digit from a whole integer(Decimal base).
 *    A N-digit decimal base number could be view as a N-dimension tensor and the sizes of the tensor are 10.
 *    So the value the whole integer is the line index. And the digits could be viewed as the indexes in
 *    different dimentions.
 *
 * 2. convert the indexs(coordinates) in the tensor to the memory offset.
 *
 *  You can get the detailes in the for-statement iterations.
 *
 * The macro is only used in the first element in each thread. For the rest, the memory offset could update
 * according to info of the tensor in order to get better performance. So we should also record the each
 * indexs in coresponding dimension of first element.
 * The recorded info is stored in the TENSOR##_counter_tmp.
 *
 */
#define __TH_TENSOR_APPLYX_CAL_MEMORY_OFFSET(TENSOR) \
  int64_t *TENSOR##_counter_tmp = (int64_t*)THAlloc(sizeof(int64_t) * TENSOR##_dim);                 \
  ptrdiff_t TENSOR##_memory_offset = 0;                                                              \
  ptrdiff_t TENSOR##_quot = line_index_start;                                                        \
  for (TENSOR##_i = TENSOR##_dim-1; TENSOR##_i>=0; --TENSOR##_i) {                                   \
    TENSOR##_counter_tmp[TENSOR##_i] = TENSOR##_quot%TENSOR##_sizes[TENSOR##_i];                     \
    TENSOR##_quot /= TENSOR##_sizes[TENSOR##_i];                                                     \
    TENSOR##_memory_offset += TENSOR##_counter_tmp[TENSOR##_i] * TENSOR##_strides[TENSOR##_i];         \
  }

/*
 * The macro update the indexes in each dimension of the elements except for the first one allocated in
 * each thread.
 * For a tensor, if the index of some dimension reaches the size of the corresponding dimension. It will carry and clear.
 * If the index of next high dimension does do, the index of next high dimension should carry and clear, too.
 *
 * The momery offset calculatation is a little confusing. If current index carries, the current index is set to 0. So
 * the offset should decrease by size*stride of the last dimension. Then the index next high dimension increases by 1. So
 * the offset should increase by stride of next high dimension.
 */
#define __TH_TENSOR_APPLYX_UPDATE_COUNTERS_OMP(TENSOR) \
  if(TENSOR##_i == TENSOR##_size && TENSOR##_dim > 1){ /*reaches the edge*/ \
    int TENSOR##_carry_coord = 1;                      /*set carry flag to true*/ \
    TENSOR##_start = 0;                                /*the current index be cleared to 0*/\
    TENSOR##_data -= TENSOR##_size * TENSOR##_stride;  /*the momery offset reset to the first one in current dimension  */\
    for(TENSOR##_i = TENSOR##_dim - 2; (TENSOR##_i >= 0) && (TENSOR##_carry_coord); TENSOR##_i--){ \
      TENSOR##_counter_tmp[TENSOR##_i]++;             /*the index of next high dimension update*/ \
      TENSOR##_data += TENSOR##_strides[TENSOR##_i];   /*memory offset increase by stride of next high dimension*/\
      if(TENSOR##_counter_tmp[TENSOR##_i] == TENSOR##_sizes[TENSOR##_i]){ /*The next high dimension also carry, continue
        to clear and carry*/\
        TENSOR##_data -= TENSOR##_sizes[TENSOR##_i] * TENSOR##_strides[TENSOR##_i]; \
        TENSOR##_counter_tmp[TENSOR##_i] = 0; \
      } else { \
        TENSOR##_carry_coord = 0; \
      } \
    } \
  } else { \
    TENSOR##_start = TENSOR##_i;                               \
  }


#define TH_TENSOR_APPLY_REDUCTION_OMP(TYPE, TENSOR, OPERATION, CODE, OMP_THRESHOLD) \
{\
  int TENSOR##Contg = THTensor_(isContiguous)(TENSOR);                      \
  ptrdiff_t TENSOR##Size = THTensor_(nElement)(TENSOR);                     \
  if(TENSOR##Contg){                                                        \
    ptrdiff_t iter = 0;                                                     \
    TYPE *rp = THTensor_getStoragePtr(TENSOR)->data<TYPE>()+TENSOR->storage_offset();         \
    PRAGMA( omp parallel for if (TENSOR##Size > OMP_THRESHOLD * 10) firstprivate(rp) reduction(OPERATION) ) \
    for (iter = 0; iter < TENSOR##Size; iter++) { \
      TYPE *TENSOR##_data = rp+iter;                    \
      CODE                                         \
    }                                              \
  } else {                                         \
    int TH_TENSOR_APPLY_hasFinished = 0;           \
    int64_t TH_TENSOR_dim_index = 0;               \
    __TH_TENSOR_APPLYX_PREAMBLE(TYPE, TENSOR, -1, 1);\
    if (0 == TH_TENSOR_APPLY_hasFinished) {          \
      PRAGMA(omp parallel if (TENSOR##Size > OMP_THRESHOLD) firstprivate(TENSOR##_data, TENSOR##_sizes, TENSOR##_strides, TENSOR##_dim, TENSOR##_stride, TENSOR##_size, TENSOR##_i) reduction(OPERATION))\
      {\
        size_t num_threads = omp_get_num_threads();\
        size_t tid = omp_get_thread_num();\
        size_t line_seg_length_avg = TENSOR##Size/num_threads;                                                     \
        ptrdiff_t line_index_start = tid * line_seg_length_avg;                                            \
        ptrdiff_t line_seg_length = (tid == num_threads - 1)? (TENSOR##Size - line_index_start):line_seg_length_avg;  \
        __TH_TENSOR_APPLYX_CAL_MEMORY_OFFSET(TENSOR);\
        TENSOR##_data += TENSOR##_memory_offset;\
        ptrdiff_t count = 0;\
        ptrdiff_t TENSOR##_start = TENSOR##_counter_tmp[TENSOR##_dim - 1];\
        while(count < line_seg_length){\
          for(TENSOR##_i=TENSOR##_start; (count < line_seg_length)&&(TENSOR##_i < TENSOR##_size); ++TENSOR##_i, ++count){\
            CODE\
            TENSOR##_data += TENSOR##_stride;\
          }\
          if(count < line_seg_length){\
            __TH_TENSOR_APPLYX_UPDATE_COUNTERS_OMP(TENSOR);\
          }\
        }\
        if(TENSOR##_counter_tmp != NULL) \
          THFree(TENSOR##_counter_tmp); \
      }\
    }\
    if(TENSOR##_counter != NULL)\
      THFree(TENSOR##_counter);\
  }\
}

#define TH_TENSOR_APPLY2_OMP(SIZE, CONTIG1, CONTIG2, TYPE1, TENSOR1, TYPE2, TENSOR2, CODE, OMP_THRESHOLD) \
{                                                                                              \
  /* for advanced searching index*/                                                            \
  if( CONTIG1 && CONTIG2 ){                                                                    \
    TYPE1 *rp = THTensor_getStoragePtr(TENSOR1)->data<TYPE1>()+TENSOR1->storage_offset();                        \
    TYPE2 *tp = THTensor_getStoragePtr(TENSOR2)->data<TYPE2>()+TENSOR2->storage_offset();                        \
    ptrdiff_t iter = 0;                                                                        \
    if(tp != (TYPE2*)rp) {                                                                             \
      PRAGMA_LOOP(ivdep) \
      PRAGMA( omp parallel for if (SIZE > OMP_THRESHOLD * 10) firstprivate(rp, tp)) \
      for (iter = 0; iter < SIZE; iter++) {                             \
        TYPE2 *TENSOR2##_data = tp+iter;                                \
        TYPE1 *TENSOR1##_data = rp+iter;                                \
        CODE                                                            \
      }\
    } else {\
      PRAGMA_LOOP(simd) \
      PRAGMA( omp parallel for if (SIZE > OMP_THRESHOLD * 10) firstprivate(rp, tp) )  \
      for (iter = 0; iter < SIZE; iter++) {\
        TYPE2* TENSOR2##_data = tp+iter;\
        TYPE1* TENSOR1##_data = rp+iter;\
        CODE                                \
      }\
    }\
  } else {                               \
    /* The following strategy is not easy to understand.
     * 1. Collapse the dimension of the tensors in order to decrease the number of nested loops.
     * 2. Calculate the numbers of elements allocated in each thread and the line index of the first one.
     * 3. Calculate the memory offset of the first element and the indexes in each dimension of the
     *    first one.
     * 4. iterate all elements in each thread. update the indexes in each dimension of the rest.
    */                                                                                             \
    int TH_TENSOR_APPLY_hasFinished = 0; \
    int64_t TH_TENSOR_dim_index = 0;     \
    /*step 1*/                           \
    __TH_TENSOR_APPLYX_PREAMBLE(TYPE2, TENSOR2, -1, 1) \
    __TH_TENSOR_APPLYX_PREAMBLE(TYPE1, TENSOR1, -1, 1) \
    if (0 == TH_TENSOR_APPLY_hasFinished) {            \
      PRAGMA(omp parallel if (SIZE > OMP_THRESHOLD) firstprivate(TENSOR2##_data, TENSOR2##_sizes, TENSOR2##_strides, TENSOR2##_dim, TENSOR2##_stride, TENSOR2##_size, TENSOR2##_i, TENSOR1##_data, TENSOR1##_sizes, TENSOR1##_strides, TENSOR1##_dim, TENSOR1##_stride, TENSOR1##_size, TENSOR1##_i)) \
      {                                   \
        /*step 2*/                                                                 \
        size_t num_threads = omp_get_num_threads();                                                        \
        size_t tid = omp_get_thread_num();                                                                 \
        size_t line_seg_length_avg = SIZE/num_threads;                                                     \
        ptrdiff_t line_index_start = tid * line_seg_length_avg;                                            \
        ptrdiff_t line_seg_length = (tid == num_threads - 1)? (SIZE - line_index_start):line_seg_length_avg;  \
        /* step 3*/                                                                                        \
        __TH_TENSOR_APPLYX_CAL_MEMORY_OFFSET(TENSOR2);                                                            \
        __TH_TENSOR_APPLYX_CAL_MEMORY_OFFSET(TENSOR1);                                                            \
        TENSOR2##_data += TENSOR2##_memory_offset;                                              \
        TENSOR1##_data += TENSOR1##_memory_offset;                                              \
        ptrdiff_t count = 0;                                                                               \
        ptrdiff_t TENSOR2##_start =  TENSOR2##_counter_tmp[TENSOR2##_dim-1];                               \
        ptrdiff_t TENSOR1##_start =  TENSOR1##_counter_tmp[TENSOR1##_dim-1];                               \
        /* step 4*/                                                                                        \
        while (count < line_seg_length) {                                                                     \
          for(TENSOR2##_i=TENSOR2##_start, TENSOR1##_i = TENSOR1##_start; ((count < line_seg_length) && (TENSOR2##_i < TENSOR2##_size) && (TENSOR1##_i < TENSOR1##_size)); ++TENSOR2##_i, ++TENSOR1##_i, ++count){ \
            CODE                                                                                               \
            TENSOR2##_data += TENSOR2##_stride;                                                                \
            TENSOR1##_data += TENSOR1##_stride;                                                                \
          }                                                                                                    \
          if (count < line_seg_length){                                                                           \
            __TH_TENSOR_APPLYX_UPDATE_COUNTERS_OMP(TENSOR2);                                                   \
            __TH_TENSOR_APPLYX_UPDATE_COUNTERS_OMP(TENSOR1);                                                   \
          }                                                                                                    \
        }                                                                                                      \
        if(TENSOR1##_counter_tmp != NULL) \
          THFree(TENSOR1##_counter_tmp); \
        if(TENSOR2##_counter_tmp != NULL) \
          THFree(TENSOR2##_counter_tmp); \
      } \
    }                                                                                                        \
    if(TENSOR2##_counter != NULL) \
      THFree(TENSOR2##_counter); \
    if(TENSOR1##_counter != NULL) \
      THFree(TENSOR1##_counter);\
  }\
}

#define TH_TENSOR_APPLY3_OMP(SIZE, CONTIG1, CONTIG2, CONTIG3, TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, CODE, OMP_THRESHOLD) \
{                                                                             \
  /* for adveanced searching index*/                                                                    \
  if(CONTIG1 && CONTIG2 && CONTIG3){                                                                    \
    TYPE1 *rp = THTensor_getStoragePtr(TENSOR1)->data<TYPE1>()+TENSOR1->storage_offset();                                 \
    TYPE2 *tp = THTensor_getStoragePtr(TENSOR2)->data<TYPE2>()+TENSOR2->storage_offset();                                 \
    TYPE3 *srcp = THTensor_getStoragePtr(TENSOR3)->data<TYPE3>()+TENSOR3->storage_offset();                               \
    ptrdiff_t iter = 0;\
    if(tp != (TYPE2*)rp) {                                                                             \
      PRAGMA_LOOP(ivdep) \
      PRAGMA( omp parallel for if (SIZE > OMP_THRESHOLD * 10) )  \
      for (iter = 0; iter < SIZE; iter++) {\
        TYPE1 *TENSOR1##_data = rp+iter;\
        TYPE2 *TENSOR2##_data = tp+iter; \
        TYPE3 *TENSOR3##_data = srcp+iter;\
        CODE                                \
      } \
    } else {\
      PRAGMA_LOOP(simd) \
      PRAGMA( omp parallel for if (SIZE > OMP_THRESHOLD * 10) )  \
      for (iter = 0; iter < SIZE; iter++) {\
        TYPE1 *TENSOR1##_data = rp+iter;\
        TYPE2 *TENSOR2##_data = tp+iter; \
        TYPE3 *TENSOR3##_data = srcp+iter;\
        CODE                                \
      } \
    }\
  } else{              \
    int TH_TENSOR_APPLY_hasFinished = 0;\
    int64_t TH_TENSOR_dim_index = 0;\
    __TH_TENSOR_APPLYX_PREAMBLE(TYPE1, TENSOR1, -1, 1) \
    __TH_TENSOR_APPLYX_PREAMBLE(TYPE2, TENSOR2, -1, 1) \
    __TH_TENSOR_APPLYX_PREAMBLE(TYPE3, TENSOR3, -1, 1) \
    if (0 == TH_TENSOR_APPLY_hasFinished) {            \
      PRAGMA(omp parallel if (SIZE > OMP_THRESHOLD) firstprivate(TENSOR1##_data, TENSOR1##_sizes, TENSOR1##_strides, TENSOR1##_dim, TENSOR1##_stride, TENSOR1##_size, TENSOR1##_i, TENSOR2##_data, TENSOR2##_sizes, TENSOR2##_strides, TENSOR2##_dim, TENSOR2##_stride, TENSOR2##_size, TENSOR2##_i, TENSOR3##_data, TENSOR3##_sizes, TENSOR3##_strides, TENSOR3##_dim, TENSOR3##_stride, TENSOR3##_size, TENSOR3##_i))\
      {\
        size_t num_threads = omp_get_num_threads();\
        size_t tid = omp_get_thread_num();\
        size_t line_seg_length_avg = SIZE/num_threads;                                                     \
        ptrdiff_t line_index_start = tid * line_seg_length_avg;                                            \
        ptrdiff_t line_seg_length = (tid == num_threads - 1)? (SIZE - line_index_start):line_seg_length_avg;  \
        __TH_TENSOR_APPLYX_CAL_MEMORY_OFFSET(TENSOR1);\
        __TH_TENSOR_APPLYX_CAL_MEMORY_OFFSET(TENSOR2);\
        __TH_TENSOR_APPLYX_CAL_MEMORY_OFFSET(TENSOR3);\
        TENSOR1##_data += TENSOR1##_memory_offset;\
        TENSOR2##_data += TENSOR2##_memory_offset;\
        TENSOR3##_data += TENSOR3##_memory_offset;\
        ptrdiff_t count = 0;\
        ptrdiff_t TENSOR1##_start = TENSOR1##_counter_tmp[TENSOR1##_dim - 1];\
        ptrdiff_t TENSOR2##_start = TENSOR2##_counter_tmp[TENSOR2##_dim - 1];\
        ptrdiff_t TENSOR3##_start = TENSOR3##_counter_tmp[TENSOR3##_dim - 1];\
        while(count < line_seg_length){\
          for(TENSOR1##_i=TENSOR1##_start, TENSOR2##_i=TENSOR2##_start,TENSOR3##_i=TENSOR3##_start; (count<line_seg_length)&&(TENSOR1##_i<TENSOR1##_size)&&(TENSOR2##_i<TENSOR2##_size)&&(TENSOR3##_i<TENSOR3##_size); ++TENSOR1##_i,++TENSOR2##_i,++TENSOR3##_i,++count){\
            CODE\
            TENSOR1##_data += TENSOR1##_stride;\
            TENSOR2##_data += TENSOR2##_stride;\
            TENSOR3##_data += TENSOR3##_stride;\
          }\
          if(count < line_seg_length){\
            __TH_TENSOR_APPLYX_UPDATE_COUNTERS_OMP(TENSOR1);\
            __TH_TENSOR_APPLYX_UPDATE_COUNTERS_OMP(TENSOR2);\
            __TH_TENSOR_APPLYX_UPDATE_COUNTERS_OMP(TENSOR3);\
          }\
        }\
        if(TENSOR1##_counter_tmp != NULL) \
          THFree(TENSOR1##_counter_tmp); \
        if(TENSOR2##_counter_tmp != NULL) \
          THFree(TENSOR2##_counter_tmp); \
        if(TENSOR3##_counter_tmp != NULL) \
          THFree(TENSOR3##_counter_tmp);\
      }\
    }\
    if(TENSOR1##_counter != NULL)\
      THFree(TENSOR1##_counter);\
    if(TENSOR2##_counter != NULL)\
      THFree(TENSOR2##_counter);\
    if(TENSOR3##_counter != NULL)\
      THFree(TENSOR3##_counter);\
  }\
}

#endif
#endif
