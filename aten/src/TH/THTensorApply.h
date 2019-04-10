#ifndef TH_TENSOR_APPLY_INC
#define TH_TENSOR_APPLY_INC

#include <ATen/Parallel.h>

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
    TENSOR##_memory_offset += TENSOR##_counter_tmp[TENSOR##_i] * TENSOR##_strides[TENSOR##_i];       \
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
#define __TH_TENSOR_APPLYX_UPDATE_COUNTERS_PARALLEL(TENSOR) \
  if(TENSOR##_i == TENSOR##_size && TENSOR##_dim > 1){ /*reaches the edge*/ \
    int TENSOR##_carry_coord = 1;                      /*set carry flag to true*/ \
    TENSOR##_start = 0;                                /*the current index be cleared to 0*/\
    TENSOR##_data -= TENSOR##_size * TENSOR##_stride;  /*the momery offset reset to the first one in current dimension  */\
    for(TENSOR##_i = TENSOR##_dim - 2; (TENSOR##_i >= 0) && (TENSOR##_carry_coord); TENSOR##_i--){ \
      TENSOR##_counter_tmp[TENSOR##_i]++;             /*the index of next high dimension update*/ \
      TENSOR##_data += TENSOR##_strides[TENSOR##_i];   /*memory offset increase by stride of next high dimension*/\
      if(TENSOR##_counter_tmp[TENSOR##_i] == TENSOR##_sizes[TENSOR##_i]){ /*The next high dimension also carry, continue
        to clear and carry*/ \
        TENSOR##_data -= TENSOR##_sizes[TENSOR##_i] * TENSOR##_strides[TENSOR##_i]; \
        TENSOR##_counter_tmp[TENSOR##_i] = 0; \
      } else { \
        TENSOR##_carry_coord = 0; \
      } \
    } \
  } else { \
    TENSOR##_start = TENSOR##_i; \
  }

#endif
