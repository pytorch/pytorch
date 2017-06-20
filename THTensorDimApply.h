#ifndef TH_TENSOR_DIM_APPLY_INC
#define TH_TENSOR_DIM_APPLY_INC

#define TH_TENSOR_DIM_APPLY3(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, DIMENSION, CODE) \
{ \
  TYPE1 *TENSOR1##_data = NULL; \
  long TENSOR1##_stride = 0, TENSOR1##_size = 0; \
  TYPE2 *TENSOR2##_data = NULL; \
  long TENSOR2##_stride = 0, TENSOR2##_size = 0; \
  TYPE3 *TENSOR3##_data = NULL; \
  long TENSOR3##_stride = 0, TENSOR3##_size = 0; \
  long *TH_TENSOR_DIM_APPLY_counter = NULL; \
  int TH_TENSOR_DIM_APPLY_hasFinished = 0; \
  int TH_TENSOR_DIM_APPLY_i; \
\
  if( (DIMENSION < 0) || (DIMENSION >= TENSOR1->nDimension) ) \
    THError("invalid dimension %d (expected to be 0 <= dim < %d)", DIMENSION, TENSOR1->nDimension); \
  int same_dims = 1;                                                    \
  if( TENSOR1->nDimension != TENSOR2->nDimension ) {                    \
    same_dims = 0;                                                      \
  } \
  if( TENSOR1->nDimension != TENSOR3->nDimension ) { \
    same_dims = 0;                                   \
  } \
  if (same_dims == 0) { \
    THDescBuff T1buff = _THSizeDesc(TENSOR1->size, TENSOR1->nDimension); \
    THDescBuff T2buff = _THSizeDesc(TENSOR2->size, TENSOR2->nDimension); \
    THDescBuff T3buff = _THSizeDesc(TENSOR3->size, TENSOR3->nDimension); \
    THError("inconsistent tensor size, expected %s %s, %s %s and %s %s to have the same " \
            "number of dimensions", #TENSOR1, T1buff.str, #TENSOR2, T2buff.str, #TENSOR3, T3buff.str); \
  }                                                                     \
  int shape_check_flag = 0;                                             \
  for(TH_TENSOR_DIM_APPLY_i = 0; TH_TENSOR_DIM_APPLY_i < TENSOR1->nDimension; TH_TENSOR_DIM_APPLY_i++) \
  { \
    if(TH_TENSOR_DIM_APPLY_i == DIMENSION) \
      continue; \
    if(TENSOR1->size[TH_TENSOR_DIM_APPLY_i] != TENSOR2->size[TH_TENSOR_DIM_APPLY_i]) \
      shape_check_flag = 1;                                             \
    if(TENSOR1->size[TH_TENSOR_DIM_APPLY_i] != TENSOR3->size[TH_TENSOR_DIM_APPLY_i]) \
      shape_check_flag = 1;                                             \
  } \
    \
  if (shape_check_flag == 1) { \
    THDescBuff T1buff = _THSizeDesc(TENSOR1->size, TENSOR1->nDimension); \
    THDescBuff T2buff = _THSizeDesc(TENSOR2->size, TENSOR2->nDimension); \
    THDescBuff T3buff = _THSizeDesc(TENSOR3->size, TENSOR3->nDimension); \
    THError("Expected %s %s, %s %s and %s %s to have the same size in dimension %d", \
            #TENSOR1, T1buff.str, #TENSOR2, T2buff.str, #TENSOR3, T3buff.str, DIMENSION); \
  } \
\
  TH_TENSOR_DIM_APPLY_counter = (long*)THAlloc(sizeof(long)*(TENSOR1->nDimension)); \
  for(TH_TENSOR_DIM_APPLY_i = 0; TH_TENSOR_DIM_APPLY_i < TENSOR1->nDimension; TH_TENSOR_DIM_APPLY_i++) \
    TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i] = 0; \
\
  TENSOR1##_data = (TENSOR1)->storage->data+(TENSOR1)->storageOffset; \
  TENSOR1##_stride = (TENSOR1)->stride[DIMENSION]; \
  TENSOR1##_size = TENSOR1->size[DIMENSION]; \
\
  TENSOR2##_data = (TENSOR2)->storage->data+(TENSOR2)->storageOffset; \
  TENSOR2##_stride = (TENSOR2)->stride[DIMENSION]; \
  TENSOR2##_size = TENSOR2->size[DIMENSION]; \
\
  TENSOR3##_data = (TENSOR3)->storage->data+(TENSOR3)->storageOffset; \
  TENSOR3##_stride = (TENSOR3)->stride[DIMENSION]; \
  TENSOR3##_size = TENSOR3->size[DIMENSION]; \
\
  while(!TH_TENSOR_DIM_APPLY_hasFinished) \
  { \
    CODE \
\
    if(TENSOR1->nDimension == 1) \
       break; \
 \
    for(TH_TENSOR_DIM_APPLY_i = 0; TH_TENSOR_DIM_APPLY_i < TENSOR1->nDimension; TH_TENSOR_DIM_APPLY_i++) \
    { \
      if(TH_TENSOR_DIM_APPLY_i == DIMENSION) \
      { \
        if(TH_TENSOR_DIM_APPLY_i == TENSOR1->nDimension-1) \
        { \
          TH_TENSOR_DIM_APPLY_hasFinished = 1; \
          break; \
        } \
        continue; \
      } \
\
      TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i]++; \
      TENSOR1##_data += TENSOR1->stride[TH_TENSOR_DIM_APPLY_i]; \
      TENSOR2##_data += TENSOR2->stride[TH_TENSOR_DIM_APPLY_i]; \
      TENSOR3##_data += TENSOR3->stride[TH_TENSOR_DIM_APPLY_i]; \
\
      if(TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i] == TENSOR1->size[TH_TENSOR_DIM_APPLY_i]) \
      { \
        if(TH_TENSOR_DIM_APPLY_i == TENSOR1->nDimension-1) \
        { \
          TH_TENSOR_DIM_APPLY_hasFinished = 1; \
          break; \
        } \
        else \
        { \
          TENSOR1##_data -= TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i]*TENSOR1->stride[TH_TENSOR_DIM_APPLY_i]; \
          TENSOR2##_data -= TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i]*TENSOR2->stride[TH_TENSOR_DIM_APPLY_i]; \
          TENSOR3##_data -= TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i]*TENSOR3->stride[TH_TENSOR_DIM_APPLY_i]; \
          TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i] = 0; \
        } \
      } \
      else \
        break; \
    } \
  } \
  THFree(TH_TENSOR_DIM_APPLY_counter); \
}

/**
 * Similar to DIM_APPLY(...) but we maintain two sets of pointers: one for the first tensor
 * and one for the second. The two tensors must have the same shape, other than at the
 * specified DIMENSION. This function makes it easy to store the output from reducing the
 * TENSOR at index. For example, in the sum example described below, we could instead do:
 *
 * long i = 0;
 * TYPE1 sum;
 *
 * for (i = 0; i < TENSOR1##_size; ++i) {
 *   sum += TENSOR1##_data[i * TENSOR1##_stride]
 * }
 * *TENSOR2##_data = (TYPE2) sum;
 *
 * In particular, we guarantee that the offset into TENSOR2 will be what you would get if
 * you applied all of the index values used to generate the offset into TENSOR1.
 */
#define TH_TENSOR_DIM_APPLY2(TYPE1, TENSOR1, TYPE2, TENSOR2, DIMENSION, CODE) \
{ \
  TYPE1 *TENSOR1##_data = NULL; \
  long TENSOR1##_stride = 0, TENSOR1##_size = 0; \
  TYPE2 *TENSOR2##_data = NULL; \
  long TENSOR2##_stride = 0, TENSOR2##_size = 0; \
  long *TH_TENSOR_DIM_APPLY_counter = NULL; \
  int TH_TENSOR_DIM_APPLY_hasFinished = 0; \
  int TH_TENSOR_DIM_APPLY_i; \
\
  if( (DIMENSION < 0) || (DIMENSION >= TENSOR1->nDimension) ) \
    THError("invalid dimension %d (expected to be 0 <= dim < %d)", DIMENSION, TENSOR1->nDimension); \
  if( TENSOR1->nDimension != TENSOR2->nDimension ) {                    \
    THDescBuff T1buff = _THSizeDesc(TENSOR1->size, TENSOR1->nDimension); \
    THDescBuff T2buff = _THSizeDesc(TENSOR2->size, TENSOR2->nDimension); \
    THError("inconsistent tensor size, expected %s %s and %s %s to have the same " \
            "number of dimensions", #TENSOR1, T1buff.str, #TENSOR2, T2buff.str);        \
  }                                                                     \
  int shape_check_flag = 0;                                             \
  for(TH_TENSOR_DIM_APPLY_i = 0; TH_TENSOR_DIM_APPLY_i < TENSOR1->nDimension; TH_TENSOR_DIM_APPLY_i++) \
  { \
    if(TH_TENSOR_DIM_APPLY_i == DIMENSION) \
      continue; \
    if(TENSOR1->size[TH_TENSOR_DIM_APPLY_i] != TENSOR2->size[TH_TENSOR_DIM_APPLY_i]) { \
      THDescBuff T1buff = _THSizeDesc(TENSOR1->size, TENSOR1->nDimension); \
      THDescBuff T2buff = _THSizeDesc(TENSOR2->size, TENSOR2->nDimension); \
      THError("Expected %s %s and %s %s to have the same size in dimension %d", \
              #TENSOR1, T1buff.str, #TENSOR2, T2buff.str, DIMENSION);   \
    }                                                                   \
  } \
\
  TH_TENSOR_DIM_APPLY_counter = (long*)THAlloc(sizeof(long)*(TENSOR1->nDimension)); \
  for(TH_TENSOR_DIM_APPLY_i = 0; TH_TENSOR_DIM_APPLY_i < TENSOR1->nDimension; TH_TENSOR_DIM_APPLY_i++) \
    TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i] = 0; \
\
  TENSOR1##_data = (TENSOR1)->storage->data+(TENSOR1)->storageOffset; \
  TENSOR1##_stride = (TENSOR1)->stride[DIMENSION]; \
  TENSOR1##_size = TENSOR1->size[DIMENSION]; \
\
  TENSOR2##_data = (TENSOR2)->storage->data+(TENSOR2)->storageOffset; \
  TENSOR2##_stride = (TENSOR2)->stride[DIMENSION]; \
  TENSOR2##_size = TENSOR2->size[DIMENSION]; \
\
  while(!TH_TENSOR_DIM_APPLY_hasFinished) \
  { \
    CODE \
\
    if(TENSOR1->nDimension == 1) \
       break; \
 \
    for(TH_TENSOR_DIM_APPLY_i = 0; TH_TENSOR_DIM_APPLY_i < TENSOR1->nDimension; TH_TENSOR_DIM_APPLY_i++) \
    { \
      if(TH_TENSOR_DIM_APPLY_i == DIMENSION) \
      { \
        if(TH_TENSOR_DIM_APPLY_i == TENSOR1->nDimension-1) \
        { \
          TH_TENSOR_DIM_APPLY_hasFinished = 1; \
          break; \
        } \
        continue; \
      } \
\
      TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i]++; \
      TENSOR1##_data += TENSOR1->stride[TH_TENSOR_DIM_APPLY_i]; \
      TENSOR2##_data += TENSOR2->stride[TH_TENSOR_DIM_APPLY_i]; \
\
      if(TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i] == TENSOR1->size[TH_TENSOR_DIM_APPLY_i]) \
      { \
        if(TH_TENSOR_DIM_APPLY_i == TENSOR1->nDimension-1) \
        { \
          TH_TENSOR_DIM_APPLY_hasFinished = 1; \
          break; \
        } \
        else \
        { \
          TENSOR1##_data -= TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i]*TENSOR1->stride[TH_TENSOR_DIM_APPLY_i]; \
          TENSOR2##_data -= TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i]*TENSOR2->stride[TH_TENSOR_DIM_APPLY_i]; \
          TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i] = 0; \
        } \
      } \
      else \
        break; \
    } \
  } \
  THFree(TH_TENSOR_DIM_APPLY_counter); \
}

/**
 * The basic idea for DIM_APPLY: Given a TENSOR and a DIMENSION, provide access to the data stored
 * at all sets of dimension values other than DIMENSION, such that we can get all the values at those
 * fixed indices for the various values at DIMENSION.
 *
 * Suppose we have a 2x3x4 Tensor A, and we have DIMENSION=2. Then we will hit CODE (2x3) times, and the
 * pointer into storage will be at:
 *
 * A[0][0]
 * A[0][1]
 * A[0][2]
 * A[1][0]
 * A[1][1]
 * A[1][2]
 *
 * And at each point, we can access the data for each of the four elements of the Tensor via
 * TENSOR##_stride. So for example, if we wanted to sum the elements there, we could do:
 *
 * long i = 0;
 * TYPE sum;
 * for (i = 0; i < TENSOR##_size; i++) {
 *  sum += TENSOR##_data[i * TENSOR##_stride]
 * }
 *
 * Note that we don't have to have DIMENSION be the last tensor. If we have DIMENSION=1, then we will hit the
 * code (2x4) times, with pointer into the storage at:
 *
 * offset +
 *   stride_0 * 0 + stride_2 * 0
 *   stride_0 * 1 + stride_2 * 0
 *   stride_0 * 0 + stride_2 * 1
 *   stride_0 * 1 + stride_2 * 1
 *   stride_0 * 0 + stride_2 * 2
 *   stride_0 * 1 + stride_2 * 2
 *   stride_0 * 0 + stride_2 * 3
 *   stride_0 * 1 + stride_2 * 3
 *
 * So we can again sum over the values at DIMENSION with the other indices fixed.
 */
#define TH_TENSOR_DIM_APPLY(TYPE, TENSOR, DIMENSION, CODE) \
{ \
  TYPE *TENSOR##_data = NULL; \
  long TENSOR##_stride = 0, TENSOR##_size = 0; \
  long *TH_TENSOR_DIM_APPLY_counter = NULL; \
  int TH_TENSOR_DIM_APPLY_hasFinished = 0; \
  int TH_TENSOR_DIM_APPLY_i; \
\
  if( (DIMENSION < 0) || (DIMENSION >= TENSOR->nDimension) ) \
    THError("invalid dimension"); \
\
  TENSOR##_data = (TENSOR)->storage->data+(TENSOR)->storageOffset; \
  TENSOR##_stride = (TENSOR)->stride[DIMENSION]; \
  TENSOR##_size = TENSOR->size[DIMENSION]; \
  /* Counter stores the indices into the Tensor at any time */ \
  TH_TENSOR_DIM_APPLY_counter = (long*)THAlloc(sizeof(long)*(TENSOR->nDimension)); \
  for(TH_TENSOR_DIM_APPLY_i = 0; TH_TENSOR_DIM_APPLY_i < TENSOR->nDimension; TH_TENSOR_DIM_APPLY_i++) \
    TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i] = 0; \
\
  while(!TH_TENSOR_DIM_APPLY_hasFinished) \
  { \
    CODE \
\
    if(TENSOR->nDimension == 1) \
       break; \
 \
    for(TH_TENSOR_DIM_APPLY_i = 0; TH_TENSOR_DIM_APPLY_i < TENSOR->nDimension; TH_TENSOR_DIM_APPLY_i++) \
    { \
       /* Check if the index is equal to DIMENSION. We don't need to update the */ \
       /* offset if this is the case, and can consider the next index. However, */ \
       /* in the case that the DIMENSION is the last index in the Tensor, then */ \
       /* we have parsed the entire tensor and can exit */ \
      if(TH_TENSOR_DIM_APPLY_i == DIMENSION) \
      { \
        if(TH_TENSOR_DIM_APPLY_i == TENSOR->nDimension-1) \
        { \
          TH_TENSOR_DIM_APPLY_hasFinished = 1; \
          break; \
        } \
        continue; \
      } \
\
      /* Bump the counter at this index, update the pointer */ \
      TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i]++; \
      TENSOR##_data += TENSOR->stride[TH_TENSOR_DIM_APPLY_i]; \
\
      if(TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i] == TENSOR->size[TH_TENSOR_DIM_APPLY_i]) \
      { \
        /* Handled TENSOR_size(dim) iterations for DIM_APPLY_i. If this is the last dimension, exit */ \
        if(TH_TENSOR_DIM_APPLY_i == TENSOR->nDimension-1) \
        { \
          TH_TENSOR_DIM_APPLY_hasFinished = 1; \
          break; \
        } \
        else \
        { \
          /* Reset the counter, and the pointer to the beginning of the storage for this combination of indices */ \
          TENSOR##_data -= TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i]*TENSOR->stride[TH_TENSOR_DIM_APPLY_i]; \
          TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i] = 0; \
        } \
      } \
      else \
        break; \
    } \
  } \
  THFree(TH_TENSOR_DIM_APPLY_counter); \
}

#endif
