#ifndef TH_TENSOR_APPLY_INC
#define TH_TENSOR_APPLY_INC

#define TH_TENSOR_APPLY3(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, CODE) \
{ \
  TYPE1 *TENSOR1##_data = NULL; \
  long *TENSOR1##_counter = NULL; \
  long TENSOR1##_stride = 0, TENSOR1##_size = 0, TENSOR1##_dim = 0, TENSOR1##_i, TENSOR1##_n; \
  TYPE2 *TENSOR2##_data = NULL; \
  long *TENSOR2##_counter = NULL; \
  long TENSOR2##_stride = 0, TENSOR2##_size = 0, TENSOR2##_dim = 0, TENSOR2##_i, TENSOR2##_n; \
  TYPE3 *TENSOR3##_data = NULL; \
  long *TENSOR3##_counter = NULL; \
  long TENSOR3##_stride = 0, TENSOR3##_size = 0, TENSOR3##_dim = 0, TENSOR3##_i, TENSOR3##_n; \
  int TH_TENSOR_APPLY_hasFinished = 0; \
\
  TENSOR1##_n = (TENSOR1->nDimension ? 1 : 0); \
  for(TENSOR1##_i = 0; TENSOR1##_i < TENSOR1->nDimension; TENSOR1##_i++) \
    TENSOR1##_n *= TENSOR1->size[TENSOR1##_i]; \
\
  TENSOR2##_n = (TENSOR2->nDimension ? 1 : 0); \
  for(TENSOR2##_i = 0; TENSOR2##_i < TENSOR2->nDimension; TENSOR2##_i++) \
    TENSOR2##_n *= TENSOR2->size[TENSOR2##_i]; \
\
  TENSOR3##_n = (TENSOR3->nDimension ? 1 : 0); \
  for(TENSOR3##_i = 0; TENSOR3##_i < TENSOR3->nDimension; TENSOR3##_i++) \
    TENSOR3##_n *= TENSOR3->size[TENSOR3##_i]; \
\
  if(TENSOR1##_n != TENSOR2##_n || TENSOR1##_n != TENSOR3##_n) /* should we do the check in the function instead? i think so */ \
    THError("inconsistent tensor size"); \
\
  if(TENSOR1->nDimension == 0) \
    TH_TENSOR_APPLY_hasFinished = 1; \
  else \
  { \
    TENSOR1##_data = TENSOR1->storage->data+TENSOR1->storageOffset; \
    for(TENSOR1##_dim = TENSOR1->nDimension-1; TENSOR1##_dim >= 0; TENSOR1##_dim--) \
    { \
      if(TENSOR1->size[TENSOR1##_dim] != 1) \
        break; \
    } \
    TENSOR1##_stride = (TENSOR1##_dim == -1 ? 0 : TENSOR1->stride[TENSOR1##_dim]); \
    TENSOR1##_size = 1; \
    for(TENSOR1##_dim = TENSOR1->nDimension-1; TENSOR1##_dim >= 0; TENSOR1##_dim--) \
    { \
      if(TENSOR1->size[TENSOR1##_dim] != 1) \
      { \
        if(TENSOR1->stride[TENSOR1##_dim] == TENSOR1##_size) \
          TENSOR1##_size *= TENSOR1->size[TENSOR1##_dim]; \
        else \
          break; \
      } \
    } \
    TENSOR1##_counter = (long*)THAlloc(sizeof(long)*(TENSOR1##_dim+1)); \
    for(TENSOR1##_i = 0; TENSOR1##_i <= TENSOR1##_dim; TENSOR1##_i++) \
      TENSOR1##_counter[TENSOR1##_i] = 0; \
\
    TENSOR2##_data = TENSOR2->storage->data+TENSOR2->storageOffset; \
    for(TENSOR2##_dim = TENSOR2->nDimension-1; TENSOR2##_dim >= 0; TENSOR2##_dim--) \
    { \
      if(TENSOR2->size[TENSOR2##_dim] != 1) \
        break; \
    } \
    TENSOR2##_stride = (TENSOR2##_dim == -1 ? 0 : TENSOR2->stride[TENSOR2##_dim]); \
    TENSOR2##_size = 1; \
    for(TENSOR2##_dim = TENSOR2->nDimension-1; TENSOR2##_dim >= 0; TENSOR2##_dim--) \
    { \
      if(TENSOR2->size[TENSOR2##_dim] != 1) \
      { \
        if(TENSOR2->stride[TENSOR2##_dim] == TENSOR2##_size) \
          TENSOR2##_size *= TENSOR2->size[TENSOR2##_dim]; \
        else \
          break; \
      } \
    } \
    TENSOR2##_counter = (long*)THAlloc(sizeof(long)*(TENSOR2##_dim+1)); \
    for(TENSOR2##_i = 0; TENSOR2##_i <= TENSOR2##_dim; TENSOR2##_i++) \
      TENSOR2##_counter[TENSOR2##_i] = 0; \
\
    TENSOR3##_data = TENSOR3->storage->data+TENSOR3->storageOffset; \
    for(TENSOR3##_dim = TENSOR3->nDimension-1; TENSOR3##_dim >= 0; TENSOR3##_dim--) \
    { \
      if(TENSOR3->size[TENSOR3##_dim] != 1) \
        break; \
    } \
    TENSOR3##_stride = (TENSOR3##_dim == -1 ? 0 : TENSOR3->stride[TENSOR3##_dim]); \
    TENSOR3##_size = 1; \
    for(TENSOR3##_dim = TENSOR3->nDimension-1; TENSOR3##_dim >= 0; TENSOR3##_dim--) \
    { \
      if(TENSOR3->size[TENSOR3##_dim] != 1) \
      { \
        if(TENSOR3->stride[TENSOR3##_dim] == TENSOR3##_size) \
          TENSOR3##_size *= TENSOR3->size[TENSOR3##_dim]; \
        else \
          break; \
      } \
    } \
    TENSOR3##_counter = (long*)THAlloc(sizeof(long)*(TENSOR3##_dim+1)); \
    for(TENSOR3##_i = 0; TENSOR3##_i <= TENSOR3##_dim; TENSOR3##_i++) \
      TENSOR3##_counter[TENSOR3##_i] = 0; \
  } \
\
  TENSOR1##_i = 0; \
  TENSOR2##_i = 0; \
  TENSOR3##_i = 0; \
  while(!TH_TENSOR_APPLY_hasFinished) \
  { \
    for(; TENSOR1##_i < TENSOR1##_size && TENSOR2##_i < TENSOR2##_size && TENSOR3##_i < TENSOR3##_size; TENSOR1##_i++, TENSOR2##_i++, TENSOR3##_i++, TENSOR1##_data += TENSOR1##_stride, TENSOR2##_data += TENSOR2##_stride, TENSOR3##_data += TENSOR3##_stride) /* 0 et pas TENSOR##_dim! */ \
    { \
      CODE \
    } \
\
    if(TENSOR1##_i == TENSOR1##_size) \
    { \
      if(TENSOR1##_dim == -1) \
         break; \
\
      TENSOR1##_data -= TENSOR1##_size*TENSOR1##_stride; \
      for(TENSOR1##_i = TENSOR1##_dim; TENSOR1##_i >= 0; TENSOR1##_i--) \
      { \
        TENSOR1##_counter[TENSOR1##_i]++; \
        TENSOR1##_data += TENSOR1->stride[TENSOR1##_i]; \
\
        if(TENSOR1##_counter[TENSOR1##_i]  == TENSOR1->size[TENSOR1##_i]) \
        { \
          if(TENSOR1##_i == 0) \
          { \
            TH_TENSOR_APPLY_hasFinished = 1; \
            break; \
          } \
            else \
          { \
            TENSOR1##_data -= TENSOR1##_counter[TENSOR1##_i]*TENSOR1->stride[TENSOR1##_i]; \
            TENSOR1##_counter[TENSOR1##_i] = 0; \
          } \
        } \
        else \
          break; \
      } \
      TENSOR1##_i = 0; \
    } \
\
    if(TENSOR2##_i == TENSOR2##_size) \
    { \
      if(TENSOR2##_dim == -1) \
         break; \
\
      TENSOR2##_data -= TENSOR2##_size*TENSOR2##_stride; \
      for(TENSOR2##_i = TENSOR2##_dim; TENSOR2##_i >= 0; TENSOR2##_i--) \
      { \
        TENSOR2##_counter[TENSOR2##_i]++; \
        TENSOR2##_data += TENSOR2->stride[TENSOR2##_i]; \
\
        if(TENSOR2##_counter[TENSOR2##_i]  == TENSOR2->size[TENSOR2##_i]) \
        { \
          if(TENSOR2##_i == 0) \
          { \
            TH_TENSOR_APPLY_hasFinished = 1; \
            break; \
          } \
            else \
          { \
            TENSOR2##_data -= TENSOR2##_counter[TENSOR2##_i]*TENSOR2->stride[TENSOR2##_i]; \
            TENSOR2##_counter[TENSOR2##_i] = 0; \
          } \
        } \
        else \
          break; \
      } \
      TENSOR2##_i = 0; \
    } \
\
    if(TENSOR3##_i == TENSOR3##_size) \
    { \
      if(TENSOR3##_dim == -1) \
         break; \
\
      TENSOR3##_data -= TENSOR3##_size*TENSOR3##_stride; \
      for(TENSOR3##_i = TENSOR3##_dim; TENSOR3##_i >= 0; TENSOR3##_i--) \
      { \
        TENSOR3##_counter[TENSOR3##_i]++; \
        TENSOR3##_data += TENSOR3->stride[TENSOR3##_i]; \
\
        if(TENSOR3##_counter[TENSOR3##_i]  == TENSOR3->size[TENSOR3##_i]) \
        { \
          if(TENSOR3##_i == 0) \
          { \
            TH_TENSOR_APPLY_hasFinished = 1; \
            break; \
          } \
            else \
          { \
            TENSOR3##_data -= TENSOR3##_counter[TENSOR3##_i]*TENSOR3->stride[TENSOR3##_i]; \
            TENSOR3##_counter[TENSOR3##_i] = 0; \
          } \
        } \
        else \
          break; \
      } \
      TENSOR3##_i = 0; \
    } \
  } \
  THFree(TENSOR1##_counter); \
  THFree(TENSOR2##_counter); \
  THFree(TENSOR3##_counter); \
}

#define TH_TENSOR_APPLY2(TYPE1, TENSOR1, TYPE2, TENSOR2, CODE) \
{ \
  TYPE1 *TENSOR1##_data = NULL; \
  long *TENSOR1##_counter = NULL; \
  long TENSOR1##_stride = 0, TENSOR1##_size = 0, TENSOR1##_dim = 0, TENSOR1##_i, TENSOR1##_n; \
  TYPE2 *TENSOR2##_data = NULL; \
  long *TENSOR2##_counter = NULL; \
  long TENSOR2##_stride = 0, TENSOR2##_size = 0, TENSOR2##_dim = 0, TENSOR2##_i, TENSOR2##_n; \
  int TH_TENSOR_APPLY_hasFinished = 0; \
\
  TENSOR1##_n = (TENSOR1->nDimension ? 1 : 0); \
  for(TENSOR1##_i = 0; TENSOR1##_i < TENSOR1->nDimension; TENSOR1##_i++) \
    TENSOR1##_n *= TENSOR1->size[TENSOR1##_i]; \
\
  TENSOR2##_n = (TENSOR2->nDimension ? 1 : 0); \
  for(TENSOR2##_i = 0; TENSOR2##_i < TENSOR2->nDimension; TENSOR2##_i++) \
    TENSOR2##_n *= TENSOR2->size[TENSOR2##_i]; \
\
  if(TENSOR1##_n != TENSOR2##_n) /* should we do the check in the function instead? i think so */ \
    THError("inconsistent tensor size"); \
\
  if(TENSOR1->nDimension == 0) \
    TH_TENSOR_APPLY_hasFinished = 1; \
  else \
  { \
    TENSOR1##_data = TENSOR1->storage->data+TENSOR1->storageOffset; \
    for(TENSOR1##_dim = TENSOR1->nDimension-1; TENSOR1##_dim >= 0; TENSOR1##_dim--) \
    { \
      if(TENSOR1->size[TENSOR1##_dim] != 1) \
        break; \
    } \
    TENSOR1##_stride = (TENSOR1##_dim == -1 ? 0 : TENSOR1->stride[TENSOR1##_dim]); \
    TENSOR1##_size = 1; \
    for(TENSOR1##_dim = TENSOR1->nDimension-1; TENSOR1##_dim >= 0; TENSOR1##_dim--) \
    { \
      if(TENSOR1->size[TENSOR1##_dim] != 1) \
      { \
        if(TENSOR1->stride[TENSOR1##_dim] == TENSOR1##_size) \
          TENSOR1##_size *= TENSOR1->size[TENSOR1##_dim]; \
        else \
          break; \
      } \
    } \
    TENSOR1##_counter = (long*)THAlloc(sizeof(long)*(TENSOR1##_dim+1)); \
    for(TENSOR1##_i = 0; TENSOR1##_i <= TENSOR1##_dim; TENSOR1##_i++) \
      TENSOR1##_counter[TENSOR1##_i] = 0; \
\
    TENSOR2##_data = TENSOR2->storage->data+TENSOR2->storageOffset; \
    for(TENSOR2##_dim = TENSOR2->nDimension-1; TENSOR2##_dim >= 0; TENSOR2##_dim--) \
    { \
      if(TENSOR2->size[TENSOR2##_dim] != 1) \
        break; \
    } \
    TENSOR2##_stride = (TENSOR2##_dim == -1 ? 0 : TENSOR2->stride[TENSOR2##_dim]); \
    TENSOR2##_size = 1; \
    for(TENSOR2##_dim = TENSOR2->nDimension-1; TENSOR2##_dim >= 0; TENSOR2##_dim--) \
    { \
      if(TENSOR2->size[TENSOR2##_dim] != 1) \
      { \
        if(TENSOR2->stride[TENSOR2##_dim] == TENSOR2##_size) \
          TENSOR2##_size *= TENSOR2->size[TENSOR2##_dim]; \
        else \
          break; \
      } \
    } \
    TENSOR2##_counter = (long*)THAlloc(sizeof(long)*(TENSOR2##_dim+1)); \
    for(TENSOR2##_i = 0; TENSOR2##_i <= TENSOR2##_dim; TENSOR2##_i++) \
      TENSOR2##_counter[TENSOR2##_i] = 0; \
  } \
\
  TENSOR1##_i = 0; \
  TENSOR2##_i = 0; \
  while(!TH_TENSOR_APPLY_hasFinished) \
  { \
    for(; TENSOR1##_i < TENSOR1##_size && TENSOR2##_i < TENSOR2##_size; TENSOR1##_i++, TENSOR2##_i++, TENSOR1##_data += TENSOR1##_stride, TENSOR2##_data += TENSOR2##_stride) /* 0 et pas TENSOR##_dim! */ \
    { \
      CODE \
    } \
\
    if(TENSOR1##_i == TENSOR1##_size) \
    { \
      if(TENSOR1##_dim == -1) \
         break; \
\
      TENSOR1##_data -= TENSOR1##_size*TENSOR1##_stride; \
      for(TENSOR1##_i = TENSOR1##_dim; TENSOR1##_i >= 0; TENSOR1##_i--) \
      { \
        TENSOR1##_counter[TENSOR1##_i]++; \
        TENSOR1##_data += TENSOR1->stride[TENSOR1##_i]; \
\
        if(TENSOR1##_counter[TENSOR1##_i]  == TENSOR1->size[TENSOR1##_i]) \
        { \
          if(TENSOR1##_i == 0) \
          { \
            TH_TENSOR_APPLY_hasFinished = 1; \
            break; \
          } \
            else \
          { \
            TENSOR1##_data -= TENSOR1##_counter[TENSOR1##_i]*TENSOR1->stride[TENSOR1##_i]; \
            TENSOR1##_counter[TENSOR1##_i] = 0; \
          } \
        } \
        else \
          break; \
      } \
      TENSOR1##_i = 0; \
    } \
\
    if(TENSOR2##_i == TENSOR2##_size) \
    { \
      if(TENSOR2##_dim == -1) \
         break; \
\
      TENSOR2##_data -= TENSOR2##_size*TENSOR2##_stride; \
      for(TENSOR2##_i = TENSOR2##_dim; TENSOR2##_i >= 0; TENSOR2##_i--) \
      { \
        TENSOR2##_counter[TENSOR2##_i]++; \
        TENSOR2##_data += TENSOR2->stride[TENSOR2##_i]; \
\
        if(TENSOR2##_counter[TENSOR2##_i]  == TENSOR2->size[TENSOR2##_i]) \
        { \
          if(TENSOR2##_i == 0) \
          { \
            TH_TENSOR_APPLY_hasFinished = 1; \
            break; \
          } \
            else \
          { \
            TENSOR2##_data -= TENSOR2##_counter[TENSOR2##_i]*TENSOR2->stride[TENSOR2##_i]; \
            TENSOR2##_counter[TENSOR2##_i] = 0; \
          } \
        } \
        else \
          break; \
      } \
      TENSOR2##_i = 0; \
    } \
  } \
  THFree(TENSOR1##_counter); \
  THFree(TENSOR2##_counter); \
}

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
 */
#define TH_TENSOR_APPLY(TYPE, TENSOR, CODE) \
{ \
  TYPE *TENSOR##_data = NULL; \
  long *TENSOR##_counter = NULL; \
  long TENSOR##_stride = 0, TENSOR##_size = 0, TENSOR##_dim = 0, TENSOR##_i; \
  int TH_TENSOR_APPLY_hasFinished = 0; \
\
  if(TENSOR->nDimension == 0) \
    TH_TENSOR_APPLY_hasFinished = 1; \
  else \
  { \
    TENSOR##_data = TENSOR->storage->data+TENSOR->storageOffset; \
\
    /* what is the first stride (ignore first dims=1)? */ \
    /* it will be used for offset updates while looping through the largest contiguous section */ \
    for(TENSOR##_dim = TENSOR->nDimension-1; TENSOR##_dim >= 0; TENSOR##_dim--) \
    { \
      if(TENSOR->size[TENSOR##_dim] != 1) \
        break; \
    } \
    TENSOR##_stride = (TENSOR##_dim == -1 ? 0 : TENSOR->stride[TENSOR##_dim]); \
\
    /* what is the largest contiguous section? size will store the size of this section */ \
    TENSOR##_size = 1; \
    for(TENSOR##_dim = TENSOR->nDimension-1; TENSOR##_dim >= 0; TENSOR##_dim--) \
    { \
      if(TENSOR->size[TENSOR##_dim] != 1) \
      { \
        if(TENSOR->stride[TENSOR##_dim] == TENSOR##_size) \
          TENSOR##_size *= TENSOR->size[TENSOR##_dim]; \
        else \
          break; \
      } \
    } \
\
    /* allocate an array of k+1 elements, where k is the first index that */ \
    /* break contiguity. Note that if the tensor is contiguous, then k is -1 and */ \
    /* this counter array is empty. */ \
\
    /* TENSOR##_counter tracks where we are in the storage. The offset into the */ \
    /* storage is given by storage_offset + (i * j), where i is the stride */ \
    /* vector and j is tensor_counter vector. This sets the starting position for the loop. */ \
    TENSOR##_counter = (long*)THAlloc(sizeof(long)*(TENSOR##_dim+1)); \
    for(TENSOR##_i = 0; TENSOR##_i <= TENSOR##_dim; TENSOR##_i++) \
      TENSOR##_counter[TENSOR##_i] = 0; \
  } \
\
  while(!TH_TENSOR_APPLY_hasFinished) \
  { \
    /* Loop through the contiguous section of the Tensor */ \
    for(TENSOR##_i = 0; TENSOR##_i < TENSOR##_size; TENSOR##_i++, TENSOR##_data += TENSOR##_stride) /* 0 et pas TENSOR##_dim! */ \
    { \
      CODE \
    } \
\
\
    /* Handle corner case where the entire Tensor was contiguous */ \
    if(TENSOR##_dim == -1) \
       break; \
 \
    /* Reset pointer to beginning of loop */ \
    TENSOR##_data -= TENSOR##_i*TENSOR##_stride; \
    for(TENSOR##_i = TENSOR##_dim; TENSOR##_i >= 0; TENSOR##_i--) \
    { \
      TENSOR##_counter[TENSOR##_i]++; \
\
      /* Jump ahread by the stride of this dimension */ \
      TENSOR##_data += TENSOR->stride[TENSOR##_i]; \
\
      if(TENSOR##_counter[TENSOR##_i]  == TENSOR->size[TENSOR##_i]) \
      { \
        if(TENSOR##_i == 0) \
        { \
          TH_TENSOR_APPLY_hasFinished = 1; \
          break; \
        } \
        else \
        { \
          /* Reset the pointer to the beginning of the chunk defined by this dimension */ \
          TENSOR##_data -= TENSOR##_counter[TENSOR##_i]*TENSOR->stride[TENSOR##_i]; \
          TENSOR##_counter[TENSOR##_i] = 0; \
        } \
      } \
      else \
        break; \
    } \
  } \
  THFree(TENSOR##_counter); \
}

#endif
