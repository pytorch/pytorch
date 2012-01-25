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
    THError("invalid dimension"); \
  if( TENSOR1->nDimension != TENSOR2->nDimension ) \
    THError("inconsistent tensor sizes"); \
  if( TENSOR1->nDimension != TENSOR3->nDimension ) \
    THError("inconsistent tensor sizes"); \
  for(TH_TENSOR_DIM_APPLY_i = 0; TH_TENSOR_DIM_APPLY_i < TENSOR1->nDimension; TH_TENSOR_DIM_APPLY_i++) \
  { \
    if(TH_TENSOR_DIM_APPLY_i == DIMENSION) \
      continue; \
    if(TENSOR1->size[TH_TENSOR_DIM_APPLY_i] != TENSOR2->size[TH_TENSOR_DIM_APPLY_i]) \
      THError("inconsistent tensor sizes"); \
    if(TENSOR1->size[TH_TENSOR_DIM_APPLY_i] != TENSOR3->size[TH_TENSOR_DIM_APPLY_i]) \
      THError("inconsistent tensor sizes"); \
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
    THError("invalid dimension"); \
  if( TENSOR1->nDimension != TENSOR2->nDimension ) \
    THError("inconsistent tensor sizes"); \
  for(TH_TENSOR_DIM_APPLY_i = 0; TH_TENSOR_DIM_APPLY_i < TENSOR1->nDimension; TH_TENSOR_DIM_APPLY_i++) \
  { \
    if(TH_TENSOR_DIM_APPLY_i == DIMENSION) \
      continue; \
    if(TENSOR1->size[TH_TENSOR_DIM_APPLY_i] != TENSOR2->size[TH_TENSOR_DIM_APPLY_i]) \
      THError("inconsistent tensor sizes"); \
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
      TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i]++; \
      TENSOR##_data += TENSOR->stride[TH_TENSOR_DIM_APPLY_i]; \
\
      if(TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i] == TENSOR->size[TH_TENSOR_DIM_APPLY_i]) \
      { \
        if(TH_TENSOR_DIM_APPLY_i == TENSOR->nDimension-1) \
        { \
          TH_TENSOR_DIM_APPLY_hasFinished = 1; \
          break; \
        } \
        else \
        { \
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
