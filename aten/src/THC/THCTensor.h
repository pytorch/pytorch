#ifndef THC_TENSOR_INC
#define THC_TENSOR_INC

#include "THTensor.h"
#include "THCStorage.h"
#include "THCGeneral.h"

#define THCTensor          TH_CONCAT_3(TH,CReal,Tensor)
#define THCTensor_(NAME)   TH_CONCAT_4(TH,CReal,Tensor_,NAME)

#define THC_DESC_BUFF_LEN 64

typedef struct THC_CLASS THCDescBuff
{
    char str[THC_DESC_BUFF_LEN];
} THCDescBuff;

#include "generic/THCTensor.h"
#include "THCGenerateAllTypes.h"

#endif
