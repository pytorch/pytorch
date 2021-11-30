#ifndef THC_TENSOR_INC
#define THC_TENSOR_INC

#include <TH/THTensor.h>
#include <THC/THCGeneral.h>

#define THCTensor_(NAME)   TH_CONCAT_4(TH,CReal,Tensor_,NAME)

#define THC_DESC_BUFF_LEN 64

typedef struct TORCH_CUDA_CU_API THCDescBuff {
  char str[THC_DESC_BUFF_LEN];
} THCDescBuff;

#include <THC/generic/THCTensor.h>
#include <THC/THCGenerateByteType.h>

#endif
