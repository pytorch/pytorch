#ifndef THCS_TENSOR_INC
#define THCS_TENSOR_INC

#include <THC/THC.h>
#include <THS/THSTensor.h>

#include "THCSparse.h"

#define THCSTensor          TH_CONCAT_3(THCS,Real,Tensor)
#define THCSTensor_(NAME)   TH_CONCAT_4(THCS,Real,Tensor_,NAME)

#define THCIndexTensor          THCudaLongTensor
#define THCIndexTensor_(NAME)   THCudaLongTensor_ ## NAME
#define integer                 long

#include "generic/THCSTensor.h"
#include "THCSGenerateAllTypes.h"

#include "generic/THCSTensorMath.h"
#include "THCSGenerateAllTypes.h"

#endif
