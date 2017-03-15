#ifndef THCS_TENSOR_INC
#define THCS_TENSOR_INC

#include <THC/THC.h>
#include <THS/THSTensor.h>

#include "THCSparse.h"

#define THCSTensor          TH_CONCAT_3(THCS,Real,Tensor)
#define THCSTensor_(NAME)   TH_CONCAT_4(THCS,Real,Tensor_,NAME)

// Using int for indices because that's what cuSparse uses...
#define THCIndexTensor          THCudaIntTensor
#define THCIndexTensor_(NAME)   THCudaIntTensor_ ## NAME
#define integer                 int

#include "generic/THCSTensor.h"
#include "THCSGenerateAllTypes.h"

#include "generic/THCSTensorMath.h"
#include "THCSGenerateAllTypes.h"

#endif
