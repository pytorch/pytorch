#ifndef THZ_TENSOR_INC
#define THZ_TENSOR_INC

#include "THZStorage.h"
#include "TH/THTensorApply.h"

#define THZTensor          TH_CONCAT_3(TH,NType,Tensor)
#define THZTensor_(NAME)   TH_CONCAT_4(TH,NType,Tensor_,NAME)

#define THZPartTensor TH_CONCAT_3(TH, Part, Tensor)
#define THZPartTensor_(NAME) TH_CONCAT_4(TH, Part, Tensor_, NAME)

/* basics */
#include "generic/THZTensor.h"
#include "THZGenerateAllTypes.h"

#include "generic/THZTensorCopy.h"
#include "THZGenerateAllTypes.h"

#include "TH/THTensorMacros.h"

/* random numbers */
#include "TH/THRandom.h"
#include "generic/THZTensorRandom.h"
#include "THZGenerateAllTypes.h"

/* maths */
#include "generic/THZTensorMath.h"
#include "THZGenerateAllTypes.h"

/* convolutions */
#include "generic/THZTensorConv.h"
#include "THZGenerateAllTypes.h"

/* lapack support */
#include "generic/THZTensorLapack.h"
#include "THZGenerateAllTypes.h"

#endif
