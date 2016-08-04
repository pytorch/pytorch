#ifndef TH_TENSOR_INC
#define TH_TENSOR_INC

#include "THStorage.h"
#include "THTensorApply.h"

#define THTensor          TH_CONCAT_3(TH,Real,Tensor)
#define THTensor_(NAME)   TH_CONCAT_4(TH,Real,Tensor_,NAME)

#define TH_DESC_BUFF_LEN 64
typedef struct {
    char str[TH_DESC_BUFF_LEN];
} THDescBuff;

/* basics */
#include "generic/THTensor.h"
#include "THGenerateAllTypes.h"

#include "generic/THTensorCopy.h"
#include "THGenerateAllTypes.h"

#include "THTensorMacros.h"

/* random numbers */
#include "THRandom.h"
#include "generic/THTensorRandom.h"
#include "THGenerateAllTypes.h"

/* maths */
#include "generic/THTensorMath.h"
#include "THGenerateAllTypes.h"

/* convolutions */
#include "generic/THTensorConv.h"
#include "THGenerateAllTypes.h"

/* lapack support */
#include "generic/THTensorLapack.h"
#include "THGenerateFloatTypes.h"

#endif
