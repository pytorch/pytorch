#ifndef TH_TENSOR_INC
#define TH_TENSOR_INC

#include "THStorageFunctions.h"
#include "THTensorApply.h"

#define THTensor_(NAME)   TH_CONCAT_4(TH,Real,Tensor_,NAME)

/* basics */
#include "generic/THTensor.h"
#include "THGenerateAllTypes.h"

#include "generic/THTensor.h"
#include "THGenerateHalfType.h"

#include "generic/THTensorCopy.h"
#include "THGenerateAllTypes.h"

#include "generic/THTensorCopy.h"
#include "THGenerateHalfType.h"

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
