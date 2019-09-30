#ifndef TH_TENSOR_INC
#define TH_TENSOR_INC

#include <TH/THStorageFunctions.h>
#include <TH/THTensorApply.h>

#define THTensor_(NAME)   TH_CONCAT_4(TH,Real,Tensor_,NAME)

/* basics */
#include <TH/generic/THTensor.h>
#include <TH/THGenerateAllTypes.h>

#include <TH/generic/THTensor.h>
#include <TH/THGenerateHalfType.h>

#include <TH/generic/THTensor.h>
#include <TH/THGenerateBoolType.h>

#include <TH/generic/THTensor.h>
#include <TH/THGenerateBFloat16Type.h>

/* random numbers */
#include <TH/generic/THTensorRandom.h>
#include <TH/THGenerateAllTypes.h>

#include <TH/generic/THTensorRandom.h>
#include <TH/THGenerateBoolType.h>

/* maths */
#include <TH/generic/THTensorMath.h>
#include <TH/THGenerateAllTypes.h>

#include <TH/generic/THTensorMath.h>
#include <TH/THGenerateBoolType.h>

#include <TH/generic/THTensorMath.h>
#include <TH/THGenerateHalfType.h>

#include <TH/generic/THTensorMath.h>
#include <TH/THGenerateBFloat16Type.h>

/* fill and zero*/
#include <TH/generic/THTensorFill.h>
#include <TH/THGenerateAllTypes.h>

#include <TH/generic/THTensorFill.h>
#include <TH/THGenerateHalfType.h>

#include <TH/generic/THTensorFill.h>
#include <TH/THGenerateBoolType.h>

#include <TH/generic/THTensorFill.h>
#include <TH/THGenerateBFloat16Type.h>

/* convolutions */
#include <TH/generic/THTensorConv.h>
#include <TH/THGenerateAllTypes.h>

/* lapack support */
#include <TH/generic/THTensorLapack.h>
#include <TH/THGenerateFloatTypes.h>
#endif
