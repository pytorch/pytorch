#ifndef THS_TENSOR_INC
#define THS_TENSOR_INC

#ifdef THS_EXPORTS
#define TH_EXPORTS
#endif

#include "TH.h"
#include <stdint.h>

#define THSTensor          TH_CONCAT_3(THS,Real,Tensor)
#define THSTensor_(NAME)   TH_CONCAT_4(THS,Real,Tensor_,NAME)

#include "generic/THSTensor.h"
#include "THSGenerateAllTypes.h"

#include "generic/THSTensorMath.h"
#include "THSGenerateAllTypes.h"

#endif
