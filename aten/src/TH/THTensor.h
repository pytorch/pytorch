#ifndef TH_TENSOR_INC
#define TH_TENSOR_INC

#include <TH/THGeneral.h>

#define THTensor_(NAME)   TH_CONCAT_4(TH,Real,Tensor_,NAME)

/* basics */
#include <TH/generic/THTensor.h>
#include <TH/THGenerateByteType.h>

#endif
