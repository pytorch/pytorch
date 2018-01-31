#ifndef THZ_BLAS_INC
#define THZ_BLAS_INC

#include "THZGeneral.h"

#define THZBlas_(NAME) TH_CONCAT_4(TH,NType,Blas_,NAME)

#include "generic/THZBlas.h"
#include "THZGenerateAllTypes.h"

#endif