#ifndef THZ_LAPACK_INC
#define THZ_LAPACK_INC

#include "THGeneral.h"
#include "TH/THLapack.h"

#define THZLapack_(NAME) TH_CONCAT_4(TH,NType,Lapack_,NAME)

#include "generic/THZLapack.h"
#include "THZGenerateAllTypes.h"


#endif