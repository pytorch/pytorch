#ifndef TH_LAPACK_INC
#define TH_LAPACK_INC

#include "THGeneral.h"

#define THLapack_(NAME) TH_CONCAT_4(TH,Real,Lapack_,NAME)

#include "generic/THLapack.h"
#include "THGenerateAllTypes.h"

#endif
