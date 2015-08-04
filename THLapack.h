#ifndef TH_LAPACK_INC
#define TH_LAPACK_INC

#include "THGeneral.h"

#define THLapack_(NAME) TH_CONCAT_4(TH,Real,Lapack_,NAME)
#define THLapackCheck(fmt, func, info , ...)						\
if (info < 0) {														\
  THError("Lapack Error in %s : Illegal Argument %d", func, -info); \
} else if(info > 0) {												\
  THError(fmt, func, info, ##__VA_ARGS__);							\
}																	\



#include "generic/THLapack.h"
#include "THGenerateAllTypes.h"

#endif
