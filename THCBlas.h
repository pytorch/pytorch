#ifndef THC_BLAS_INC
#define THC_BLAS_INC

#include "THCGeneral.h"

#undef TH_API
#define TH_API THC_API
#define real float
#define Real Cuda
#define THBlas_(NAME) TH_CONCAT_4(TH,Real,Blas_,NAME)

#define TH_GENERIC_FILE "generic/THBlas.h"
#include "generic/THBlas.h"
#undef TH_GENERIC_FILE

#undef THBlas_
#undef real
#undef Real
#undef TH_API

#ifdef WIN32
# define TH_API THC_EXTERNC __declspec(dllimport)
#else
# define TH_API THC_EXTERNC
#endif

#endif
