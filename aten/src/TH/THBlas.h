#ifndef TH_BLAS_INC
#define TH_BLAS_INC

#include <TH/THGeneral.h>

#define THBlas_(NAME) TH_CONCAT_4(TH, Real, Blas_, NAME)

// clang-format off
#include <TH/generic/THBlas.h>
#include <TH/THGenerateAllTypes.h>

#include <TH/generic/THBlas.h>
#include <TH/THGenerateBFloat16Type.h>

#include <TH/generic/THBlas.h>
#include <TH/THGenerateHalfType.h>
// clang-format on

#endif
