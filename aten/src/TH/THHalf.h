#ifndef TH_HALF_H
#define TH_HALF_H

#include <TH/THGeneral.h>

#ifdef __cplusplus
#include <ATen/TensorImpl.h>
#endif

#ifdef __cplusplus
#define THHalf at::Half
#else
typedef struct at_Half at_Half;
#define THHalf at_Half
#endif

TH_API void TH_float2halfbits(float*, unsigned short*);
TH_API void TH_halfbits2float(unsigned short*, float*);

TH_API THHalf TH_float2half(float);
TH_API float TH_half2float(THHalf);

#endif
