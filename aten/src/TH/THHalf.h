#ifndef TH_HALF_H
#define TH_HALF_H

#include <TH/THGeneral.h>
#include <ATen/core/Half.h>

TH_API void TH_float2halfbits(float*, unsigned short*);
TH_API void TH_halfbits2float(unsigned short*, float*);

TH_API at::Half TH_float2half(float);
TH_API float TH_half2float(at::Half);

#endif
