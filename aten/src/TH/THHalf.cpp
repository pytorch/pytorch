#include "THHalf.h"
#include <ATen/core/Half.h>

/* Copyright 1993-2014 NVIDIA Corporation.  All rights reserved. */

THHalf TH_float2half(float f)
{
  THHalf h;
  TH_float2halfbits(&f, &h.x);
  return h;
}

TH_API float TH_half2float(THHalf h)
{
  float f;
  TH_halfbits2float(&h.x, &f);
  return f;
}


void TH_halfbits2float(unsigned short* src, float* res)
{
  *res = at::detail::halfbits2float(*src);
}


void TH_float2halfbits(float* src, unsigned short* dest)
{
  *dest = at::detail::float2halfbits(*src);
}
