#include "THHalf.h"

/* Copyright 1993-2014 NVIDIA Corporation.  All rights reserved. */

at::Half TH_float2half(float f)
{
  at::Half h;
  TH_float2halfbits(&f, &h.x);
  return h;
}

TH_API float TH_half2float(at::Half h)
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
