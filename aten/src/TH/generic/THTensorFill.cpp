#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensorFill.cpp"
#else

#include <TH/generic/THTensorApply.hpp>

void THTensor_(fill)(THTensor *r_, scalar_t value)
{
  if (THTensor_(isContiguous)(r_) || THTensor_(isTransposed)(r_)) {
    TH_TENSOR_APPLY_CONTIG(scalar_t, r_, THVector_(fill)(r__data, value, r__len););
  } else {
    TH_TENSOR_APPLY(scalar_t, r_,
      if (r__stride == 1) {
        THVector_(fill)(r__data, value, r__size);
        r__i = r__size;
        r__data += r__stride * r__size;
        break;
      } else {
        *r__data = value;
      }
      );
  }
}

void THTensor_(zero)(THTensor *r_)
{
  THTensor_(fill)(r_, 0);
}

#endif
