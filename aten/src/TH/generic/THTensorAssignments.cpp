#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensorAssignments.cpp"
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

void THTensor_(eye)(THTensor *r_, int64_t n, int64_t m)
{
  scalar_t *r__data;
  int64_t i, sz;

  THArgCheck(n > 0, 1, "invalid argument");

  if(m <= 0)
    m = n;

  THTensor_(resize2d)(r_, n, m);
  THTensor_(zero)(r_);

  i = 0;
  r__data = r_->data<scalar_t>();
  sz = THMin(THTensor_(size)(r_, 0), THTensor_(size)(r_, 1));
  for(i = 0; i < sz; i++)
    r__data[i*(r_->stride(0)+r_->stride(1))] = 1;
}

#endif
