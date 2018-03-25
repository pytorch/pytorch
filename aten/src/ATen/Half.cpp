#include "ATen/Half.h"

#include <TH/TH.h>

#include "ATen/Tensor.h"
#include "ATen/Context.h"

namespace at {

template<> AT_API Half convert(float f) {
  Half t;
  TH_float2halfbits(&f,&t.x);
  return t;
}
template<> AT_API float convert(Half f) {
  float t;
  TH_halfbits2float(&f.x,&t);
  return t;
}

template<> AT_API Half convert(double f) {
  return convert<Half, float>(f);
}
template<> AT_API double convert(Half f) {
  return convert<float, Half>(f);
}

template<> AT_API Half convert(int64_t f) {
  return convert<Half,double>(static_cast<double>(f));
}
template<> AT_API int64_t convert(Half f) {
  return static_cast<int64_t>(convert<double,Half>(f));
}

template<> bool overflows<Half, double>(double f) {
  return f > 65504 || f < -65504;
}
template<> bool overflows<Half, int64_t>(int64_t f) {
  return f > 65504 || f < -65504;
}
} // namespace at
