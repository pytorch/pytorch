#include "ATen/Scalar.h"
#include <TH/TH.h>

namespace at {

template<> Half convert(double f) {
  float t = static_cast<float>(f);
  Half h;
  TH_float2halfbits(&t,&h.x);
  return h;
}
template<> double convert(Half f) {
  float t;
  TH_halfbits2float(&f.x,&t);
  return t;
}
template<> Half convert(int64_t f) {
  return convert<Half,double>(static_cast<double>(f));
}
template<> int64_t convert(Half f) {
  return static_cast<int64_t>(convert<double,Half>(f));
}


#ifdef AT_CUDA_ENABLED
template<> half convert(double d) {

#if CUDA_VERSION < 9000
  return half {convert<Half,double>(d).x};
#else
  __half_raw raw;
  raw.x = convert<Half,double>(d).x;
  return half {raw};
#endif
}
#endif

}
