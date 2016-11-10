#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "common/generic/_THTensor.h"
#else

template<>
struct th_traits<real> {
  using tensor_type = THRealTensor;
};

#endif
