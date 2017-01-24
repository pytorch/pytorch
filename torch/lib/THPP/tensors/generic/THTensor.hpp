#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "tensors/generic/THTensor.hpp"
#else

template<>
struct th_tensor_traits<real> {
  using tensor_type = THRealTensor;
};

#endif
