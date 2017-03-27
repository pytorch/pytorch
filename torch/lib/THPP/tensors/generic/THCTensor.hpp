#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "tensors/generic/THCTensor.hpp"
#else

template<>
struct thc_tensor_traits<real> {
  using tensor_type = THCRealTensor;
};

#endif
