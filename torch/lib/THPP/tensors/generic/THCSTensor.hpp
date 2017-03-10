#ifndef THCS_GENERIC_FILE
#define THCS_GENERIC_FILE "tensors/generic/THCSTensor.hpp"
#else

template<>
struct thcs_tensor_traits<real> {
  using tensor_type = THCSRealTensor;
};

#endif
