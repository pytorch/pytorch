#ifndef THS_GENERIC_FILE
#define THS_GENERIC_FILE "tensors/generic/THSTensor.hpp"
#else

template<>
struct ths_tensor_traits<real> {
  using tensor_type = THSRealTensor;
};

#endif
