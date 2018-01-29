#include "ATen/Config.h"

#include "ATen/Scalar.h"

#include <TH/TH.h>

#include "ATen/Tensor.h"
#include "ATen/Context.h"

#if AT_CUDA_ENABLED()
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

namespace at {
Tensor Scalar::toTensor() const {
  if (Tag::HAS_t == tag) {
    return Tensor(t);
  } else if (Tag::HAS_d == tag) {
    return CPU(kDouble).scalarTensor(*this);
  } else {
    assert(Tag::HAS_i == tag);
    return CPU(kLong).scalarTensor(*this);
  }
}
}
