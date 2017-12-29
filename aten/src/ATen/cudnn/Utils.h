#pragma once

#include <ATen/ATen.h>
#include "THC/THC.h"
#include "cudnn-wrapper.h"
#include "Handles.h"

namespace at { namespace native {

inline void setCuDNNStreamToCurrent() {
  // TODO: Should getCurrentStream be a method on Context?
  CUDNN_CHECK(cudnnSetStream(getCudnnHandle(), THCState_getCurrentStream(globalContext().thc_state)));
}

// cuDNN has a buggy check for tensor being contiguous (that is, it does
// not ignore stride for dimension that is equal to 0).  This function
// makes tensors which have zero stride contiguous, by setting the
// strides to 1 as cuDNN likes.
inline Tensor contiguousIfZeroInStrides(const Tensor& t) {
  for (auto s : t.strides()) {
    if (s == 0) return t.contiguous();
  }
  return t;
}

}}
