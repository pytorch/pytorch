#include <ATen/ATen.h>

namespace at { namespace native {

// These native operations are not "really" native; they're actually just bridge
// functions that decide whether or not to call native sparse functions, or
// TH functions.  This file should be temporary; when all of TH gets ported, we
// can just use the native mechanism straight.

Tensor norm(const Tensor & self, Scalar p) {
  if (self.is_sparse() && !self.is_cuda()) {
    return native_norm(self, p);
  } else {
    return th_norm(self, p);
  }
}

}} // namespace at::native
