#include <ATen/ATen.h>

namespace at { namespace native {

namespace {
  static bool _has_native(const Tensor& self) {
    return self.is_sparse() && !self.is_cuda();
  }
}

// These native operations are not "really" native; they're actually just bridge
// functions that decide whether or not to call native sparse functions, or
// TH functions.  This file should be temporary; when all of TH gets ported, we
// can just use the native mechanism straight.

Tensor norm(const Tensor & self, Scalar p) {
  if (_has_native(self)) {
    return native_norm(self, p);
  } else {
    return th_norm(self, p);
  }
}

Tensor clone(const Tensor& self) {
  if (_has_native(self)) {
    return native_clone(self);
  } else {
    return th_clone(self);
  }
}

Tensor& resize_as_(Tensor& self, const Tensor& the_template) {
  if (_has_native(self)) {
    return native_resize_as_(self, the_template);
  } else {
    return th_resize_as_(self, the_template);
  }
}

}} // namespace at::native
