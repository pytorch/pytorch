#include <ATen/ATen.h>
#include <ATen/SparseTensorRef.h>

namespace at { namespace native {

namespace {
  // NB: Even though some of the functions we have ported are CUDA
  // friendly, flipping the switch between native and non-native is
  // an all or nothing affair, because the internal representation
  // is different
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

Tensor& pow_out(Tensor& result, const Tensor& self, Scalar exponent) {
  if (_has_native(self)) {
    return native_pow_out(result, self, exponent);
  } else {
    return th_pow_out(result, self, exponent);
  }
}

Tensor pow(const Tensor& self, Scalar exponent) {
  Tensor r = self.type().tensor();
  native::pow_out(r, self, exponent);
  return r;
}

Tensor& zero_(Tensor& self) {
  if (_has_native(self)) {
    return native_zero_(self);
  } else {
    return th_zero_(self);
  }
}

Tensor& add_out(Tensor& result, const Tensor& self, const Tensor& other, Scalar alpha) {
  if (_has_native(self)) {
    return native_add_out(result, self, other, alpha);
  } else {
    return th_add_out(result, self, other, alpha);
  }
}

Tensor add(const Tensor& self, const Tensor& other, Scalar alpha) {
  Tensor r = self.type().tensor();
  native::add_out(r, self, other, alpha);
  return r;
}

Tensor& add_out(Tensor& result, const Tensor& self, SparseTensorRef other, Scalar alpha) {
  if (_has_native(self)) {
    return native_add_out(result, self, other, alpha);
  } else {
    return th_add_out(result, self, other, alpha);
  }
}

Tensor add(const Tensor& self, SparseTensorRef other, Scalar alpha) {
  Tensor r = self.type().tensor();
  native::add_out(r, self, other, alpha);
  return r;
}

}} // namespace at::native
