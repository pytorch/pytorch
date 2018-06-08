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

// TODO: Maybe the foo_ variants should call th_foo_

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

Tensor& add_(Tensor& self, const Tensor& other, Scalar alpha) {
  return native::add_out(self, self, other, alpha);
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

Tensor& add_(Tensor& self, SparseTensorRef other, Scalar alpha) {
  return native::add_out(self, self, other, alpha);
}


Tensor& sub_out(Tensor& result, const Tensor& self, const Tensor& other, Scalar alpha) {
  if (_has_native(self)) {
    return native_sub_out(result, self, other, alpha);
  } else {
    return th_sub_out(result, self, other, alpha);
  }
}

Tensor sub(const Tensor& self, const Tensor& other, Scalar alpha) {
  Tensor r = self.type().tensor();
  return native::sub_out(r, self, other, alpha);
  return r;
}

Tensor& sub_(Tensor& self, const Tensor& other, Scalar alpha) {
  return native::sub_out(self, self, other, alpha);
}


Tensor& mul_out(Tensor& result, const Tensor& self, const Tensor& other) {
  if (_has_native(self)) {
    return native_mul_out(result, self, other);
  } else {
    return th_mul_out(result, self, other);
  }
}

Tensor mul(const Tensor& self, const Tensor& other) {
  Tensor r = self.type().tensor();
  return native::mul_out(r, self, other);
  return r;
}

Tensor& mul_(Tensor& self, const Tensor& other) {
  return native::mul_out(self, self, other);
}

Tensor& mul_out(Tensor& result, const Tensor& self, Scalar other) {
  if (_has_native(self)) {
    return native_mul_out(result, self, other);
  } else {
    return th_mul_out(result, self, other);
  }
}

Tensor mul(const Tensor& self, Scalar other) {
  Tensor r = self.type().tensor();
  return native::mul_out(r, self, other);
  return r;
}

Tensor& mul_(Tensor& self, Scalar other) {
  return native::mul_out(self, self, other);
}


Tensor& div_out(Tensor& result, const Tensor& self, Scalar other) {
  if (_has_native(self)) {
    return native_div_out(result, self, other);
  } else {
    return th_div_out(result, self, other);
  }
}

Tensor div(const Tensor& self, Scalar other) {
  Tensor r = self.type().tensor();
  return native::div_out(r, self, other);
  return r;
}

Tensor& div_(Tensor& self, Scalar other) {
  return native::div_out(self, self, other);
}

Tensor& addmm_out(Tensor& result, const Tensor& self, SparseTensorRef mat1, const Tensor& mat2, Scalar beta, Scalar alpha) {
  if (_has_native(self)) {
    return native_addmm_out(result, self, mat1, mat2, beta, alpha);
  } else {
    return th_addmm_out(result, self, mat1, mat2, beta, alpha);
  }
}

Tensor addmm(const Tensor& self, SparseTensorRef mat1, const Tensor& mat2, Scalar beta, Scalar alpha) {
  Tensor r = self.type().tensor();
  return native::addmm_out(r, self, mat1, mat2, beta, alpha);
  return r;
}

}} // namespace at::native
