#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/SparseTensorRef.h>
#include <ATen/ExpandUtils.h>

namespace at { namespace native {

namespace {
  static bool _type_has_native(const Type& dtype) {
    return dtype.is_sparse();
  }

  static bool _has_native(const Tensor& self) {
    return _type_has_native(self.type());
  }
}

// These native operations are not "really" native; they're actually just bridge
// functions that decide whether or not to call native sparse functions, or
// TH functions.  This file should be temporary; when all of TH gets ported, we
// can just use the native mechanism straight.

// TODO: Maybe the foo_ variants should call th_foo_

Tensor clone(const Tensor& self) {
  if (_has_native(self)) {
    return native_clone(self);
  } else {
    return _th_clone(self);
  }
}

Tensor& resize_as_(Tensor& self, const Tensor& the_template) {
  if (_has_native(self)) {
    return native_resize_as_(self, the_template);
  } else {
    return _th_resize_as_(self, the_template);
  }
}

Tensor& pow_out(Tensor& result, const Tensor& self, Scalar exponent) {
  if (_has_native(self)) {
    return native_pow_out(result, self, exponent);
  } else {
    return _th_pow_out(result, self, exponent);
  }
}

Tensor pow(const Tensor& self, Scalar exponent) {
  if (_has_native(self)) {
    return native_pow(self, exponent);
  } else {
    return _th_pow(self, exponent);
  }
}

Tensor& zero_(Tensor& self) {
  if (_has_native(self)) {
    return native_zero_(self);
  } else {
    return _th_zero_(self);
  }
}

// Note [Multiple dispatch to sparse]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// In an ideal world, we would use direct support for multiple dispatch to
// say that add(Dense, Dense) should dispatch to one function, while
// add(Dense, Sparse) should dispatch to another function.
//
// In a world where we only have single dispatch, we can single dispatch on
// the first function, and then do an is_sparse() test on the second argument
// to direct ourselves to the correct argument.
//
// We are in neither of those worlds.  Instead, we have a _th_addmm function
// which has legacy implementations in the single dispatch world, BUT our
// actual addmm function needs to call s_native_addmm if the function *would have*
// utilized a sparse kernel that is natively implemented.
//
// _th_addmm is "good old single dispatch" which internally handles the is_sparse()
// test and also handles broadcasting.  s_native_addmm works asymmetrically:
// it doesn't handle broadcasting at all, and it ASSUMES that the relevant
// argument is a sparse tensor.  Why the asymmetry?  It turns out it is not
// so easy to figure out if a kernel is implemented in THS; it's not as simple
// as testing if the first argument is sparse, because, e.g.,
// in addmm(Dense, Sparse), the sparse kernel is in the second argument.  So,
// the trampoline function is going to know about the overloads *anyway*; it
// might as well also handle is_sparse() and broadcasting while it's at it.
//
// Why not change TH to follow this new scheme?  We could... but since it's
// all going away when we finish porting the TH functions to ATen, we haven't
// done it.

// NB: You may be tempted to implement addmm and addmm_ just as calls to addmm_out, but
// calling the actual implementing function matters, because broadcast
// will be handled differently depending on if you call addmm_ or (a seemingly
// equivalent) add_out.  Arguably this mismatch in treatment is a bug,
// c.f., https://github.com/pytorch/pytorch/issues/8308 but fixing this
// bug would involve changing a lot of other places, so we leave it
// alone for now.

Tensor& addmm_out(Tensor& result, const Tensor& self, const Tensor& mat1, const Tensor& mat2, Scalar beta, Scalar alpha) {
  // See Note [Multiple dispatch to sparse]
  auto mat1_sparse = mat1.is_sparse();
  if (mat1_sparse) {
    Tensor b_self;
    std::tie(b_self) = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm_out");
    return s_native_addmm_out(result, b_self, mat1, mat2, beta, alpha);
  } else {
    return _th_addmm_out(result, self, mat1, mat2, beta, alpha);
  }
}

Tensor addmm(const Tensor& self, const Tensor& mat1, const Tensor& mat2, Scalar beta, Scalar alpha) {
  // See Note [Multiple dispatch to sparse]
  auto mat1_sparse = mat1.is_sparse();
  if (mat1_sparse) {
    Tensor b_self;
    std::tie(b_self) = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm");
    return s_native_addmm(b_self, mat1, mat2, beta, alpha);
  } else {
    return _th_addmm(self, mat1, mat2, beta, alpha);
  }
}

Tensor& addmm_(Tensor& self, const Tensor& mat1, const Tensor& mat2, Scalar beta, Scalar alpha) {
  // See Note [Multiple dispatch to sparse]
  auto mat1_sparse = mat1.is_sparse();
  if (mat1_sparse) {
    // inplace is not broadcasting
    return s_native_addmm_(self, mat1, mat2, beta, alpha);
  } else {
    return _th_addmm_(self, mat1, mat2, beta, alpha);
  }
}

}} // namespace at::native
