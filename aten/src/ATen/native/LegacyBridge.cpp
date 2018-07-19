#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/SparseTensorRef.h>
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
  if (_has_native(self)) {
    return native_pow(self, exponent);
  } else {
    return th_pow(self, exponent);
  }
}

Tensor& zero_(Tensor& self) {
  if (_has_native(self)) {
    return native_zero_(self);
  } else {
    return th_zero_(self);
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
// We are in neither of those worlds.  Instead, we have a th_add function
// which has legacy implementations in the single dispatch world, BUT our
// actual add function needs to call s_native_add if the function *would have*
// utilized a sparse kernel that is natively implemented.
//
// th_add is "good old single dispatch" which internally handles the is_sparse()
// test and also handles broadcasting.  s_native_add works asymmetrically:
// it doesn't handle broadcasting at all, and it ASSUMES that the relevant
// argument is a sparse tensor.  Why the asymmetry?  It turns out it is not
// so easy to figure out if a kernel is implemented in THS; it's not as simple
// as testing if the first argument is sparse, because, e.g.,
// in add(Dense, Sparse), the sparse kernel is in the second argument.  So,
// the trampoline function is going to know about the overloads *anyway*; it
// might as well also handle is_sparse() and broadcasting while it's at it.
//
// Why not change TH to follow this new scheme?  We could... but since it's
// all going away when we finish porting the TH functions to ATen, we haven't
// done it.

Tensor& add_out(Tensor& result, const Tensor& self, const Tensor& other, Scalar alpha) {
  // See Note [Multiple dispatch to sparse]
  auto self_sparse = self.is_sparse();
  auto other_sparse = other.is_sparse();
  if (self_sparse && other_sparse) {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "add_out");
    return s_native_add_out(result, b_self, b_other, alpha);
  } else if (!self_sparse && other_sparse) {
    // TODO: Perhaps doing overload selection with SparseTensorRef is
    // confusing, and we should have given these overloads different names.
    // For now, we do it this way for consistency with the TH bindings
    // (not that it is terribly consistent anyway).
    return native_add_out(result, self, SparseTensorRef(other), alpha);
  } else {
    return th_add_out(result, self, other, alpha);
  }
}

// NB: You may be tempted to implement add and add_ just as calls to add_out, but
// calling the actual implementing function matters, because broadcast
// will be handled differently depending on if you call add_ or (a seemingly
// equivalent) add_out.  Arguably this mismatch in treatment is a bug,
// c.f., https://github.com/pytorch/pytorch/issues/8308 but fixing this
// bug would involve changing a lot of other places, so we leave it
// alone for now.

Tensor add(const Tensor& self, const Tensor& other, Scalar alpha) {
  // See Note [Multiple dispatch to sparse]
  auto self_sparse = self.is_sparse();
  auto other_sparse = other.is_sparse();
  if (self_sparse && other_sparse) {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "add");
    return s_native_add(b_self, b_other, alpha);
  } else if (!self_sparse && other_sparse) {
    return native_add(self, SparseTensorRef(other), alpha);
  } else {
    return th_add(self, other, alpha);
  }
}

Tensor& add_(Tensor& self, const Tensor& other, Scalar alpha) {
  // See Note [Multiple dispatch to sparse]
  auto self_sparse = self.is_sparse();
  auto other_sparse = other.is_sparse();
  if (self_sparse && other_sparse) {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "add_");
    return s_native_add_(self, b_other, alpha);
  } else if (!self_sparse && other_sparse) {
    return native_add_(self, SparseTensorRef(other), alpha);
  } else {
    return th_add_(self, other, alpha);
  }
}


Tensor& sub_out(Tensor& result, const Tensor& self, const Tensor& other, Scalar alpha) {
  if (_has_native(self)) {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "sub_out");
    return s_native_sub_out(result, b_self, b_other, alpha);
  } else {
    return th_sub_out(result, self, other, alpha);
  }
}

Tensor sub(const Tensor& self, const Tensor& other, Scalar alpha) {
  if (_has_native(self)) {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "sub");
    return s_native_sub(b_self, b_other, alpha);
  } else {
    return th_sub(self, other, alpha);
  }
}

Tensor& sub_(Tensor& self, const Tensor& other, Scalar alpha) {
  if (_has_native(self)) {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "sub_");
    return s_native_sub_(self, b_other, alpha);
  } else {
    return th_sub_(self, other, alpha);
  }
}


Tensor& mul_out(Tensor& result, const Tensor& self, const Tensor& other) {
  if (_has_native(self)) {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "mul_out");
    return s_native_mul_out(result, self, other);
  } else {
    return th_mul_out(result, self, other);
  }
}

Tensor mul(const Tensor& self, const Tensor& other) {
  if (_has_native(self)) {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "mul");
    return s_native_mul(self, other);
  } else {
    return th_mul(self, other);
  }
}

Tensor& mul_(Tensor& self, const Tensor& other) {
  if (_has_native(self)) {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "mul_");
    return s_native_mul_(self, b_other);
  } else {
    return th_mul_(self, other);
  }
}

Tensor& mul_out(Tensor& result, const Tensor& self, Scalar other) {
  if (_has_native(self)) {
    return native_mul_out(result, self, other);
  } else {
    return th_mul_out(result, self, other);
  }
}

Tensor mul(const Tensor& self, Scalar other) {
  if (_has_native(self)) {
    return native_mul(self, other);
  } else {
    return th_mul(self, other);
  }
}

Tensor& mul_(Tensor& self, Scalar other) {
  if (_has_native(self)) {
    return native_mul_(self, other);
  } else {
    return th_mul_(self, other);
  }
}


Tensor& div_out(Tensor& result, const Tensor& self, Scalar other) {
  if (_has_native(self)) {
    return native_div_out(result, self, other);
  } else {
    return th_div_out(result, self, other);
  }
}

Tensor div(const Tensor& self, Scalar other) {
  if (_has_native(self)) {
    return native_div(self, other);
  } else {
    return th_div(self, other);
  }
}

Tensor& div_(Tensor& self, Scalar other) {
  if (_has_native(self)) {
    return native_div_(self, other);
  } else {
    return th_div_(self, other);
  }
}

Tensor& addmm_out(Tensor& result, const Tensor& self, const Tensor& mat1, const Tensor& mat2, Scalar beta, Scalar alpha) {
  // See Note [Multiple dispatch to sparse]
  auto mat1_sparse = mat1.is_sparse();
  if (mat1_sparse) {
    Tensor b_self;
    std::tie(b_self) = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm_out");
    return s_native_addmm_out(result, b_self, mat1, mat2, beta, alpha);
  } else {
    return th_addmm_out(result, self, mat1, mat2, beta, alpha);
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
    return th_addmm(self, mat1, mat2, beta, alpha);
  }
}

Tensor& addmm_(Tensor& self, const Tensor& mat1, const Tensor& mat2, Scalar beta, Scalar alpha) {
  // See Note [Multiple dispatch to sparse]
  auto mat1_sparse = mat1.is_sparse();
  if (mat1_sparse) {
    // inplace is not broadcasting
    return s_native_addmm_(self, mat1, mat2, beta, alpha);
  } else {
    return th_addmm_(self, mat1, mat2, beta, alpha);
  }
}


Tensor tensor(const Type& dtype) {
  if (_type_has_native(dtype)) {
    return dtype.native_tensor();
  } else {
    return dtype.th_tensor();
  }
}

Tensor tensor(const Type& dtype, ArrayRef<int64_t> size) {
  if (_type_has_native(dtype)) {
    return dtype.native_tensor(size);
  } else {
    return dtype.th_tensor(size);
  }
}

Tensor sparse_coo_tensor(const Tensor& indices, const Tensor& values) {
  return values.type().toSparse().native_sparse_coo_tensor(indices, values);
}

Tensor sparse_coo_tensor(const Tensor& indices, const Tensor& values, ArrayRef<int64_t> size) {
  return values.type().toSparse().native_sparse_coo_tensor(indices, values, size);
}

Tensor _sparse_coo_tensor_unsafe(const Tensor& indices, const Tensor& values, ArrayRef<int64_t> size) {
  return values.type().toSparse()._native_sparse_coo_tensor_unsafe(indices, values, size);
}

int64_t get_device(const Tensor& self) {
  if (_has_native(self)) {
    return native_get_device(self);
  } else {
    return _th_get_device(self);
  }
}

}} // namespace at::native
