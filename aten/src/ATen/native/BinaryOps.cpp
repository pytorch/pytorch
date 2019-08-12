#include <ATen/native/BinaryOps.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {

DEFINE_DISPATCH(add_stub);
DEFINE_DISPATCH(sub_stub);
DEFINE_DISPATCH(mul_stub);
DEFINE_DISPATCH(div_stub);
DEFINE_DISPATCH(atan2_stub);
DEFINE_DISPATCH(logical_xor_stub);

Tensor& add_out(Tensor& result, const Tensor& self, const Tensor& other, Scalar alpha) {
  if (other.is_sparse()) {
    if (self.is_sparse()) {
      at::_sparse_add_out(result, self, other, alpha);
    } else {
      at::_sparse_dense_add_out(result, self, other, alpha);
    }
    return result;
  } else if (self.is_sparse()) {
    AT_ERROR("add(sparse, dense) is not supported. Use add(dense, sparse) instead.");
  }
  auto iter = TensorIterator::binary_op(result, self, other,
    /*check_internal_overlap=*/true);
  add_stub(iter.device_type(), iter, alpha);
  return result;
}

Tensor add(const Tensor& self, const Tensor& other, Scalar alpha) {
  Tensor result;
  if (other.is_sparse()) {
    result = at::empty({0}, self.options());
    return native::add_out(result, self, other, alpha);
  }
  auto iter = TensorIterator::binary_op(result, self, other);
  add_stub(iter.device_type(), iter, alpha);
  return iter.output();
}

Tensor& add_(Tensor& self, const Tensor& other, Scalar alpha) {
  return native::add_out(self, self, other, alpha);
}

Tensor& div_out(Tensor& result, const Tensor& self, const Tensor& other) {
  if (self.is_sparse()) {
    if (other.dim() != 0) {
      AT_ERROR("div(): sparse division only supports division by a scalar ",
        "(got shape ", other.sizes(), " for argument 'other')");
    }
    return at::_sparse_div_zerodim_out(result, self, other);
  }
  auto iter = TensorIterator::binary_op(result, self, other,
    /*check_internal_overlap=*/true);
  div_stub(iter.device_type(), iter);
  return result;
}

Tensor div(const Tensor& self, const Tensor& other) {
  Tensor result;
  if (self.is_sparse()) {
    result = at::empty({0}, self.options());
    return native::div_out(result, self, other);
  }
  auto iter = TensorIterator::binary_op(result, self, other);
  div_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& div_(Tensor& self, const Tensor& other) {
  return native::div_out(self, self, other);
}

Tensor& mul_out(Tensor& result, const Tensor& self, const Tensor& other) {
  if (self.is_sparse() || other.is_sparse()) {
    return at::_sparse_mul_out(result, self, other);
  }
  auto iter = TensorIterator::binary_op(result, self, other,
    /*check_internal_overlap=*/true);
  mul_stub(iter.device_type(), iter);
  return result;
}

Tensor mul(const Tensor& self, const Tensor& other) {
  Tensor result;
  if (self.is_sparse() || other.is_sparse()) {
    result = at::empty({0}, self.options());
    return native::mul_out(result, self, other);
  }
  auto iter = TensorIterator::binary_op(result, self, other);
  mul_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& mul_(Tensor& self, const Tensor& other) {
  return native::mul_out(self, self, other);
}

// Basic checking for all sub functions.
static inline void sub_check(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(self.scalar_type() != kBool || other.scalar_type() != kBool,
              "Subtraction, the `-` operator, with two bool tensors is not supported. "
              "Use the `^` or `logical_xor()` operator instead.")
  TORCH_CHECK(self.scalar_type() != kBool && other.scalar_type() != kBool,
              "Subtraction, the `-` operator, with a bool tensor is not supported. "
              "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
}

Tensor& sub_out(Tensor& result, const Tensor& self, const Tensor& other, Scalar alpha) {
  sub_check(self, other);
  if (other.is_sparse()) {
    if (!self.sizes().equals(other.sizes())) {
      AT_ERROR("sizes do not match");
    }
    if (self.is_sparse()) {
      at::_sparse_add_out(result, self, other, -alpha);
    } else {
      at::_sparse_dense_add_out(result, self, other, -alpha);
    }
    return result;
  } else if (self.is_sparse()) {
    AT_ERROR("sub(sparse, dense) is not supported. Use sub(dense, sparse) instead.");
  }
  auto iter = TensorIterator::binary_op(result, self, other,
    /*check_internal_overlap=*/true);
  sub_stub(iter.device_type(), iter, alpha);
  return result;
}

Tensor sub(const Tensor& self, const Tensor& other, Scalar alpha) {
  sub_check(self, other);
  Tensor result;
  if (other.is_sparse()) {
    result = at::empty({0}, self.options());
    return native::sub_out(result, self, other, alpha);
  }
  auto iter = TensorIterator::binary_op(result, self, other);
  sub_stub(iter.device_type(), iter, alpha);
  return iter.output();
}

Tensor& sub_(Tensor& self, const Tensor& other, Scalar alpha) {
  return native::sub_out(self, self, other, alpha);
}

Tensor rsub(const Tensor& self, const Tensor& other, Scalar alpha) {
  return native::sub(other, self, alpha);
}

Tensor& atan2_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other);
  atan2_stub(iter.device_type(), iter);
  return result;
}

Tensor atan2(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty_like(self);
  return native::atan2_out(result, self, other);
}

Tensor& atan2_(Tensor& self, const Tensor& other) {
  return native::atan2_out(self, self, other);
}

// These are still needed because we don't have C++ conversions from number
// types (int, float, etc.) to Tensor (only to Scalar). They're not exposed
// to Python.

static Tensor wrapped_scalar_tensor(Scalar scalar) {
  auto tensor = scalar_to_tensor(scalar);
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  return tensor;
}

Tensor add(const Tensor& self, Scalar other, Scalar alpha) {
  return native::add(self, wrapped_scalar_tensor(other), alpha);
}

Tensor& add_(Tensor& self, Scalar other, Scalar alpha) {
  return native::add_(self, wrapped_scalar_tensor(other), alpha);
}

Tensor div(const Tensor& self, Scalar other) {
  return native::div(self, wrapped_scalar_tensor(other));
}

Tensor& div_(Tensor& self, Scalar other) {
  return native::div_(self, wrapped_scalar_tensor(other));
}

Tensor mul(const Tensor& self, Scalar other) {
  return native::mul(self, wrapped_scalar_tensor(other));
}

Tensor& mul_(Tensor& self, Scalar other) {
  return native::mul_(self, wrapped_scalar_tensor(other));
}

Tensor sub(const Tensor& self, Scalar other, Scalar alpha) {
  return native::sub(self, wrapped_scalar_tensor(other), alpha);
}

Tensor& sub_(Tensor& self, Scalar other, Scalar alpha) {
  return native::sub_(self, wrapped_scalar_tensor(other), alpha);
}

Tensor rsub(const Tensor& self, Scalar other, Scalar alpha) {
  return native::rsub(self, wrapped_scalar_tensor(other), alpha);
}

Tensor& logical_xor_out(Tensor& result, const Tensor& self, const Tensor& other) {
  TensorIterator iter;
  iter.dont_compute_common_dtype();
  iter.check_and_add_output(result);
  iter.add_input(self);
  iter.add_input(other);
  iter.build();
  logical_xor_stub(iter.device_type(), iter);
  return result;
}

Tensor logical_xor(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  at::logical_xor_out(result, self, other);
  return result;
}

Tensor& logical_xor_(Tensor& self, const Tensor& other) {
  return at::logical_xor_out(self, self, other);
}

}
}  // namespace at
