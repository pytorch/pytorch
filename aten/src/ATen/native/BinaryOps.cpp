#include "ATen/native/BinaryOps.h"

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {

DEFINE_DISPATCH(add_stub);
DEFINE_DISPATCH(sub_stub);
DEFINE_DISPATCH(mul_stub);
DEFINE_DISPATCH(div_stub);

Tensor& add_out(Tensor& result, const Tensor& self, const Tensor& other, Scalar alpha) {
  if (other.is_sparse()) {
    if (!result.defined()) {
      result = at::empty({0}, self.options());
    }
    if (self.is_sparse()) {
      at::_sparse_add_out(result, self, other, alpha);
    } else {
      at::_sparse_dense_add_out(result, self, SparseTensorRef(other), alpha);
    }
    return result;
  } else if (self.is_sparse()) {
    AT_ERROR("add(sparse, dense) is not supported. Use add(dense, sparse) instead.");
  }
  auto iter = TensorIterator::binary_op(result, self, other);
  add_stub(iter->device_type(), *iter, alpha);
  result = iter->output();
  return result;
}

Tensor add(const Tensor& self, const Tensor& other, Scalar alpha) {
  Tensor result;
  return native::add_out(result, self, other, alpha);
}

Tensor& add_(Tensor& self, const Tensor& other, Scalar alpha) {
  return native::add_out(self, self, other, alpha);
}

Tensor& div_out(Tensor& result, const Tensor& self, const Tensor& other) {
  if (self.is_sparse()) {
    if (!result.defined()) {
      result = at::empty({0}, self.options());
    }
    if (other.dim() != 0) {
      AT_ERROR("div(): sparse division only supports division by a scalar ",
        "(got shape ", other.sizes(), " for argument 'other')");
    }
    return at::_sparse_div_zerodim_out(result, self, other);
  }
  auto iter = TensorIterator::binary_op(result, self, other);
  div_stub(iter->device_type(), *iter);
  result = iter->output();
  return result;
}

Tensor div(const Tensor& self, const Tensor& other) {
  Tensor result;
  return native::div_out(result, self, other);
}

Tensor& div_(Tensor& self, const Tensor& other) {
  return native::div_out(self, self, other);
}

Tensor& mul_out(Tensor& result, const Tensor& self, const Tensor& other) {
  if (self.is_sparse() || other.is_sparse()) {
    if (!result.defined()) {
      result = at::empty({0}, self.options());
    }
    return at::_sparse_mul_out(result, self, other);
  }
  auto iter = TensorIterator::binary_op(result, self, other);
  mul_stub(iter->device_type(), *iter);
  result = iter->output();
  return result;
}

Tensor mul(const Tensor& self, const Tensor& other) {
  Tensor result;
  return native::mul_out(result, self, other);
}

Tensor& mul_(Tensor& self, const Tensor& other) {
  return native::mul_out(self, self, other);
}

Tensor& sub_out(Tensor& result, const Tensor& self, const Tensor& other, Scalar alpha) {
  if (other.is_sparse()) {
    if (!result.defined()) {
      result = at::empty({0}, self.options());
    }
    if (!self.sizes().equals(other.sizes())) {
      AT_ERROR("sizes do not match");
    }
    if (self.is_sparse()) {
      at::_sparse_add_out(result, self, other, -alpha);
    } else {
      at::_sparse_dense_add_out(result, self, SparseTensorRef(other), -alpha);
    }
    return result;
  } else if (self.is_sparse()) {
    AT_ERROR("sub(sparse, dense) is not supported. Use sub(dense, sparse) instead.");
  }
  auto iter = TensorIterator::binary_op(result, self, other);
  sub_stub(iter->device_type(), *iter, alpha);
  result = iter->output();
  return result;
}

Tensor sub(const Tensor& self, const Tensor& other, Scalar alpha) {
  Tensor result;
  return native::sub_out(result, self, other, alpha);
}

Tensor& sub_(Tensor& self, const Tensor& other, Scalar alpha) {
  return native::sub_out(self, self, other, alpha);
}

Tensor rsub(const Tensor& self, const Tensor& other, Scalar alpha) {
  return native::sub(other, self, alpha);
}

// These are still needed because we don't have C++ conversions from number
// types (int, float, etc.) to Tensor (only to Scalar). They're not exposed
// to Python.

static Tensor scalar_tensor(Scalar scalar) {
  auto tensor = scalar_to_tensor(scalar);
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  return tensor;
}

Tensor add(const Tensor& self, Scalar other, Scalar alpha) {
  return native::add(self, scalar_tensor(other), alpha);
}

Tensor& add_(Tensor& self, Scalar other, Scalar alpha) {
  return native::add_(self, scalar_tensor(other), alpha);
}

Tensor div(const Tensor& self, Scalar other) {
  return native::div(self, scalar_tensor(other));
}

Tensor& div_(Tensor& self, Scalar other) {
  return native::div_(self, scalar_tensor(other));
}

Tensor mul(const Tensor& self, Scalar other) {
  return native::mul(self, scalar_tensor(other));
}

Tensor& mul_(Tensor& self, Scalar other) {
  return native::mul_(self, scalar_tensor(other));
}

Tensor sub(const Tensor& self, Scalar other, Scalar alpha) {
  return native::sub(self, scalar_tensor(other), alpha);
}

Tensor& sub_(Tensor& self, Scalar other, Scalar alpha) {
  return native::sub_(self, scalar_tensor(other), alpha);
}

Tensor rsub(const Tensor& self, Scalar other, Scalar alpha) {
  return native::rsub(self, scalar_tensor(other), alpha);
}

}
}  // namespace at
