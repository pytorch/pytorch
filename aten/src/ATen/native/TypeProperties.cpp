#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <type_traits>

namespace at { namespace native {

bool is_cuda(const Tensor& self) {
  return self.is_cuda();
}

bool is_distributed(const Tensor& self) {
  return false;
}

bool is_complex(const Tensor& self) {
  return at::isComplexType(self.scalar_type());
}

bool is_floating_point(const Tensor& self) {
  return at::isFloatingType(self.scalar_type());
}

bool is_signed(const Tensor &self) {
  return at::isSignedType(self.scalar_type());
}

bool is_sparse(const Tensor& self) {
  return self.is_sparse();
}

bool is_quantized(const Tensor& self) {
  return self.is_quantized();
}

// True if `self` and `from` have compatible tensor type so that `from`'s
// TensorImpl can be copied to `self`.
bool _has_compatible_shallow_copy_type(const Tensor& self, const Tensor& from) {
  return self.unsafeGetTensorImpl()->has_compatible_shallow_copy_type(
      from.type_set());
}

Tensor type_as(const Tensor& self, const Tensor& other) {
  return self.to(other.options());
}

static inline ScalarType promote_skip_undefined(ScalarType a, ScalarType b) {
  if (a == ScalarType::Undefined) {
    return b;
  }
  if (b == ScalarType::Undefined) {
    return a;
  }
  return promoteTypes(a, b);
}


static inline ScalarType combine_categories(ScalarType higher, ScalarType lower) {
  if (isFloatingType(higher)) {
    return higher;
  }
  if (higher == ScalarType::Bool || isFloatingType(lower)) {
    return promote_skip_undefined(higher, lower);
  }
  if (higher != ScalarType::Undefined) {
      return higher;
  }
  return lower;
}

ScalarType result_type(TensorList tensors) {
  auto dimResult = ScalarType::Undefined;
  auto zeroResult = ScalarType::Undefined;
  auto wrappedResult = ScalarType::Undefined;
  for (Tensor tensor : tensors) {
    if (!tensor.defined()) {
      continue;
    }
    ScalarType current = tensor.scalar_type();
    if (tensor.unsafeGetTensorImpl()->is_wrapped_number() && isFloatingType(current)) {
      current = typeMetaToScalarType(at::get_default_dtype());
    }
    if ( tensor.dim() > 0 ) {
      dimResult = promote_skip_undefined(dimResult, current);
    } else if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
      wrappedResult = promote_skip_undefined(wrappedResult, current);
    } else {
      zeroResult = promote_skip_undefined(zeroResult, current);
    }
  }
  return combine_categories(dimResult, combine_categories(zeroResult, wrappedResult));
}

ScalarType result_type(const Tensor &tensor, const Tensor &other) {
  std::vector<Tensor> tensors({tensor, other});
  return native::result_type(tensors);
}

ScalarType result_type(const Tensor &tensor, const Scalar other) {
  auto tensor2 = scalar_to_tensor(other);
  tensor2.unsafeGetTensorImpl()->set_wrapped_number(true);
  std::vector<Tensor> tensors({tensor, tensor2});
  return native::result_type(tensors);
}

ScalarType result_type(const Scalar scalar, const Tensor &tensor) {
  return at::result_type(tensor, scalar);
}

ScalarType result_type(const Scalar scalar1, const Scalar scalar2) {
  auto tensor1 = scalar_to_tensor(scalar1);
  tensor1.unsafeGetTensorImpl()->set_wrapped_number(true);
  return at::result_type(tensor1, scalar2);
}

ScalarType promote_types(ScalarType type1, ScalarType type2) {
  return promoteTypes(type1, type2);
}

}} // namespace at::native
