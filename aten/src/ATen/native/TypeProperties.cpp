#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/TypeProperties.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_has_compatible_shallow_copy_type_native.h>
#include <ATen/ops/_is_zerotensor_native.h>
#include <ATen/ops/can_cast_native.h>
#include <ATen/ops/is_complex_native.h>
#include <ATen/ops/is_conj_native.h>
#include <ATen/ops/is_distributed_native.h>
#include <ATen/ops/is_floating_point_native.h>
#include <ATen/ops/is_inference_native.h>
#include <ATen/ops/is_neg_native.h>
#include <ATen/ops/is_signed_native.h>
#include <ATen/ops/promote_types_native.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/result_type_native.h>
#include <ATen/ops/type_as_native.h>
#endif

namespace at::native {

static bool is_cuda(const Tensor& self) {
  return self.is_cuda();
}

bool is_distributed(const Tensor& self) {
  return false;
}

bool is_complex(const Tensor& self) {
  return self.is_complex();
}

bool is_floating_point(const Tensor& self) {
  return self.is_floating_point();
}

bool is_inference(const Tensor& self) {
  return self.is_inference();
}

bool is_signed(const Tensor &self) {
  return self.is_signed();
}

bool _is_zerotensor(const Tensor& self) {
  return self._is_zerotensor();
}

bool is_conj(const Tensor& self) {
  return self.is_conj();
}

bool is_neg(const Tensor& self) {
  return self.is_neg();
}

static bool is_sparse(const Tensor& self) {
  return self.is_sparse();
}

static bool is_sparse_csr(const Tensor& self) {
  return self.is_sparse_csr();
}

static bool is_quantized(const Tensor& self) {
  return self.is_quantized();
}

// True if `self` and `from` have compatible tensor type so that `from`'s
// TensorImpl can be copied to `self`.
bool _has_compatible_shallow_copy_type(const Tensor& self, const Tensor& from) {
  return self.unsafeGetTensorImpl()->has_compatible_shallow_copy_type(
      from.key_set());
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
  // NOLINTNEXTLINE(bugprone-branch-clone)
  if(isComplexType(higher)) {
    return higher;
  } else if (isComplexType(lower)) {
    // preserve value type of higher if it is floating type.
    if (isFloatingType(higher)) {
      return toComplexType(higher);
    }
    // in case of integral input
    // lower complex takes precedence.
    return lower;
  } else if (isFloatingType(higher)) {
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

ResultTypeState update_result_type_state(const Tensor& tensor, const ResultTypeState& in_state) {
  if (!tensor.defined()) {
    return in_state;
  }
  ResultTypeState new_state = in_state;
  ScalarType current = tensor.scalar_type();
  if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    if(isComplexType(current)) {
      current = typeMetaToScalarType(at::get_default_complex_dtype());
    }
    else if(isFloatingType(current)) {
      current = typeMetaToScalarType(at::get_default_dtype());
    }
  }
  if ( tensor.dim() > 0 ) {
    new_state.dimResult = promote_skip_undefined(in_state.dimResult, current);
  } else if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    new_state.wrappedResult = promote_skip_undefined(in_state.wrappedResult, current);
  } else {
    new_state.zeroResult = promote_skip_undefined(in_state.zeroResult, current);
  }
  return new_state;
}

ResultTypeState update_result_type_state(const Scalar& scalar, const ResultTypeState& in_state) {
  ResultTypeState new_state = in_state;
  ScalarType current = scalar.type();
  if (isComplexType(current)) {
    current = typeMetaToScalarType(at::get_default_complex_dtype());
  } else if (isFloatingType(current)) {
    current = typeMetaToScalarType(at::get_default_dtype());
  }
  new_state.wrappedResult = promote_skip_undefined(in_state.wrappedResult, current);
  return new_state;
}

ScalarType result_type(const ResultTypeState& in_state) {
  return combine_categories(in_state.dimResult, combine_categories(in_state.zeroResult, in_state.wrappedResult));
}

ScalarType result_type(ITensorListRef tensors) {
  ResultTypeState state = {};
  for (const Tensor& tensor : tensors) {
    state = update_result_type_state(tensor, state);
  }
  return result_type(state);
}

ScalarType result_type(const Tensor &tensor, const Tensor &other) {
  ResultTypeState state = {};
  state = update_result_type_state(tensor, state);
  state = update_result_type_state(other, state);
  return result_type(state);
}

ScalarType result_type(const Tensor &tensor, const Scalar& other) {
  ResultTypeState state = {};
  state = update_result_type_state(tensor, state);
  state = update_result_type_state(other, state);
  return result_type(state);
}

ScalarType result_type(const Scalar& scalar, const Tensor &tensor) {
  return at::result_type(tensor, scalar);
}

ScalarType result_type(const Scalar& scalar1, const Scalar& scalar2) {
  ResultTypeState state = {};
  state = update_result_type_state(scalar1, state);
  state = update_result_type_state(scalar2, state);
  return result_type(state);
}

bool can_cast(const at::ScalarType from_, const at::ScalarType to) {
  return at::canCast(from_, to);
}

ScalarType promote_types(ScalarType type1, ScalarType type2) {
  ScalarType ret = promoteTypes(type1, type2);
  TORCH_CHECK(ret != ScalarType::Undefined, "Promotion from ", type1, " and ", type2, " is unsupported.");
  return ret;
}

} // namespace at::native
