#include <ATen/native/nested/NestedTensorMath.h>
#include  <ATen/native/nested/NestedTensorBinaryOps.h>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/nested/NestedTensorUtils.h>

#include <tuple>

namespace at {
namespace native {

DEFINE_DISPATCH(nested_dense_elementwise_stub);

std::pair<NestedTensorImpl*, NestedTensorImpl*>
get_elementwise_nested_tensor_impl(
    const Tensor& self,
    const Tensor& other,
    const std::string& op_name) {
  if (self.is_nested() && !(other.is_nested())) {
    TORCH_CHECK(
        false,
        "Expected both self and other to be nested, but got a nested self and non-nested other");
  } else if (!(self.is_nested()) && other.is_nested()) {
    TORCH_CHECK(
        false,
        "Expected both self and other to be nested, but got a non-nested self and nested other");
  } else if (!(self.is_nested()) || !(other.is_nested())) {
    TORCH_CHECK(
        false,
        "Expected both self and other to be nested, but got a non-nested self and non-nested other");
  }

  auto self_ptr = get_nested_tensor_impl(self);
  auto other_ptr = get_nested_tensor_impl(other);

  TORCH_CHECK(
      self.dim() == other.dim(),
      op_name,
      " does not support broadcasting when given a NestedTensor");
  TORCH_CHECK(
      at::equal(
          self_ptr->get_nested_size_tensor(),
          other_ptr->get_nested_size_tensor()),
      op_name,
      " does not support broadcasting when given a NestedTensor");
  TORCH_CHECK(
      at::equal(
          self_ptr->get_nested_stride_tensor(),
          other_ptr->get_nested_stride_tensor()),
      op_name,
      " requires strides to match when given NestedTensors");
  auto self_offsets = self_ptr->get_storage_offsets();
  auto other_offsets = other_ptr->get_storage_offsets();
  bool offsets_match = true;
  for (size_t i = 0; i < self_offsets.size(); i++) {
    offsets_match = offsets_match && (self_offsets[i] == other_offsets[i]);
  }
  TORCH_CHECK(
      offsets_match,
      op_name,
      " requires offsets to match when given NestedTensors");
  return std::make_pair(self_ptr, other_ptr);
}

template <typename Func>
Tensor NestedTensor_elementwise_Tensor(
    const Tensor& self,
    const Tensor& other,
    const std::string& op_name,
    Func f) {
  // self is a scalar
  if (!self.is_nested() && self.dim() == 0 && self.numel() == 1) {
    auto other_impl = get_nested_tensor_impl(other);
    return wrap_buffer(
      f(self, other_impl->get_unsafe_storage_as_tensor()),
      other_impl->get_nested_size_tensor().clone(),
      other_impl->get_nested_stride_tensor().clone(),
      other_impl->get_storage_offsets()
    );
  }
  // other is a scalar
  if (!other.is_nested() && other.dim() == 0 && other.numel() == 1) {
    auto self_impl = get_nested_tensor_impl(self);
    return wrap_buffer(
      f(self_impl->get_unsafe_storage_as_tensor(), other),
      self_impl->get_nested_size_tensor().clone(),
      self_impl->get_nested_stride_tensor().clone(),
      self_impl->get_storage_offsets()
    );
  }
  // special case when other is dense
  if (self.is_nested() && !other.is_nested()) {
    // check for the [B, *, D], [B, 1, D] esuhm case
    // this if statement is ugly and should be refactored
    auto self_ptr = get_nested_tensor_impl(self);
    if (self_ptr->dim() == 3 &&
        other.dim() == 3 &&
        self_ptr->size(0) == other.size(0) &&
        other.size(1) == 1 &&
        self_ptr->opt_size(2).has_value() &&
        self_ptr->opt_size(2).value() == other.size(2) &&
        self.is_cuda() &&
        other.is_cuda()) {
      if (!nested_tensor_impl_is_contiguous(self_ptr)) {
        self_ptr = get_nested_tensor_impl(self.contiguous());
      }
      const auto self_buffer = self_ptr->get_buffer();
      const auto self_sizes = self_ptr->get_nested_size_tensor();
      auto result_buffer = self_buffer.clone();
      auto result = wrap_buffer(result_buffer, self_sizes);
      if (op_name == "add") {
        nested_dense_elementwise_stub(self.device().type(), result, self, other, NESTED_DENSE_OP::ADD);
      } else if (op_name == "mul") {
        nested_dense_elementwise_stub(self.device().type(), result, self, other, NESTED_DENSE_OP::MUL);
      } else {
        TORCH_CHECK(false, "Unsupported nested dense elementwise op");
      }
      return result;
    }
    TORCH_CHECK(false, "Expected both self and other to be nested, but got a nested self and non-nested other.");
  }

  NestedTensorImpl* self_impl = nullptr;
  NestedTensorImpl* other_impl = nullptr;
  std::tie(self_impl, other_impl) =
      get_elementwise_nested_tensor_impl(self, other, op_name);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self_impl);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(other_impl);
  return wrap_buffer(
      f(self_impl->get_unsafe_storage_as_tensor(),
        other_impl->get_unsafe_storage_as_tensor()),
      self_impl->get_nested_size_tensor(),
      self_impl->get_nested_stride_tensor(),
      self_impl->get_storage_offsets());
}

Tensor NestedTensor_add_Tensor(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  return NestedTensor_elementwise_Tensor(
      self, other, "add", [alpha](const Tensor& b1, const Tensor& b2) {
        return at::add(b1, b2, alpha);
      });
}

Tensor NestedTensor_mul_Tensor(const Tensor& self, const Tensor& other) {
  return NestedTensor_elementwise_Tensor(
      self, other, "mul", [](const Tensor& b1, const Tensor& b2) {
        return at::mul(b1, b2);
      });
}

// Only usable on the C++ side; scalars are converted to tensors coming from Python.
Tensor NestedTensor_mul_Scalar(const Tensor& self, const Scalar& other) {
  return NestedTensor_mul_Tensor(self, wrapped_scalar_tensor(other));
}

Tensor NestedTensor_div_Tensor(const Tensor& self, const Tensor& other) {
  return NestedTensor_elementwise_Tensor(
      self, other, "div", [](const Tensor& b1, const Tensor& b2) {
        return at::div(b1, b2);
      });
}

// Only usable on the C++ side; scalars are converted to tensors coming from Python.
Tensor NestedTensor_div_Scalar(const Tensor& self, const Scalar& other) {
  return NestedTensor_div_Tensor(self, wrapped_scalar_tensor(other));
}

template <typename Func>
Tensor& NestedTensor_elementwise__Tensor(
    Tensor& self,
    const Tensor& other,
    const std::string& op_name,
    Func f) {
  // self is a scalar
  if (!self.is_nested() && self.dim() == 0 && self.numel() == 1) {
    auto other_impl = get_nested_tensor_impl(other);
    f(self, other_impl->get_buffer());
    return self;
  }
  // other is a scalar
  if (!other.is_nested() && other.dim() == 0 && other.numel() == 1) {
    auto self_impl = get_nested_tensor_impl(self);
    f(self_impl->get_buffer(), other);
    return self;
  }
  NestedTensorImpl* self_impl = nullptr;
  NestedTensorImpl* other_impl = nullptr;
  std::tie(self_impl, other_impl) =
      get_elementwise_nested_tensor_impl(self, other, op_name);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self_impl);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(other_impl);
  const auto& nt_self = *self_impl;
  const auto& nt_other = *other_impl;
  f(nt_self.get_buffer().view({-1}), nt_other.get_buffer().view({-1}));
  return self;
}

Tensor& NestedTensor_add__Tensor(
    Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  return NestedTensor_elementwise__Tensor(
      self, other, "add_", [alpha](const Tensor& b1, const Tensor& b2) {
        return b1.add_(b2, alpha);
      });
}

Tensor& NestedTensor_mul__Tensor(Tensor& self, const Tensor& other) {
  return NestedTensor_elementwise__Tensor(
      self, other, "mul_", [](const Tensor& b1, const Tensor& b2) {
        return b1.mul_(b2);
      });
}

// Only usable on the C++ side; scalars are converted to tensors coming from Python.
Tensor& NestedTensor_mul__Scalar(Tensor& self, const Scalar& other) {
  return NestedTensor_mul__Tensor(self, wrapped_scalar_tensor(other));
}

Tensor& fill_nested_(Tensor& self, const Scalar& value) {
  const auto& self_buf = get_nested_tensor_impl(self)->get_buffer();
  self_buf.fill_(value);
  return self;
}

Tensor& fill_nested_(Tensor& self, const Tensor& value) {
  const auto& self_buf = get_nested_tensor_impl(self)->get_buffer();
  self_buf.fill_(value);
  return self;
}

void _nested_op_dense_esuhm_cpu(Tensor& result, const Tensor& self, const Tensor& other, const NESTED_DENSE_OP& op) {
  // TODO: implement CPU kernel
  TORCH_CHECK(false);
}

REGISTER_ARCH_DISPATCH(nested_dense_elementwise_stub, DEFAULT, &_nested_op_dense_esuhm_cpu);
REGISTER_AVX512_DISPATCH(nested_dense_elementwise_stub, &_nested_op_dense_esuhm_cpu);
REGISTER_AVX2_DISPATCH(nested_dense_elementwise_stub, &_nested_op_dense_esuhm_cpu);
REGISTER_VSX_DISPATCH(nested_dense_elementwise_stub, &_nested_op_dense_esuhm_cpu);
REGISTER_ZVECTOR_DISPATCH(nested_dense_elementwise_stub, &_nested_op_dense_esuhm_cpu);

} // namespace native
} // namespace at
