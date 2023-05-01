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
REGISTER_NO_CPU_DISPATCH(nested_dense_elementwise_stub);

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
          self_ptr->get_nested_sizes(),
          other_ptr->get_nested_sizes()),
      op_name,
      " does not support broadcasting when given a NestedTensor");
  TORCH_CHECK(
      at::equal(
          self_ptr->get_nested_strides(),
          other_ptr->get_nested_strides()),
      op_name,
      " requires strides to match when given NestedTensors");
  const auto self_offsets = self_ptr->get_storage_offsets();
  int64_t *self_offsets_ptr = self_offsets.data_ptr<int64_t>();
  int64_t *other_offsets_ptr = other_ptr->get_storage_offsets().data_ptr<int64_t>();
  bool offsets_match = true;
  for (auto i = 0; i < self_offsets.size(0); i++) {
    offsets_match = offsets_match && (self_offsets_ptr[i] == other_offsets_ptr[i]);
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
    bool supports_striding,
    Func f) {
  Tensor self_contiguous = self;
  Tensor other_contiguous = other;
  // self is a scalar
  if (!self.is_nested() && self.dim() == 0 && self.numel() == 1) {
    auto other_impl = get_nested_tensor_impl(other);
    return wrap_buffer(
      f(self, other_impl->get_unsafe_storage_as_tensor()),
      other_impl->get_nested_sizes().clone(),
      other_impl->get_nested_strides().clone(),
      other_impl->get_storage_offsets()
    );
  }
  // other is a scalar
  if (!other.is_nested() && other.dim() == 0 && other.numel() == 1) {
    auto self_impl = get_nested_tensor_impl(self);
    return wrap_buffer(
      f(self_impl->get_unsafe_storage_as_tensor(), other),
      self_impl->get_nested_sizes().clone(),
      self_impl->get_nested_strides().clone(),
      self_impl->get_storage_offsets()
    );
  }
  // special case when other is dense (CUDA only for now)
  if (self.is_nested() && !other.is_nested() && self.is_cuda() && other.is_cuda()) {
    auto self_ptr = get_nested_tensor_impl(self);
    auto other_ = other;
    // check for the [B, *, D], [B, 1, D] case -> use custom kernel
    // TODO: this if statement is ugly and hopefully we will remove this in the near future
    bool is_broadcastable_3d = (
        self_ptr->dim() == 3 &&
        other.dim() == 3 &&
        self_ptr->size(0) == other.size(0) &&
        other.size(1) == 1 &&
        self_ptr->opt_size(2).has_value() &&
        self_ptr->opt_size(2).value() == other.size(2));
    // check for the [B, *], [B, 1] case -> treat as 3D with [B, *, 1], [B, 1, 1]
    bool is_broadcastable_2d = (
        self_ptr->dim() == 2 &&
        other.dim() == 2 &&
        self_ptr->size(0) == other.size(0) &&
        other.size(1) == 1);
    if(is_broadcastable_2d) {
        other_ = other.unsqueeze(-1);
        is_broadcastable_3d = true;
    }

    if (is_broadcastable_3d) {
      self_contiguous = self.contiguous();
      self_ptr = get_nested_tensor_impl(self_contiguous);
      const auto self_buffer = self_ptr->get_buffer();
      const auto self_sizes = self_ptr->get_nested_sizes();
      auto result_buffer = at::empty_like(self_buffer);
      auto result = wrap_buffer(result_buffer, self_sizes);
      if (op_name == "add") {
        nested_dense_elementwise_stub(self.device().type(), result, self, other_, NESTED_DENSE_OP::ADD);
      } else if (op_name == "mul") {
        nested_dense_elementwise_stub(self.device().type(), result, self, other_, NESTED_DENSE_OP::MUL);
      } else {
        TORCH_CHECK(false, "Unsupported nested dense elementwise op: ", op_name, ".");
      }
      return result;
    }
    TORCH_CHECK(
        false,
        "Expected both self and other to be nested, but got a nested self and non-nested other for op: ",
        op_name,
        ".");
  }

  NestedTensorImpl* self_impl = nullptr;
  NestedTensorImpl* other_impl = nullptr;

  self_contiguous = supports_striding ? self.contiguous() : self;
  other_contiguous = supports_striding ? other.contiguous() : other;

  std::tie(self_impl, other_impl) =
      get_elementwise_nested_tensor_impl(self_contiguous, other_contiguous, op_name);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self_impl);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(other_impl);
  return wrap_buffer(
      f(self_impl->get_unsafe_storage_as_tensor(),
        other_impl->get_unsafe_storage_as_tensor()),
      self_impl->get_nested_sizes(),
      self_impl->get_nested_strides(),
      self_impl->get_storage_offsets());
}

Tensor NestedTensor_add_Tensor(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  return NestedTensor_elementwise_Tensor(
      self, other, "add", true /* supports_striding*/, [alpha](const Tensor& b1, const Tensor& b2) {
        return at::add(b1, b2, alpha);
      });
}

Tensor NestedTensor_sub_Tensor(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  return NestedTensor_elementwise_Tensor(
      self, other, "sub", true /* supports_striding*/, [alpha](const Tensor& b1, const Tensor& b2) {
        return at::sub(b1, b2, alpha);
      });
}

Tensor NestedTensor_mul_Tensor(const Tensor& self, const Tensor& other) {
  return NestedTensor_elementwise_Tensor(
      self, other, "mul", false /* supports_striding*/, [](const Tensor& b1, const Tensor& b2) {
        return at::mul(b1, b2);
      });
}

// Only usable on the C++ side; scalars are converted to tensors coming from Python.
Tensor NestedTensor_mul_Scalar(const Tensor& self, const Scalar& other) {
  return NestedTensor_mul_Tensor(self, wrapped_scalar_tensor(other));
}

Tensor NestedTensor_div_Tensor(const Tensor& self, const Tensor& other) {
  return NestedTensor_elementwise_Tensor(
      self, other, "div", false /* supports_striding*/, [](const Tensor& b1, const Tensor& b2) {
        return at::div(b1, b2);
      });
}

// Only usable on the C++ side; scalars are converted to tensors coming from Python.
Tensor NestedTensor_div_Scalar(const Tensor& self, const Scalar& other) {
  return NestedTensor_div_Tensor(self, wrapped_scalar_tensor(other));
}
Tensor NestedTensor_masked_fill(
    const Tensor& self,
    const Tensor& mask,
    const Scalar& value) {
  return NestedTensor_elementwise_Tensor(
      self, mask, "masked_fill", false /* supports_striding*/, [value](const Tensor& b1, const Tensor& b2) {
        return at::masked_fill(b1, b2, value);
      });
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

} // namespace native
} // namespace at
