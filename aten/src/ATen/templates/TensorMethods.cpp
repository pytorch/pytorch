#include <c10/core/Scalar.h>
#include <ATen/core/TensorBody.h>

namespace at {

#define DEFINE_CAST(T, name)                                         \
   template <>                                                       \
   TORCH_API T* TensorBase::data_ptr() const {                       \
     TORCH_CHECK(                                                    \
         scalar_type() == ScalarType::name                           \
         || (isQIntType(scalar_type())                               \
         && toUnderlying(scalar_type()) == ScalarType::name),        \
         "expected scalar type "                                     \
         #name                                                       \
         " but found ",                                              \
         scalar_type());                                             \
     return this->unsafeGetTensorImpl()->data_ptr_impl<T>();         \
   }

 AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CAST)
 AT_FORALL_QINT_TYPES(DEFINE_CAST)
 #undef DEFINE_CAST

 #define DEFINE_ITEM(T, name)      \
   template <>                     \
   TORCH_API T Tensor::item() const { \
     return item().to##name();     \
   }

 AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_ITEM)
 #undef DEFINE_ITEM
}

#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>

at::Tensor create_out(at::IntArrayRef sizes, at::IntArrayRef strides, const at::TensorOptions &options) {
  if (strides.empty()) {
      return at::detail::empty_cpu(sizes, options);
  } else {
      return at::detail::empty_strided_cpu(sizes, strides, options);
  }
}

void resize_out(const at::Tensor &out, at::IntArrayRef sizes, at::IntArrayRef strides, const at::TensorOptions &options) {
  TORCH_CHECK(options.dtype() == out.dtype(),
      "Expected out tensor to have dtype ", options.dtype(), ", but got ", out.dtype(), " instead");
  TORCH_CHECK(options.device() == out.device(),
      "Expected out tensor to have device ", options.device(), ", but got ", out.device(), " instead");
  const bool resized = at::native::resize_output(out, sizes);
  // Only restride if a resize occurred; otherwise we ignore the (advisory)
  // strides from the meta function and directly use the output tensor's
  // preexisting strides
  if (resized) {
    if (!strides.empty()) {
      TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
      // TODO: avoid the redispatch here
      out.as_strided_(sizes, strides);
    } else if (options.memory_format_opt().has_value()) {
      out.unsafeGetTensorImpl()->empty_tensor_restride(*options.memory_format_opt());
    }
  }
}

c10::optional<at::Tensor> maybe_create_proxy(const at::Tensor &out, at::IntArrayRef sizes, at::IntArrayRef strides, const at::TensorOptions &options) {
  if (out.strides() != strides) {
    return at::detail::empty_strided_cpu(sizes, strides, options);
  }
  return c10::nullopt;
}

void check_inplace(const at::Tensor &self, at::IntArrayRef sizes, const at::TensorOptions &options) {
  // These checks are needed on those operators that:
  //   1) don't use 'TensorIterator' (e.g. 'addmm' and 'baddbmm')
  //   2) have particular typing rules (e.g. 'cumsum' and 'cumprod')
  // For other operators (e.g. 'add'), 'TensorIterator' already checks
  // these things separately.
  TORCH_CHECK(options.dtype() == self.dtype(),
      "Bad in-place call: ",
      "input tensor dtype ", self.dtype(), " and output tensor dtype ", options.dtype(), " should match");
  TORCH_CHECK(options.device() == self.device(),
      "Bad in-place call: ",
      "input tensor device ", self.device(), " and output tensor device ", options.device(), " should match");
  TORCH_CHECK(sizes == self.sizes(),
      "Bad in-place call: ",
      "input tensor size ", self.sizes(), " and output tensor size ", sizes, " should match");
}

struct structured_mul_out_functional final : public at::native::structured_mul_out {
  void set_output_strided(
      int64_t output_idx, at::IntArrayRef sizes, at::IntArrayRef strides,
      at::TensorOptions options, at::DimnameList names
  ) override {
      outputs_[output_idx] = create_out(sizes, strides, options);
      if (!names.empty()) {
        at::namedinference::propagate_names(*outputs_[output_idx], names);
      }
      // super must happen after, so that downstream can use maybe_get_output
      // to retrieve the output
      at::native::structured_mul_out::set_output_raw_strided(output_idx, sizes, strides, options, names);
  }
  void set_output_raw_strided(
      int64_t output_idx, at::IntArrayRef sizes, at::IntArrayRef strides,
      at::TensorOptions options, at::DimnameList names
  ) override {
      outputs_[output_idx] = create_out(sizes, strides, options);
      if (!names.empty()) {
        at::namedinference::propagate_names(*outputs_[output_idx], names);
      }
      // super must happen after, so that downstream can use maybe_get_output
      // to retrieve the output
      at::native::structured_mul_out::set_output_raw_strided(output_idx, sizes, strides, options, names);
  }
  const at::Tensor& maybe_get_output(int64_t output_idx) override {
    return *outputs_[output_idx];
  }
  std::array<c10::ExclusivelyOwned<at::Tensor>, 1> outputs_;
};
at::Tensor wrapper_mul_Tensor(const at::Tensor & self, const at::Tensor & other) {
structured_mul_out_functional op;
op.meta(self, other);
op.impl(self, other, *op.outputs_[0]);
return std::move(op.outputs_[0]).take();
}

struct structured_mul_out_out final : public at::native::structured_mul_out {
  structured_mul_out_out(at::Tensor& out0) : outputs_{ std::ref(out0) } {}
  void set_output_strided(
      int64_t output_idx, at::IntArrayRef sizes, at::IntArrayRef strides,
      at::TensorOptions options, at::DimnameList names
  ) override {
      const auto& out = outputs_[output_idx].get();
      resize_out(out, sizes, strides, options);
      auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
      if (C10_UNLIKELY(maybe_proxy.has_value())) {
          proxy_outputs_[output_idx] = c10::ExclusivelyOwned<at::Tensor>(std::move(maybe_proxy).value());
      }
      if (!names.empty()) {
        at::namedinference::propagate_names(outputs_[output_idx], names);
      }
      // super must happen after, so that downstream can use maybe_get_output
      // to retrieve the output
      at::native::structured_mul_out::set_output_raw_strided(output_idx, sizes, strides, options, names);
  }
  void set_output_raw_strided(
      int64_t output_idx, at::IntArrayRef sizes, at::IntArrayRef strides,
      at::TensorOptions options, at::DimnameList names
  ) override {
      const auto& out = outputs_[output_idx].get();
      resize_out(out, sizes, strides, options);
      if (!names.empty()) {
        at::namedinference::propagate_names(outputs_[output_idx], names);
      }
      // super must happen after, so that downstream can use maybe_get_output
      // to retrieve the output
      at::native::structured_mul_out::set_output_raw_strided(output_idx, sizes, strides, options, names);
  }
  const at::Tensor& maybe_get_output(int64_t output_idx) override {
    return proxy_outputs_[output_idx].has_value() ? **proxy_outputs_[output_idx] : outputs_[output_idx].get();
  }
  std::array<std::reference_wrapper<at::Tensor>, 1> outputs_;
  std::array<c10::optional<c10::ExclusivelyOwned<at::Tensor>>, 1> proxy_outputs_;
};
at::Tensor & wrapper_mul_out_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
structured_mul_out_out op(out);
op.meta(self, other);
op.impl(self, other, op.maybe_get_output(0));
if (op.proxy_outputs_[0].has_value()) op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
return out;
}
struct structured_mul_out_inplace final : public at::native::structured_mul_out {
  structured_mul_out_inplace(at::Tensor& self) : outputs_{std::ref(self)} {}
  void set_output_strided(
      int64_t output_idx, at::IntArrayRef sizes, at::IntArrayRef strides,
      at::TensorOptions options, at::DimnameList names
  ) override {
      const auto& out = outputs_[output_idx].get();
      check_inplace(out, sizes, options);
      auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
      if (C10_UNLIKELY(maybe_proxy.has_value())) {
          proxy_outputs_[output_idx] = c10::ExclusivelyOwned<at::Tensor>(std::move(maybe_proxy).value());
      }
      if (!names.empty()) {
        at::namedinference::propagate_names(outputs_[output_idx], names);
      }
      // super must happen after, so that downstream can use maybe_get_output
      // to retrieve the output
      at::native::structured_mul_out::set_output_raw_strided(output_idx, sizes, strides, options, names);
  }
  void set_output_raw_strided(
      int64_t output_idx, at::IntArrayRef sizes, at::IntArrayRef strides,
      at::TensorOptions options, at::DimnameList names
  ) override {
      const auto& out = outputs_[output_idx].get();
      check_inplace(out, sizes, options);
      if (!names.empty()) {
        at::namedinference::propagate_names(outputs_[output_idx], names);
      }
      // super must happen after, so that downstream can use maybe_get_output
      // to retrieve the output
      at::native::structured_mul_out::set_output_raw_strided(output_idx, sizes, strides, options, names);
  }
  const at::Tensor& maybe_get_output(int64_t output_idx) override {
    return proxy_outputs_[output_idx].has_value() ? **proxy_outputs_[output_idx] : outputs_[output_idx].get();
  }
  std::array<std::reference_wrapper<at::Tensor>, 1> outputs_;
  std::array<c10::optional<c10::ExclusivelyOwned<at::Tensor>>, 1> proxy_outputs_;
};
at::Tensor & wrapper_mul__Tensor(at::Tensor & self, const at::Tensor & other) {
structured_mul_out_inplace op(self);
op.meta(self, other);
op.impl(self, other, op.outputs_[0]);
if (op.proxy_outputs_[0].has_value()) op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
return self;
}

namespace at {
 at::Tensor Tensor::mul(const at::Tensor & other) const {
  return wrapper_mul_Tensor(*this, other);
 }
 at::Tensor & Tensor::mul_(const at::Tensor & other) const {
  return wrapper_mul__Tensor(const_cast<Tensor&>(*this), other);
 }

 } //namespace at
