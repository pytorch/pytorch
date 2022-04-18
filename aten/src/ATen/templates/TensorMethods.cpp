#include <c10/core/Scalar.h>
#include <ATen/core/TensorBody.h>

namespace at {

#define DEFINE_CAST(T, name)                                         \
   template <>                                                       \
   TORCH_API T* TensorBase::data_ptr() const {                       \
     TORCH_CHECK(                                                    \
         scalar_type() == ScalarType::name,                          \
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

} //namespace at

#include <ATen/NativeFunctions.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/EmptyTensor.h>


namespace at {
namespace {
// copied from RegisterCPU.cpp
Tensor create_out(IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {
  if (strides.empty()) {
      return at::detail::empty_cpu(sizes, options);
  } else {
      return at::detail::empty_strided_cpu(sizes, strides, options);
  }
}

// copied from RegisterCPU.cpp
void check_inplace(const Tensor &self, IntArrayRef sizes, const TensorOptions &options) {
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

    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options, DimnameList names) override {

        outputs_[output_idx] = create_out(sizes, strides, options);
        if (!names.empty()) {
          namedinference::propagate_names(*outputs_[output_idx], names);
        }
        // super must happen after, so that downstream can use maybe_get_output
        // to retrieve the output
        at::native::structured_mul_out::set_output(output_idx, sizes, strides, options, names);
    }

    const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
    }
    std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;
};

at::Tensor wrapper_mul_Tensor(const at::Tensor & self, const at::Tensor & other) {
structured_mul_out_functional op;
op.meta(self, other);
op.impl(self, other, *op.outputs_[0]);
return std::move(op.outputs_[0]).take();
}

struct structured_mul_out_inplace final : public at::native::structured_mul_out {
    structured_mul_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}

    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options, DimnameList names) override {

        const auto& out = outputs_[output_idx].get();
        check_inplace(out, sizes, options);
        if (!names.empty()) {
          namedinference::propagate_names(outputs_[output_idx], names);
        }
        // super must happen after, so that downstream can use maybe_get_output
        // to retrieve the output
        at::native::structured_mul_out::set_output(output_idx, sizes, strides, options, names);
    }

    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};

at::Tensor & wrapper_mul__Tensor(at::Tensor & self, const at::Tensor & other) {
structured_mul_out_inplace op(self);
op.meta(self, other);
op.impl(self, other, op.outputs_[0]);
return self;
}

} // namespace anonymous

Tensor Tensor::mul(const Tensor & other) const{
  return wrapper_mul_Tensor(const_cast<Tensor&>(*this), other);
}

Tensor & Tensor::mul_(const Tensor & other) const{
  return wrapper_mul__Tensor(const_cast<Tensor&>(*this), other);
}

}//namespace at
