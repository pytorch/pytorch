#pragma once

#include <ATen/core/TensorBody.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/function_schema.h>
#include <c10/core/TensorImpl.h>

namespace at {

struct CAFFE2_API LazyTensorImpl : public TensorImpl {
  explicit LazyTensorImpl(TensorTypeSet ts)
      : TensorImpl(
            ts.add(TensorTypeId::LazyTensorId),
            caffe2::TypeMeta::Make<float>(), // Hack to support gradients
            c10::Device(c10::DeviceType::CPU)) {}

  explicit LazyTensorImpl(at::Tensor const& inp)
      : LazyTensorImpl(inp.type_set()) {
    inps_.emplace_back(inp);
  }

  explicit LazyTensorImpl(
      c10::optional<c10::FunctionSchema> schema,
      std::vector<c10::IValue> const& inps,
      size_t index = 0)
      : LazyTensorImpl([&inps]() {
          TensorTypeSet ts(TensorTypeId::LazyTensorId);
          for (const auto& inp : inps) {
            if (inp.isTensor()) {
              ts = ts | inp.toTensor().type_set();
            }
          }
          return ts;
        }()) {
    schema_.swap(schema);
    for (const auto& inp : inps) {
      inps_.emplace_back(inp);
    }
    index_ = index;
  }

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override {
    auto impl = c10::make_intrusive<LazyTensorImpl>(schema_, inps_, index_);
    return impl;
  }

  c10::optional<c10::FunctionSchema> schema_;
  std::vector<c10::IValue> inps_;
  // Index of this tensor as an output
  size_t index_ = 0;

  Tensor to_eager() const;
};

} // namespace at
