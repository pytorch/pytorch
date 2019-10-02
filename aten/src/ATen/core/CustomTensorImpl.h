#pragma once

#include <string>
#include <unordered_map>
#include <ATen/core/stack.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/TensorImpl.h>

using Impl = void(const c10::FunctionSchema& schema, torch::jit::Stack*);

namespace at {

class CAFFE2_API CustomTensorImpl : public TensorImpl {
  size_t tag_;
  std::shared_ptr<void> storage_;
 public:
  inline size_t tag() {
    return tag_;
  }
  inline std::shared_ptr<void> storage() {
    return storage_;
  }
  CustomTensorImpl(
    size_t tag, std::shared_ptr<void> storage, TensorTypeSet ts,
    caffe2::TypeMeta dtype, c10::Device device) :
    TensorImpl(ts.add(TensorTypeId::CustomTensorId),
        dtype,
        device) {
        tag_ = tag;
        storage_ = storage;
        }
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override;
};

struct CAFFE2_API RegisterCustomTensorImpl {
  RegisterCustomTensorImpl(std::string name, std::string method, Impl* impl);
};

CAFFE2_API std::unordered_map<std::string, size_t>& getTagNameMap();

}

#define REGISTER_CUSTOM_TENSOR_METHOD(name, method, impls) \
  static at::RegisterCustomTensorImpl implementations_##name_##method(#name, #method, impls);

#define GET_CUSTOM_TENSOR_TAG(name) \
  at::getTagNameMap()[name];



