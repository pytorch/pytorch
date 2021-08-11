#pragma once

#include <ATen/Tensor.h>

namespace at {

struct TORCH_API FunctionalTensorImplBase : public c10::TensorImpl {
  explicit FunctionalTensorImplBase(c10::DispatchKeySet keyset, caffe2::TypeMeta dtype, c10::Device device);
  explicit FunctionalTensorImplBase(caffe2::TypeMeta dtype, c10::Device device);

  // Different backends need to override what it means to "replace" a tensor in the functionalization pass.
  virtual void replace_(const at::Tensor& other) = 0;

  void maybe_add_update(const Tensor& updated_val);
  void set_view_meta(const Tensor& other, at::ViewMeta meta);
  bool is_view() const;
  std::shared_ptr<at::Alias> alias() { return alias_; }
  bool is_up_to_date() const;
  void sync_();

 private:
  const char* tensorimpl_type_name() const override;

  size_t generation_ = 0;
  std::vector<at::ViewMeta> view_metas_;
  std::shared_ptr<at::Alias> alias_;
};

TORCH_API inline FunctionalTensorImplBase* unsafeGetFunctionalImplBase(const at::Tensor& tensor) {
  return static_cast<FunctionalTensorImplBase*>(tensor.unsafeGetTensorImpl());
}

} // namespace at

