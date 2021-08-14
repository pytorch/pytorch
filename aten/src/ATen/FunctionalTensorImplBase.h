#pragma once

#include <ATen/Tensor.h>

namespace at {

struct TORCH_API FunctionalTensorImplBase : public c10::TensorImpl {
  explicit FunctionalTensorImplBase(c10::DispatchKeySet keyset, caffe2::TypeMeta dtype, c10::Device device);
  explicit FunctionalTensorImplBase(caffe2::TypeMeta dtype, c10::Device device);

  void maybe_add_update(const Tensor& updated_val);
  void set_view_meta(const Tensor& other, at::ViewMeta meta);
  bool is_view() const;
  bool is_alias_tensor() const { return is_alias_tensor_; }
  std::shared_ptr<at::Alias> alias() { return alias_; }
  bool is_up_to_date() const;
  // See Note [Marking Alias Tensors]
  void mark_as_alias();
  void sync_();

 private:
  const char* tensorimpl_type_name() const override;

  size_t generation_ = 0;
  std::vector<at::ViewMeta> view_metas_;
  std::shared_ptr<at::Alias> alias_;
  bool is_alias_tensor_;
};

TORCH_API inline FunctionalTensorImplBase* unsafeGetFunctionalImplBase(const at::Tensor& tensor) {
  return static_cast<FunctionalTensorImplBase*>(tensor.unsafeGetTensorImpl());
}

namespace functionalization {
namespace impl {

// Utility functions for the functionalization pass.

void maybe_sync(const at::Tensor& t);
void maybe_sync(const c10::optional<Tensor>& t);
void maybe_sync(const c10::List<Tensor>& t_list);
void maybe_sync(const at::TensorList t_list);
void maybe_sync(const c10::List<c10::optional<Tensor>>& t_list);

void maybe_add_update(Tensor& self);

void set_view_meta(const Tensor& out, const Tensor& t, ViewMeta meta);
void set_view_meta(const c10::List<Tensor>& outs, const Tensor& t, ViewMeta meta);

bool is_alias_tensor(const Tensor& t);
bool is_alias_tensor(const c10::List<Tensor>& t_list);

// We'll need one of these per view op. Consider code-generating the declarations?
at::ViewMeta get_meta_view(const at::Tensor& self, at::IntArrayRef size);

} // namespace impl
} // namespace functionalization
} // namespace at

