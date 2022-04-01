#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>

#include <torch/csrc/lazy/core/tensor.h>

namespace torch {
namespace lazy {

// Tensor implementation class used to be fed to the at::Tensor.
// Its scope is just to handle an LazyTensor.
class TORCH_API LTCTensorImpl final : public c10::TensorImpl {
 public:
  explicit LTCTensorImpl(const LazyTensor& tensor);
  explicit LTCTensorImpl(LazyTensor&& tensor);

  LazyTensor& tensor() { return tensor_; }

  void set_tensor(const LazyTensor& lazy_tensor);

  void force_refresh_sizes() { generation_ = 0; }

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override;

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override;

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override;

  int64_t size(int64_t d) const override;

  int64_t stride(int64_t d) const override;

#ifndef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
  at::IntArrayRef sizes() const override;
  at::IntArrayRef strides() const override;
  int64_t dim() const override;
  int64_t numel() const override;

  bool is_contiguous(at::MemoryFormat memory_format) const override;
  const at::Storage& storage() const override;
  bool has_storage() const override { return false; }
#endif  // C10_DISABLE_TENSORIMPL_EXTENSIBILITY

 private:
  void setup_size_properties();

  LazyTensor tensor_;
  size_t generation_ {0};
};

}  // namespace lazy
}  // namespace torch
