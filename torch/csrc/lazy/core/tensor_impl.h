#pragma once

#include <ATen/Tensor.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/TensorImpl.h>

#include <torch/csrc/lazy/core/tensor.h>

namespace torch {
namespace lazy {

// Tensor implementation class used to be fed to the at::Tensor.
// Its scope is just to handle an LazyTensor.
class TORCH_API LTCTensorImpl final : public c10::TensorImpl {
 public:
  explicit LTCTensorImpl(const LazyTensorPtr& tensor);
  explicit LTCTensorImpl(const LazyTensor& tensor);
  explicit LTCTensorImpl(LazyTensor&& tensor);

  LazyTensorPtr tensor() {
    return tensor_;
  }

  void set_tensor(const LazyTensorPtr& lazy_tensor);

  void force_refresh_sizes() {
    generation_ = 0;
  }

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override;

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override;

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override;

  at::IntArrayRef sizes_custom() const override;
  at::IntArrayRef strides_custom() const override;
  int64_t numel_custom() const override;
  int64_t storage_offset_custom() const override;
  int64_t dim_custom() const override;
  bool is_contiguous_custom(at::MemoryFormat memory_format) const override;
  bool is_strides_like_custom(at::MemoryFormat memory_format) const override;
  bool is_non_overlapping_and_dense_custom() const override;

  c10::SymIntArrayRef sym_sizes_custom() const override;
  c10::SymIntArrayRef sym_strides_custom() const override;
  c10::SymInt sym_numel_custom() const override;

 private:
  void setup_size_properties();

  LazyTensorPtr tensor_;
  mutable std::optional<std::vector<c10::SymInt>> sym_sizes_;
  size_t generation_{0};
};

} // namespace lazy
} // namespace torch
