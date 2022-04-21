#pragma once
#include <ATen/Tensor.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <ATen/MemoryOverlap.h>
#include <c10/core/MemoryFormat.h>
#include <c10/util/Metaprogramming.h>
#include <iostream>

namespace at {
namespace native {

struct TORCH_API NestedTensorImpl : public c10::TensorImpl {
  explicit NestedTensorImpl(at::Tensor buffer, at::Tensor nested_size_tensor);

#ifndef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
  int64_t numel() const override {
    TORCH_CHECK(
        false, "numel is disabled. These methods are not virtual in fbcode.");
  }
#endif
#ifndef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
  bool is_contiguous(at::MemoryFormat memory_format) const override {
    TORCH_CHECK(
        false,
        "is_contiguous is disabled. These methods are not virtual in fbcode.");
  }
#endif
  // TODO: don't expose private implementation details like this; in
  // particular, resizing this tensor will mess up our dim() and
  // callers cannot fix it.
  const Tensor& get_nested_size_tensor() const {
    return nested_size_tensor_;
  }
  c10::optional<int64_t> get_opt_size(int64_t i) const {
    return opt_sizes_[i];
  }
#ifndef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
  IntArrayRef sizes() const override {
    TORCH_CHECK(
        false,
        "Internal error: NestedTensorImpl doesn't support sizes. Please file an issue on https://github.com/pytorch/nestedtensor");
    return IntArrayRef();
  }
#endif
#ifndef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
  IntArrayRef strides() const override {
    TORCH_CHECK(
        false,
        "Internal error: NestedTensorImpl doesn't support strides. Please file an issue on https://github.com/pytorch/nestedtensor");
    return IntArrayRef();
  }
#endif

  const at::Tensor& get_buffer() const {
    return buffer_;
  }

 protected:
  const char* tensorimpl_type_name() const override;

 private:
  // Must be called after any changes to our dim() to sync the state
  // to TensorImpl.
  void refresh_dim();

  at::Tensor buffer_;
  const at::Tensor nested_size_tensor_;
  std::vector<c10::optional<int64_t>> opt_sizes_;
};

inline NestedTensorImpl* get_nested_tensor_impl_or_null(const at::Tensor& tensor) {
  if (tensor.is_nested()) {
    return static_cast<NestedTensorImpl*>(tensor.unsafeGetTensorImpl());
  }
  return nullptr;
}

inline NestedTensorImpl* get_nested_tensor_impl(
    const at::Tensor& tensor) {
  TORCH_CHECK(
      tensor.is_nested(),
      "get_nested_tensor_impl requires a NestedTensor.");
  return static_cast<NestedTensorImpl*>(
      tensor.unsafeGetTensorImpl());
}


// TODO: real implementation once we support strides.
inline bool nested_tensor_impl_is_contiguous(
    const NestedTensorImpl* nt,
    at::MemoryFormat memory_format = MemoryFormat::Contiguous) {
  return memory_format == MemoryFormat::Contiguous;
}

inline std::vector<int64_t> NestedTensor_get_max_size_from_size_tensor(const Tensor& sizes) {
  if (sizes.dim() == 0) {
    return {};
  }
  const auto sizes_ptr = sizes.data_ptr<int64_t>();
  const auto sizes_size_0 = sizes.sizes()[0];
  const auto sizes_size_1 = sizes.sizes()[1];
  TORCH_INTERNAL_ASSERT(sizes_size_1 > 0);
  std::vector<int64_t> results(sizes_size_1, 0);
  for (const auto ii : c10::irange(sizes_size_0)) {
    for (const auto jj : c10::irange(sizes_size_1)) {
      auto val = sizes_ptr[ii * sizes_size_1 + jj];
      if (results[jj] < val) {
        results[jj] = val;
      }
    }
  }
  return results;
}

inline std::vector<int64_t> NestedTensor_get_max_size(const NestedTensorImpl& nt) {
  const auto& sizes = nt.get_nested_size_tensor();
  return NestedTensor_get_max_size_from_size_tensor(sizes);
}

} // namespace native
} // namespace at
