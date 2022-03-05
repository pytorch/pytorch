#pragma once
#include <ATen/Tensor.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <ATen/MemoryOverlap.h>
#include <c10/core/MemoryFormat.h>
#include <c10/util/Metaprogramming.h>

namespace at {
namespace native {

struct NestedTensorImpl : public c10::TensorImpl {
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
  const Tensor& get_nested_size_tensor() {
    return nested_size_tensor_;
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
};

} // namespace native
} // namespace at
