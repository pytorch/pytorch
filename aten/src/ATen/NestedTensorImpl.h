#pragma once
#include <ATen/MemoryOverlap.h>
#include <ATen/Tensor.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/irange.h>

namespace at {
namespace native {

struct TORCH_API NestedTensorImpl : public c10::TensorImpl {
  explicit NestedTensorImpl(at::Tensor buffer, at::Tensor nested_size_tensor);

  // Note: in current implementation,
  // empty nested tensor = no underlying tensor
  // nesting empty tensors is not considered empty nested tensor
  // both cases have `buffer_.numel() = 0`, though
  // TODO: for now, we use `nested_size_tensor_.dim()`
  // to determine whether a nested tensor is empty
  // when empty, `nested_size_tensor_.dim() = 0`
  // otherwise `nested_size_tensor_.dim() = 2`
  // maybe there is a better indicator for emptiness?

  // TODO: don't expose private implementation details like this; in
  // particular, resizing this tensor will mess up our dim() and
  // callers cannot fix it.
  const Tensor& get_nested_size_tensor() const {
    return nested_size_tensor_;
  }
  // TODO: don't expose private implementation details like this
  const Tensor& get_nested_stride_tensor() const {
    return nested_stride_tensor_;
  }
  // Returns nullopt if the ith dimension is irregular. The ith dimension
  // of a NestedTensor is regular if the unbound tensors match in
  // size at the (i-1)th dimension.
  c10::optional<int64_t> opt_size(int64_t d) const {
    d = at::maybe_wrap_dim(d, dim(), false);
    if (opt_sizes_[d] == -1) {
      return c10::nullopt;
    }
    return opt_sizes_[d];
  }

  int64_t size(int64_t d) const {
    c10::optional<int64_t> optional_size = this->opt_size(d);
    TORCH_CHECK(
        optional_size.has_value(),
        "Given dimension ",
        d,
        " is irregular and does not have a size.");
    return *optional_size;
  }

  const at::Tensor& get_buffer() const {
    return buffer_;
  }

 protected:
  const char* tensorimpl_type_name() const override;

  // TODO: numel_custom and is_contiguous_custom can be profitably overridden
  // with real implementations
  int64_t numel_custom() const override;
  bool is_contiguous_custom(MemoryFormat) const override;
  int64_t size_custom(int64_t d) const override {
    return this->size(d);
  }
  c10::SymInt sym_size_custom(int64_t d) const override {
    return c10::SymInt{this->size(d)};
  }
  IntArrayRef sizes_custom() const override;
  c10::SymIntArrayRef sym_sizes_custom() const override;
  c10::SymIntArrayRef sym_sizes() const override;
  IntArrayRef strides_custom() const override;

  // this one is real
  int64_t dim_custom() const override;

 private:
  // Must be called after any changes to our dim() to sync the state
  // to TensorImpl.
  void refresh_dim();

  at::Tensor buffer_;
  const at::Tensor nested_size_tensor_, nested_stride_tensor_;
  // NOTE: -1 here means the size is missing
  std::vector<int64_t> opt_sizes_;
};

inline NestedTensorImpl* get_nested_tensor_impl_or_null(
    const at::Tensor& tensor) {
  if (tensor.is_nested()) {
    return static_cast<NestedTensorImpl*>(tensor.unsafeGetTensorImpl());
  }
  return nullptr;
}

inline NestedTensorImpl* get_nested_tensor_impl(const at::Tensor& tensor) {
  TORCH_CHECK(
      tensor.is_nested(), "get_nested_tensor_impl requires a NestedTensor.");
  return static_cast<NestedTensorImpl*>(tensor.unsafeGetTensorImpl());
}

// TODO: real implementation once we support strides.
inline bool nested_tensor_impl_is_contiguous(
    const NestedTensorImpl* nt,
    at::MemoryFormat memory_format = MemoryFormat::Contiguous) {
  return memory_format == MemoryFormat::Contiguous;
}

inline const at::Tensor& get_nested_size_tensor(const at::Tensor& tensor) {
  return get_nested_tensor_impl(tensor)->get_nested_size_tensor();
}

} // namespace native
} // namespace at
