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
  explicit NestedTensorImpl(
      at::Tensor buffer,
      at::Tensor nested_size_tensor,
      at::Tensor nested_stride_tensor,
      const std::vector<int64_t>& offsets);
  // assume contiguous, `nested_stride_tensor` and `offsets`
  // can be infered from `nested_size_tensor`
  explicit NestedTensorImpl(at::Tensor buffer, at::Tensor nested_size_tensor);

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
  const std::vector<int64_t>& get_offsets() const {
    return offsets_;
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

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override;

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override;

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override {
    copy_tensor_metadata(
        /*src_impl=*/impl.get(),
        /*dest_impl=*/this,
        /*version_counter=*/version_counter(),
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());
  }

 private:
  // Must be called after any changes to our dim() to sync the state
  // to TensorImpl.
  void refresh_dim();

  at::Tensor buffer_;
  const at::Tensor nested_size_tensor_, nested_stride_tensor_;
  // The starting positions of the underlying tensors in contiguous buffer
  // i.e. the buffer memory offsets to get the underlying tensors
  // The reason to keep this metadata is that, without strong enough constraint
  // it cannot be derived from `nested_size_tensor_`
  // and `nested_stride_tensor_`:
  // 1. when buffer has blanks, e.g. [tensor1, blank, tensor2]
  //    this can happen e.g. after slicing a nested tensor
  // 2. when multiple tensors share a same memory
  // 3. when the nesting ordering is changed, e.g. [tensor1, tensor3, tensor2]
  // Some strong enough constraints are:
  // 1. every underlying tensor is contiguous in memory
  //    && nesting in ascending order
  std::vector<int64_t> offsets_;
  // NOTE: -1 here means the size is missing
  // TODO: maybe we can remove this metadata since
  //       we can compute it from `nested_size_tensor_`
  std::vector<int64_t> opt_sizes_;

  template <typename VariableVersion>
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach_core(
      VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const;
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

inline bool nested_tensor_impl_is_contiguous(const NestedTensorImpl* nt) {
  int64_t ntensors = nt->size(0);
  if (ntensors == 0) {
    return true;
  }
  // if any underlying tensor is noncontiguous
  const Tensor &sizemat = nt->get_nested_size_tensor(),
               &stridemat = nt->get_nested_stride_tensor();
  const int64_t *sizemat_ptr = sizemat.data_ptr<int64_t>(),
                *stridemat_ptr = stridemat.data_ptr<int64_t>();
  int64_t orig_dim = sizemat.size(1);
  for (int64_t i = 0; i < ntensors; i++) {
    if (stridemat_ptr[orig_dim - 1] != 1) {
      return false;
    }
    int64_t product = sizemat_ptr[orig_dim - 1];
    for (int64_t j = orig_dim - 2; j >= 0; j--) {
      if (stridemat_ptr[j] != product) {
        return false;
      }
      product *= sizemat_ptr[j];
    }
    sizemat_ptr += orig_dim;
    stridemat_ptr += orig_dim;
  }
  // if there is blanck memory between underlying tensors
  const auto& offsets = nt->get_offsets();
  if (offsets[0] != 0) {
    return false;
  }
  sizemat_ptr = sizemat.data_ptr<int64_t>();
  stridemat_ptr = stridemat.data_ptr<int64_t>();
  for (int64_t i = 1; i < ntensors; i++) {
    if (offsets[i] != offsets[i - 1] + *sizemat_ptr * *stridemat_ptr) {
      return false;
    }
    sizemat_ptr += orig_dim;
    stridemat_ptr += orig_dim;
  }
  // everything is fine
  return true;
}

inline const at::Tensor& get_nested_size_tensor(const at::Tensor& tensor) {
  return get_nested_tensor_impl(tensor)->get_nested_size_tensor();
}

} // namespace native
} // namespace at
