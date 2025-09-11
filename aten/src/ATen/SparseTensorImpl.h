#pragma once

#include <ATen/Tensor.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/resize.h>
#endif

namespace at {
struct TORCH_API SparseTensorImpl : public TensorImpl {
  // Stored in COO format, indices + values.

  // INVARIANTS:
  // sparse_dim: range [0, len(shape)]; sparse_dim + dense_dim = len(shape)
  // dense_dim : range [0, len(shape)]; sparse_dim + dense_dim = len(shape)
  // _indices.shape: dimensionality: 2,  shape: (sparse_dim, nnz)
  // _values.shape:  dimensionality: 1 + dense_dim.  shape: (nnz,
  // shape[sparse_dim:])

  int64_t sparse_dim_ = 0; // number of sparse dimensions
  int64_t dense_dim_ = 0; // number of dense dimensions

  Tensor indices_; // always a LongTensor
  Tensor values_;

  // A sparse tensor is 'coalesced' if every index occurs at most once in
  // the indices tensor, and the indices are in sorted order.  (This means
  // that it is very easy to convert a coalesced tensor to CSR format: you
  // need only compute CSR format indices.)
  //
  // Most math operations can only be performed on coalesced sparse tensors,
  // because many algorithms proceed by merging two sorted lists (of indices).
  bool coalesced_ = false;

  // compute_numel with integer multiplication overflow check, see gh-57542
  void refresh_numel() {
    TensorImpl::safe_refresh_numel();
  }

 public:
  // Public for now...
  explicit SparseTensorImpl(at::DispatchKeySet, const caffe2::TypeMeta);

  void release_resources() override;

  int64_t nnz() const {
    return values_.size(0);
  }

  c10::SymInt sym_nnz() const {
    return values_.sym_size(0);
  }
  int64_t sparse_dim() const {
    return sparse_dim_;
  }
  int64_t dense_dim() const {
    return dense_dim_;
  }
  bool coalesced() const {
    return coalesced_;
  }
  Tensor indices() const {
    return indices_;
  }
  Tensor values() const {
    return values_;
  }

  void set_size(int64_t dim, int64_t new_size) override;
  void set_stride(int64_t dim, int64_t new_stride) override;
  void set_storage_offset(int64_t storage_offset) override;

#ifdef DEBUG
  bool has_storage() const override;
#endif

  // WARNING: This function does NOT preserve invariants of sparse_dim/dense_dim
  // with respect to indices and values
  void raw_resize_(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size) {
    TORCH_CHECK(
        allow_tensor_metadata_change(),
        "raw_resize_ ",
        err_msg_tensor_metadata_change_not_allowed);
    TORCH_CHECK(
        !has_symbolic_sizes_strides_,
        "raw_resize_ called on tensor with symbolic shape")
    set_sizes_and_strides(size, std::vector<int64_t>(size.size()));
    sparse_dim_ = sparse_dim;
    dense_dim_ = dense_dim;
    refresh_numel();
  }

  // NOTE: This function preserves invariants of sparse_dim/dense_dim with
  // respect to indices and values.
  //
  // NOTE: This function supports the following cases:
  // 1. When we keep the number of dense dimensions unchanged, and NOT shrinking
  // the size of any of the dense dimensions.
  // 2. When we keep the number of sparse dimensions unchanged, and NOT
  // shrinking the size of any of the sparse dimensions.
  // 3. When the sparse tensor has zero nnz, in which case we are free to change
  // the shapes of both its sparse and dense dimensions.
  //
  // This function DOESN'T support (and will throw an error) the following
  // cases:
  // 1. When we attempt to change the number of sparse dimensions on a non-empty
  // sparse tensor (such an operation will invalidate the indices stored).
  // 2. When we attempt to change the number of dense dimensions on a non-empty
  // sparse tensor (such an operation will behave differently from an equivalent
  // dense tensor's resize method, and for API consistency we don't support it).
  // 3. When we attempt to shrink the size of any of the dense dimensions on a
  // non-empty sparse tensor (such an operation will behave differently from an
  // equivalent dense tensor's resize method, and for API consistency we don't
  // support it).
  // 4. When we attempt to shrink the size of any of the sparse dimensions on a
  // non-empty sparse tensor (this could make some of the stored indices
  // out-of-bound and thus unsafe).
  template <typename T>
  void _resize_(int64_t sparse_dim, int64_t dense_dim, ArrayRef<T> size) {
    TORCH_CHECK(
        allow_tensor_metadata_change(),
        "resize_ ",
        err_msg_tensor_metadata_change_not_allowed);
    TORCH_CHECK(
        !has_symbolic_sizes_strides_,
        "resize_ called on tensor with symbolic shape")
    TORCH_CHECK(
        sparse_dim + dense_dim == static_cast<int64_t>(size.size()),
        "'len(size) == sparse_dim + dense_dim' is not satisfied: len(size) = ",
        size.size(),
        ", sparse_dim = ",
        sparse_dim,
        ", dense_dim = ",
        dense_dim);
    if (nnz() > 0) {
      [[maybe_unused]] auto constexpr alt_options_msg =
          "You could try the following options:\n\
1. If you need an empty sparse tensor of this size, call `x = torch.sparse_coo_tensor(size)`.\n\
2. If you need to resize this tensor, you have the following options:\n\
    1. For both sparse and dense dimensions, keep the number of them constant and the size of them non-shrinking, and then try the same call again.\n\
    2. Or, create a new sparse tensor with the correct indices and values from this sparse tensor.";

      TORCH_CHECK(
          sparse_dim == sparse_dim_,
          "changing the number of sparse dimensions (from ",
          sparse_dim_,
          " to ",
          sparse_dim,
          ") on a non-empty sparse tensor is not supported.\n",
          alt_options_msg);

      TORCH_CHECK(
          dense_dim == dense_dim_,
          "changing the number of dense dimensions (from ",
          dense_dim_,
          " to ",
          dense_dim,
          ") on a non-empty sparse tensor is not supported.\n",
          alt_options_msg);

      bool shrinking_sparse_dims = false;
      bool shrinking_dense_dim = false;
      auto sparse_size_original = generic_sizes<T>().slice(0, sparse_dim);
      auto sparse_size_new = size.slice(0, sparse_dim);
      for (const auto i : c10::irange(sparse_dim)) {
        if (sparse_size_new[i] < sparse_size_original[i]) {
          shrinking_sparse_dims = true;
          break;
        }
      }
      auto dense_size_original = generic_sizes<T>().slice(sparse_dim);
      auto dense_size_new = size.slice(sparse_dim);
      for (const auto i : c10::irange(dense_dim)) {
        if (dense_size_new[i] < dense_size_original[i]) {
          shrinking_dense_dim = true;
          break;
        }
      }

      TORCH_CHECK(
          !shrinking_sparse_dims,
          "shrinking the size of sparse dimensions (from ",
          sparse_size_original,
          " to ",
          sparse_size_new,
          ") on a non-empty sparse tensor is not supported.\n",
          alt_options_msg);

      TORCH_CHECK(
          !shrinking_dense_dim,
          "shrinking the size of dense dimensions (from ",
          dense_size_original,
          " to ",
          dense_size_new,
          ") on a non-empty sparse tensor is not supported.\n",
          alt_options_msg);
    }

    auto sizes_and_strides = generic_sizes<T>();
    const bool size_equals_sizes = std::equal(
        size.begin(),
        size.end(),
        sizes_and_strides.begin(),
        sizes_and_strides.end());
    if ((!size_equals_sizes) || (sparse_dim != sparse_dim_) ||
        (dense_dim != dense_dim_)) {
      auto nnz = at::symint::sizes<T>(values())[0];
      std::vector<T> values_size = {nnz};
      auto dense_size = size.slice(sparse_dim);
      values_size.insert(
          values_size.end(), dense_size.begin(), dense_size.end());
      at::symint::resize_<T>(values_, values_size);
      at::symint::resize_<T>(indices_, {T(sparse_dim), nnz});
    }

    if (!size_equals_sizes) {
      set_sizes_and_strides(size, std::vector<T>(size.size()));
    }
    sparse_dim_ = sparse_dim;
    dense_dim_ = dense_dim;
    refresh_numel();
  }

  void resize_(int64_t sparse_dim, int64_t dense_dim, ArrayRef<int64_t> size) {
    return _resize_(sparse_dim, dense_dim, size);
  }

  void resize_(
      int64_t sparse_dim,
      int64_t dense_dim,
      ArrayRef<c10::SymInt> size) {
    return _resize_(sparse_dim, dense_dim, size);
  }

  // NOTE: this function will resize the sparse tensor and also set `indices`
  // and `values` to empty.
  void resize_and_clear_(
      int64_t sparse_dim,
      int64_t dense_dim,
      IntArrayRef size) {
    TORCH_CHECK(
        allow_tensor_metadata_change(),
        "resize_and_clear_ ",
        err_msg_tensor_metadata_change_not_allowed);
    TORCH_CHECK(
        !has_symbolic_sizes_strides_,
        "resize_and_clear_ called on tensor with symbolic shape")
    TORCH_CHECK(
        sparse_dim + dense_dim == static_cast<int64_t>(size.size()),
        "'len(size) == sparse_dim + dense_dim' is not satisfied: len(size) = ",
        size.size(),
        ", sparse_dim = ",
        sparse_dim,
        ", dense_dim = ",
        dense_dim);

    set_sizes_and_strides(size, std::vector<int64_t>(size.size()));
    sparse_dim_ = sparse_dim;
    dense_dim_ = dense_dim;

    auto empty_indices = at::empty({sparse_dim, 0}, indices().options());
    std::vector<int64_t> values_size = {0};
    auto dense_size = sizes().slice(sparse_dim);
    values_size.insert(values_size.end(), dense_size.begin(), dense_size.end());
    auto empty_values = at::empty(values_size, values().options());
    set_indices_and_values_unsafe(empty_indices, empty_values);
    refresh_numel();
  }

  void set_coalesced(bool coalesced) {
    TORCH_CHECK(
        allow_tensor_metadata_change(),
        "set_coalesced ",
        err_msg_tensor_metadata_change_not_allowed);
    coalesced_ = coalesced;
  }

  // NOTE: this function is only used internally and not exposed to Python
  // frontend
  void set_nnz_and_narrow(int64_t new_nnz) {
    TORCH_CHECK(
        allow_tensor_metadata_change(),
        "set_nnz_and_narrow ",
        err_msg_tensor_metadata_change_not_allowed);
    AT_ASSERT(new_nnz <= nnz());
    indices_ = indices_.narrow(1, 0, new_nnz);
    values_ = values_.narrow(0, 0, new_nnz);
    if (new_nnz < 2) {
      coalesced_ = true;
    }
  }

  // Takes indices and values and directly puts them into the sparse tensor, no
  // copy. NOTE: this function is unsafe because it doesn't check whether any
  // indices are out of boundaries of `sizes`, so it should ONLY be used where
  // we know that the indices are guaranteed to be within bounds. This used to
  // be called THSTensor_(_move) NB: This used to be able to avoid a refcount
  // bump, but I was too lazy to make it happen
  void set_indices_and_values_unsafe(
      const Tensor& indices,
      const Tensor& values);

  template <typename VariableVersion>
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach_core(
      VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const {
    const auto mode_stack_len = c10::impl::TorchDispatchModeTLS::stack_len();
    c10::impl::PyInterpreter&& interpreter = nullptr;
    if (mode_stack_len > 0 &&
        !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::Python)) {
      const auto& cur_torch_dispatch_mode_state =
          c10::impl::TorchDispatchModeTLS::get_stack_at(mode_stack_len - 1);
      interpreter = cur_torch_dispatch_mode_state->pyinterpreter();
    } else if (
        key_set_.has(DispatchKey::Python) &&
        !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::Python)) {
      interpreter = pyobj_slot_.load_pyobj_interpreter();
    } else {
      // otherwise just copy the SparseTensorImpl and not the PyObject.
      auto impl = c10::make_intrusive<SparseTensorImpl>(key_set(), dtype());
      copy_tensor_metadata(
          /*src_sparse_impl=*/this,
          /*dest_sparse_impl=*/impl.get(),
          /*version_counter=*/version_counter,
          /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
      impl->refresh_numel();
      return impl;
    }
    auto r = interpreter->detach(this);
    r->set_version_counter(std::forward<VariableVersion>(version_counter));
    r->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
    return r;
  }

  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override {
    return shallow_copy_and_detach_core(
        version_counter, allow_tensor_metadata_change);
  }

  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override {
    return shallow_copy_and_detach_core(
        std::move(version_counter), allow_tensor_metadata_change);
  }

  /**
   * Shallow-copies data from another TensorImpl into this TensorImpl.
   *
   * For why this function doesn't check this TensorImpl's
   * `allow_tensor_metadata_change_`, see NOTE [ TensorImpl Shallow-Copying ].
   */
  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override {
    AT_ASSERT(has_compatible_shallow_copy_type(impl->key_set()));
    auto sparse_impl = static_cast<const SparseTensorImpl*>(impl.get());
    copy_tensor_metadata(
        /*src_sparse_impl=*/sparse_impl,
        /*dest_sparse_impl=*/this,
        /*version_counter=*/version_counter(),
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());
    refresh_numel();
  }

 private:
  explicit SparseTensorImpl(
      at::DispatchKeySet,
      const caffe2::TypeMeta,
      at::Tensor indices,
      at::Tensor values);

  /**
   * Copy the tensor metadata fields (e.g. sizes / strides / storage pointer /
   * storage_offset) from one TensorImpl to another TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`, see NOTE
   * [ TensorImpl Shallow-Copying ].
   */
  static void copy_tensor_metadata(
      const SparseTensorImpl* src_sparse_impl,
      SparseTensorImpl* dest_sparse_impl,
      c10::VariableVersion version_counter,
      bool allow_tensor_metadata_change) {
    TensorImpl::copy_tensor_metadata(
        src_sparse_impl,
        dest_sparse_impl,
        std::move(version_counter),
        allow_tensor_metadata_change);

    // Sparse-specific fields
    dest_sparse_impl->sparse_dim_ = src_sparse_impl->sparse_dim();
    dest_sparse_impl->dense_dim_ = src_sparse_impl->dense_dim();
    dest_sparse_impl->indices_ = src_sparse_impl->indices();
    dest_sparse_impl->values_ = src_sparse_impl->values();
    dest_sparse_impl->coalesced_ = src_sparse_impl->coalesced();
  }

  const char* tensorimpl_type_name() const override;
};

} // namespace at
