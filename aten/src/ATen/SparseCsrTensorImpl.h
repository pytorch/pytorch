#pragma once

#include <ATen/Tensor.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>
namespace at {

// Struct implementing a sparse CSR tensor. It uses three 1-D tensors for
// denoting the data: `crow_indices_`, `col_indices_` and `values_`.
// The `crow_indices_` tensor is a integer tensor of shape `(size(0) + 1)`
// that represents the compressed row indices of the CSR tensor. The
// `col_indices_` tensor is an integer tensor of shape `(nnz())`
// that explicitly stores the column indices of each value of the sparse
// tensor. The `values_` tensor can be of any pytorch-supported data type
// and has shape `(nnz())`.
//
// Since the main advantage of the CSR format over the COO format is speed of
// computation, care must be taken to facilitate smooth interfacing of
// these data structures with optimized libraries such as MKL and MAGMA.
// Since the MKL interface for pytorch currently uses indexing with int32
// type, it is important to make sure that the `crow_indices` and `col_indices`
// are of type int32 when calling MKL routines such as SPMM or SPMV.
//
// If not calling MKL, it should be alright to use 64 bit integer tensors
// for indexing.
struct TORCH_API SparseCsrTensorImpl : public TensorImpl {
  Tensor crow_indices_;
  Tensor col_indices_;
  Tensor values_;
  Layout layout_;

 public:
  explicit SparseCsrTensorImpl(
      at::DispatchKeySet,
      at::Device device,
      Layout layout,
      const caffe2::TypeMeta);

  void resize_(int64_t nnz, IntArrayRef size);
  void resize_and_clear_(
      int64_t sparse_dim,
      int64_t dense_dim,
      IntArrayRef size);
  void resize_as_sparse_compressed_tensor_(const Tensor& src);
  void set_member_tensors(
      const Tensor& crow_indices,
      const Tensor& col_indices,
      const Tensor& values,
      c10::SymIntArrayRef size);
  void set_member_tensors(
      const Tensor& crow_indices,
      const Tensor& col_indices,
      const Tensor& values,
      IntArrayRef size);
  const Tensor& compressed_indices() const {
    return crow_indices_;
  }
  const Tensor& plain_indices() const {
    return col_indices_;
  }
  const Tensor& values() const {
    return values_;
  }
  int64_t nnz() {
    return col_indices_.size(-1);
  }

  inline int64_t batch_dim() const noexcept {
    return crow_indices_.dim() - 1;
  }

  inline int64_t sparse_dim() const noexcept {
    return 2;
  }

  inline int64_t dense_dim() const noexcept {
    return values_.dim() - batch_dim() - block_dim() - 1;
  }

 private:
  inline int64_t block_dim() const noexcept {
    return (layout_ == kSparseBsr || layout_ == kSparseBsc ? 2 : 0);
  }

 protected:
  IntArrayRef strides_custom() const override;
  SymIntArrayRef sym_strides_custom() const override;
  bool is_contiguous_custom(MemoryFormat) const override;

 public:
  void set_size(int64_t dim, int64_t new_size) override;
  void set_stride(int64_t dim, int64_t new_stride) override;
  void set_storage_offset(int64_t storage_offset) override;
  Layout layout_impl() const override {
    return layout_;
  }
  void set_layout(Layout layout) {
    switch (layout) {
      case kSparseCsr:
      case kSparseCsc:
      case kSparseBsr:
      case kSparseBsc:
        layout_ = layout;
        break;
      default:
        TORCH_CHECK(false, "unsupported layout ", layout);
    }
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
    auto impl = c10::make_intrusive<SparseCsrTensorImpl>(
        key_set(), device(), layout_impl(), dtype());
    copy_tensor_metadata(
        /*src_impl=*/this,
        /*dest_impl=*/impl.get(),
        /*version_counter=*/version_counter,
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    impl->refresh_numel();
    return impl;
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
    auto impl = c10::make_intrusive<SparseCsrTensorImpl>(
        key_set(), device(), layout_impl(), dtype());
    copy_tensor_metadata(
        /*src_impl=*/this,
        /*dest_impl=*/impl.get(),
        /*version_counter=*/std::move(version_counter),
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    impl->refresh_numel();
    return impl;
  }

 private:
  explicit SparseCsrTensorImpl(
      at::DispatchKeySet key_set,
      const caffe2::TypeMeta data_type,
      at::Tensor crow_indices,
      at::Tensor col_indices,
      at::Tensor values,
      at::Layout layout);

  const char* tensorimpl_type_name() const override;

  /**
   * Copy the tensor metadata fields (e.g. sizes / strides / storage pointer /
   * storage_offset) from one TensorImpl to another TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`, see NOTE
   * [ TensorImpl Shallow-Copying ].
   */
  static void copy_tensor_metadata(
      const SparseCsrTensorImpl* src_sparse_impl,
      SparseCsrTensorImpl* dest_sparse_impl,
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) {
    TensorImpl::copy_tensor_metadata(
        src_sparse_impl,
        dest_sparse_impl,
        version_counter,
        allow_tensor_metadata_change);

    // Sparse-specific fields
    dest_sparse_impl->crow_indices_ = src_sparse_impl->compressed_indices();
    dest_sparse_impl->col_indices_ = src_sparse_impl->plain_indices();
    dest_sparse_impl->values_ = src_sparse_impl->values();
    dest_sparse_impl->layout_ = src_sparse_impl->layout_impl();
  }
};
} // namespace at
