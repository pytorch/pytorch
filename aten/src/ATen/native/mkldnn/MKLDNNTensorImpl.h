#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()

#include <ideep.hpp>

#include "c10/core/TensorImpl.h"

namespace c10 { namespace mkldnn {

struct CAFFE2_API MKLDNNTensorImpl : public c10::TensorImpl {
private:
  ideep::tensor it_;

public:
  explicit MKLDNNTensorImpl(c10::TensorTypeId type_id, const caffe2::TypeMeta& data_type);

  IntArrayRef sizes() const override;
  IntArrayRef strides() const override;
  bool is_contiguous() const override;
  int64_t size(int64_t d) const override;
  int64_t stride(int64_t d) const override;
  void resize_dim(int64_t ndim) override;
  void set_size(int64_t dim, int64_t new_size) override;
  void set_stride(int64_t dim, int64_t new_stride) override;
  void set_storage_offset(int64_t storage_offset) override;

  int64_t dim() const override;
  TensorImpl* maybe_zero_dim(bool condition_when_zero_dim) override;
  bool has_storage() const override;
  const Storage& storage() const override;
  int64_t storage_offset() const override;

  // NOTE: `shallow_copy_and_detach()` does not copy the AutogradMeta pointer
  // because it is unique for each Variable.
  // NOTE: We don't set `allow_tensor_metadata_change_` to false here, because there are call sites
  // to this function that need to change the shallow copy's size or storage afterwards, and setting
  // `allow_tensor_metadata_change_` to false would prevent those changes from happening and is
  // undesirable.
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach() const override {
    auto impl = c10::make_intrusive<MKLDNNTensorImpl>(type_id(), dtype());
    // TensorImpl general fields
    // Note that these fields are not used in mkldnn tensor code, and we copy them here only for completeness.
    impl->sizes_ = sizes_;
    impl->strides_ = strides_;
    impl->storage_offset_ = storage_offset_;
    impl->is_contiguous_ = is_contiguous_;
    impl->is_wrapped_number_ = is_wrapped_number_;
    impl->reserved_ = reserved_;

    // Mkldnn-specific fields
    impl->it_ = it_;
    return impl;
  }

  ideep::tensor& get_ideep_tensor() {
    return it_;
  }

};


} }// namespace c10::mkldnn

#endif // AT_MKLDNN_ENABLED()
