#pragma once

#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>

namespace at {

// An "Opaque" TensorImpl -- there are no strides and (for now)
// even data() is not supported (thus no pointer arithmetic).

// NOTE: We could allow data() in the future, but would have to ensure pointer
// arithmetic code is properly guarded.
//
// NOTE: This does not support resize_ (and other metadata-changing ops) because of
// `shallow_copy_and_detach`. We would need to define an interface to  "shallow copy"
// in order to add support.

template <typename OpaqueHandle>
struct CAFFE2_API OpaqueTensorImpl : public TensorImpl {
  // public constructor for now...
  OpaqueTensorImpl(at::TensorTypeId type_id, const caffe2::TypeMeta& data_type, c10::Device device,
                   OpaqueHandle opaque_handle, c10::IntArrayRef sizes)
  :   TensorImpl(type_id, data_type, device),
      opaque_handle_(std::move(opaque_handle))
  {
    sizes_ = sizes.vec();
    refresh_numel();
  }

  void release_resources() override {
    TensorImpl::release_resources();
    opaque_handle_ = {};
  }

  IntArrayRef strides() const override {
    AT_ERROR("opaque tensors do not have strides");
  }

  bool is_contiguous() const override {
    AT_ERROR("opaque tensors do not have is_contiguous");
  }

  int64_t stride(int64_t d) const override {
    AT_ERROR("opaque tensors do not have strides");
  }

  void resize_dim(int64_t ndim) override {
    AT_ERROR("opaque tensors do not have resize_dim");
  }

  void set_size(int64_t dim, int64_t new_size) override {
    AT_ERROR("opaque tensors do not have set_size");
  }

  void set_stride(int64_t dim, int64_t new_stride) override {
    AT_ERROR("opaque tensors do not have set_stride");
  }

  void set_storage_offset(int64_t storage_offset) override {
    AT_ERROR("opaque tensors do not have set_storage_offset");
  }

  TensorImpl* maybe_zero_dim(bool condition_when_zero_dim) override {
      AT_ERROR("opaque tensors do not support maybe_zero_dim");
  }

  bool has_storage() const override {
    return false;
    }

  const Storage& storage() const override{
    AT_ERROR("opaque tensors do not have storage");
  }

  int64_t storage_offset() const override {
    AT_ERROR("opaque tensors do not have storage");
  }

// NOTE: `shallow_copy_and_detach()` does not copy the following TensorImpl fields:
// 1. the AutogradMeta pointer, because it is unique for each Variable.
// 2. the version counter, because although it lives in TensorImpl, the version counter is managed
// by autograd, and the call sites of `shallow_copy_and_detach()` (from autograd) should decide what
// the version counter should be for each new TensorImpl. See NOTE [ Version Counter Sharing ] for details.
//
// NOTE: We don't set `allow_tensor_metadata_change_` to false here, because there are call sites
// to this function that need to change the shallow copy's size or storage afterwards, and setting
// `allow_tensor_metadata_change_` to false would prevent those changes from happening and is
// undesirable.
c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach() const override {
  //AT_ASSERT(false);
  auto impl = c10::make_intrusive<OpaqueTensorImpl<OpaqueHandle>>(
    type_id(), dtype(), device(), opaque_handle_, sizes_);
  // TensorImpl general fields
  // Note that some of these fields are not used in opaque tensor code,
  // and we copy them here only for completeness.
  impl->sizes_ = sizes_;
  impl->strides_ = strides_;
  impl->storage_offset_ = storage_offset_;
  impl->is_contiguous_ = is_contiguous_;
  impl->is_wrapped_number_ = is_wrapped_number_;
  impl->reserved_ = reserved_;

  // OpaqueTensorImpl-specific fields (none currently).
  return impl;
}
  OpaqueHandle& unsafe_opaque_handle() {
    return opaque_handle_;
  }

private:
  OpaqueHandle opaque_handle_;
};

} // namespace at
