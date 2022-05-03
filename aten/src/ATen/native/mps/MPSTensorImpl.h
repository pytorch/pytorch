//  Copyright © 2022 Apple Inc.

#pragma once

#include <ATen/Tensor.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <sys/_types/_size_t.h>
#include <memory>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/intrusive_ptr.h>

namespace at {
namespace native {
namespace mps {

// MPSTensor implementation encapsulated by at::Tensor when using MPS device.
class TORCH_API MPSTensorImpl : public c10::TensorImpl {

 public:
  MPSTensorImpl(const at::Tensor& tensor) : TensorImpl(c10::DispatchKey::MPS, tensor.dtype(), Device(DeviceType::MPS, 0)) {
    TORCH_INTERNAL_ASSERT(tensor.defined());
    contents_ = std::make_shared<Contents>();
    set_tensor(tensor);
    refresh_numel();
    refresh_contiguous();
    storage_ = at::Storage();
  }

  MPSTensorImpl(
      at::DispatchKeySet key_set,
      const caffe2::TypeMeta data_type,
      c10::Device device,
      c10::IntArrayRef sizes)
      : TensorImpl(key_set, data_type, device) {
    refresh_numel();
    storage_ = at::Storage();
  }

  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(c10::VariableVersion&& version_counter,
                                                         bool allow_tensor_metadata_change) const override {
    auto impl = c10::make_intrusive<MPSTensorImpl>(key_set(), dtype(), device(), sizes());
    copy_tensor_metadata(
      /*src=*/this,
      /*dest=*/impl.get(),
      /*version_counter=*/std::move(version_counter),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    impl->refresh_numel();
    impl->refresh_contiguous();
    return impl;
  }

  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(const c10::VariableVersion& version_counter,
                                                         bool allow_tensor_metadata_change) const override {
    auto impl = c10::make_intrusive<MPSTensorImpl>(key_set(), dtype(), device(), sizes());
    copy_tensor_metadata(
        /*src=*/this,
        /*dest=*/impl.get(),
        /*version_counter=*/version_counter,
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    impl->refresh_numel();
    return impl;
  }

  /**
   * Shallow-copies data from another TensorImpl into this TensorImpl.
   *
   * For why this function doesn't check this TensorImpl's `allow_tensor_metadata_change_`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override {
    AT_ASSERT(has_compatible_shallow_copy_type(impl->key_set()));
    auto mps_impl = static_cast<const MPSTensorImpl*>(impl.get());
    copy_tensor_metadata(
      /*src=*/mps_impl,
      /*dest=*/this,
      /*version_counter=*/version_counter(),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());
    refresh_numel();
    refresh_contiguous();
  }

  const char* tensorimpl_type_name() const override;

  void release_resources() override;

  // storage
  const at::Storage& storage() const override;

  bool has_storage() const override {
    return true;
  }

  bool is_contiguous(at::MemoryFormat memory_format) const override {
    return true;
  }

 private:
  struct Contents {
    at::Tensor attensor_;
  };

  at::Storage storage_;

  std::shared_ptr<Contents> contents_;
  std::vector<int64_t> strides_;
  // int64_t offset_ = 0;
  std::vector<int64_t> sizes_;

  virtual void set_tensor(const at::Tensor& tensor) {
    if (tensor.device().type() == DeviceType::MPS) {
      throw std::logic_error(
          "Unable to set at::Tensor tensor for MPSTensorImpl as provided tensor is already MPSTensor.");
    }
    contents_->attensor_ = tensor;
  }

  /**
   * Copy the tensor metadata fields (e.g. sizes / strides / storage pointer / storage_offset)
   * from one TensorImpl to another TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`, see NOTE [ TensorImpl Shallow-Copying ].
   */
  static void copy_tensor_metadata(
      const MPSTensorImpl* src,
      MPSTensorImpl* dest,
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) {
    TensorImpl::copy_tensor_metadata(src, dest, version_counter, allow_tensor_metadata_change);
    // copy any MPSTensorImpl specific fields
    dest->storage_ = src->storage_;
  }

  /**
   * Copy the tensor metadata fields (e.g. sizes / strides / storage pointer / storage_offset)
   * from one TensorImpl to another TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`, see NOTE [ TensorImpl Shallow-Copying ].
   */
  static void copy_tensor_metadata(
      const MPSTensorImpl* src,
      MPSTensorImpl* dest,
      const c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) {
    TensorImpl::copy_tensor_metadata(src, dest, version_counter, allow_tensor_metadata_change);
    // copy any MPSTensorImpl specific fields
    dest->storage_ = src->storage_;
  }
};

} // namespace mps
} // namespace native
} // namespace at
