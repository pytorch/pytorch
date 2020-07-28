#pragma once

//#include <ATen/OpaqueTensorImpl.h>

namespace at {
// The only difference from OpaqueTensorImpl is faking strides(), stride(),
// is_contiguous(). The main intention for this is to be able to run torchscript
// model on Vulkan backend. Strides are not supported on Vulkan side, plan to
// support them.
template <typename OpaqueHandle>
struct VulkanOpaqueTensorImpl : public TensorImpl {
  VulkanOpaqueTensorImpl(
      at::DispatchKeySet key_set,
      const caffe2::TypeMeta& data_type,
      c10::Device device,
      OpaqueHandle opaque_handle,
      c10::IntArrayRef sizes,
      c10::IntArrayRef strides)
      : TensorImpl(key_set, data_type, device),
        opaque_handle_(std::move(opaque_handle)),
        strides_(strides.vec()) {
    sizes_ = sizes.vec();
    refresh_numel();
  }

  void release_resources() override {
    TensorImpl::release_resources();
    opaque_handle_ = {};
  }

  IntArrayRef strides() const override {
    return strides_;
  }

  bool is_contiguous(
      c10::MemoryFormat memory_format =
          c10::MemoryFormat::Contiguous) const override {
    return true;
  }

  int64_t stride(int64_t d) const override {
    d = at::maybe_wrap_dim(d, this->dim(), false);
    return strides_[d];
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

  bool has_storage() const override {
    return false;
  }

  const Storage& storage() const override {
    AT_ERROR("opaque tensors do not have storage");
  }

  int64_t storage_offset() const override {
    AT_ERROR("opaque tensors do not have storage");
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
    auto impl = c10::make_intrusive<VulkanOpaqueTensorImpl<OpaqueHandle>>(
        key_set(), dtype(), device(), opaque_handle_, sizes_, strides_);
    copy_tensor_metadata(
        /*src_impl=*/this,
        /*dest_impl=*/impl.get(),
        /*version_counter=*/version_counter,
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    impl->refresh_numel();
    return impl;
  }

  /**
   * Shallow-copies data from another TensorImpl into this TensorImpl.
   *
   * For why this function doesn't check this TensorImpl's
   * `allow_tensor_metadata_change_`, see NOTE [ TensorImpl Shallow-Copying ].
   */
  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override {
    AT_ASSERT(has_compatible_shallow_copy_type(impl->key_set()));
    auto opaque_impl =
        static_cast<const VulkanOpaqueTensorImpl<OpaqueHandle>*>(impl.get());
    copy_tensor_metadata(
        /*src_impl=*/opaque_impl,
        /*dest_impl=*/this,
        /*version_counter=*/version_counter(),
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());
    refresh_numel();
  }

  const OpaqueHandle& opaque_handle() const {
    return opaque_handle_;
  }

  OpaqueHandle& unsafe_opaque_handle() {
    return opaque_handle_;
  }

 protected:
  /**
   * Copy the tensor metadata fields (e.g. sizes / strides / storage pointer /
   * storage_offset) from one TensorImpl to another TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`, see NOTE
   * [ TensorImpl Shallow-Copying ].
   */
  static void copy_tensor_metadata(
      const VulkanOpaqueTensorImpl<OpaqueHandle>* src_opaque_impl,
      VulkanOpaqueTensorImpl<OpaqueHandle>* dest_opaque_impl,
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) {
    TensorImpl::copy_tensor_metadata(
        src_opaque_impl,
        dest_opaque_impl,
        version_counter,
        allow_tensor_metadata_change);

    // OpaqueTensorImpl-specific fields.
    dest_opaque_impl->opaque_handle_ = src_opaque_impl->opaque_handle_;
  }

 private:
  OpaqueHandle opaque_handle_;
  SmallVector<int64_t, 5> strides_;
};

} // namespace at
