#pragma once

#include <c10/core/MemoryFormat.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>

namespace at {

// An "Opaque" TensorImpl -- there are no strides and (for now)
// even data() is not supported (thus no pointer arithmetic).

// NOTE: We could allow data() in the future, but would have to ensure pointer
// arithmetic code is properly guarded.
//
// NOTE: This does not support resize_ (and other metadata-changing ops) because
// of `shallow_copy_and_detach`. We would need to define an interface to
// "shallow copy" in order to add support.

template <typename OpaqueHandle>
struct TORCH_API OpaqueTensorImpl : public TensorImpl {
  // public constructor for now...
  OpaqueTensorImpl(
      at::DispatchKeySet key_set,
      const caffe2::TypeMeta data_type,
      c10::Device device,
      OpaqueHandle opaque_handle,
      c10::IntArrayRef sizes,
      bool is_non_overlapping_and_dense = true)
      : TensorImpl(key_set, data_type, device),
        opaque_handle_(std::move(opaque_handle)) {
    set_storage_access_should_throw();
    set_has_contiguity_policy(HasContiguityPolicy::ContiguityNotSupported);
    sizes_and_strides_.set_sizes(sizes);
    refresh_numel();
    is_non_overlapping_and_dense_ = is_non_overlapping_and_dense;
  }

  void release_resources() override {
    TensorImpl::release_resources();
    opaque_handle_ = {};
  }

  IntArrayRef strides() const override {
    AT_ERROR("opaque tensors do not have strides");
  }

  int64_t stride(int64_t d) const override {
    AT_ERROR("opaque tensors do not have strides");
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

#ifdef DEBUG
  bool has_storage() const override {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!storage_, "OpaqueTensorImpl assumes that storage_ is never set");
    return false;
  }
#endif

  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override {
    auto impl = c10::make_intrusive<OpaqueTensorImpl<OpaqueHandle>>(
        key_set(), dtype(), device(), opaque_handle_, sizes_and_strides_.sizes_arrayref());
    copy_tensor_metadata(
        /*src_opaque_impl=*/this,
        /*dest_opaque_impl=*/impl.get(),
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
    auto impl = c10::make_intrusive<OpaqueTensorImpl<OpaqueHandle>>(
        key_set(), dtype(), device(), opaque_handle_, sizes_and_strides_.sizes_arrayref());
    copy_tensor_metadata(
        /*src_opaque_impl=*/this,
        /*dest_opaque_impl=*/impl.get(),
        /*version_counter=*/std::move(version_counter),
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
        static_cast<const OpaqueTensorImpl<OpaqueHandle>*>(impl.get());
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
      const OpaqueTensorImpl<OpaqueHandle>* src_opaque_impl,
      OpaqueTensorImpl<OpaqueHandle>* dest_opaque_impl,
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

  static void copy_tensor_metadata(
      const OpaqueTensorImpl<OpaqueHandle>* src_opaque_impl,
      OpaqueTensorImpl<OpaqueHandle>* dest_opaque_impl,
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) {
    TensorImpl::copy_tensor_metadata(
        src_opaque_impl,
        dest_opaque_impl,
        std::move(version_counter),
        allow_tensor_metadata_change);

    // OpaqueTensorImpl-specific fields.
    dest_opaque_impl->opaque_handle_ = src_opaque_impl->opaque_handle_;
  }

 private:
  const char* tensorimpl_type_name() const override {
    return "OpaqueTensorImpl";
  }

  OpaqueHandle opaque_handle_;
};

} // namespace at
