#pragma once

#include <ATen/OpaqueTensorImpl.h>

namespace at {
// The only difference from OpaqueTensorImpl is faking strides(), stride(),
// is_contiguous(). The main intention for this is to be able to run torchscript
// model on Vulkan backend. Strides are not supported on Vulkan side, plan to
// support them.
template <typename OpaqueHandle>
struct VulkanOpaqueTensorImpl : public OpaqueTensorImpl<OpaqueHandle> {
  VulkanOpaqueTensorImpl(
      at::DispatchKeySet key_set,
      const caffe2::TypeMeta data_type,
      c10::Device device,
      OpaqueHandle opaque_handle,
      c10::IntArrayRef sizes,
      c10::IntArrayRef strides)
      : OpaqueTensorImpl<OpaqueHandle>(
            key_set,
            data_type,
            device,
            opaque_handle,
            sizes,
            false),
        strides_(strides.vec()) {
    // VulkanOpContext objects store some vTensor objects cast as at::Tensor in
    // the packed_context_ member variable. When trying to save a Torchscript
    // model that contains VulkanOpContext objects, the pickler (serializer)
    // will invoke the .storage() member of the converted at::Tensor objects.
    // However, the issue is that OpaqueTensorImpl calls
    // TensorImpl::set_storage_access_should_throw() in its constructor since
    // opaque tensors never set the storage_ member TensorImpl (i.e. storage_
    // will be a default constructed c10::Storage struct). This means calling
    // the .storage() function on opaque tensors will throw an exception. To
    // circumvent that, the flags is turned off the flag to enable
    // serialization.
    this->storage_access_should_throw_ = false;
  }

  IntArrayRef strides_custom() const override {
    return strides_;
  }

  bool is_contiguous_custom(c10::MemoryFormat memory_format) const override {
    return true;
  }

 private:
  const char* tensorimpl_type_name() const override {
    return "VulkanOpaqueTensorImpl";
  }

  // TODO: storing strides separately is unnecessary, the base TensorImpl
  // has space for them
  SmallVector<int64_t, 5> strides_;
};

} // namespace at
