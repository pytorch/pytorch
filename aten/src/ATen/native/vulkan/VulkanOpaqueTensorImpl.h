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
        strides_(strides.vec()) {}

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

 private:
  const char* tensorimpl_type_name() const override {
    return "VulkanOpaqueTensorImpl";
  }

  SmallVector<int64_t, 5> strides_;
};

} // namespace at
