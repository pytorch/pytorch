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

  IntArrayRef strides_custom() const override {
    return strides_;
  }

  SymIntArrayRef sym_strides_custom() const override {
    return c10::fromIntArrayRefKnownNonNegative(strides_);
  }

  bool is_contiguous_custom(c10::MemoryFormat memory_format) const override {
    (void)memory_format;
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
