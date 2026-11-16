#ifndef MetalTensorImpl_h
#define MetalTensorImpl_h

#include <ATen/OpaqueTensorImpl.h>
#include <ATen/WrapDimUtils.h>
#import <ATen/native/metal/MetalTensorImplStorage.h>
#import <ATen/native/metal/mpscnn/MPSImageWrapper.h>

namespace at {
template <typename OpaqueHandle>
struct TORCH_API MetalTensorImpl : public OpaqueTensorImpl<OpaqueHandle> {
  MetalTensorImpl(
      at::DispatchKeySet key_set,
      const caffe2::TypeMeta& data_type,
      c10::Device device,
      OpaqueHandle opaque_handle,
      c10::IntArrayRef sizes,
      c10::IntArrayRef strides)
      : OpaqueTensorImpl<OpaqueHandle>(
            key_set,
            data_type,
            device,
            opaque_handle,
            sizes),
        strides_(strides.vec()) {
  }

  // TODO: manually storing strides here is dumb

  IntArrayRef strides_custom() const override {
    return strides_;
  }

  c10::SymIntArrayRef sym_strides_custom() const override {
    return c10::fromIntArrayRefKnownNonNegative(strides_);
  }

  c10::SymBool sym_is_contiguous_custom(c10::MemoryFormat memory_format) const override {
    return true;
  }

 private:
  const char* tensorimpl_type_name() const override {
    return "MetalTensorImpl";
  }

  SmallVector<int64_t, 5> strides_;
};
} // namespace at

#endif /* MetalTensorImpl_h*/
