// WARNING! WARNING! WARNING!
// This file is a temporary hack to enable development of pytorch quantization
//
// It's a stub for wrapping arbitrary cpp types in TorchScript. Proper
// implementation (under development) is to use TorchScript custom types.
// In the meantime, we abuse ByteTensor with custom deleter for this purpose.
//
// Template argument <T> has to be registered with CAFFE_KNOWN_TYPE mechanism.

#include <ATen/ATen.h>
#include <ATen/TracerMode.h>

namespace at {
namespace cpp_custom_type_hack {

template <typename T>
bool isa(const Tensor& packed) {
  return (packed.scalar_type() == kByte) &&
      (packed.storage().data_ptr().get_deleter() ==
       caffe2::TypeMeta::Make<T>().deleteFn());
}

template <typename T>
T& cast(const Tensor& packed) {
  TORCH_CHECK(
      packed.scalar_type() == kByte, "Expected temporary cpp type wrapper");
  TORCH_CHECK(
      packed.storage().data_ptr().get_deleter() ==
          caffe2::TypeMeta::Make<T>().deleteFn(),
      "Expected temporary cpp type wrapper of type ",
      caffe2::TypeMeta::TypeName<T>());
  return *reinterpret_cast<T*>(packed.storage().data_ptr().get());
}

template <typename T>
Tensor create(std::unique_ptr<T> ptr, TensorOptions options) {
  // None of this should trace, so turn off Tracer dispatching
  at::AutoNonVariableTypeMode guard;  // TODO: remove
  at::tracer::impl::NoTracerDispatchMode tracer_guard;

  // We store this instance away in a Tensor and register a deleter function
  // so that we do not leak memory. On the other side, we pull out the storage's
  // data_ptr and get the right typed pointer.
  void* raw_ptr = ptr.release();
  at::DataPtr at_ptr(
      raw_ptr, raw_ptr, caffe2::TypeMeta::Make<T>().deleteFn(), at::kCPU);

  // size doesn't really matter, but we can align it to the actual size
  // returning variables because one likely want to use this hack from python
  auto retval = at::empty({sizeof(T)}, options.device(kCPU).dtype(at::kByte));
  retval.storage().set_data_ptr(std::move(at_ptr));
  return retval;
}

} // namespace cpp_custom_type_hack
} // namespace at
