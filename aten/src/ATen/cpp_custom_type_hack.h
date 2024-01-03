// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP

// YOU ARE IN THE WRONG PLACE! TURN BACK NOW!

// This code was a temporary hack to enable embedding arbitrary C++ structures
// into Tensors. THIS IS UNSAFE AND IS NOT SUPPORTED. IF YOU USE THIS CODE,
// IT __WILL__ BREAK.

// This code has been superseded by custom classes:
// https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html

// Please use custom classes and **DO NOT ADD MORE CALLSITES TO THINGS DEFINED
// IN THIS FILE**.

// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP

#include <ATen/TracerMode.h>
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace at {
namespace cpp_custom_type_hack {

template <typename T>
[[deprecated(
    "Use custom classes instead: "
    "https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html")]] bool
isa(const Tensor& packed) {
  return (packed.scalar_type() == kByte) &&
      (packed.storage().data_ptr().get_deleter() ==
       caffe2::TypeMeta::Make<T>().deleteFn());
}

template <typename T>
[[deprecated(
    "Use custom classes instead: "
    "https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html")]] T&
cast(const Tensor& packed) {
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
[[deprecated(
    "Use custom classes instead: "
    "https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html")]] Tensor
create(std::unique_ptr<T> ptr, TensorOptions options) {
  // None of this should trace, so turn off Tracer dispatching
  at::AutoDispatchBelowADInplaceOrView guard; // TODO: remove
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
  retval.storage().set_data_ptr_noswap(std::move(at_ptr));
  return retval;
}

} // namespace cpp_custom_type_hack
} // namespace at
