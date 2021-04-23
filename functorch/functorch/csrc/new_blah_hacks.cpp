#include <functorch/csrc/DynamicLayer.h>
#include <functorch/csrc/Constants.h>

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace at { namespace functorch {

#define NEW_BLAH_HACK(new_blah) \
  static Tensor new_blah##_hack( \
      const Tensor& self, \
      IntArrayRef size, \
      c10::optional<ScalarType> dtype, \
      c10::optional<Layout> layout, \
      c10::optional<Device> device, \
      c10::optional<bool> pin_memory \
      ) { \
    static auto op = c10::Dispatcher::singleton() \
      .findSchemaOrThrow("functorch::"#new_blah"_hack", "") \
      .typed<decltype(new_blah##_hack)>(); \
    return op.call(self, size, dtype, layout, device, pin_memory); \
  } \
  static Tensor new_blah##_hack_impl( \
      const Tensor& self, \
      IntArrayRef size, \
      c10::optional<ScalarType> dtype, \
      c10::optional<Layout> layout, \
      c10::optional<Device> device, \
      c10::optional<bool> pin_memory \
      ) { \
    auto layer = maybeCurrentDynamicLayer(); \
    if (!layer.has_value()) { \
      return self.new_blah(size, dtype, layout, device, pin_memory); \
    } \
    AutoNonVariableTypeMode dispatch_after_grad_guard; \
    c10::impl::ExcludeDispatchKeyGuard dispatch_after_vmap_guard(kBatchedKey); \
    return new_blah##_hack(self, size, dtype, layout, device, pin_memory); \
  }

NEW_BLAH_HACK(new_zeros);
NEW_BLAH_HACK(new_empty);

#undef NEW_BLAH_HACK 

TORCH_LIBRARY(functorch, m) {
  m.def("new_zeros_hack", new_zeros_hack_impl);
  m.def("new_empty_hack", new_empty_hack_impl);
}

TORCH_LIBRARY_IMPL(aten, DynamicLayerFront, m) {
  m.impl("new_zeros", new_zeros_hack);
  m.impl("new_empty", new_empty_hack);
}


}}

