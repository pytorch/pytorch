#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#include <ATen/Operators.h>
#else
#include <ATen/ops/add_ops.h>
#include <ATen/ops/contiguous_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/fill_ops.h>
#include <ATen/ops/mul_ops.h>
#include <ATen/ops/round_ops.h>
#include <ATen/ops/to_ops.h>
#include <ATen/ops/zero_ops.h>
#endif
/**
 * We are removing Operators.cpp and RegisterBackendSelect.cpp from lightweight dispatch since C++ APIs that calls into
 * the dispatcher is not useful in lightweight dispatch. It turns out there are several exceptions, in these cases we
 * still calls into the dispatcher even with lightweight dispatch enabled, because the callsites are not codegen'd but
 * handcrafted. Here I'm adding this file to avoid undefined symbols.
 */

namespace at {
namespace {
at::Tensor wrapper_contiguous(
    const at::Tensor& self,
    at::MemoryFormat memory_format = MemoryFormat::Contiguous) {
  return at::native::contiguous(self, memory_format);
}

at::Tensor& wrapper_Scalar_fill__Scalar(
    at::Tensor& self,
    const at::Scalar& value) {
  return at::native::fill_(self, value);
}

at::Tensor& wrapper_Scalar_fill__Scalar_quantized(
    at::Tensor& self,
    const at::Scalar& value) {
  return at::native::fill_quantized_(self, value);
}

TORCH_LIBRARY_FRAGMENT(aten, m) {
m.def("fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)");
m.def(
"contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)");
}

TORCH_LIBRARY_IMPL(aten, CatchAll, m) {
m.impl("contiguous", TORCH_FN(wrapper_contiguous));
}

TORCH_LIBRARY_IMPL(aten, CPU, m) {
m.impl("fill_.Scalar", TORCH_FN(wrapper_Scalar_fill__Scalar));
}

TORCH_LIBRARY_IMPL(aten, QuantizedCPU, m) {
m.impl("fill_.Scalar", TORCH_FN(wrapper_Scalar_fill__Scalar_quantized));
}

} // namespace

namespace _ops {

// aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fill__Scalar::schema>
create_fill__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fill__Scalar::name, fill__Scalar::overload_name)
      .typed<fill__Scalar::schema>();
}
// aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)
at::Tensor& fill__Scalar::call(at::Tensor& self, const at::Scalar& value) {
  static auto op = create_fill__Scalar_typed_handle();
  return op.call(self, value);
}
// aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)
at::Tensor& fill__Scalar::redispatch(
    c10::DispatchKeySet dispatchKeySet,
    at::Tensor& self,
    const at::Scalar& value) {
  static auto op = create_fill__Scalar_typed_handle();
  return op.redispatch(dispatchKeySet, self, value);
}

// aten::contiguous(Tensor(a) self, *, MemoryFormat
// memory_format=contiguous_format) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<contiguous::schema>
create_contiguous_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(contiguous::name, contiguous::overload_name)
      .typed<contiguous::schema>();
}
// aten::contiguous(Tensor(a) self, *, MemoryFormat
// memory_format=contiguous_format) -> Tensor(a)
at::Tensor contiguous::call(
    const at::Tensor& self,
    at::MemoryFormat memory_format) {
  static auto op = create_contiguous_typed_handle();
  return op.call(self, memory_format);
}
// aten::contiguous(Tensor(a) self, *, MemoryFormat
// memory_format=contiguous_format) -> Tensor(a)
at::Tensor contiguous::redispatch(
    c10::DispatchKeySet dispatchKeySet,
    const at::Tensor& self,
    at::MemoryFormat memory_format) {
  static auto op = create_contiguous_typed_handle();
  return op.redispatch(dispatchKeySet, self, memory_format);
}

// The following definitions are workaround to avoid undefined symbols, these
// are not used in the code. We need proper fix for these errors.

// aten::zero_(Tensor(a!) self) -> Tensor(a!)
at::Tensor& zero_::call(at::Tensor& self) {
  TORCH_CHECK(false, "at::_ops::zero_::call Not supported");
}

// aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
at::Tensor add_Tensor::redispatch(
    c10::DispatchKeySet dispatchKeySet,
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  TORCH_CHECK(false, "at::_ops::add_Tensor::redispatch Not supported");
}

// aten::round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor& round_out::call(const at::Tensor& self, at::Tensor& out) {
  TORCH_CHECK(false, "at::_ops::round_out::call Not supported");
}

// aten::mul.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor mul_Tensor::redispatch(
    c10::DispatchKeySet dispatchKeySet,
    const at::Tensor& self,
    const at::Tensor& other) {
  TORCH_CHECK(false, "at::_ops::mul_Tensor::redispatch Not supported");
}

// aten::div.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor div_Tensor::redispatch(
    c10::DispatchKeySet dispatchKeySet,
    const at::Tensor& self,
    const at::Tensor& other) {
  TORCH_CHECK(false, "at::_ops::div_Tensor::redispatch Not supported");
}

// aten::to.dtype_layout(Tensor(a) self, *, ScalarType? dtype=None, Layout?
// layout=None, Device? device=None, bool? pin_memory=None, bool
// non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) ->
// Tensor(a)
at::Tensor to_dtype_layout::call(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    bool non_blocking,
    bool copy,
    c10::optional<at::MemoryFormat> memory_format) {
  TORCH_CHECK(false, "at::_ops::to_dtype_layout::call Not supported");
}

} // namespace _ops
} // namespace at
