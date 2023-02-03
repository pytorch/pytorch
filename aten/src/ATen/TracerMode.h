#pragma once

#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>

// NOTE [Tracing Mode Switches]
//
// Historically, tracing function was controlled by two switches:
//
// - `AutoDispatchBelowADInplaceOrView` guard
//
//    Tracing function used to be script-generated inside `VariableType_*.cpp`
//    kernels, sharing the same `Autograd` dispatch key with autograd function.
//    Therefore, before tracing function was moved out of VariableType,
//    `AutoDispatchBelowADInplaceOrView` guard can also disable tracing as a
//    side effect of disabling `Autograd` dispatching.
//
// - `setTracingState()` API in `torch/csrc/jit/frontend/tracer.h`
//
//    It stores tracing data in a `TracingState` object in TLS. If the
//    `TracingState` object in TLS is `null`, then tracing is paused.
//
//    The `TracingState` object is created in `tracer::trace()` - the main
//    entrance of tracing function. It's temporarily set to `null` inside
//    generated VariableType (now TraceType) to bypass tracing for intermediate
//    ops (ops being called by other ops). After the intermediate op call
//    finishes it's set back to the original `TracingState` object.
//
//    The `TracingState` obect in TLS can also be read/written via its Python
//    binding in `python_tracer.cpp`, and `get/setTracingState()` C++ APIs,
//    which are also exposed as `TORCH_API`.
//
// Two new switches were introduced since tracing function was moved out of
// VariableType:
//
// - `tracer::impl::set_dispatch_enabled()` API
//
//    Unlike the special `Autograd` dispatch key which is included in dispatch
//    key set by default, `Tracer` dispatch key is off by default. The
//    dispatching switch can be toggled via this new API.
//
// - `tracer::impl::NoTracerDispatchMode` guard
//
//    It's used to cover the old semantics of `AutoDispatchBelowADInplaceOrView`
//    after tracing was moved out of VariableType.
//
// Before tracing function was moved out of VariableType, tracing was enabled
// when the following conditions are satisfied:
//
//    1) `TracingState` object in TLS != null;
//       - Either inside the execution scope of `tracer::trace()`, or
//       - Eagerly called `setTracingState()` with non-null object.
//    2) Not inside `AutoDispatchBelowADInplaceOrView` scope;
//
// After:
//
//    1) `TracingState` object in TLS != null;
//    2) Has called `tracer::impl::set_dispatch_enabled(true)`;
//    3) Not inside `tracer::impl::NonDispatchGuard` scope;
//
// [TODOs]
//
// - `setTracingState()` v.s. `tracer::impl::set_dispatch_enabled()`
//
//   Currently `set_dispatch_enabled()` is set/unset inside `setTracingState()`
//   to keep the semantics exactly the same as before - it's confusing to keep
//   both switches, though. We should consider simplifying/limiting the exposed
//   `setTracingState()` Python/C++ APIs (and other APIs calling it) so that
//   these two can be unified.
//
// - `AutoDispatchBelowADInplaceOrView` v.s.
// `tracer::impl::NoTracerDispatchMode`
//
//   We don't need to always set both guards together to keep semantics
//   unchanged. For the follow use cases of `AutoDispatchBelowADInplaceOrView`
//   we don't need set the new tracer guard:
//
//   * Script-generated VariableType kernels. The guard is not necessary as
//     tracing is already disabled explicitly by `setTracingState(null)` in
//     generated TraceType kernels - we could keep it as is or use the new guard
//     instead.
//
//   * Custom ops. Will be handled by fallback kernel for `Tracer`.
//
//   * Functions that are not likely to be called in tracing context (no python
//     binding / not an operator), e.g.: all mobile forward() wrappers, test
//     binaries, and etc.
//
//   * Where new threads are spawned, e.g.: ATen/native/ConvolutionMM2d.cpp.
//     It's not necessary as tracing is off by default.
//
//   For the rest of cases we might need have both:
//
//   * Functions that might be reachable from eager mode python (especially
//     factory methods), e.g.:
//     `internal_new_from_data()` in `torch/csrc/utils/tensor_new.cpp`.
//     Without the new guard it will add `aten::empty` to the traced graph.
//
//   * Some manually maintained functions, e.g.:
//     `torch/csrc/autograd/VariableTypeManual.cpp`.
//     Set the new guard if it's not obvious whether `setTracingState(null)`
//     has been called before it reaches the `AutoDispatchBelowADInplaceOrView`
//     guard.
//
//   We might need tweak the usage of the new guard to optimize/fix things.
//   It should only affect the correctness of tracing function, because the
//   guard is essentially no-op when the master `setTracingState()` switch is
//   off.

namespace at {
// TODO: move this from `at::` to `jit::torch::` after
// `aten/src/ATen/cpp_custom_type_hack.h` is removed.

namespace tracer {
namespace impl {

static inline bool is_dispatch_enabled() {
  return c10::impl::tls_is_dispatch_key_included(at::DispatchKey::Tracer) &&
      !c10::impl::tls_is_dispatch_key_excluded(at::DispatchKey::Tracer);
}

static inline void set_dispatch_enabled(bool enabled) {
  TORCH_INTERNAL_ASSERT(
      !c10::impl::tls_is_dispatch_key_excluded(at::DispatchKey::Tracer),
      "Cannot enable tracing within the scope of NoTracerDispatchMode!");
  c10::impl::tls_set_dispatch_key_included(at::DispatchKey::Tracer, enabled);
}

struct NoTracerDispatchMode {
  c10::impl::ExcludeDispatchKeyGuard guard_{at::DispatchKey::Tracer};
};

} // namespace impl
} // namespace tracer
} // namespace at
