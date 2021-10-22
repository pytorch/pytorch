#pragma once
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/boxing/impl/boxing.h>
#include <ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h>

namespace c10 {

// NOTE: This function is in a separate header _NOT_ included in KernelFunction.h
// this is needed to that KernelFunction.h doesn't depend on c10::IValue.
namespace kernel_function {

template<class Return, class... Args>
C10_ALWAYS_INLINE Return call(
    const KernelFunction &kernel, const OperatorHandle& opHandle,
    DispatchKeySet dispatchKeySet, Args... args) {
  // note: Args above is intentionally not Args&&. We don't want perfect
  // forwarding, which would require Args to be deduced, but instead we
  // want callers to explicitly specify the Args.

  if (C10_LIKELY(kernel.unboxed_kernel_func_ != nullptr)) {
    return callUnboxedKernelFunction<Return, Args...>(
        kernel.unboxed_kernel_func_, kernel.functor_.get(), dispatchKeySet, std::forward<Args>(args)...);
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      kernel.boxed_kernel_func_ != nullptr,
      "Tried to call KernelFunction::call() on an uninitialized KernelFunction."
    );

  return impl::BoxedKernelWrapper<Return(Args...)>::call(
      kernel.boxed_kernel_func_,
      kernel.functor_.get(),
      opHandle,
      dispatchKeySet,
      std::forward<Args>(args)...
  );
}

}
}  // namespace c10
