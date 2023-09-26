#pragma once

#include <ATen/core/stack.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace at::native {

// This function implements a boxed fallback to CPU.
// External backends can add their own custom logging on top if it to customize their own CPU fallbacks.
TORCH_API void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack, bool error_on_views = false);

// This is a helper function that backends can use to directly call their boxed CPU fallback
// TODO: update and add a usage example after https://github.com/pytorch/pytorch/pull/58092 lands.
template<c10::KernelFunction::BoxedKernelFunction* fallback_fn, class Op, bool symint, class ReturnType, class... ParameterTypes>
struct _call_fallback_fn final {};

template<c10::KernelFunction::BoxedKernelFunction* fallback_fn, class Op, bool symint, class ReturnType, class... ParameterTypes>
struct _call_fallback_fn<fallback_fn, Op, symint, ReturnType(ParameterTypes...)> final {
    static ReturnType call(typename c10::maybe_keep_symint<symint, ParameterTypes>::type... args) {
        auto op = c10::Dispatcher::singleton()
            // TODO: figure out how to make compiler happy without dynamic casts
            .findSchemaOrThrow((const char*) Op::name, (const char*) Op::overload_name)
            //.findSchemaOrThrow("a", "b")
            .typed<ReturnType (typename c10::maybe_keep_symint<symint, ParameterTypes>::type...)>();
        return c10::impl::BoxedKernelWrapper<ReturnType (typename c10::maybe_keep_symint<symint, ParameterTypes>::type...)>::call(
            c10::BoxedKernel::makeFromFunction<fallback_fn>(),
            op,
            c10::DispatchKeySet(), // we know that the cpu_fallback doesn't use the dispatch keyset.
            // TODO: get std::forward<> to work
            args...
            );
    }
};

template<c10::KernelFunction::BoxedKernelFunction* fallback_fn, class Op>
using call_fallback_fn_symint = _call_fallback_fn<fallback_fn, Op, true, typename Op::schema>;

template<c10::KernelFunction::BoxedKernelFunction* fallback_fn, class Op>
using call_fallback_fn = _call_fallback_fn<fallback_fn, Op, false, typename Op::schema>;

} // namespace at::native
