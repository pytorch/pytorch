#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace at { namespace native {

// This function implements a boxed fallback to CPU.
// External backends can add their own custom logging on top if it to customize their own CPU fallbacks.
TORCH_API void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);

// This is a helper function that backends can use to directly call their boxed CPU fallback
// TODO: update and add a usage example after https://github.com/pytorch/pytorch/pull/58092 lands.
template<c10::KernelFunction::BoxedKernelFunction* fallback_fn, class ReturnType, class... ParameterTypes>
TORCH_API ReturnType call_fallback_fn(const char* name, const char* overload_name, ParameterTypes... args) {
    auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow(name, overload_name)
        .typed<ReturnType (ParameterTypes...)>();
    return c10::impl::BoxedKernelWrapper<ReturnType(ParameterTypes...)>::call(
        c10::KernelFunction::make_boxed_function<fallback_fn>,
        nullptr,
        op,
        c10::DispatchKeySet(), // we know that the cpu_fallback doesn't use the dispatch keyset.
        std::forward<ParameterTypes>(args)...
    );
}

// Version for ops with no overload name.
template<c10::KernelFunction::BoxedKernelFunction* fallback_fn, class ReturnType, class... ParameterTypes>
TORCH_API ReturnType call_fallback_fn(const char* name, ParameterTypes... args) {
    return call_fallback_fn<fallback_fn, ReturnType, ParameterTypes...>(name, "", args...);
}

template<c10::KernelFunction::BoxedKernelFunction* fallback_fn, class F2>
struct call_fallback_fn2 final {};

template<c10::KernelFunction::BoxedKernelFunction* fallback_fn, class ReturnType, class... ParameterTypes>
struct call_fallback_fn2<fallback_fn, ReturnType(ParameterTypes...)> final {
    static ReturnType call(const char* name, const char* overload_name, ParameterTypes... args) {
        auto op = c10::Dispatcher::singleton()
            .findSchemaOrThrow(name, overload_name)
            .typed<ReturnType (ParameterTypes...)>();
        return c10::impl::BoxedKernelWrapper<ReturnType(ParameterTypes...)>::call(
            c10::KernelFunction::make_boxed_function<fallback_fn>,
            nullptr,
            op,
            c10::DispatchKeySet(), // we know that the cpu_fallback doesn't use the dispatch keyset.
            std::forward<ParameterTypes>(args)...
            );
    }

    // Version for ops with no overload name.
    static ReturnType call(const char* name, ParameterTypes... args) {
        return call(name, "", args...);
    }
};

} // namespace native
} // namespace at
