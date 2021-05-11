#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace at { namespace native {

// This function implements a boxed fallback to CPU.
// External backends can add their own custom logging on top if it to customize their own CPU fallbacks.
void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);

template<c10::KernelFunction::BoxedKernelFunction* fallback_fn, class ReturnType, class... ParameterTypes>
ReturnType call_fallback_fn(const char* name, const char* overload_name, ParameterTypes... args) {
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

} // namespace native
} // namespace at
