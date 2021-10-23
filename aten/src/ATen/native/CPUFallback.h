#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/Metaprogramming.h>
#include <torch/library.h>

namespace at { namespace native {

// This function implements a boxed fallback to CPU.
// External backends can add their own custom logging on top if it to customize their own CPU fallbacks.
TORCH_API void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);

// This is a helper function that backends can use to directly call their boxed CPU fallback
// TODO: update and add a usage example after https://github.com/pytorch/pytorch/pull/58092 lands.
template<c10::KernelFunction::BoxedKernelFunction* fallback_fn, class Op, class ReturnType, class... ParameterTypes>
struct _call_fallback_fn final {};

template<c10::KernelFunction::BoxedKernelFunction* fallback_fn, class Op, class ReturnType, class... ParameterTypes>
struct _call_fallback_fn<fallback_fn, Op, ReturnType(ParameterTypes...)> final {
    static_assert(std::is_same<ReturnType, typename guts::infer_function_traits_t<typename Op::schema>::return_type>::value,
      "Return type mismatch");
    static_assert(std::is_same<guts::typelist::typelist<ParameterTypes...>, typename guts::infer_function_traits_t<typename Op::schema>::parameter_types>::value,
      "Parameter types mismatch");

    static ReturnType call(ParameterTypes... args) {
        auto op = c10::Dispatcher::singleton()
            // TODO: figure out how to make compiler happy without dynamic casts
            .findSchemaOrThrow((const char*) Op::name, (const char*) Op::overload_name)
            //.findSchemaOrThrow("a", "b")
            .typed<ReturnType (ParameterTypes...)>();
        return c10::impl::BoxedKernelWrapper<ReturnType (ParameterTypes...)>::call(
            c10::BoxedKernel::makeFromFunction<fallback_fn>(),
            op,
            c10::DispatchKeySet(), // we know that the cpu_fallback doesn't use the dispatch keyset.
            //std::forward<ParameterTypes...>(args...)
            // TODO: get std::forward<> to work
            args...
            );
    }
};

template<c10::KernelFunction::BoxedKernelFunction* fallback_fn, class Op>
using call_fallback_fn = _call_fallback_fn<fallback_fn, Op, typename Op::schema>;

} // namespace native
} // namespace at
