#pragma once

/**
 * This file implements c10::kernel(stack_based_kernel) which is used in the
 * kernel registration API to set the dispatch key for a registered kernel.
 * You probably don't want to use this API, stack based kernels are internal
 * only. There's other, better kernel APIs which are built on top of this one.
 *
 * You probably don't want to include this file directly but include
 * op_registration.h instead since that adds more functionality you'll
 * likely need to register your operators.
 */

#include <ATen/core/op_registration/base.h>

namespace c10 {

namespace detail {

  struct NoFunctionSchemaInference final {
    std::unique_ptr<FunctionSchema> operator()() const {
      return nullptr;
    }
  };

  template<class KernelCacheCreatorFunction_, class InferFunctionSchemaFunction>
  struct KernelRegistrationConfigParameter final {
    template<class KernelCacheCreatorFunction__>
    constexpr KernelRegistrationConfigParameter(KernelFunction* kernel_func, KernelCacheCreatorFunction__&& cache_creator_func, InferFunctionSchemaFunction&& infer_function_schema_func)
    : kernel_func_(kernel_func)
    , cache_creator_func_(std::forward<KernelCacheCreatorFunction__>(cache_creator_func))
    , infer_function_schema_func_(std::forward<InferFunctionSchemaFunction>(infer_function_schema_func)) {
    }

    void apply(KernelRegistrationConfig* registration) const & {
      registration->kernel_func = kernel_func_;
      registration->cache_creator_func = cache_creator_func_;
      registration->inferred_function_schema = infer_function_schema_func_();
    }

    void apply(KernelRegistrationConfig* registration) && {
      registration->kernel_func = kernel_func_;
      registration->cache_creator_func = std::move(cache_creator_func_);
      registration->inferred_function_schema = std::move(infer_function_schema_func_)();
    }

  private:
    KernelFunction* kernel_func_;
    KernelCacheCreatorFunction_ cache_creator_func_;
    InferFunctionSchemaFunction infer_function_schema_func_;
  };

  static_assert(is_registration_config_parameter<KernelRegistrationConfigParameter<KernelCacheCreatorFunction, NoFunctionSchemaInference>>::value, "KernelRegistrationConfigParameter must fulfill the registration config parameter concept");
}

/**
 * Use this to register an operator whose kernel is implemented by a stack
 * based function. This is meant to be used internally, for example for writing
 * wrappers for other ways of writing operators. This is not part of the
 * public API.
 *
 * Example:
 *
 * > namespace {
 * >   void my_kernel_cpu(Stack* stack, KernelCache* cache) {...}
 * >   unique_ptr<KernelCache> my_cache_creator() {...}
 * > }
 * >
 * > static auto registry = c10::RegisterOperators()
 * >     .op("my_op",
 * >         c10::kernel(&my_kernel_cpu, &my_cache_creator),
 * >         c10::dispatchKey(CPUTensorId()));
 */
template<class KernelCacheCreatorFunction_>
inline constexpr detail::KernelRegistrationConfigParameter<guts::decay_t<KernelCacheCreatorFunction_>, detail::NoFunctionSchemaInference> kernel(KernelFunction* kernel_func, KernelCacheCreatorFunction_&& cache_creator) {
  static_assert(detail::is_registration_config_parameter<detail::KernelRegistrationConfigParameter<guts::decay_t<KernelCacheCreatorFunction_>, detail::NoFunctionSchemaInference>>::value, "KernelRegistrationConfigParameter must fulfill the registration config parameter concept");

  return {kernel_func, std::forward<KernelCacheCreatorFunction_>(cache_creator), detail::NoFunctionSchemaInference()};
}

}
