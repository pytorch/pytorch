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
  template<class Cache>
  inline std::unique_ptr<c10::KernelCache> cacheCreator() {
    static_assert(std::is_default_constructible<Cache>::value, "Cache class must be default constructible");
    return guts::make_unique<Cache>();
  }

  template<>
  inline std::unique_ptr<c10::KernelCache> cacheCreator<void>() {
    return nullptr;
  }

  // TODO If this was templated on KernelCacheCreatorFunction, it (and c10::kernel()) could be constexpr
  struct KernelRegistrationConfigParameter final {
    explicit KernelRegistrationConfigParameter(KernelFunction* kernel_func, KernelCacheCreatorFunction cache_creator_func)
    : kernel_func_(kernel_func), cache_creator_func_(std::move(cache_creator_func)) {
    }

    void apply(KernelRegistrationConfig* registration) && {
      registration->kernel_func = kernel_func_;
      registration->cache_creator_func = std::move(cache_creator_func_);
    }

  private:
    KernelFunction* kernel_func_;
    KernelCacheCreatorFunction cache_creator_func_;
  };
}

/**
 * Use this to register an operator whose kernel is implemented by a stack
 * based function. This is meant to be used internally, for example for writing
 * wrappers for other ways of writing operators. This is not part of the
 * public API.
 *
 * Example:
 *
 * > namespace { void my_kernel_cpu(Stack* stack, KernelCache* cache) {...} }
 * >
 * > static auto registry = c10::RegisterOperators()
 * >     .op("my_op",
 * >         c10::kernel(my_kernel_cpu),
 * >         c10::dispatchKey(CPUTensorId()));
 */
inline detail::KernelRegistrationConfigParameter kernel(KernelFunction* kernel_func) {
  return detail::KernelRegistrationConfigParameter(kernel_func, &detail::cacheCreator<void>);
}


}
