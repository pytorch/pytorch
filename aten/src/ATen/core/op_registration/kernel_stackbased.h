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
  struct KernelRegistrationConfigParameter final {
    explicit constexpr KernelRegistrationConfigParameter(KernelFunction* kernel_func, KernelCacheCreatorFunction* cache_creator_func)
    : kernel_func_(kernel_func), cache_creator_func_(std::move(cache_creator_func)) {
    }

    void apply(KernelRegistrationConfig* registration) const {
      registration->kernel_func = kernel_func_;
      registration->cache_creator_func = cache_creator_func_;
    }

  private:
    KernelFunction* kernel_func_;
    KernelCacheCreatorFunction* cache_creator_func_;
  };

  static_assert(is_registration_config_parameter<KernelRegistrationConfigParameter>::value, "KernelRegistrationConfigParameter must fulfill the registration config parameter concept");
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
inline constexpr detail::KernelRegistrationConfigParameter kernel(KernelFunction* kernel_func, KernelCacheCreatorFunction* cache_creator) {
  return detail::KernelRegistrationConfigParameter(kernel_func, cache_creator);
}

}
