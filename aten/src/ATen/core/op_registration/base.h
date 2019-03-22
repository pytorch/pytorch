#pragma once

/**
 * This file sets up the basics for operator registration.
 *
 * You probably don't want to include this file directly but include
 * op_registration.h instead since that adds more functionality you'll
 * likely need to register your operators.
 */

#include <ATen/core/dispatch/Dispatcher.h>

namespace c10 {

namespace detail {

  // KernelRegistrationConfig accumulates all information from the config
  // parameters passed to a RegisterOperators::op() call into one object.
  struct KernelRegistrationConfig final {
    c10::optional<TensorTypeId> dispatch_key = c10::nullopt;
    KernelFunction* kernel_func = nullptr;
    KernelCacheCreatorFunction cache_creator_func = nullptr;
    std::unique_ptr<FunctionSchema> inferred_function_schema = nullptr;
  };

  // Take a list of configuration parameters and return a
  // KernelRegistrationConfig accumulating all their configurations.
  template<class... ConfigParameters>
  KernelRegistrationConfig make_registration_config(ConfigParameters&&... configParameters) {
    KernelRegistrationConfig config;

    // apply all configParameters
    (void)std::initializer_list<int>{(std::move(configParameters).apply(&config), 0)...};

    return config;
  }
}

}
