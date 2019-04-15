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

  // is_registration_config_parameter is a concept that returns true_type iff its argument is
  // a valid parameter to be passed to c10::RegisterOperators().op(parameters...)
  // That is, it must have an apply method that takes a KernelRegistrationConfig*.
  template<class ConfigParameter, class Enable = void>
  struct is_registration_config_parameter : std::false_type {
    static_assert(std::is_same<ConfigParameter, guts::decay_t<ConfigParameter>>::value, "is_registration_config_parameter doesn't work with reference types");
  };
  template<class ConfigParameter>
  struct is_registration_config_parameter<ConfigParameter, guts::void_t<decltype(
    std::declval<ConfigParameter>().apply(std::declval<KernelRegistrationConfig*>()),
    std::declval<const ConfigParameter&>().apply(std::declval<KernelRegistrationConfig*>())
  )>> : std::true_type {
    static_assert(std::is_same<ConfigParameter, guts::decay_t<ConfigParameter>>::value, "is_registration_config_parameter doesn't work with reference types");
  };
  static_assert(!is_registration_config_parameter<KernelRegistrationConfig>::value, "For classes that aren't registration parameters, this concept should return false");
  // note: the corresponding asserts that the concept returns true are next to the definition of the corresponding classes

  // Take a list of configuration parameters and return a
  // KernelRegistrationConfig accumulating all their configurations.
  template<class... ConfigParameters>
  KernelRegistrationConfig make_registration_config(ConfigParameters&&... configParameters) {
    static_assert(guts::conjunction<is_registration_config_parameter<guts::decay_t<ConfigParameters>>...>::value, "One of the parameters isn't a valid registration config parameter.");

    KernelRegistrationConfig config;

    // apply all configParameters
    (void)std::initializer_list<int>{(std::forward<ConfigParameters>(configParameters).apply(&config), 0)...};

    return config;
  }
}

}
