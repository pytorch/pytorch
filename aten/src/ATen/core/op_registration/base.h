#pragma once

/**
 * This file sets up the basics for operator registration like the
 * c10::RegisterOperators() class.
 *
 * You probably don't want to include this file directly but include
 * op_registration.h instead since that adds more functionality you'll
 * likely need to register your operators.
 */

#include <ATen/core/dispatch/Dispatcher.h>

namespace c10 {

namespace detail {

  // OperatorRegistrar in its constructor registers an operator in the dispatch
  // table deregisters it in the destructor. The intent is that this class is
  // constructed at static initialization time so that operators automatically
  // get registered when a dlopen() occurs.
  // You shouldn't call this directly; instead, use the RegisterOperators class.
  class OperatorRegistrar final {
  public:
    explicit OperatorRegistrar(FunctionSchema&& schema, TensorTypeId dispatch_key, KernelFunction* kernel, KernelCacheCreatorFunction&& cache_creator)
    : op_(Dispatcher::singleton().registerSchema(std::move(schema))), dispatch_key_(std::move(dispatch_key)), owns_registration_(true) {
      Dispatcher::singleton().registerKernel(op_, dispatch_key_, kernel, std::move(cache_creator));
    }

    OperatorRegistrar(OperatorRegistrar&& rhs) noexcept
    :  op_(std::move(rhs.op_)), dispatch_key_(std::move(rhs.dispatch_key_)), owns_registration_(rhs.owns_registration_) {
      rhs.owns_registration_ = false;
    }

    // not needed and would break RAII if defaulted.
    OperatorRegistrar& operator=(OperatorRegistrar&& rhs) noexcept = delete;
    OperatorRegistrar(const OperatorRegistrar& rhs) = delete;
    OperatorRegistrar& operator=(const OperatorRegistrar& rhs) = delete;

    ~OperatorRegistrar() {
      if (owns_registration_) {
        Dispatcher::singleton().deregisterKernel(op_, dispatch_key_);
        Dispatcher::singleton().deregisterSchema(op_);
      }
    }

  private:
    const OperatorHandle op_;
    const TensorTypeId dispatch_key_;
    bool owns_registration_;
  };

  // KernelRegistrationConfig accumulates all information from the config
  // parameters passed to a RegisterOperators::op() call into one object.
  struct KernelRegistrationConfig final {
    TensorTypeId dispatch_key;
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

    // TODO Allow this for just registering the schema?
    AT_CHECK(config.kernel_func != nullptr, "Cannot register operator without kernel");

    // if kernel_func is set, so must be cache_creator_func,
    // the API shouldn't allow anything else.
    AT_ASSERT(static_cast<bool>(config.cache_creator_func));

    return config;
  }
}

}
