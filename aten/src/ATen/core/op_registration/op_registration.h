#pragma once

/**
 * Include this file if you want to register operators. It includes all
 * functionality needed to do so for you.
 */

#include <ATen/core/op_registration/base.h>
#include <ATen/core/op_registration/dispatch_key.h>
#include <ATen/core/op_registration/kernel_stackbased.h>
#include <ATen/core/op_registration/kernel_functor.h>
#include <ATen/core/op_registration/kernel_function.h>
#include <ATen/core/op_registration/kernel_function_legacy.h>
#include <ATen/core/op_registration/infer_schema.h>

namespace c10 {

/**
 * An instance of this class handles the registration for one or more operators.
 * Make sure you keep the RegisterOperators instance around since it will
 * deregister the operator it's responsible for in its destructor.
 *
 * Example:
 *
 * > namespace {
 * >   class my_kernel_cpu final : public c10::OperatorKernel {
 * >   public:
 * >     Tensor operator()(Tensor a, Tensor b) {...}
 * >   };
 * > }
 * >
 * > static auto registry = c10::RegisterOperators()
 * >     .op("my_op",
 * >         c10::kernel<my_kernel_cpu>(),
 * >         c10::dispatchKey(CPUTensorId()));
 */
class C10_API RegisterOperators final {
public:
  RegisterOperators() = default;
  RegisterOperators(const RegisterOperators&) = delete;
  RegisterOperators(RegisterOperators&&) = default;
  RegisterOperators& operator=(const RegisterOperators&) = delete;
  RegisterOperators& operator=(RegisterOperators&&) = default;

  /**
   * Register an operator based on a function schema and a set of configuration
   * parameters (i.e. kernel function, dispatch key, ...).
   *
   * Example:
   *
   * > namespace {
   * >   class my_kernel_cpu final : public c10::OperatorKernel {
   * >   public:
   * >     Tensor operator()(Tensor a, Tensor b) {...}
   * >   };
   * > }
   * >
   * > static auto registry = c10::RegisterOperators()
   * >     .op("my_op",
   * >         c10::kernel<my_kernel_cpu>(),
   * >         c10::dispatchKey(CPUTensorId()));
   */
  template<class... ConfigParameters>
  RegisterOperators op(FunctionSchema schema, ConfigParameters&&... configParameters) && {
    detail::KernelRegistrationConfig config = make_registration_config(configParameters...);

    if (config.inferred_function_schema.get() != nullptr) {
      assertSchemasHaveSameSignature(*config.inferred_function_schema, schema);
    }

    registrars_.emplace_back(std::move(schema), config.dispatch_key, config.kernel_func, std::move(config.cache_creator_func));
    return std::move(*this);
  }

  // TODO allow input schema to be just the operator name + overload name, in that case use schema generated from kernel function
  // TODO if schema is fully specified, still generate schema from kernel function and make sure it's correct

  // TODO error if dispatch key is not specified

  // Deprecated. For backwards compatibility only.
  // Don't use this, it introduces a performance overhead on each kernel call
  // due to the kernel being stored in the wrapper as a runtime function pointer.
  template<class KernelFunc>
  RegisterOperators op(FunctionSchema schema, KernelFunc* func) && {
    return op(std::move(schema), kernel<detail::WrapKernelFunctionRuntime>(func));
  }

  // TODO Add deprecated lambda-based API

private:
  std::vector<detail::OperatorRegistrar> registrars_;
};

}
