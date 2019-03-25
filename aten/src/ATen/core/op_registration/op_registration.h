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
  RegisterOperators();
  RegisterOperators(RegisterOperators&&) noexcept;
  RegisterOperators& operator=(RegisterOperators&&) noexcept;
  RegisterOperators(const RegisterOperators&) = delete;
  RegisterOperators& operator=(const RegisterOperators&) = delete;
  ~RegisterOperators();

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
    registerOp_(std::move(schema), make_registration_config(configParameters...));
    return std::move(*this);
  }

  // TODO allow input schema to be just the operator name + overload name, in that case use schema generated from kernel function

  /**
   * Deprecated. For backwards compatibility only.
   * Don't use this, it introduces a performance overhead on each kernel call
   * due to the kernel being stored in the wrapper as a runtime function pointer.
   *
   * Given a kernel
   *
   * > namespace { Tensor my_kernel_cpu(Tensor a, Tensor b) {...} }
   *
   * This deprecated API looks like:
   *
   * > static auto registry = c10::RegisterOperators()
   * >     .op("my_op", &my_kernel_cpu);
   *
   * But you should use the new API instead:
   *
   * > static auto registry = c10::RegisterOperators()
   * >     .op("my_op", kernel<decltype(my_kernel_cpu), &my_kernel_cpu>());
   *
   * Or, alternatively, write your kernel as a functor:
   *
   * > namespace {
   * >   class my_kernel_cpu final : public c10::OperatorKernel {
   * >   public:
   * >     Tensor operator()(Tensor a, Tensor b) {...}
   * >   };
   * > }
   * >
   * > static auto registry = c10::RegisterOperators()
   * >     .op("my_op", c10::kernel<my_kernel_cpu>());
   */
  template<class KernelFunc>
  RegisterOperators op(FunctionSchema schema, KernelFunc* func) && {
    // We intentionally don't extend this deprecated API to support dispatch keys
    // and the like to push people towards using the new API.
    return std::move(*this).op(std::move(schema), kernel<detail::WrapKernelFunctionRuntime<KernelFunc>>(func));
  }

  // TODO Add deprecated lambda-based API

private:
  void registerOp_(FunctionSchema&& schema, detail::KernelRegistrationConfig&& config);

  class OperatorRegistrar;

  std::vector<OperatorRegistrar> registrars_;
};

}
