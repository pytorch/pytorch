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
#include <ATen/core/op_registration/kernel_lambda.h>
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
  ~RegisterOperators();

  RegisterOperators(const RegisterOperators&) = delete;
  RegisterOperators& operator=(const RegisterOperators&) = delete;
  RegisterOperators(RegisterOperators&&) noexcept;
  RegisterOperators& operator=(RegisterOperators&&);

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
  RegisterOperators op(const std::string& schemaOrName, ConfigParameters&&... configParameters) && {
    static_assert(guts::conjunction<detail::is_registration_config_parameter<guts::decay_t<ConfigParameters>>...>::value,
      "Invalid argument passed to op(). Examples for valid arguments are c10::kernel(...) for defining a kernel "
      " and c10::dispatchKey(...) for defining a dispatch key. Please see the documentation for registering c10 operators.");

    op_(schemaOrName, std::forward<ConfigParameters>(configParameters)...);
    return std::move(*this);
  }

  // This FunctionSchema based variant is only meant to be used for internal
  // purposes when we already have a pre-parsed FunctionSchema.
  // This is for example used for exposing legacy caffe2 operators to c10.
  template<class... ConfigParameters>
  RegisterOperators op(FunctionSchema schema, ConfigParameters&&... configParameters) && {
    static_assert(guts::conjunction<detail::is_registration_config_parameter<guts::decay_t<ConfigParameters>>...>::value,
      "Invalid argument passed to op(). Examples for valid arguments are c10::kernel(...) for defining a kernel "
      " and c10::dispatchKey(...) for defining a dispatch key. Please see the documentation for registering c10 operators.");

    op_(std::move(schema), std::forward<ConfigParameters>(configParameters)...);
    return std::move(*this);
  }

  template<class FuncType>
  C10_DEPRECATED_MESSAGE("Registering kernels via passing arguments to RegisterOperators(...) is deprecated. " \
                         "Please use RegisterOperators().op(...) instead.")
  // enable_if: only enable it if FuncType is actually a function, but not a stack based KernelFunction.
  explicit RegisterOperators(guts::enable_if_t<guts::is_function_type<FuncType>::value && !std::is_same<FuncType, KernelFunction>::value, const std::string&> schemaOrName, FuncType* func)
  : RegisterOperators() {
    legacyAPIOp_(schemaOrName, func);
  }

  template<class FuncType>
  C10_DEPRECATED_MESSAGE("Registering kernels via passing arguments to RegisterOperators(...) is deprecated. " \
                         "Please use RegisterOperators().op(...) instead.")
  // enable_if: only enable it if FuncType is actually a functor
  explicit RegisterOperators(guts::enable_if_t<guts::is_functor<FuncType>::value, const std::string&> schemaOrName, FuncType&& func)
  : RegisterOperators() {
    legacyAPIOp_(schemaOrName, std::forward<FuncType>(func));
  }

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
   template<class FuncType, class...  OtherArgs>
   C10_DEPRECATED_MESSAGE("Registering kernels via passing function pointers to op() directly is deprecated. " \
                          "Please use the new c10::kernel() based API instead.")
   // enable_if: only enable it if FuncType is actually a function, but not a stack based KernelFunction.
   guts::enable_if_t<guts::is_function_type<FuncType>::value && !std::is_same<FuncType, KernelFunction>::value, RegisterOperators>
   op(const std::string& schemaOrName, FuncType* func, OtherArgs...) && {
     // We intentionally don't extend this deprecated API to support dispatch keys
     // and the like to push people towards using the new API.
     static_assert(sizeof...(OtherArgs) == 0, "The deprecated function pointer based API to register kernels doesn't allow additional arguments for dispatch keys or other things. Please use the new c10::kernel() based API instead.");

     legacyAPIOp_(schemaOrName, func);
     return std::move(*this);
   }

   /**
    * Deprecated. For backwards compatibility only.
    *
    * This deprecated API looks like:
    *
    * > static auto registry = c10::RegisterOperators()
    * >     .op("my_op", [] (Tensor a, Tensor b) {...});
    *
    * But you should use the new API instead:
    *
    * > static auto registry = c10::RegisterOperators()
    * >     .op("my_op", kernel([] (Tensor a, Tensor b) {...}));
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
    template<class FuncType, class...  OtherArgs>
    C10_DEPRECATED_MESSAGE("Registering kernels via passing lambdas to op() directly is deprecated. " \
                           "Please use the new c10::kernel() based API instead.")
    // enable_if: only enable it if FuncType is actually a functor
    guts::enable_if_t<guts::is_functor<FuncType>::value, RegisterOperators>
    op(const std::string& schemaOrName, FuncType&& func, OtherArgs...) && {
      // We intentionally don't extend this deprecated API to support dispatch keys
      // and the like to push people towards using the new API.
      static_assert(sizeof...(OtherArgs) == 0, "The deprecated lambda based API to register kernels doesn't allow additional arguments for dispatch keys or other things. Please use the new c10::kernel() based API instead.");

      static_assert(!std::is_base_of<OperatorKernel, FuncType>::value, "c10::OperatorKernel is part of the new kernel registration API and shouldn't be used together with the deprecated registration API. Please use the new c10::kernel() based API instead.");

      legacyAPIOp_(schemaOrName, std::forward<FuncType>(func));
      return std::move(*this);
    }

private:
  template<class... ConfigParameters>
  void op_(FunctionSchema&& schema, ConfigParameters&&... configParameters) {
    checkSchemaAndRegisterOp_(std::move(schema), detail::make_registration_config(std::forward<ConfigParameters>(configParameters)...));
  }
  template<class... ConfigParameters>
  void op_(const std::string& schemaOrName, ConfigParameters&&... configParameters) {
    checkSchemaAndRegisterOp_(schemaOrName, detail::make_registration_config(std::forward<ConfigParameters>(configParameters)...));
  }

  template<class FuncType>
  void legacyAPIOp_(const std::string& schemaOrName, FuncType&& func) {
    op_(schemaOrName, kernel<detail::WrapRuntimeKernelFunctor<guts::decay_t<FuncType>>>(std::forward<FuncType>(func)));
  }

  void checkSchemaAndRegisterOp_(FunctionSchema&& schema, detail::KernelRegistrationConfig&& config);
  void checkSchemaAndRegisterOp_(const std::string& schemaOrName, detail::KernelRegistrationConfig&& config);

  void registerOp_(FunctionSchema&& schema, detail::KernelRegistrationConfig&& config);

  class OperatorRegistrar;

  std::vector<OperatorRegistrar> registrars_;
};

}
