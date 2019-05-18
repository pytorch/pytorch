#pragma once

/**
 * Include this file if you want to register operators. It includes all
 * functionality needed to do so for you.
 */

#include <ATen/core/dispatch/Dispatcher.h>
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
 * >     .op("my_op", c10::RegisterOperators::options()
 * >         .kernel<my_kernel_cpu>()
 * >         .dispatchKey(CPUTensorId()));
 */
class CAFFE2_API RegisterOperators final {
public:
  RegisterOperators();
  ~RegisterOperators();

  RegisterOperators(const RegisterOperators&) = delete;
  RegisterOperators& operator=(const RegisterOperators&) = delete;
  RegisterOperators(RegisterOperators&&) noexcept;
  RegisterOperators& operator=(RegisterOperators&&) noexcept;

  class CAFFE2_API Options final {
  public:
    Options(const Options&) = delete;
    Options(Options&&) noexcept = delete;
    Options& operator=(const Options&) = delete;
    Options& operator=(Options&&) noexcept = delete;

    // internal-only for registering stack based kernels
    Options&& kernel(KernelFunction* kernel_func, KernelCacheCreatorFunction&& cache_creator) && {
      return std::move(*this).kernel(kernel_func, std::move(cache_creator), nullptr);
    }

    /**
     * Use this to register an operator whose kernel is implemented as a functor
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
     * >     .op("my_op", c10::RegisterOperators::options()
     * >         .kernel<my_kernel_cpu>()
     * >         .dispatchKey(CPUTensorId()));
     *
     * The functor constructor can take arguments to configure the kernel.
     * The arguments are defined in the kernel registration.
     * Example:
     *
     * > namespace {
     * >   class my_kernel_cpu final : public c10::OperatorKernel {
     * >   public:
     * >     explicit my_kernel_cpu(std::string some_configuration, int a, bool b)
     * >         : ... {...}
     * >
     * >     Tensor operator()(Tensor a, Tensor b) {...}
     * >   };
     * > }
     * >
     * > static auto registry = c10::RegisterOperators()
     * >     .op("my_op", c10::RegisterOperators::options()
     * >         .kernel<my_kernel_cpu>("some_configuration", 3, true)
     * >         .dispatchKey(CPUTensorId()));
     */
    template<class KernelFunctor, class... ConstructorParameters>
    // enable_if: only enable it if KernelFunctor is actually a functor
    guts::enable_if_t<guts::is_functor<KernelFunctor>::value, Options&&> kernel(ConstructorParameters&&... constructorParameters) {
      static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to register a kernel functor using the kernel<Functor>() API, but it doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");
      static_assert(std::is_constructible<KernelFunctor, ConstructorParameters...>::value, "Wrong argument list for constructor of kernel functor. The arguments to kernel<Functor>(arguments...) must match one of the constructors of Functor.");

      return std::move(*this).kernelFunctor<KernelFunctor, false>(std::forward<ConstructorParameters>(constructorParameters)...);
    }

    /**
     * Use this to register an operator whose kernel is implemented by a function:
     *
     * Example:
     *
     * > namespace { Tensor my_kernel_cpu(Tensor a, Tensor b) {...} }
     * >
     * > static auto registry = c10::RegisterOperators()
     * >     .op("my_op", c10::RegisterOperators()
     * >         .kernel<decltype(my_kernel_cpu), &my_kernel_cpu>()
     * >         .dispatchKey(CPUTensorId()));
     */
    template<class FuncType, FuncType* kernel_func>
    // enable_if: only enable it if FuncType is actually a function
    guts::enable_if_t<guts::is_function_type<FuncType>::value, Options&&> kernel() {
      static_assert(!std::is_same<FuncType, KernelFunction>::value, "Tried to register a stackbased (i.e. internal) kernel function using the public kernel<...>() API. Please either use the internal kernel(...) API or also implement the kernel function as defined by the public API.");

      return kernel<typename detail::WrapKernelFunction<FuncType, kernel_func>::type>();
    }

    /**
     * Use this to register an operator whose kernel is implemented as a stateless lambda.
     *
     * Example:
     *
     * > static auto registry = c10::RegisterOperators()
     * >     .op("my_op", c10::RegisterOperators::options()
     * >         .kernel([] (Tensor a) -> Tensor {...})
     * >         .dispatchKey(CPUTensorId()));
     */
    template<class Lambda>
    // enable_if: only enable it if Lambda is a functor (note: lambdas are functors)
    guts::enable_if_t<guts::is_functor<guts::decay_t<Lambda>>::value, Options&&> kernel(Lambda&& functor) {
      static_assert(!std::is_base_of<OperatorKernel, Lambda>::value, "The kernel(x) API for registering a kernel is only meant to be used with lambdas. Your kernel is a functor. Please use the kernel<Functor>() API instead.");

      // We don't support stateful lambdas (i.e. lambdas with a capture), because their
      // behavior would be nonobvious. A functor kernel with cache gets a new instance of
      // its cache each time the kernel is looked up from the dispatch table.
      // A lambda with a capture would be global and share its capture between all kernel lookups.
      // So, instead of making users having to think about it (including the thread-safety
      // issues this causes), let's just forbid stateful lambdas alltogether.
      static_assert(guts::is_stateless_lambda<guts::decay_t<Lambda>>::value, "The kernel(x) API for registering a kernel only works for stateless lambdas (i.e. lambdas without captures). If you need a cache, please use the functor based API kernel<Functor>() instead.");

      return std::move(*this).kernelFunctor<detail::WrapRuntimeKernelFunctor<guts::decay_t<Lambda>>>(std::forward<Lambda>(functor));
    }

    /**
     * Use this to register an operator with a kernel for a certain dispatch key.
     *
     * Example:
     *
     * > namespace {
     * >   class my_kernel_cpu final : public c10::OperatorKernel {
     * >   public:
     * >     Tensor operator()(Tensor a, Tensor b) {...}
     * >   };
     * >   class my_kernel_cuda final : public c10::OperatorKernel {
     * >   public:
     * >     Tensor operator()(Tensor a, Tensor b) {...}
     * >   };
     * > }
     * >
     * > static auto registry = c10::RegisterOperators()
     * >     .op("my_op", c10::RegisterOperators::options()
     * >         .kernel<my_kernel_cpu>()
     * >         .dispatchKey(CPUTensorId()))
     * >     .op("my_op", c10::RegisterOperators::options()
     * >         .kernel<my_kernel_cuda>()
     * >         .dispatchKey(CUDATensorId()));
     */
    Options&& dispatchKey(TensorTypeId dispatch_key) && {
      config.dispatch_key = dispatch_key;
      return std::move(*this);
    }

  private:
    Options&& kernel(KernelFunction* kernel_func, KernelCacheCreatorFunction&& cache_creator, std::unique_ptr<FunctionSchema>&& inferred_function_schema) && {
      config.kernel_func = kernel_func;
      config.cache_creator_func = std::move(cache_creator);
      config.inferred_function_schema = std::move(inferred_function_schema);
      return std::move(*this);
    }

    template<class KernelFunctor, bool AllowDeprecatedTypes = false, class... ConstructorParameters>
    Options&& kernelFunctor(ConstructorParameters&&... constructorParameters) && {
      return std::move(*this).kernel(
        &detail::wrap_kernel_functor<KernelFunctor, AllowDeprecatedTypes>::call,
        detail::KernelFactory<KernelFunctor, guts::decay_t<ConstructorParameters>...>(std::forward<ConstructorParameters>(constructorParameters)...),
        detail::FunctionSchemaInferer<KernelFunctor>()()
      );
    }

    Options() = default;

    // KernelRegistrationConfig accumulates all information from the config
    // parameters passed to a RegisterOperators::op() call into one object.
    struct KernelRegistrationConfig final {
      KernelRegistrationConfig()
        : dispatch_key(c10::nullopt)
        , kernel_func(nullptr)
        , cache_creator_func(nullptr)
        , inferred_function_schema(nullptr)
      {}

      c10::optional<TensorTypeId> dispatch_key;
      KernelFunction* kernel_func;
      KernelCacheCreatorFunction cache_creator_func;
      std::unique_ptr<FunctionSchema> inferred_function_schema;
    };

    KernelRegistrationConfig config;
    friend class RegisterOperators;
  };

  /**
   * Call this to get an instance of registration options, which
   * can be passed to a call to RegisterOperators::op() to specify
   * these options for the operator registration.
   * See class doc comment for examples.
   */
  static Options options() {
    return {};
  }

  /**
   * Call this to register an operator. See class doc comment for examples.
   */
  RegisterOperators op(const std::string& schemaOrName, Options&& options = {}) && {
    checkSchemaAndRegisterOp_(schemaOrName, std::move(options));
    return std::move(*this);
  }

  // internal only for registering caffe2 ops
  RegisterOperators op(FunctionSchema schema, Options&& options) && {
    checkSchemaAndRegisterOp_(std::move(schema), std::move(options));
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
   * >     .op("my_op", c10::RegisterOperators::options().kernel<decltype(my_kernel_cpu), &my_kernel_cpu>());
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
   * >     .op("my_op", c10::RegisterOperators::options().kernel<my_kernel_cpu>());
   */
   template<class FuncType, class...  OtherArgs>
   C10_DEPRECATED_MESSAGE("Registering kernels via passing function pointers to op() directly is deprecated. " \
                          "Please use the new RegisterOperators::options().kernel() based API instead.")
   // enable_if: only enable it if FuncType is actually a function, but not a stack based KernelFunction.
   guts::enable_if_t<guts::is_function_type<FuncType>::value && !std::is_same<FuncType, KernelFunction>::value, RegisterOperators>
   op(const std::string& schemaOrName, FuncType* func, OtherArgs...) && {
     // We intentionally don't extend this deprecated API to support dispatch keys
     // and the like to push people towards using the new API.
     static_assert(sizeof...(OtherArgs) == 0, "The deprecated function pointer based API to register kernels doesn't allow additional arguments for dispatch keys or other things. Please use the new RegisterOperators::options().kernel() based API instead.");

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
    * >     .op("my_op", c10::RegisterOperators::options().kernel([] (Tensor a, Tensor b) {...}));
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
    * >     .op("my_op", c10::RegisterOperators::options().kernel<my_kernel_cpu>());
    */
    template<class FuncType, class...  OtherArgs>
    C10_DEPRECATED_MESSAGE("Registering kernels via passing lambdas to op() directly is deprecated. " \
                           "Please use the new RegisterOperators::options().kernel() based API instead.")
    // enable_if: only enable it if FuncType is actually a functor
    guts::enable_if_t<guts::is_functor<FuncType>::value, RegisterOperators>
    op(const std::string& schemaOrName, FuncType&& func, OtherArgs...) && {
      // We intentionally don't extend this deprecated API to support dispatch keys
      // and the like to push people towards using the new API.
      static_assert(sizeof...(OtherArgs) == 0, "The deprecated lambda based API to register kernels doesn't allow additional arguments for dispatch keys or other things. Please use the new RegisterOperators::options().kernel() based API instead.");

      static_assert(!std::is_base_of<OperatorKernel, FuncType>::value, "c10::OperatorKernel is part of the new kernel registration API and shouldn't be used together with the deprecated registration API. Please use the new RegisterOperators::options().kernel() based API instead.");

      legacyAPIOp_(schemaOrName, std::forward<FuncType>(func));
      return std::move(*this);
    }

private:
  template<class FuncType>
  void legacyAPIOp_(const std::string& schemaOrName, FuncType&& func) {
    constexpr bool AllowLegacyTypes = true;
    checkSchemaAndRegisterOp_(schemaOrName, options().kernelFunctor<detail::WrapRuntimeKernelFunctor<guts::decay_t<FuncType>>, AllowLegacyTypes>(std::forward<FuncType>(func)));
  }

  void checkSchemaAndRegisterOp_(FunctionSchema&& schema, Options&& config);
  void checkSchemaAndRegisterOp_(const std::string& schemaOrName, Options&& config);

  void registerOp_(FunctionSchema&& schema, Options&& config);

  class OperatorRegistrar;

  std::vector<OperatorRegistrar> registrars_;

  static_assert(std::is_nothrow_move_constructible<std::vector<OperatorRegistrar>>::value, "");
  static_assert(std::is_nothrow_move_assignable<std::vector<OperatorRegistrar>>::value, "");
};

}
