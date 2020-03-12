#pragma once

/**
 * Include this file if you want to register operators. It includes all
 * functionality needed to do so for you.
 */

#include <c10/core/DispatchKey.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/op_registration/infer_schema.h>
#if defined(EXPOSE_C2_OPS) || !defined(CAFFE2_IS_XPLAT_BUILD)
#include <torch/csrc/jit/frontend/function_schema_parser.h>
#endif
#include <ATen/core/OpsAlreadyMovedToC10.h>

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
 * >     .op(c10::RegisterOperators::options()
 * >         .schema("my_op")
 * >         .kernel<my_kernel_cpu>(DispatchKey::CPUTensorId));
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
    template<KernelFunction::BoxedKernelFunction* kernel_func>
    Options&& kernel(DispatchKey dispatch_key) && {
      return std::move(*this).kernel(dispatch_key, KernelFunction::makeFromBoxedFunction<kernel_func>(), nullptr);
    }

    // internal-only for registering stack based catch-all kernels
    template<KernelFunction::BoxedKernelFunction* kernel_func>
    Options&& catchAllKernel() && {
      return std::move(*this).kernel(c10::nullopt, KernelFunction::makeFromBoxedFunction<kernel_func>(), nullptr);
    }

    // internal only for registering caffe2 ops
    Options&& schema(FunctionSchema&& schema) {
        TORCH_CHECK(!schemaOrName_.has_value(), "You can only specify the schema once per operator registration.");
        schemaOrName_ = c10::make_right<OperatorName, FunctionSchema>(std::move(schema));
        return std::move(*this);
    }

    /**
     * Use this to specify the schema for an operator. You can also specify
     * the operator name only to have the function signature part of the
     * schema be inferred from the kernel function.
     *
     * Example:
     *
     * > // Infer function signature from my_kernel_cpu
     * > static auto registry = c10::RegisterOperators()
     * >     .op(c10::RegisterOperators::options()
     * >         .schema("my_op")
     * >         .kernel<my_kernel_cpu>(DispatchKey::CPUTensorId));
     * >
     * >
     * > // Explicitly specify full schema
     * > static auto registry = c10::RegisterOperators()
     * >     .op(c10::RegisterOperators::options()
     * >         .schema("my_op(Tensor a) -> Tensor")
     * >         .kernel<my_kernel_cpu>(DispatchKey::CPUTensorId));
     */
    Options&& schema(const std::string& schemaOrName) {
      TORCH_CHECK(!schemaOrName_.has_value(), "Tried to register operator ", schemaOrName," but specified schema multiple times. You can only specify the schema once per operator registration.");

      #if !defined(EXPOSE_C2_OPS) && defined(CAFFE2_IS_XPLAT_BUILD)
        throw std::logic_error("Tried to register operator " + schemaOrName + ". We don't support registering c10 ops on mobile yet because the function schema parser isn't present in the mobile build.");
      #else
        schemaOrName_ = torch::jit::parseSchemaOrName(schemaOrName);
      #endif

      return std::move(*this);
    }

    /**
     * Use this to register an operator whose kernel is implemented as a functor.
     * The kernel is only called for inputs matching the given dispatch key.
     * You can register multiple kernels for different dispatch keys.
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
     * >     .op(c10::RegisterOperators::options()
     * >         .schema("my_op")
     * >         .kernel<my_kernel_cpu>(DispatchKey::CPUTensorId));
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
     * >     .op(c10::RegisterOperators::options()
     * >         .schema("my_op")
     * >         .kernel<my_kernel_cpu>(DispatchKey::CPUTensorId, "some_configuration", 3, true));
     */
    template<class KernelFunctor, class... ConstructorParameters>
    // enable_if: only enable it if KernelFunctor is actually a functor
    std::enable_if_t<guts::is_functor<KernelFunctor>::value, Options&&> kernel(DispatchKey dispatch_key, ConstructorParameters&&... constructorParameters) && {
      static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to register a kernel functor using the kernel<Functor>() API, but it doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");
      static_assert(std::is_constructible<KernelFunctor, ConstructorParameters...>::value, "Wrong argument list for constructor of kernel functor. The arguments to kernel<Functor>(arguments...) must match one of the constructors of Functor.");

      return std::move(*this).kernel(
        std::move(dispatch_key),
        KernelFunction::makeFromUnboxedFunctorFactory<KernelFunctor>(detail::KernelFactory<KernelFunctor, std::decay_t<ConstructorParameters>...>(std::forward<ConstructorParameters>(constructorParameters)...)),
        detail::FunctionSchemaInferer<KernelFunctor>()()
      );
    }

    /**
     * Use this to register an operator whose kernel is implemented as a functor.
     * The kernel is a catch-all kernel, meaning it's called independent from
     * the input. Dispatch is disabled for this operator.
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
     * >     .op(c10::RegisterOperators::options()
     * >         .schema("my_op")
     * >         .catchAllKernel<my_kernel_cpu>());
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
     * >     .op(c10::RegisterOperators::options()
     * >         .schema("my_op")
     * >         .catchAllKernel<my_kernel_cpu>("some_configuration", 3, true));
     */
    template<class KernelFunctor, class... ConstructorParameters>
    // enable_if: only enable it if KernelFunctor is actually a functor
    std::enable_if_t<guts::is_functor<KernelFunctor>::value, Options&&> catchAllKernel(ConstructorParameters&&... constructorParameters) && {
      static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to register a kernel functor using the kernel<Functor>() API, but it doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");
      static_assert(std::is_constructible<KernelFunctor, ConstructorParameters...>::value, "Wrong argument list for constructor of kernel functor. The arguments to kernel<Functor>(arguments...) must match one of the constructors of Functor.");

      return std::move(*this).kernel(
        c10::nullopt,
        KernelFunction::makeFromUnboxedFunctorFactory<KernelFunctor>(detail::KernelFactory<KernelFunctor, std::decay_t<ConstructorParameters>...>(std::forward<ConstructorParameters>(constructorParameters)...)),
        detail::FunctionSchemaInferer<KernelFunctor>()()
      );
    }

    /**
     * Use this to register an operator whose kernel is implemented by a function.
     * The kernel is only called for inputs matching the given dispatch key.
     * You can register multiple kernels for different dispatch keys.
     *
     * Example:
     *
     * > namespace { Tensor my_kernel_cpu(Tensor a, Tensor b) {...} }
     * >
     * > static auto registry = c10::RegisterOperators()
     * >     .op(c10::RegisterOperators::options()
     * >         .schema("my_op")
     * >         .kernel<decltype(my_kernel_cpu), &my_kernel_cpu>(DispatchKey::CPUTensorId));
     */
    template<class FuncType, FuncType* kernel_func>
    // enable_if: only enable it if FuncType is actually a function
    std::enable_if_t<guts::is_function_type<FuncType>::value, Options&&> kernel(DispatchKey dispatch_key) && {
      static_assert(!std::is_same<FuncType, KernelFunction::BoxedKernelFunction>::value, "Tried to register a stackbased (i.e. internal) kernel function using the public kernel<...>() API. Please either use the internal kernel(...) API or also implement the kernel function as defined by the public API.");
      static_assert(kernel_func != nullptr, "Kernel function cannot be nullptr");

      return std::move(*this).kernel(
        std::move(dispatch_key),
        KernelFunction::makeFromUnboxedFunction<FuncType, kernel_func>(),
        // TODO Do schema inference without relying on WrapKernelFunction
        detail::FunctionSchemaInferer<typename detail::WrapKernelFunction<FuncType, kernel_func>::type>()()
      );
    }

    /**
     * Use this to register an operator whose kernel is implemented by a function.
     * The kernel is a catch-all kernel, meaning it's called independent from
     * the input. Dispatch is disabled for this operator.
     *
     * Example:
     *
     * > namespace { Tensor my_kernel_cpu(Tensor a, Tensor b) {...} }
     * >
     * > static auto registry = c10::RegisterOperators()
     * >     .op(c10::RegisterOperators::options()
     * >         .schema("my_op")
     * >         .catchAllKernel<decltype(my_kernel_cpu), &my_kernel_cpu>());
     */
    template<class FuncType, FuncType* kernel_func>
    // enable_if: only enable it if FuncType is actually a function
    std::enable_if_t<guts::is_function_type<FuncType>::value, Options&&> catchAllKernel() && {
      static_assert(!std::is_same<FuncType, KernelFunction::BoxedKernelFunction>::value, "Tried to register a stackbased (i.e. internal) kernel function using the public kernel<...>() API. Please either use the internal kernel(...) API or also implement the kernel function as defined by the public API.");
      static_assert(kernel_func != nullptr, "Kernel function cannot be nullptr");

      return std::move(*this).kernel(
        c10::nullopt,
        KernelFunction::makeFromUnboxedFunction<FuncType, kernel_func>(),
        // TODO Do schema inference without relying on WrapKernelFunction
        detail::FunctionSchemaInferer<typename detail::WrapKernelFunction<FuncType, kernel_func>::type>()()
      );
    }

    template<class FuncType>
    // enable_if: only enable it if FuncType is actually a function
    std::enable_if_t<guts::is_function_type<FuncType>::value, Options&&> kernel(DispatchKey dispatch_key, FuncType* kernel_func) && {
      static_assert(!std::is_same<FuncType, KernelFunction::BoxedKernelFunction>::value, "Tried to register a stackbased (i.e. internal) kernel function using the public kernel<...>() API. Please either use the internal kernel(...) API or also implement the kernel function as defined by the public API.");
      TORCH_INTERNAL_ASSERT(kernel_func != nullptr, "Kernel function cannot be nullptr");

      return std::move(*this).kernel(
        std::move(dispatch_key),
        KernelFunction::makeFromUnboxedRuntimeFunction(kernel_func),
        // TODO Do schema inference without relying on WrapKernelFunction
        detail::FunctionSchemaInferer<detail::WrapRuntimeKernelFunctor<std::decay_t<FuncType>>>()()
      );
    }

    template<class FuncType>
    // enable_if: only enable it if FuncType is actually a function
    std::enable_if_t<guts::is_function_type<FuncType>::value, Options&&> catchAllKernel(FuncType* kernel_func) && {
      static_assert(!std::is_same<FuncType, KernelFunction::BoxedKernelFunction>::value, "Tried to register a stackbased (i.e. internal) kernel function using the public kernel<...>() API. Please either use the internal kernel(...) API or also implement the kernel function as defined by the public API.");
      TORCH_INTERNAL_ASSERT(kernel_func != nullptr, "Kernel function cannot be nullptr");

      return std::move(*this).kernel(
        c10::nullopt,
        KernelFunction::makeFromUnboxedRuntimeFunction(kernel_func),
        // TODO Do schema inference without relying on WrapKernelFunction
        detail::FunctionSchemaInferer<detail::WrapRuntimeKernelFunctor<std::decay_t<FuncType>>>()()
      );
    }

    // TODO Remove impl_unboxedOnlyKernel once all of aten can generate boxed kernels
    template<class FuncType, FuncType* kernel_func>
    // enable_if: only enable it if FuncType is actually a function
    std::enable_if_t<guts::is_function_type<FuncType>::value, Options&&> impl_unboxedOnlyKernel(DispatchKey dispatch_key) && {
      static_assert(!std::is_same<FuncType, KernelFunction::BoxedKernelFunction>::value, "Tried to register a stackbased (i.e. internal) kernel function using the public kernel<...>() API. Please either use the internal kernel(...) API or also implement the kernel function as defined by the public API.");
      static_assert(kernel_func != nullptr, "Kernel function cannot be nullptr");

      return std::move(*this).kernel(
        std::move(dispatch_key),
        KernelFunction::makeFromUnboxedOnlyRuntimeFunction(kernel_func),
        nullptr // disable function schema inference because some ops from native_functions.yaml don't support it yet
      );
    }

    // TODO Remove impl_unboxedOnlyCatchAllKernel once all of aten can generate boxed kernels
    template<class FuncType, FuncType* kernel_func>
    // enable_if: only enable it if FuncType is actually a function
    std::enable_if_t<guts::is_function_type<FuncType>::value, Options&&> impl_unboxedOnlyCatchAllKernel() && {
      static_assert(!std::is_same<FuncType, KernelFunction::BoxedKernelFunction>::value, "Tried to register a stackbased (i.e. internal) kernel function using the public kernel<...>() API. Please either use the internal kernel(...) API or also implement the kernel function as defined by the public API.");
      static_assert(kernel_func != nullptr, "Kernel function cannot be nullptr");

      return std::move(*this).kernel(
        c10::nullopt,
        KernelFunction::makeFromUnboxedOnlyRuntimeFunction(kernel_func),
        nullptr // disable function schema inference because some ops from native_functions.yaml don't support it yet
      );
    }

    /**
     * Use this to register an operator whose kernel is implemented as a lambda.
     * The kernel is only called for inputs matching the given dispatch key.
     * You can register multiple kernels for different dispatch keys.
     *
     * The lambda must be stateless, i.e. not have a capture. If your kernel
     * needs to store some configuration parameters, write the kernel as a
     * functor instead.
     *
     * Example:
     *
     * > static auto registry = c10::RegisterOperators()
     * >     .op(c10::RegisterOperators::options()
     * >         .schema("my_op")
     * >         .kernel(DispatchKey::CPUTensorId, [] (Tensor a) -> Tensor {...}));
     */
    template<class Lambda>
    // enable_if: only enable it if Lambda is a functor (note: lambdas are functors)
    std::enable_if_t<
        guts::is_functor<std::decay_t<Lambda>>::value
        && !std::is_same<typename guts::infer_function_traits_t<std::decay_t<Lambda>>::func_type, KernelFunction::BoxedKernelFunction>::value,
        Options&&> kernel(DispatchKey dispatch_key, Lambda&& functor) && {
      static_assert(!std::is_base_of<OperatorKernel, std::decay_t<Lambda>>::value, "The kernel(x) API for registering a kernel is only meant to be used with lambdas. Your kernel is a functor. Please use the kernel<Functor>() API instead.");

      // We don't support stateful lambdas (i.e. lambdas with a capture), because their
      // behavior would be nonobvious. A functor kernel with cache gets a new instance of
      // its cache each time the kernel is looked up from the dispatch table.
      // A lambda with a capture would be global and share its capture between all kernel lookups.
      // So, instead of making users having to think about it (including the thread-safety
      // issues this causes), let's just forbid stateful lambdas altogether.
      static_assert(guts::is_stateless_lambda<std::decay_t<Lambda>>::value, "The kernel(x) API for registering a kernel only works for stateless lambdas (i.e. lambdas without captures). If you need a cache, please use the functor based API kernel<Functor>() instead.");

      return std::move(*this).kernel(
        std::move(dispatch_key),
        KernelFunction::makeFromUnboxedLambda(std::forward<Lambda>(functor)),
        // TODO Do schema inference without relying on WrapRuntimeKernelFunctor
        detail::FunctionSchemaInferer<detail::WrapRuntimeKernelFunctor<std::decay_t<Lambda>>>()()
      );
    }

    /**
     * Use this to register an operator whose kernel is implemented as a lambda.
     * The kernel is a catch-all kernel, meaning it's called independent from
     * the input. Dispatch is disabled for this operator.
     *
     * The lambda must be stateless, i.e. not have a capture. If your kernel
     * needs to store some configuration parameters, write the kernel as a
     * functor instead.
     *
     * Example:
     *
     * > static auto registry = c10::RegisterOperators()
     * >     .op(c10::RegisterOperators::options()
     * >         .schema("my_op")
     * >         .catchAllKernel([] (Tensor a) -> Tensor {...}));
     */
    template<class Lambda>
    // enable_if: only enable it if Lambda is a functor (note: lambdas are functors)
    std::enable_if_t<
        guts::is_functor<std::decay_t<Lambda>>::value
        && !std::is_same<typename guts::infer_function_traits_t<std::decay_t<Lambda>>::func_type, KernelFunction::BoxedKernelFunction>::value,
        Options&&> catchAllKernel(Lambda&& lambda) && {
      static_assert(!std::is_base_of<OperatorKernel, std::decay_t<Lambda>>::value, "The kernel(x) API for registering a kernel is only meant to be used with lambdas. Your kernel is a functor. Please use the kernel<Functor>() API instead.");

      // We don't support stateful lambdas (i.e. lambdas with a capture), because their
      // behavior would be nonobvious.
      // A lambda with a capture would be global and share its capture between all kernel lookups.
      // This would be a likely source for unexpected race conditions, so we forbid it.
      // If a kernel really needs global state, they can just have regular global state
      // in their .cpp file next to the kernel lambda.
      static_assert(guts::is_stateless_lambda<std::decay_t<Lambda>>::value, "The kernel(x) API for registering a kernel only works for stateless lambdas (i.e. lambdas without captures). If you need a cache, please use the functor based API kernel<Functor>() instead.");

      return std::move(*this).kernel(
        c10::nullopt,
        KernelFunction::makeFromUnboxedLambda(std::forward<Lambda>(lambda)),
        // TODO Do schema inference without relying on WrapRuntimeKernelFunctor
        detail::FunctionSchemaInferer<detail::WrapRuntimeKernelFunctor<std::decay_t<Lambda>>>()()
      );
    }

    Options&& aliasAnalysis(AliasAnalysisKind aliasAnalysisKind) && {
      TORCH_CHECK(!aliasAnalysisKind_.has_value(), "You can only call aliasAnalysis() once per operator registration.");
      aliasAnalysisKind_ = aliasAnalysisKind;
      return std::move(*this);
    }

  private:
    Options&& kernel(c10::optional<DispatchKey> dispatch_key, KernelFunction&& func, std::unique_ptr<FunctionSchema>&& inferred_function_schema) && {
      KernelRegistrationConfig config;
      config.dispatch_key = dispatch_key;
      config.func = std::move(func);
      config.inferred_function_schema = std::move(inferred_function_schema);
      kernels.push_back(std::move(config));
      return std::move(*this);
    }

    Options()
    : schemaOrName_(c10::nullopt)
    , kernels()
    , aliasAnalysisKind_(c10::nullopt)
    {}

    // KernelRegistrationConfig accumulates all information from the config
    // parameters passed to a RegisterOperators::op() call into one object.
    struct KernelRegistrationConfig final {
      KernelRegistrationConfig()
        : dispatch_key(c10::nullopt)
        , func()
        , inferred_function_schema(nullptr)
      {}

      c10::optional<DispatchKey> dispatch_key;
      KernelFunction func;
      std::unique_ptr<FunctionSchema> inferred_function_schema;
    };

    c10::optional<c10::either<OperatorName, FunctionSchema>> schemaOrName_;

    std::vector<KernelRegistrationConfig> kernels;
    optional<AliasAnalysisKind> aliasAnalysisKind_;
    friend class RegisterOperators;
    friend class Module;
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
  RegisterOperators&& op(Options&& options) && {
    checkSchemaAndRegisterOp_(std::move(options));
    return std::move(*this);
  }

  // Regular mutator version of the && version above
  RegisterOperators& op(Options&& options) & {
    checkSchemaAndRegisterOp_(std::move(options));
    return *this;
  }

  /**
   * This is a shorthand for RegisterOperators::op(Options) where you can
   * specify the operator schema outside of the options parameter.
   * See class doc comment for examples.
   */
  RegisterOperators&& op(const std::string& schemaOrName, Options&& options = RegisterOperators::options()) && {
    return std::move(*this).op(std::move(options).schema(schemaOrName));
  }

  // internal only for registering caffe2 ops
  RegisterOperators&& op(FunctionSchema schema, Options&& options) && {
    return std::move(*this).op(std::move(options).schema(std::move(schema)));
  }

  template<class FuncType>
  explicit RegisterOperators(const std::string& schemaOrName, FuncType&& func, Options&& options = RegisterOperators::options())
  : RegisterOperators() {
    std::move(*this).op(schemaOrName, std::forward<FuncType>(func), std::move(options));
  }

  /**
   * This API registers an operator based on a kernel function pointer.
   *
   * Given a kernel
   *
   * > namespace { Tensor my_kernel_cpu(Tensor a, Tensor b) {...} }
   *
   * This API looks like:
   *
   * > static auto registry = c10::RegisterOperators()
   * >     .op("my_op", &my_kernel_cpu);
   *
   * If your kernel is small and the overhead of calling it matters,
   * then this API might be the wrong choice since the following API
   * has a slightly lower overhead for calling into the kernel:
   *
   * > static auto registry = c10::RegisterOperators()
   * >     .op("my_op", c10::RegisterOperators::options()
   * >         .kernel<decltype(my_kernel_cpu), &my_kernel_cpu>());
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
   * >     .op("my_op", c10::RegisterOperators::options()
   * >         .kernel<my_kernel_cpu>());
   */
   template<class FuncType>
   // enable_if: only enable it if FuncType is actually a function, but not a stack based BoxedKernelFunction.
   std::enable_if_t<guts::is_function_type<FuncType>::value && !std::is_same<FuncType, KernelFunction::BoxedKernelFunction>::value, RegisterOperators&&>
   op(const std::string& schemaOrName, FuncType* func, Options&& options = RegisterOperators::options()) && {
     constexpr bool AllowLegacyTypes = true;
     return std::move(*this).op(std::move(options).schema(schemaOrName).kernel(
       c10::nullopt,
       KernelFunction::makeFromUnboxedRuntimeFunction<AllowLegacyTypes>(func),
       // TODO Do schema inference without relying on WrapRuntimeKernelFunctor
       detail::FunctionSchemaInferer<detail::WrapRuntimeKernelFunctor<std::decay_t<FuncType>>>()()
     ));
   }

   /**
    * This API registers an operator based on a kernel lambda.
    *
    * This API looks like:
    *
    * > static auto registry = c10::RegisterOperators()
    * >     .op("my_op", [] (Tensor a, Tensor b) {...});
    *
    * This is equivalent to:
    *
    * > static auto registry = c10::RegisterOperators()
    * >     .op("my_op", c10::RegisterOperators::options()
    * >         .catchAllKernel([] (Tensor a, Tensor b) {...}));
    *
    */
    template<class Lambda>
    // enable_if: only enable it if Lambda is actually a stateless lambda
    std::enable_if_t<guts::is_functor<Lambda>::value && guts::is_stateless_lambda<std::decay_t<Lambda>>::value, RegisterOperators&&>
    op(const std::string& schemaOrName, Lambda&& lambda, Options&& options = RegisterOperators::options()) && {
      static_assert(!std::is_base_of<OperatorKernel, Lambda>::value, "c10::OperatorKernel is part of the new kernel registration API and shouldn't be used together with the deprecated registration API. Please use the new RegisterOperators::options().kernel() based API instead.");

      constexpr bool AllowLegacyTypes = true;
      return std::move(*this).op(std::move(options).schema(schemaOrName).kernel(
        c10::nullopt,
        KernelFunction::makeFromUnboxedLambda<AllowLegacyTypes>(std::forward<Lambda>(lambda)),
        // TODO Do schema inference without relying on WrapRuntimeKernelFunctor
        detail::FunctionSchemaInferer<detail::WrapRuntimeKernelFunctor<std::decay_t<Lambda>>>()()
      ));
    }

    template<class Lambda>
    C10_DEPRECATED_MESSAGE("Registering operator kernels with stateful lambdas (i.e. lambdas with a capture) has non-obvious behavior. This is deprecated. Please use a lambda without a capture or a functor class instead.")
    // enable_if: only enable it if Lambda is actually a functor but not a stateless lambda
    std::enable_if_t<guts::is_functor<Lambda>::value && !guts::is_stateless_lambda<std::decay_t<Lambda>>::value, RegisterOperators&&>
    op(const std::string& schemaOrName, Lambda&& lambda, Options&& options = RegisterOperators::options()) && {
      static_assert(!std::is_base_of<OperatorKernel, Lambda>::value, "c10::OperatorKernel is part of the new kernel registration API and shouldn't be used together with the deprecated registration API. Please use the new RegisterOperators::options().kernel() based API instead.");

      constexpr bool AllowLegacyTypes = true;
      return std::move(*this).op(std::move(options).schema(schemaOrName).kernel(
        c10::nullopt,
        KernelFunction::makeFromUnboxedLambda<AllowLegacyTypes>(std::forward<Lambda>(lambda)),
        // TODO Do schema inference without relying on WrapRuntimeKernelFunctor
        detail::FunctionSchemaInferer<detail::WrapRuntimeKernelFunctor<std::decay_t<Lambda>>>()()
      ));
    }

private:
  void checkSchemaAndRegisterOp_(Options&& config);

  static c10::FunctionSchema inferSchemaFromKernels_(const OperatorName& opNameStr, const Options& options);
  void checkNoDuplicateKernels_(const Options& options);
  void registerOp_(Options&& options);

  std::vector<RegistrationHandleRAII> registrars_;
};

// --------------------------------------------------------------------------
//
// New style API
//
// --------------------------------------------------------------------------
//
// The basic concept behind the new style API is to be as similar to pybind11's
// API as possible.
//
// A quick tour of a few usage examples:
//
//  auto register = torch::import("aten")
//
//    // Define a schema for an operator, but provide no implementation
//    .def("mul(Tensor self, Tensor other) -> Tensor")
//
//    // Define a operator with exactly one implementation for all backends.
//    .def("add(Tensor self, Tensor other) -> Tensor", &add_impl)
//
//    // Provide an implementation for a defined operator (you can
//    // provide multiple; one per backend).  We'll take care of calling
//    // the correct implementation depending on if we get a CPU
//    // tensor or a CUDA tensor
//    .impl("mul", torch::dispatch(torch::kCPU, &mul_cpu_impl))
//    .impl("mul", torch::dispatch(torch::kCUDA, &mul_cuda_impl))
//
//    // Define an "optimized" operator with extra inlining
//    // (you can use this with torch::dispatch too); mostly only used for
//    // our internal stuff
//    .impl("sub", TORCH_OPTIMIZED_FN(sub_impl))
//
// Also, you can omit the top level namespace and specify it explicitly in
// the sub-definitions, e.g.,  torch::import().impl("aten::mul", ...)


// Represents a C++ function that implements an operator.  Most users won't
// interact directly with this class, except via error messages: the
// constructors this function define the set of permissible "function"-like
// things you can bind via the interface.
//
// This class erases the type of the passed in function, but durably records
// the type via an inferred schema for the function.
//
// TODO: This is morally the same thing as KernelRegistrationConfig, but it's
// opaque to the user.
class CAFFE2_API CppFunction final {
public:
  // This overload accepts function pointers, e.g., CppFunction(&add_impl)
  template <typename Func>
  explicit CppFunction(Func* f, std::enable_if_t<guts::is_function_type<Func>::value, std::nullptr_t> = nullptr)
    : func_(c10::KernelFunction::makeFromUnboxedRuntimeFunction(f))
    // TODO: Don't go through WrapRuntimeKernelFunctor
    , schema_(detail::FunctionSchemaInferer<detail::WrapRuntimeKernelFunctor<std::decay_t<Func>>>()())
    {}

  // This overload accepts lambdas, e.g., CppFunction([](const Tensor& self) { ... })
  template <typename Lambda>
  explicit CppFunction(Lambda&& f, std::enable_if_t<guts::is_functor<std::decay_t<Lambda>>::value, std::nullptr_t> = nullptr)
    : func_(c10::KernelFunction::makeFromUnboxedLambda(std::forward<Lambda>(f)))
    // TODO: Don't go through WrapRuntimeKernelFunctor
    , schema_(detail::FunctionSchemaInferer<detail::WrapRuntimeKernelFunctor<std::decay_t<Lambda>>>()())
    {}

  // This static factory lets you create CppFunctions that (1) don't have boxing
  // wrappers (because we don't support it yet) and (2) don't have schema
  // inference (because some ops don't support it).
  //
  // TODO: Eliminate the necessity for this function entirely.
  template <typename Func>
  static CppFunction makeUnboxedOnly(Func* f) {
    return CppFunction(
      c10::KernelFunction::makeFromUnboxedOnlyRuntimeFunction(f),
      /* schema */ nullptr
    );
  }

  // TODO: more user friendly API
  static CppFunction makeFallthrough() {
    return CppFunction(
      c10::KernelFunction::makeFallthrough(),
      /* schema */ nullptr
    );
  }

  // TODO: more user friendly API
  template<KernelFunction::BoxedKernelFunction* func>
  static CppFunction makeFromBoxedFunction() {
    return CppFunction(
      c10::KernelFunction::makeFromBoxedFunction<func>(),
      /* schema */ nullptr
    );
  }

private:
  c10::optional<c10::DispatchKey> dispatch_key_;
  c10::KernelFunction func_;
  std::unique_ptr<c10::FunctionSchema> schema_;

  // The "setter" for dispatch_key_
  template <typename Func>
  friend CppFunction dispatch(c10::DispatchKey, Func&&);

  // The only class which actually pulls out values from CppFunction (does so
  // destructively, felt too lazy to write accessors that I don't even
  // want users to use)
  friend class Module;

  CppFunction(KernelFunction func, std::unique_ptr<c10::FunctionSchema> schema);
};

// Create a CppFunction which is associated with a specific dispatch key.
// CppFunctions that are tagged with a DispatchKey don't get invoked /unless/
// the dispatcher determines that the DispatchKey is the best choice for
// a function
template <typename Func>
inline CppFunction dispatch(c10::DispatchKey k, Func&& raw_f) {
  CppFunction f(std::forward<Func>(raw_f));
  f.dispatch_key_ = k;
  return f;
}

// Convenience overload of dispatch which accepts DeviceType
template <typename Func>
inline CppFunction dispatch(DeviceType type, Func&& raw_f) {
  auto deviceTypeToDispatchKey = [](DeviceType t){
    switch (t) {
      case DeviceType::CPU:
        return c10::DispatchKey::CPUTensorId;
      case DeviceType::CUDA:
        return c10::DispatchKey::CUDATensorId;
      case DeviceType::XLA:
        return c10::DispatchKey::XLATensorId;
      default:
        TORCH_CHECK(false,
          "Device type ", t, " cannot be overloaded at dispatch time, "
          "please file a bug report explaining what you were trying to do.");
    }
  };
  return dispatch(deviceTypeToDispatchKey(type), std::forward<Func>(raw_f));
}

// Represents a namespace in which we can define operators.  Conventionally
// constructed using "torch::import".  This object lets you avoid repeatedly
// having to specify a namespace, instead you specify it once with:
//
//      torch::import("aten")
//        .def("add", [](const Tensor& a, const Tensor& b) { return a + b; })
//        .def("mul", torch::dispatch(torch::kCPU, &cpu_add))
//
// versus
//
//      torch::import()
//        .def("aten::add", ...)
//        .def("aten::mul", ...)
//
class CAFFE2_API Module final {
  // Could be nullptr, in which case it represents the "top-level" namespace
  // (all subsequent definitions must be explicitly namespaced).
  // TODO: Could store a optional<std::string> if you want to support dynamically computed
  // namespaces; for now don't support
  const char* ns_;

  std::vector<RegistrationHandleRAII> registrars_;

  Module(const char* ns);

  // Use these as the constructors
  friend Module import(const char* ns);
  friend Module import();

public:
  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;

  Module(Module&&) noexcept;
  Module& operator=(Module&&) noexcept;

  // Declare an operator with a schema, but don't provide any implementations
  // for it.  You're expected to then provide implementations using the
  // impl() method.
  Module&& def(const char* schema) &&;

  // Convenience method to define an operator for a schema and then register
  // an implementation for it.  def(n, f) is almost equivalent to def(n).impl(f),
  // except that if n is not a schema, then the schema is inferred from the
  // static type of f.
  Module&& def(const char* name_or_schema, CppFunction&& f) &&;
  template <typename Func>
  Module&& def(const char* name_or_schema, Func&& raw_f) && {
    CppFunction f(std::forward<Func>(raw_f));
    return std::move(*this).def(name_or_schema, std::move(f));
  }

  // Register an implementation for an operator.  You may register multiple
  // implementations for a single operator at different dispatch keys
  // (see torch::dispatch).  Implementations must have a corresponding
  // declaration (from def), otherwise they are invalid.
  Module&& impl(const char* name, CppFunction&& f) &&;
  template <typename Func>
  Module&& impl(const char* name, Func&& raw_f) && {
    CppFunction f(std::forward<Func>(raw_f));
    return std::move(*this).impl(name, std::move(f));
  }

  // Register a fallback implementation for all operators which will be used
  // if there is not a specific implementation for an operator available.
  // At the moment, you must specify a dispatch key (see torch::dispatch) for
  // your fallback.
  Module&& fallback(CppFunction&& f) &&;
  template <typename Func>
  Module&& fallback(Func&& raw_f) && {
    CppFunction f(std::forward<Func>(raw_f));
    return std::move(*this).fallback(std::move(f));
  }
};

// Return the namespace corresponding to the string 'ns'
inline Module import(const char* ns) {
  return Module(ns);
}

// Return the "top-level" namespace; subsequent definitions must be explicitly
// namespaced
inline Module import() {
  return Module(nullptr);
}

} // namespace c10

namespace torch {
  // Old-style API
  using RegisterOperators = c10::RegisterOperators;

  // New-style API
  using c10::dispatch;
  using c10::import;
}
