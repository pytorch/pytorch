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

namespace detail {
template<class KernelFunctor>
std::unique_ptr<FunctionSchema> inferFunctionSchemaFromFunctor() {
  using func_type = typename c10::guts::infer_function_traits_t<KernelFunctor>::func_type;
  return std::make_unique<FunctionSchema>(inferFunctionSchemaFlattenedReturns<func_type>("", ""));
}
}

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
 * >         .kernel<my_kernel_cpu>(DispatchKey::CPU));
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
     * >         .kernel<my_kernel_cpu>(DispatchKey::CPU));
     * >
     * >
     * > // Explicitly specify full schema
     * > static auto registry = c10::RegisterOperators()
     * >     .op(c10::RegisterOperators::options()
     * >         .schema("my_op(Tensor a) -> Tensor")
     * >         .kernel<my_kernel_cpu>(DispatchKey::CPU));
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
     * >         .kernel<my_kernel_cpu>(DispatchKey::CPU));
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
     * >         .kernel<my_kernel_cpu>(DispatchKey::CPU, "some_configuration", 3, true));
     */
    template<class KernelFunctor, class... ConstructorParameters>
    // enable_if: only enable it if KernelFunctor is actually a functor
    std::enable_if_t<guts::is_functor<KernelFunctor>::value, Options&&> kernel(DispatchKey dispatch_key, ConstructorParameters&&... constructorParameters) && {
      static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to register a kernel functor using the kernel<Functor>() API, but it doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");
      static_assert(std::is_constructible<KernelFunctor, ConstructorParameters...>::value, "Wrong argument list for constructor of kernel functor. The arguments to kernel<Functor>(arguments...) must match one of the constructors of Functor.");

      return std::move(*this).kernel(
        std::move(dispatch_key),
        KernelFunction::makeFromUnboxedFunctor<false, KernelFunctor>(std::make_unique<KernelFunctor>(std::forward<ConstructorParameters>(constructorParameters)...)),
        detail::inferFunctionSchemaFromFunctor<KernelFunctor>()
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
        KernelFunction::makeFromUnboxedFunctor<false, KernelFunctor>(std::make_unique<KernelFunctor>(std::forward<ConstructorParameters>(constructorParameters)...)),
        detail::inferFunctionSchemaFromFunctor<KernelFunctor>()
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
     * >         .kernel<decltype(my_kernel_cpu), &my_kernel_cpu>(DispatchKey::CPU));
     */
    template<class FuncType, FuncType* kernel_func>
    // enable_if: only enable it if FuncType is actually a function
    std::enable_if_t<guts::is_function_type<FuncType>::value, Options&&> kernel(DispatchKey dispatch_key) && {
      static_assert(!std::is_same<FuncType, KernelFunction::BoxedKernelFunction>::value, "Tried to register a stackbased (i.e. internal) kernel function using the public kernel<...>() API. Please either use the internal kernel(...) API or also implement the kernel function as defined by the public API.");
      static_assert(kernel_func != nullptr, "Kernel function cannot be nullptr");

      return std::move(*this).kernel(
        std::move(dispatch_key),
        KernelFunction::makeFromUnboxedFunction<FuncType, kernel_func>(),
        // TODO Do schema inference without relying on WrapFunctionIntoFunctor
        detail::inferFunctionSchemaFromFunctor<typename impl::WrapFunctionIntoFunctor<FuncType, kernel_func>::type>()
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
        // TODO Do schema inference without relying on WrapFunctionIntoFunctor
        detail::inferFunctionSchemaFromFunctor<typename impl::WrapFunctionIntoFunctor<FuncType, kernel_func>::type>()
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
        // TODO Do schema inference without relying on WrapFunctionIntoFunctor
        detail::inferFunctionSchemaFromFunctor<impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<FuncType>>>()
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
        // TODO Do schema inference without relying on WrapFunctionIntoFunctor
        detail::inferFunctionSchemaFromFunctor<impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<FuncType>>>()
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
     * >         .kernel(DispatchKey::CPU, [] (Tensor a) -> Tensor {...}));
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
        // TODO Do schema inference without relying on WrapFunctionIntoRuntimeFunctor
        detail::inferFunctionSchemaFromFunctor<impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<Lambda>>>()
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
        // TODO Do schema inference without relying on WrapFunctionIntoRuntimeFunctor
        detail::inferFunctionSchemaFromFunctor<impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<Lambda>>>()
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
       // TODO Do schema inference without relying on WrapFunctionIntoRuntimeFunctor
       detail::inferFunctionSchemaFromFunctor<impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<FuncType>>>()
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
        // TODO Do schema inference without relying on WrapFunctionIntoRuntimeFunctor
        detail::inferFunctionSchemaFromFunctor<impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<Lambda>>>()
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
        // TODO Do schema inference without relying on WrapFunctionIntoRuntimeFunctor
        detail::inferFunctionSchemaFromFunctor<impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<Lambda>>>()
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
//  auto register = torch::import()
//
//    // Define a schema for an operator, but provide no implementation
//    .def("aten::mul(Tensor self, Tensor other) -> Tensor")
//
//    // Define a operator with exactly one implementation for all backends.
//    .def("aten::add(Tensor self, Tensor other) -> Tensor", &add_impl)
//
//    // Provide an implementation for a defined operator (you can
//    // provide multiple; one per backend).  We'll take care of calling
//    // the correct implementation depending on if we get a CPU
//    // tensor or a CUDA tensor
//    .impl("aten::mul", torch::kCPU, &mul_cpu_impl)
//    .impl("aten::mul", torch::kCUDA, &mul_cuda_impl)
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
    , schema_(detail::inferFunctionSchemaFromFunctor<impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<Func>>>())
    , debug_("Func")
    {}

  // This overload accepts lambdas, e.g., CppFunction([](const Tensor& self) { ... })
  template <typename Lambda>
  explicit CppFunction(Lambda&& f, std::enable_if_t<guts::is_functor<std::decay_t<Lambda>>::value, std::nullptr_t> = nullptr)
    : func_(c10::KernelFunction::makeFromUnboxedLambda(std::forward<Lambda>(f)))
    // TODO: Don't go through WrapRuntimeKernelFunctor
    , schema_(detail::inferFunctionSchemaFromFunctor<impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<Lambda>>>())
    , debug_("Lambda")
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
      /* schema */ nullptr,
      "UnboxedOnly"
    );
  }

  // TODO: more user friendly API
  static CppFunction makeFallthrough() {
    return CppFunction(
      c10::KernelFunction::makeFallthrough(),
      /* schema */ nullptr,
      "Fallthrough"
    );
  }

  // TODO: more user friendly API
  template<KernelFunction::BoxedKernelFunction* func>
  static CppFunction makeFromBoxedFunction() {
    return CppFunction(
      c10::KernelFunction::makeFromBoxedFunction<func>(),
      /* schema */ nullptr,
      "BoxedFunction"
    );
  }

  CppFunction&& debug(std::string d) && {
    debug_ = std::move(d);
    return std::move(*this);
  }

private:
  c10::optional<c10::DispatchKey> dispatch_key_;
  c10::KernelFunction func_;
  std::unique_ptr<c10::FunctionSchema> schema_;
  std::string debug_;

  // The "setter" for dispatch_key_
  template <typename Func>
  friend CppFunction dispatch(c10::DispatchKey, Func&&);

  // The only class which actually pulls out values from CppFunction (does so
  // destructively, felt too lazy to write accessors that I don't even
  // want users to use)
  friend class Module;

  CppFunction(KernelFunction func, std::unique_ptr<c10::FunctionSchema> schema, std::string debug);
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
      // This list is synchronized with the k-constants in c10/core/DeviceType.h
      case DeviceType::CPU:
        return c10::DispatchKey::CPU;
      case DeviceType::CUDA:
        return c10::DispatchKey::CUDA;
      case DeviceType::XLA:
        return c10::DispatchKey::XLA;
      case DeviceType::HIP:
        return c10::DispatchKey::HIP;
      case DeviceType::MSNPU:
        return c10::DispatchKey::MSNPU;
      default:
        TORCH_CHECK(false,
          "Device type ", t, " cannot be overloaded at dispatch time, "
          "please file a bug report explaining what you were trying to do.");
    }
  };
  return dispatch(deviceTypeToDispatchKey(type), std::forward<Func>(raw_f));
}

inline FunctionSchema schema(const char* str, AliasAnalysisKind k) {
  FunctionSchema s = torch::jit::parseSchema(str);
  s.setAliasAnalysis(k);
  return s;
}
inline FunctionSchema schema(const char* s) {
  return schema(s, AliasAnalysisKind::FROM_SCHEMA);
}
inline FunctionSchema&& schema(FunctionSchema&& s) { return std::move(s); }

namespace detail {

  inline c10::either<OperatorName, FunctionSchema> constructSchemaOrName(FunctionSchema&& s) {
    return c10::make_right<OperatorName, FunctionSchema>(std::move(s));
  }
  inline c10::either<OperatorName, FunctionSchema> constructSchemaOrName(OperatorName&& n) {
    return c10::make_left<OperatorName, FunctionSchema>(std::move(n));
  }
  inline c10::either<OperatorName, FunctionSchema> constructSchemaOrName(const char* str) {
    auto s = torch::jit::parseSchemaOrName(str);
    if (s.is_right()) {
      s.right().setAliasAnalysis(AliasAnalysisKind::FROM_SCHEMA);
    }
    return s;
  }

}

// Represents a namespace in which we can define operators.  Conventionally
// constructed using "torch::import".
//
//      torch::import()
//        .def("aten::add", ...)
//        .def("aten::mul", ...)
//
class CAFFE2_API Module final {
  c10::optional<std::string> ns_;

  std::vector<RegistrationHandleRAII> registrars_;

  Module(std::string ns);
  Module();

  // Use these as the constructors
  friend Module _import_DOES_NOT_WORK_WITH_MOBILE_CUSTOM_BUILD(std::string ns);
  friend Module import();

private:
  // Non-user visible actual implementations of functions.  These aren't
  // public because we only implement & qualifier and not && qualifier
  Module& _def(FunctionSchema&& schema) &;
  Module& _def(c10::either<OperatorName, FunctionSchema>&&, CppFunction&& f) &;
  Module& _impl(const char* name, CppFunction&& f) &;
  Module& _fallback(CppFunction&& f) &;

public:
  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;

  Module(Module&&);
  Module& operator=(Module&&);

  // Some notes about the API design here.  We had the following constraints:
  //
  //  - We wanted to support both method chaining to a static variable (&& ref
  //    qualifier) as well as regular allocate the object and then mutate it (&
  //    ref qualifier)
  //  - We need to support multiple "types" of arguments for schema and
  //    functions (e.g., unnamed lambda types, regular functions, const char*,
  //    fully instantiated schemas)
  //  - We don't want to write exponentially many overloads
  //  - We don't want to rely on implicit conversion to a common type,
  //    because the C++ compiler will only be willing to do a single
  //    implicit conversion (reducing the set of valid types which you
  //    can invoke with); also error messages are worse when an implicit
  //    conversion is not selected (as the compiler will not explain
  //    why it didn't select an implicit conversion; this is different
  //    from overloads where it will explain each candidate overload and
  //    why it didn't apply)
  //
  // To solve all of these constraints at the same time, we use a trick taken
  // from the pybind11 library: template over the argument in the user visible
  // API, and inside of the templated function explicitly call an overloaded
  // function to resolve the argument to a real type.  You get the good error
  // messages from overloads, but at the same time you only need to write the
  // overload for any given argument type once.
  //
  // We still have to 2x all functions in the API so we can do both && and &
  // ref qualifiers, c'est la vie.

  // Declare an operator with a schema, but don't provide any implementations
  // for it.  You're expected to then provide implementations using the
  // impl() method.
  template <typename Schema>
  Module& def(Schema&& raw_schema) & {
    FunctionSchema s = schema(std::forward<Schema>(raw_schema));
    return _def(std::move(s));
  }
  template <typename Schema>
  Module&& def(Schema&& raw_schema) && {
    def(std::forward<Schema>(raw_schema));
    return std::move(*this);
  }

  // Convenience method to define an operator for a schema and then register
  // an implementation for it.  def(n, f) is almost equivalent to def(n).impl(f),
  // except that if n is not a schema, then the schema is inferred from the
  // static type of f.
  template <typename NameOrSchema, typename Func>
  Module& def(NameOrSchema&& raw_name_or_schema, Func&& raw_f) & {
    CppFunction f(std::forward<Func>(raw_f));
    auto name_or_schema = detail::constructSchemaOrName(std::forward<NameOrSchema>(raw_name_or_schema));
    return _def(std::move(name_or_schema), std::move(f));
  }
  template <typename NameOrSchema, typename Func>
  Module&& def(NameOrSchema&& raw_name_or_schema, Func&& raw_f) && {
    def(std::forward<NameOrSchema>(raw_name_or_schema), std::forward<Func>(raw_f));
    return std::move(*this);
  }

  // Register an implementation for an operator.  You may register multiple
  // implementations for a single operator at different dispatch keys
  // (see torch::dispatch).  Implementations must have a corresponding
  // declaration (from def), otherwise they are invalid.
  template <typename Func>
  Module& impl(const char* name, Func&& raw_f) & {
    CppFunction f(std::forward<Func>(raw_f));
    return _impl(name, std::move(f));
  }
  template <typename Func>
  Module&& impl(const char* name, Func&& raw_f) && {
    impl(name, std::forward<Func>(raw_f));
    return std::move(*this);
  }
  // Convenience overload for directly specifying the dispatch key.  Dispatch
  // can validly be either DeviceType or DispatchKey; check torch::dispatch for
  // the canonical list of accepted overloads.
  template <typename Dispatch, typename Func>
  Module& impl(const char* name, Dispatch&& key, Func&& raw_f) & {
    return impl(name, dispatch(std::forward<Dispatch>(key), std::forward<Func>(raw_f)));
  }
  template <typename Dispatch, typename Func>
  Module&& impl(const char* name, Dispatch&& key, Func&& raw_f) && {
    impl(name, std::forward<Dispatch>(key), std::forward<Func>(raw_f));
    return std::move(*this);
  }
  // Convenience overload for unboxed only kernels.  These are quite common
  // but will be eventually eliminated; this function makes it easy to grep for
  // them.
  //
  // If you're looking how to def or impl with no DispatchKey, use the
  // CppFunction::makeUnboxedOnly factory directly (the API is compositional;
  // it's just that we have a LOT of impl calls that are unboxed, so there is
  // just a little syntax sugar for this case.)
  //
  // TODO: Remove these overloads once the makeUnboxedOnly incidence rate
  // goes way down
  template <typename Dispatch, typename Func>
  Module& impl_UNBOXED(const char* name, Dispatch&& key, Func* raw_f) & {
    return impl(name, dispatch(std::forward<Dispatch>(key), CppFunction::makeUnboxedOnly(raw_f)));
  }
  template <typename Dispatch, typename Func>
  Module&& impl_UNBOXED(const char* name, Dispatch&& key, Func* raw_f) && {
    impl_UNBOXED(name, std::forward<Dispatch>(key), raw_f);
    return std::move(*this);
  }

  // Register a fallback implementation for all operators which will be used
  // if there is not a specific implementation for an operator available.
  // Providing a DispatchKey is MANDATORY for fallback at the moment.
  //
  // Dispatch can validly be either DeviceType or DispatchKey; check
  // torch::dispatch for the canonical list of accepted overloads.
  template <typename Dispatch, typename Func>
  Module& fallback(Dispatch&& key, Func&& raw_f) & {
    return fallback(c10::dispatch(std::forward<Dispatch>(key), std::forward<Func>(raw_f)));
  }
  template <typename Dispatch, typename Func>
  Module&& fallback(Dispatch&& key, Func&& raw_f) && {
    fallback(std::forward<Dispatch>(key), std::forward<Func>(raw_f));
    return std::move(*this);
  }
  // NB: these overloads are here for completeness, but you'll probably want to
  // use the direct Dispatch overload
  template <typename Func>
  Module& fallback(Func&& raw_f) & {
    CppFunction f((std::forward<Func>(raw_f)));
    return _fallback(std::move(f));
  }
  template <typename Func>
  Module&& fallback(Func&& raw_f) && {
    fallback(std::forward<Func>(raw_f));
    return std::move(*this);
  }
};

// TODO: We'd like to support this, but custom mobile build doesn't
// understand how to interpret registration calls that have a namespace
// call (as this requires some non-local reasoning.)  Tracked in
// https://github.com/pytorch/pytorch/issues/35397
//
// So for now, we give this a scary internal name to discourage people
// from using it

// Return the namespace corresponding to the string 'ns'
inline Module _import_DOES_NOT_WORK_WITH_MOBILE_CUSTOM_BUILD(std::string ns) {
  return Module(std::move(ns));
}

// Return the "top-level" namespace; subsequent definitions must be explicitly
// namespaced
inline Module import() {
  return Module();
}

} // namespace c10

namespace torch {
  // Old-style API
  using RegisterOperators = c10::RegisterOperators;

  // New-style API
  using c10::dispatch;
  using c10::schema;
  using c10::import;
}
