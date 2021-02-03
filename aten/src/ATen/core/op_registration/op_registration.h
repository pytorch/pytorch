#pragma once

/**
 * Include this file if you want to register operators. It includes all
 * functionality needed to do so for you.
 */

#include <c10/core/DispatchKey.h>
#include <c10/core/CompileTimeFunctionPointer.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/dispatch/CppSignature.h>
#include <ATen/core/dispatch/RegistrationHandleRAII.h>
#include <ATen/core/op_registration/infer_schema.h>
#if defined(EXPOSE_C2_OPS) || !defined(CAFFE2_IS_XPLAT_BUILD)
#include <torch/csrc/jit/frontend/function_schema_parser.h>
#endif
#include <ATen/core/ATenOpList.h>

namespace c10 {

namespace detail {
template<class KernelFunctor>
std::unique_ptr<FunctionSchema> inferFunctionSchemaFromFunctor() {
  using func_type = typename c10::guts::infer_function_traits_t<KernelFunctor>::func_type;
  return std::make_unique<FunctionSchema>(inferFunctionSchemaFlattenedReturns<func_type>());
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
class TORCH_API RegisterOperators final {
public:
  RegisterOperators();
  ~RegisterOperators();

  RegisterOperators(const RegisterOperators&) = delete;
  RegisterOperators& operator=(const RegisterOperators&) = delete;
  RegisterOperators(RegisterOperators&&) noexcept;
  RegisterOperators& operator=(RegisterOperators&&) noexcept;

  class TORCH_API Options final {
  public:
    Options(const Options&) = delete;
    Options(Options&&) noexcept = delete;
    Options& operator=(const Options&) = delete;
    Options& operator=(Options&&) noexcept = delete;

    // internal-only for registering stack based kernels
    template<KernelFunction::BoxedKernelFunction* kernel_func>
    Options&& kernel(DispatchKey dispatch_key) && {
      return std::move(*this).kernel(dispatch_key, KernelFunction::makeFromBoxedFunction<kernel_func>(), nullopt, nullptr);
    }

    // internal-only for registering stack based catch-all kernels
    template<KernelFunction::BoxedKernelFunction* kernel_func>
    Options&& catchAllKernel() && {
      return std::move(*this).kernel(c10::nullopt, KernelFunction::makeFromBoxedFunction<kernel_func>(), nullopt, nullptr);
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
        impl::CppSignature::make<KernelFunctor>(),
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
        impl::CppSignature::make<KernelFunctor>(),
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
        KernelFunction::makeFromUnboxedFunction(TORCH_FN(kernel_func)),
        impl::CppSignature::make<FuncType>(),
        // TODO Do schema inference without relying on WrapFunctionIntoFunctor
        detail::inferFunctionSchemaFromFunctor<typename impl::WrapFunctionIntoFunctor<CompileTimeFunctionPointer<FuncType, kernel_func>>::type>()
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
        KernelFunction::makeFromUnboxedFunction(TORCH_FN(kernel_func)),
        impl::CppSignature::make<FuncType>(),
        // TODO Do schema inference without relying on WrapFunctionIntoFunctor
        detail::inferFunctionSchemaFromFunctor<typename impl::WrapFunctionIntoFunctor<CompileTimeFunctionPointer<FuncType, kernel_func>>::type>()
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
        impl::CppSignature::make<FuncType>(),
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
        impl::CppSignature::make<FuncType>(),
        // TODO Do schema inference without relying on WrapFunctionIntoFunctor
        detail::inferFunctionSchemaFromFunctor<impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<FuncType>>>()
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
        impl::CppSignature::make<Lambda>(),
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
        impl::CppSignature::make<Lambda>(),
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
    Options&& kernel(c10::optional<DispatchKey> dispatch_key, KernelFunction&& func, c10::optional<impl::CppSignature> cpp_signature, std::unique_ptr<FunctionSchema>&& inferred_function_schema) && {
      KernelRegistrationConfig config;
      config.dispatch_key = dispatch_key;
      config.func = std::move(func);
      config.cpp_signature = std::move(cpp_signature);
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
        , cpp_signature(c10::nullopt)
        , inferred_function_schema(nullptr)
      {}

      c10::optional<DispatchKey> dispatch_key;
      KernelFunction func;
      c10::optional<impl::CppSignature> cpp_signature;
      std::unique_ptr<FunctionSchema> inferred_function_schema;
    };

    c10::optional<c10::either<OperatorName, FunctionSchema>> schemaOrName_;

    std::vector<KernelRegistrationConfig> kernels;
    optional<AliasAnalysisKind> aliasAnalysisKind_;
    friend class RegisterOperators;
    friend class Library;
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
       impl::CppSignature::make<FuncType>(),
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
        impl::CppSignature::make<Lambda>(),
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
        impl::CppSignature::make<Lambda>(),
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

} // namespace c10

namespace torch {
  // Old-style API
  using RegisterOperators = c10::RegisterOperators;
}
