#pragma once

#include <ATen/core/dispatch/Dispatcher.h>

namespace c10 {

/**
 * Inherit from OperatorKernel to implement a c10 kernel.
 *
 * Example:
 * > namespace {
 * >   class my_kernel_cpu final : public c10::OperatorKernel {
 * >   public:
 * >     Tensor operator()(Tensor a, Tensor b) {...}
 * >   };
 * > }
 *
 * The kernel class is allowed to have members to cache things between calls
 * but it is not allowed to change behavior based on the cache.
 * The cache is purely a performance optimization and the kernel must
 * return the same outputs regardless of what's in the cache.
 *
 * See below for how to register this kernel with PyTorch.
 */
class OperatorKernel : public c10::KernelCache {};

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

    // not needed for now
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

  // ivalue_to_arg_type<T>: Take an IValue that is an argument to a kernel and
  // cast it to the type that should be passed to the kernel function.
  // Examples: If the IValue contains a plain type like an int, return that.
  //           If the IValue contains an IntList, return it as ArrayRef<int>.
  template<class T>
  struct ivalue_to_arg_type {
    static T call(const IValue& v) {
      return std::move(v).to<T>();
    }
  };
  template<class T>
  struct ivalue_to_arg_type<ArrayRef<T>> {
    static ArrayRef<T> call(const IValue& v) {
      return v.to<intrusive_ptr<ivalue::List<T>>>()->elements();
    }
  };

  template<class Functor, size_t... ivalue_arg_indices>
  typename guts::infer_function_traits_t<Functor>::return_type call_functor_with_ivalue_args_(Functor* functor, ArrayRef<IValue> ivalue_args, guts::index_sequence<ivalue_arg_indices...>) {
    using IValueArgTypes = typename guts::infer_function_traits_t<Functor>::parameter_types;
    return (*functor)(ivalue_to_arg_type<guts::remove_cv_t<guts::remove_reference_t<guts::typelist::element_t<ivalue_arg_indices, IValueArgTypes>>>>::call(ivalue_args[ivalue_arg_indices])...);
  }

  template<class Functor>
  typename guts::infer_function_traits_t<Functor>::return_type call_functor_with_ivalue_args(Functor* functor, ArrayRef<IValue> ivalue_args) {
    constexpr size_t num_ivalue_args = guts::infer_function_traits_t<Functor>::number_of_parameters;
    AT_ASSERTM(num_ivalue_args == ivalue_args.size(), "Wrong number of ivalue arguments");
    return call_functor_with_ivalue_args_<Functor>(functor, ivalue_args, guts::make_index_sequence<num_ivalue_args>());
  }

  template<class OutputType>
  struct push_outputs final {
    static void call(OutputType&& output, Stack* stack) {
      push_outputs<std::tuple<OutputType>>(std::tuple<OutputType>(std::move(output)), stack);
    }
  };
  template<class... OutputTypes>
  struct push_outputs<std::tuple<OutputTypes...>> final {
    static void call(std::tuple<OutputTypes...>&& output, Stack* stack) {
      for (size_t i = 0; i < sizeof...(OutputTypes); ++i) {
        torch::jit::push(return_type_to_ivalue(std::move(output)));
      }
    }
  };

  template<class KernelFunctor, class Enable = void> struct wrap_kernel_functor final {};

  // SFINAE version for kernels that return an output
  template<class KernelFunctor>
  struct wrap_kernel_functor<KernelFunctor, guts::enable_if_t<!std::is_same<void, typename guts::infer_function_traits_t<KernelFunctor>::return_type>::value>> final {
    static void call(Stack* stack, KernelCache* cache) {
      static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Kernel functor must inherit from c10::OperatorKernel");

      constexpr size_t num_inputs = guts::infer_function_traits_t<KernelFunctor>::number_of_parameters;
      KernelFunctor* functor = static_cast<KernelFunctor*>(cache);
      auto output = call_functor_with_ivalue_args<KernelFunctor>(functor, torch::jit::last(*stack, num_inputs));
      push_outputs<typename guts::infer_function_traits_t<KernelFunctor>::return_type>(std::move(output), stack);
    }
  };

  // SFINAE version for kernels that don't return an output
  template<class KernelFunctor>
  struct wrap_kernel_functor<KernelFunctor, guts::enable_if_t<std::is_same<void, typename guts::infer_function_traits_t<KernelFunctor>::return_type>::value>> final {
    static void call(Stack* stack, KernelCache* cache) {
      static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Kernel functor must inherit from c10::OperatorKernel");

      constexpr size_t num_inputs = guts::infer_function_traits_t<KernelFunctor>::number_of_parameters;
      KernelFunctor* functor = static_cast<KernelFunctor*>(cache);
      call_functor_with_ivalue_args<KernelFunctor>(functor, torch::jit::last(*stack, num_inputs));
    }
  };

  template<class Cache>
  inline std::unique_ptr<c10::KernelCache> cacheCreator() {
    static_assert(std::is_default_constructible<Cache>::value, "Cache class must be default constructible");
    return guts::make_unique<Cache>();
  }

  template<>
  inline std::unique_ptr<c10::KernelCache> cacheCreator<void>() {
    return nullptr;
  }

  // WrapKernelFunction: Wraps a compile time function pointer into a kernel functor.
  // Since it is a compile time function pointer, many compilers can inline it
  // into the wrapper and you don't get any performance overhead for wrapping.
  template<class FuncType, FuncType* kernel_func, class ReturnType, class ParameterList> class WrapKernelFunction_ {};
  template<class FuncType, FuncType* kernel_func, class ReturnType, class... Parameters>
  class WrapKernelFunction_<FuncType, kernel_func, ReturnType, guts::typelist::typelist<Parameters...>> final : public c10::OperatorKernel {
  public:
    auto operator()(Parameters&&... args) -> decltype(kernel_func(std::forward<Parameters>(args)...)) {
      return (*kernel_func)(std::forward<Parameters>(args)...);
    }
  };
  template<class FuncType, FuncType* kernel_func>
  using WrapKernelFunction = WrapKernelFunction_<
      FuncType,
      kernel_func,
      typename guts::function_traits<FuncType>::return_type,
      typename guts::function_traits<FuncType>::parameter_types
  >;

  // WrapKernelFunctionRuntime: Wraps a runtime function pointer into a kernel functor.
  // Since it is a runtime function pointer, there is an overhead for calling
  // the function pointer whenever the kernel is invoked.
  // TODO Enable this and use it for deprecated API
  /*template<class FuncType, class ReturnType, class ParameterList> class WrapKernelFunctionRuntime_ {};
  template<class FuncType, class ReturnType, class... Parameters>
  class WrapKernelFunctionRuntime_<FuncType, ReturnType, guts::typelist::typelist<Parameters...>> final : public c10::OperatorKernel {
  public:
    explicit WrapKernelFunctionRuntime_(FuncType* kernel_func)
    : kernel_func_(kernel_func) {}

    auto operator()(Parameters&&... args) -> decltype(kernel_func(std::forward<Parameters>(args)...)) {
      return (*kernel_func_)(std::forward<Parameters>(args)...);
    }

  private:
    FuncType* kernel_func_;
  };
  template<class FuncType>
  using WrapKernelFunctionRuntime = WrapKernelFunctionRuntime_<
      FuncType,
      typename guts::function_traits<FuncType>::return_type,
      typename guts::function_traits<FuncType>::parameter_types
  >;*/

  struct KernelRegistrationConfig final {
    TensorTypeId dispatch_key;
    KernelFunction* kernel_func = nullptr;
    KernelCacheCreatorFunction cache_creator_func = nullptr;
  };

  // TODO If this was templated on KernelCacheCreatorFunction, it could be constexpr again.
  struct KernelRegistrationConfigParameter final {
    explicit KernelRegistrationConfigParameter(KernelFunction* kernel_func, KernelCacheCreatorFunction&& cache_creator_func)
    : kernel_func_(kernel_func), cache_creator_func_(std::move(cache_creator_func)) {
    }

    void apply(KernelRegistrationConfig* registration) && {
      registration->kernel_func = kernel_func_;
      registration->cache_creator_func = std::move(cache_creator_func_);
    }

  private:
    KernelFunction* kernel_func_;
    KernelCacheCreatorFunction cache_creator_func_;
  };

  struct DispatchKeyConfigParameter final {
    explicit constexpr DispatchKeyConfigParameter(TensorTypeId dispatch_key)
    : dispatch_key_(dispatch_key) {}

    void apply(KernelRegistrationConfig* registration) && {
      registration->dispatch_key = std::move(dispatch_key_);
    }

  private:
    TensorTypeId dispatch_key_;
  };
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
 * >     .op("my_op",
 * >         c10::kernel<my_kernel_cpu>(),
 * >         c10::dispatchKey(CPUTensorId()));
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
 * >     .op("my_op",
 * >         c10::kernel<my_kernel_cpu>("some_configuration", 3, true),
 * >         c10::dispatchKey(CPUTensorId()));
 */
template<class KernelFunctor, class... ConstructorParameters>
inline detail::KernelRegistrationConfigParameter kernel(ConstructorParameters&&... constructorParameters) {
  // TODO We're only doing this make_shared nonsense so we're able to perfectly
  //      forward the constructorParameters into the lambda below and don't have
  //      to copy them. Once we have C++14, we should change this to proper
  //      capture like:
  //      [parameters = std::make_tuple(std::forward<ConstructorParameters>(constructorParameters)...)]
  //      With C++20, we could even directly forward-capture the parameter pack
  //      without converting it into a tuple.
  auto parameters = std::make_shared<std::tuple<guts::decay_t<ConstructorParameters>...>>(
      std::forward<ConstructorParameters>(constructorParameters)...);
  return detail::KernelRegistrationConfigParameter(
      &detail::wrap_kernel_functor<KernelFunctor>::call,
      [parameters] {
        return guts::apply(&guts::make_unique<KernelFunctor>, *parameters);
      }
  );
}

/**
 * Use this to register an operator whose kernel is implemented by a function:
 *
 * Example:
 *
 * > namespace { Tensor my_kernel_cpu(Tensor a, Tensor b) {...} }
 * >
 * > static auto registry = c10::RegisterOperators()
 * >     .op("my_op",
 * >         c10::kernel<decltype(my_kernel_cpu), &my_kernel_cpu>(),
 * >         c10::dispatchKey(CPUTensorId()));
 */
template<class FuncType, FuncType* kernel_func>
inline detail::KernelRegistrationConfigParameter kernel() {
  return kernel<detail::WrapKernelFunction<FuncType, kernel_func>>();
}

/**
 * Use this to register an operator whose kernel is implemented by a stack
 * based function. This is meant to be used internally, for example for writing
 * wrappers for other ways of writing operators. This is not part of the
 * public API.
 *
 * Example:
 *
 * > namespace { void my_kernel_cpu(Stack* stack, KernelCache* cache) {...} }
 * >
 * > static auto registry = c10::RegisterOperators()
 * >     .op("my_op",
 * >         c10::kernel(my_kernel_cpu),
 * >         c10::dispatchKey(CPUTensorId()));
 */
inline detail::KernelRegistrationConfigParameter kernel(KernelFunction* kernel_func) {
  return detail::KernelRegistrationConfigParameter(kernel_func, &detail::cacheCreator<void>);
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
 * >     .op("my_op",
 * >         c10::kernel<my_kernel_cpu>(),
 * >         c10::dispatchKey(CPUTensorId()))
 * >     .op("my_op",
 * >         c10::kernel<my_kernel_cuda>(),
 * >         c10::dispatchKey(CUDATensorId()));
 */
inline constexpr detail::DispatchKeyConfigParameter dispatchKey(TensorTypeId dispatch_key) {
  return detail::DispatchKeyConfigParameter(dispatch_key);
}

namespace detail {

// Take a list of configuration parameters and return a KernelRegistrationConfig
// accumulating all their configurations.
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

// TODO doc comments
class RegisterOperators final {
public:
  RegisterOperators() = default;
  RegisterOperators(const RegisterOperators&) = delete;
  RegisterOperators(RegisterOperators&&) = default;
  RegisterOperators& operator=(const RegisterOperators&) = delete;
  RegisterOperators& operator=(RegisterOperators&&) = default;

  // TODO doc comments
  template<class... ConfigParameters>
  RegisterOperators op(FunctionSchema schema, ConfigParameters&&... configParameters) && {
    detail::KernelRegistrationConfig config = make_registration_config(configParameters...);
    registrars_.emplace_back(std::move(schema), config.dispatch_key, config.kernel_func, std::move(config.cache_creator_func));
    return std::move(*this);
  }

  // TODO allow input schema to be just the operator name + overload name, in that case use schema generated from kernel function
  // TODO if schema is fully specified, still generate schema from kernel function and make sure it's correct

  // Deprecated. For backwards compatibility only.
  // Don't use this, it introduces a performance overhead on each kernel call
  // due to the kernel being stored in the wrapper as a runtime function pointer.
  // TODO Enable this
  /*template<class KernelFunc>
  RegisterOperators op(FunctionSchema schema, KernelFunc* func) && {
    return op(std::move(schema), kernel(func));
  }*/

private:
  std::vector<c10::detail::OperatorRegistrar> registrars_;
};

}
