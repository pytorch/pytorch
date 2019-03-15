#pragma once

#include <ATen/core/dispatch/Dispatcher.h>

namespace c10 {

namespace detail {
  /**
   * Class which, on construction, registers an operator in the dispatch table
   * and on destruction deregisters it. The intent is that this class is
   * constructed at static initialization time so that operators automatically
   * get registered when a dlopen() occurs.
   *
   * You shouldn't call this directly; instead, use the RegisterOperators class.
   */
  class OperatorRegistrar final {
  public:
    /**
     * @param schema The operator schema to register the kernel for
     * @param dispatch_key  The dispatch key to register the function to
     * @param kernel The concrete function implementation to register
     * @param cache_creator A function initializing the cache for the kernel
     */
    explicit OperatorRegistrar(FunctionSchema&& schema, TensorTypeId dispatch_key, KernelFunction* kernel, KernelCacheCreatorFunction* cache_creator)
    : op_(Dispatcher::singleton().registerSchema(std::move(schema))), dispatch_key_(std::move(dispatch_key)), owns_registration_(true) {
      Dispatcher::singleton().registerKernel(op_, dispatch_key_, kernel, cache_creator);
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

  // call_with_ivalue_args: Take a function pointer and an ArrayRef<IValue>
  // containing the arguments to call the function pointer with, and call it.
  // The extra_args are appended as additional arguments at the end of the function call.
  // Example:
  // int myfunc(int a, ArrayRef<int> b, string c);
  // int main() {
  //   std::vector<IValue> ivalue_args = {IValue(2), IntList::create(3, 4)};
  //   call_with_ivalue_args<decltype(myfunc), &myfunc>(ivalue_args, "extra_arg");
  // }
  template<class FuncType, FuncType* func, class... ExtraArgs, size_t... ivalue_arg_indices>
  typename guts::function_traits<FuncType>::return_type call_with_ivalue_args_(ArrayRef<IValue> ivalue_args, guts::index_sequence<ivalue_arg_indices...>, ExtraArgs&&... extra_args) {
    using IValueArgTypes = typename guts::function_traits<FuncType>::parameter_types;
    return (*func)(ivalue_to_arg_type<guts::remove_cv_t<guts::remove_reference_t<guts::typelist::element_t<ivalue_arg_indices, IValueArgTypes>>>>::call(ivalue_args[ivalue_arg_indices])..., std::forward<ExtraArgs>(extra_args)...);
  }

  template<class FuncType, FuncType* func, class... ExtraArgs>
  typename guts::function_traits<FuncType>::return_type call_with_ivalue_args(ArrayRef<IValue> ivalue_args, ExtraArgs&&... extra_args) {
    constexpr size_t num_ivalue_args = guts::function_traits<FuncType>::number_of_parameters - sizeof...(ExtraArgs);
    AT_ASSERTM(num_ivalue_args == ivalue_args.size(), "Wrong number of ivalue arguments");
    return call_with_ivalue_args_<FuncType, func>(ivalue_args, guts::make_index_sequence<num_ivalue_args>(), std::forward<ExtraArgs>(extra_args)...);
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

  // SFINAE over (1) does the operator kernel have a cache and (2) does it return a value or void
  template<class CacheTypeOrVoid, class FuncType, FuncType* kernel, class Enable = void> struct wrap_kernel {};
  // SFINAE version for kernels with output and with cache
  template<class CacheTypeOrVoid, class FuncType, FuncType* kernel>
  struct wrap_kernel<CacheTypeOrVoid, FuncType, kernel, guts::enable_if_t<!std::is_same<void, CacheTypeOrVoid>::value && !std::is_same<void, typename guts::function_traits<FuncType>::return_type>::value>> final {
    static typename guts::function_traits<FuncType>::return_type call(Stack* stack, KernelCache* cache) {
      constexpr size_t num_inputs = guts::function_traits<FuncType>::number_of_parameters - 1; // -1 because it takes the kernel cache as last argument
      auto output = call_with_ivalue_args<FuncType, kernel>(torch::jit::last(*stack, num_inputs), static_cast<CacheTypeOrVoid*>(cache));
      push_outputs<typename guts::function_traits<FuncType>::return_type>(std::move(output), stack);
    }
  };
  // SFINAE version for kernels with output and without a cache
  template<class CacheTypeOrVoid, class FuncType, FuncType* kernel>
  struct wrap_kernel<CacheTypeOrVoid, FuncType, kernel, guts::enable_if_t<std::is_same<void, CacheTypeOrVoid>::value && !std::is_same<void, typename guts::function_traits<FuncType>::return_type>::value>> final {
    static typename guts::function_traits<FuncType>::return_type call(Stack* stack, c10::KernelCache* /*cache*/) {
      constexpr size_t num_inputs = guts::function_traits<FuncType>::number_of_parameters;
      auto output = call_with_ivalue_args<FuncType, kernel>(torch::jit::last(*stack, num_inputs));
      push_outputs<typename guts::function_traits<FuncType>::return_type>(std::move(output), stack);
    }
  };
  // SFINAE version for kernels without output and with a cache
  template<class CacheTypeOrVoid, class FuncType, FuncType* kernel>
  struct wrap_kernel<CacheTypeOrVoid, FuncType, kernel, guts::enable_if_t<!std::is_same<void, CacheTypeOrVoid>::value && std::is_same<void, typename guts::function_traits<FuncType>::return_type>::value>> final {
    static typename guts::function_traits<FuncType>::return_type call(Stack* stack, c10::KernelCache* cache) {
      constexpr size_t num_inputs = guts::function_traits<FuncType>::number_of_parameters - 1; // -1 because it takes the kernel cache as last argument
      call_with_ivalue_args<FuncType, kernel>(torch::jit::last(*stack, num_inputs), static_cast<CacheTypeOrVoid*>(cache));
    }
  };
  // SFINAE version for kernels without output and without a cache
  template<class CacheTypeOrVoid, class FuncType, FuncType* kernel>
  struct wrap_kernel<CacheTypeOrVoid, FuncType, kernel, guts::enable_if_t<std::is_same<void, CacheTypeOrVoid>::value && std::is_same<void, typename guts::function_traits<FuncType>::return_type>::value>> final {
    static typename guts::function_traits<FuncType>::return_type call(Stack* stack, c10::KernelCache* /*cache*/) {
      constexpr size_t num_inputs = guts::function_traits<FuncType>::number_of_parameters;
      call_with_ivalue_args<FuncType, kernel>(torch::jit::last(*stack, num_inputs));
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

  struct KernelForRegistration final {
    TensorTypeId dispatch_key;
    KernelFunction* kernel_func = nullptr;
    KernelCacheCreatorFunction* cache_creator_func = nullptr;
  };

  struct KernelRegistrationConfigParameter final {
    static constexpr bool is_c10_operator_registration_config_parameter = true;

    explicit constexpr KernelRegistrationConfigParameter(KernelFunction* kernel_func, KernelCacheCreatorFunction* cache_creator_func)
    : kernel_func_(kernel_func), cache_creator_func_(cache_creator_func) {
    }

    void apply(KernelForRegistration* registration) const {
      registration->kernel_func = kernel_func_;
      registration->cache_creator_func = cache_creator_func_;
    }

  private:
    KernelFunction* kernel_func_;
    KernelCacheCreatorFunction* cache_creator_func_;
  };

  struct DispatchKeyConfigParameter final {
    static constexpr bool is_c10_operator_registration_config_parameter = true;

    explicit constexpr DispatchKeyConfigParameter(TensorTypeId dispatch_key): dispatch_key_(dispatch_key) {}

    void apply(KernelForRegistration* registration) const {
      registration->dispatch_key = dispatch_key_;
    }

  private:
    TensorTypeId dispatch_key_;
  };
}

template<class FuncType, FuncType* kernel_func, class CacheTypeOrVoid = void>
inline constexpr detail::KernelRegistrationConfigParameter kernel() {
  return detail::KernelRegistrationConfigParameter(&detail::wrap_kernel<CacheTypeOrVoid, FuncType, kernel_func>::call, &detail::cacheCreator<CacheTypeOrVoid>);
}

inline constexpr detail::KernelRegistrationConfigParameter kernel(KernelFunction* kernel_func) {
  return detail::KernelRegistrationConfigParameter(kernel_func, &detail::cacheCreator<void>);
}

inline constexpr detail::DispatchKeyConfigParameter dispatchKey(TensorTypeId dispatch_key) {
  return detail::DispatchKeyConfigParameter(dispatch_key);
}

namespace detail {

template<class... ConfigParameters>
KernelForRegistration make_registration_config(const ConfigParameters&... configParameters) {
  KernelForRegistration registration;

  // apply all configParameters
  (void)std::initializer_list<int>{(configParameters.apply(&registration), 0)...};

  // TODO Allow this for just registering the schema?
  AT_CHECK(registration.kernel_func != nullptr, "Cannot register operator without kernel");

  // if kernel_func is set, so must be cache_creator_func
  AT_ASSERT(registration.cache_creator_func != nullptr);

  return registration;
}
}

class RegisterOperators final {
public:
  RegisterOperators() = default;
  RegisterOperators(const RegisterOperators&) = delete;
  RegisterOperators(RegisterOperators&&) = default;
  RegisterOperators& operator=(const RegisterOperators&) = delete;
  RegisterOperators& operator=(RegisterOperators&&) = default;

  template<class... ConfigParameters>
  RegisterOperators op(FunctionSchema schema, const ConfigParameters&... configParameters) && {
    const detail::KernelForRegistration registration = make_registration_config(configParameters...);
    registrars_.emplace_back(std::move(schema), registration.dispatch_key, registration.kernel_func, registration.cache_creator_func);
    return std::move(*this);
  }

private:
  std::vector<c10::detail::OperatorRegistrar> registrars_;
};

}
