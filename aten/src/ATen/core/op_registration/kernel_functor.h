#pragma once

#include <ATen/core/op_registration/kernel_stackbased.h>
#include <ATen/core/op_registration/infer_schema.h>

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
class OperatorKernel : public KernelCache {};

namespace detail {
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

  template<class T>
  IValue return_type_to_ivalue(T&& t) {
    return IValue(std::forward<T>(t));
  }

  template<class Functor, size_t... ivalue_arg_indices>
  typename guts::infer_function_traits_t<Functor>::return_type call_functor_with_ivalue_args_(Functor* functor, ArrayRef<IValue> ivalue_args, guts::index_sequence<ivalue_arg_indices...>) {
    (void)(ivalue_args); // when sizeof...(ivalue_arg_indices) == 0, this argument would be unused and we have to silence the compiler warning.
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
      torch::jit::push(*stack, return_type_to_ivalue(std::move(output)));
    }
  };
  template<class... OutputTypes>
  struct push_outputs<std::tuple<OutputTypes...>> final {
    static void call(std::tuple<OutputTypes...>&& output, Stack* stack) {
      call_(std::move(output), stack, guts::make_index_sequence<sizeof...(OutputTypes)>());
    }

  private:
    template<size_t... indices>
    static void call_(std::tuple<OutputTypes...>&& output, Stack* stack, guts::index_sequence<indices...>) {
      (void)(stack); // when sizeof...(indices) == 0, this argument would be unused and we have to silence the compiler warning.
      // iterate over all outputs and push them
      (void)std::initializer_list<int>{(
        torch::jit::push(*stack, return_type_to_ivalue(std::move(std::get<indices>(output))))
      , 0)...};
    }
  };

  template<class KernelFunctor, class Enable = void> struct wrap_kernel_functor final {};

  // SFINAE version for kernels that return an output
  template<class KernelFunctor>
  struct wrap_kernel_functor<KernelFunctor, guts::enable_if_t<!std::is_same<void, typename guts::infer_function_traits_t<KernelFunctor>::return_type>::value>> final {
    static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to register a kernel functor using the kernel<Functor>() API, but it doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");

    static void call(Stack* stack, KernelCache* cache) {
      constexpr size_t num_inputs = guts::infer_function_traits_t<KernelFunctor>::number_of_parameters;
      KernelFunctor* functor = static_cast<KernelFunctor*>(cache);
      auto output = call_functor_with_ivalue_args<KernelFunctor>(functor, torch::jit::last(*stack, num_inputs));
      torch::jit::pop(*stack, num_inputs);
      push_outputs<typename guts::infer_function_traits_t<KernelFunctor>::return_type>::call(std::move(output), stack);
    }
  };

  // SFINAE version for kernels that don't return an output
  template<class KernelFunctor>
  struct wrap_kernel_functor<KernelFunctor, guts::enable_if_t<std::is_same<void, typename guts::infer_function_traits_t<KernelFunctor>::return_type>::value>> final {
    static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to register a kernel functor using the kernel<Functor>() API, but it doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");

    static void call(Stack* stack, KernelCache* cache) {
      constexpr size_t num_inputs = guts::infer_function_traits_t<KernelFunctor>::number_of_parameters;
      KernelFunctor* functor = static_cast<KernelFunctor*>(cache);
      call_functor_with_ivalue_args<KernelFunctor>(functor, torch::jit::last(*stack, num_inputs));
      torch::jit::pop(*stack, num_inputs);
    }
  };

  template<class KernelFunctor, class... Args>
  class KernelFactory final {
    static_assert(std::is_constructible<KernelFunctor, Args...>::value, "Wrong argument types for constructor of kernel functor.");

  public:
    explicit constexpr KernelFactory(Args... args)
    : constructor_parameters_(std::move(args)...) {}

    std::unique_ptr<KernelCache> operator()() const {
      return guts::apply(
        [] (const Args&... params) {return guts::make_unique<KernelFunctor>(params...); },
        constructor_parameters_);
    }

  private:
    std::tuple<Args...> constructor_parameters_;
  };

  template<class KernelFunctor>
  class FunctionSchemaInferer final {
  public:
    std::unique_ptr<FunctionSchema> operator()() const {
      return guts::make_unique<FunctionSchema>(inferFunctionSchema<KernelFunctor>("", ""));
    }
  };

  template<class KernelFunctor, class... ConstructorParameters>
  detail::KernelRegistrationConfigParameter<detail::KernelFactory<KernelFunctor, guts::decay_t<ConstructorParameters>...>, detail::FunctionSchemaInferer<KernelFunctor>>
  kernelFunctor(ConstructorParameters&&... constructorParameters) {
    return {
      &detail::wrap_kernel_functor<KernelFunctor>::call,
      detail::KernelFactory<KernelFunctor, guts::decay_t<ConstructorParameters>...>(std::forward<ConstructorParameters>(constructorParameters)...),
      detail::FunctionSchemaInferer<KernelFunctor>()
    };
  }
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
// enable_if: only enable it if KernelFunctor is actually a functor
inline constexpr guts::enable_if_t<guts::is_functor<KernelFunctor>::value,
detail::KernelRegistrationConfigParameter<detail::KernelFactory<KernelFunctor, guts::decay_t<ConstructorParameters>...>, detail::FunctionSchemaInferer<KernelFunctor>>>
kernel(ConstructorParameters&&... constructorParameters) {
  static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to register a kernel functor using the kernel<Functor>() API, but it doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");
  static_assert(std::is_constructible<KernelFunctor, ConstructorParameters...>::value, "Wrong argument list for constructor of kernel functor. The arguments to kernel<Functor>(arguments...) must match one of the constructors of Functor.");

  return detail::kernelFunctor<KernelFunctor>(std::forward<ConstructorParameters>(constructorParameters)...);
}

}
