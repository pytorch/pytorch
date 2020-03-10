#pragma once

#include <ATen/core/op_registration/infer_schema.h>
#include <ATen/core/ivalue.h>

namespace c10 {

using Stack = torch::jit::Stack; // TODO Instead of this, move torch::jit::Stack to the c10 namespace.
class OperatorHandle;

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
 * The kernel class is allowed to have members but these are equivalent
 * to global variables. The kernel implementation is responsible for
 * preventing race conditions on them.
 *
 * See below for how to register this kernel with PyTorch.
 */
struct CAFFE2_API OperatorKernel {
  virtual ~OperatorKernel() = default;
};

namespace detail {
  // supported_primitive_arg_types defines which primitive types we allow in
  // kernel functions as arguments or returns.
  // Additionally, we support lists, dicts and optionals containing these types.
  using supported_primitive_arg_types = guts::typelist::typelist<
    int64_t,
    double,
    bool,
    std::string,
    at::Tensor,
    at::Scalar,
    c10::QScheme
  >;

  template<class T, bool AllowDeprecatedTypes, class Enable = void> struct assert_is_valid_input_type {
    assert_is_valid_input_type() {
      auto tmap = c10::getCustomClassTypeMap();
      TORCH_CHECK(c10::isCustomClassRegistered<T>(), "Tried to use undefined class as input argument");
    }
  };

  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<T, AllowDeprecatedTypes, std::enable_if_t<guts::typelist::contains<supported_primitive_arg_types, T>::value>> {
    // everything is ok, this is a primitive type
  };

  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<c10::optional<T>, AllowDeprecatedTypes>
  : assert_is_valid_input_type<T, AllowDeprecatedTypes> {};

  template <bool AllowDeprecatedTypes, class... Args>
  struct TypeCheckHelper;

  template <bool AllowDeprecatedTypes>
  struct TypeCheckHelper<AllowDeprecatedTypes> {};

  template <bool AllowDeprecatedTypes, class Head, class... Rest>
  struct TypeCheckHelper<AllowDeprecatedTypes, Head, Rest...> : TypeCheckHelper<AllowDeprecatedTypes, Rest...> {
    assert_is_valid_input_type<Head, AllowDeprecatedTypes> check;
  };

  template<class... Contained, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<std::tuple<Contained...>, AllowDeprecatedTypes> : TypeCheckHelper<AllowDeprecatedTypes, Contained...> {};

  template<class Key, class Value, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<Dict<Key, Value>, AllowDeprecatedTypes>
  : assert_is_valid_input_type<Value, AllowDeprecatedTypes> {
    static_assert(guts::typelist::contains<impl::valid_dict_key_types, Key>::value, "You tried to register a kernel with an unsupported input type: Dict<Key, Value> where Key is invalid. We only support int64_t, double, bool, and string.");
  };

  template<class Key, class Value, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<std::unordered_map<Key, Value>, AllowDeprecatedTypes>
  : assert_is_valid_input_type<Value, AllowDeprecatedTypes> {
    static_assert(AllowDeprecatedTypes, "You tried to register a kernel with an unsupported input type: std::unordered_map<Key, Value>. Please use Dict<Key, Value> instead.");
    static_assert(guts::typelist::contains<impl::valid_dict_key_types, Key>::value, "You tried to register a kernel with an unsupported input type: std::unordered_map<Key, Value> where Key is invalid. We only support int64_t, double, bool, and string.");
  };

  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<List<T>, AllowDeprecatedTypes>
  : assert_is_valid_input_type<T, AllowDeprecatedTypes> {
    static_assert(!std::is_same<T, at::Scalar>::value, "You tried to register a kernel with an unsupported input type: List<Scalar>. Please use List<int64_t>, List<double> or Tensor instead.");
  };

  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<std::vector<T>, AllowDeprecatedTypes>
  : assert_is_valid_input_type<T, AllowDeprecatedTypes> {
    static_assert(!std::is_same<T, at::Scalar>::value, "You tried to register a kernel with an unsupported input type: std::vector<Scalar>. Please use List<int64_t>, List<double> or Tensor instead.");
    // TODO static_assert(AllowDeprecatedTypes, "You tried to register a kernel with an unsupported input type: std::vector<T>. Please use List<T> instead.");
  };

  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<c10::ArrayRef<T>, AllowDeprecatedTypes>
  : assert_is_valid_input_type<T, AllowDeprecatedTypes> {
    static_assert(!std::is_same<T, at::Scalar>::value, "You tried to register a kernel with an unsupported input type: ArrayRef<Scalar>. Please use List<int64_t>, List<double> or Tensor instead.");
  };

  // The following specialisations of assert_is_valid_input_type are technically not
  // necessary since we would hit the base case and show an error message
  // there if they didn't exist, but we can show a better error message
  // in some common error scenarios.
  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_same<float, T>::value>> {
    // There is no reason to support float when we have double. Keep the API lean.
    static_assert(guts::false_t<T>::value, "You tried to register a kernel with an unsupported input type: float. Please use double instead.");
  };
  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_same<const char*, T>::value>> {
    static_assert(guts::false_t<T>::value, "You tried to register a kernel with an unsupported input type: const char*. Please use std::string instead.");
  };
  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_same<std::vector<bool>, T>::value>> {
    static_assert(guts::false_t<T>::value, "You tried to register a kernel with an unsupported input type: vector<bool>. Please use List<bool> instead.");
  };
  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_integral<T>::value && !guts::typelist::contains<supported_primitive_arg_types, T>::value>> {
    static_assert(guts::false_t<T>::value, "You tried to register a kernel with an unsupported integral input type. Please use int64_t instead.");
  };

  template<class T, bool AllowDeprecatedTypes, class Enable = void> struct assert_is_valid_output_type {
    assert_is_valid_output_type() {
      auto tmap = getCustomClassTypeMap();
      TORCH_CHECK(c10::isCustomClassRegistered<T>(), "Tried to use undefined class as output");
    }
  };

  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_output_type<T, AllowDeprecatedTypes, std::enable_if_t<guts::typelist::contains<supported_primitive_arg_types, T>::value>> {
    // everything is ok, this is a primitive type
  };

  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_output_type<c10::optional<T>, AllowDeprecatedTypes>
  : assert_is_valid_output_type<T, AllowDeprecatedTypes> {};

  template<class Key, class Value, bool AllowDeprecatedTypes>
  struct assert_is_valid_output_type<Dict<Key, Value>, AllowDeprecatedTypes>
  : assert_is_valid_output_type<Value, AllowDeprecatedTypes> {
    static_assert(guts::typelist::contains<impl::valid_dict_key_types, Key>::value, "You tried to register a kernel with an unsupported output type: Dict<Key, Value> where Key is invalid. We only support int64_t, double, bool, and string.");
    static_assert(!std::is_same<Value, at::Scalar>::value, "You tried to register a kernel with an unsupported output type: Dict<Key, Scalar>. Please use Dict<Key, int64_t> or Dict<Key, double>.");
  };

  template<class Key, class Value, bool AllowDeprecatedTypes>
  struct assert_is_valid_output_type<std::unordered_map<Key, Value>, AllowDeprecatedTypes>
  : assert_is_valid_output_type<Value, AllowDeprecatedTypes> {
    static_assert(AllowDeprecatedTypes, "You tried to register a kernel with an unsupported output type: std::unordered_map<Key, Value>. Please use Dict<Key, Value> instead.");
    static_assert(guts::typelist::contains<impl::valid_dict_key_types, Key>::value, "You tried to register a kernel with an unsupported output type: std::unordered_map<Key, Value> where Key is invalid. We only support int64_t, double, bool, and string.");
    static_assert(!std::is_same<Value, at::Scalar>::value, "You tried to register a kernel with an unsupported output type: std::unordered_map<Key, Scalar>. Please use Dict<Key, int64_t> or Dict<Key, double>.");
  };

  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_output_type<List<T>, AllowDeprecatedTypes>
  : assert_is_valid_output_type<T, AllowDeprecatedTypes> {
    static_assert(!std::is_same<T, at::Scalar>::value, "You tried to register a kernel with an unsupported output type: List<Scalar>. Please use List<int64_t>, List<double> or Tensor instead.");
  };

  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_output_type<std::vector<T>, AllowDeprecatedTypes>
  : assert_is_valid_output_type<T, AllowDeprecatedTypes> {
    static_assert(!std::is_same<T, at::Scalar>::value, "You tried to register a kernel with an unsupported output type: std::vector<Scalar>. Please use List<int64_t>, List<double> or Tensor instead.");
    // TODO static_assert(AllowDeprecatedTypes, "You tried to register a kernel with an unsupported output type: std::vector<T>. Please use List<T> instead.");
  };

  // The following specialisations of assert_is_valid_output_type are technically not
  // necessary since we would hit the base case and show an error message
  // there if they didn't exist, but we can show a better error message
  // in some common error scenarios.
  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_output_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_same<float, T>::value>> {
    // There is no reason to support float when we have double. Keep the API lean.
    static_assert(guts::false_t<T>::value, "You tried to register a kernel with an unsupported output type: float. Please use double instead.");
  };
  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_output_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_same<const char*, T>::value>> {
    static_assert(guts::false_t<T>::value, "You tried to register a kernel with an unsupported output type: const char*. Please use std::string instead.");
  };
  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_output_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_same<std::vector<bool>, T>::value>> {
    static_assert(guts::false_t<T>::value, "You tried to register a kernel with an unsupported output type: vector<bool>. Please use List<bool> instead.");
  };
  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_output_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_integral<T>::value && !guts::typelist::contains<supported_primitive_arg_types, T>::value>> {
    static_assert(guts::false_t<T>::value, "You tried to register a kernel with an unsupported integral output type. Please use int64_t instead.");
  };


  template<class T, bool AllowDeprecatedTypes>
  struct ivalue_to_arg final {
    static T call(IValue&& v) {
      assert_is_valid_input_type<T, AllowDeprecatedTypes>();
      return std::move(v).to<T>();
    }
  };

  template<class T, bool AllowDeprecatedTypes>
  struct ivalue_to_arg<ArrayRef<T>, AllowDeprecatedTypes> final {
    // If an argument is ArrayRef<T>, convert the IValue to a std::vector<T> and pass that
    // to the operator. std::vector<T> is implicitly convertible to ArrayRef<T>.
    static std::vector<T> call(IValue&& v) {
      return ivalue_to_arg<std::vector<T>, AllowDeprecatedTypes>::call(std::move(v));
    }
  };

  template<class T, bool AllowDeprecatedTypes>
  IValue return_to_ivalue(T&& v) {
    assert_is_valid_output_type<T, AllowDeprecatedTypes>();
    return c10::ivalue::from(std::forward<T>(v));
  }

  template<class Functor, bool AllowDeprecatedTypes, size_t... ivalue_arg_indices>
  typename guts::infer_function_traits_t<Functor>::return_type call_functor_with_args_from_stack_(Functor* functor, Stack* stack, std::index_sequence<ivalue_arg_indices...>) {
    (void)(stack); // when sizeof...(ivalue_arg_indices) == 0, this argument would be unused and we have to silence the compiler warning.

    constexpr size_t num_ivalue_args = sizeof...(ivalue_arg_indices);

    using IValueArgTypes = typename guts::infer_function_traits_t<Functor>::parameter_types;
    return (*functor)(ivalue_to_arg<std::remove_cv_t<std::remove_reference_t<guts::typelist::element_t<ivalue_arg_indices, IValueArgTypes>>>, AllowDeprecatedTypes>::call(
      std::move(torch::jit::peek(*stack, ivalue_arg_indices, num_ivalue_args))
    )...);
  }

  template<class Functor, bool AllowDeprecatedTypes>
  typename guts::infer_function_traits_t<Functor>::return_type call_functor_with_args_from_stack(Functor* functor, Stack* stack) {
    constexpr size_t num_ivalue_args = guts::infer_function_traits_t<Functor>::number_of_parameters;
    return call_functor_with_args_from_stack_<Functor, AllowDeprecatedTypes>(functor, stack, std::make_index_sequence<num_ivalue_args>());
  }

  template<class OutputType, bool AllowDeprecatedTypes>
  struct push_outputs final {
    static void call(OutputType&& output, Stack* stack) {
      torch::jit::push(*stack, return_to_ivalue<OutputType, AllowDeprecatedTypes>(std::move(output)));
    }
  };
  template<class... OutputTypes, bool AllowDeprecatedTypes>
  struct push_outputs<std::tuple<OutputTypes...>, AllowDeprecatedTypes> final {
    static void call(std::tuple<OutputTypes...>&& output, Stack* stack) {
      call_(std::move(output), stack, std::make_index_sequence<sizeof...(OutputTypes)>());
    }

  private:
    template<size_t... indices>
    static void call_(std::tuple<OutputTypes...>&& output, Stack* stack, std::index_sequence<indices...>) {
      torch::jit::push(*stack, return_to_ivalue<OutputTypes, AllowDeprecatedTypes>(std::move(std::get<indices>(output)))...);
    }
  };

  template<class KernelFunctor, bool AllowDeprecatedTypes, class Enable = void> struct make_boxed_from_unboxed_functor final {};

  // SFINAE version for kernels that return an output
  template<class KernelFunctor, bool AllowDeprecatedTypes>
  struct make_boxed_from_unboxed_functor<KernelFunctor, AllowDeprecatedTypes, std::enable_if_t<!std::is_same<void, typename guts::infer_function_traits_t<KernelFunctor>::return_type>::value>> final {
    static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to register a kernel functor using the kernel<Functor>() API, but it doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");

    static void call(OperatorKernel* functor, const OperatorHandle&, Stack* stack) {
      constexpr size_t num_inputs = guts::infer_function_traits_t<KernelFunctor>::number_of_parameters;
      KernelFunctor* functor_ = static_cast<KernelFunctor*>(functor);
      auto output = call_functor_with_args_from_stack<KernelFunctor, AllowDeprecatedTypes>(functor_, stack);
      torch::jit::drop(*stack, num_inputs);
      push_outputs<typename guts::infer_function_traits_t<KernelFunctor>::return_type, AllowDeprecatedTypes>::call(std::move(output), stack);
    }
  };

  // SFINAE version for kernels that don't return an output
  template<class KernelFunctor, bool AllowDeprecatedTypes>
  struct make_boxed_from_unboxed_functor<KernelFunctor, AllowDeprecatedTypes, std::enable_if_t<std::is_same<void, typename guts::infer_function_traits_t<KernelFunctor>::return_type>::value>> final {
    static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to register a kernel functor using the kernel<Functor>() API, but it doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");

    static void call(OperatorKernel* functor, const OperatorHandle&, Stack* stack) {
      constexpr size_t num_inputs = guts::infer_function_traits_t<KernelFunctor>::number_of_parameters;
      KernelFunctor* functor_ = static_cast<KernelFunctor*>(functor);
      call_functor_with_args_from_stack<KernelFunctor, AllowDeprecatedTypes>(functor_, stack);
      torch::jit::pop(*stack, num_inputs);
    }
  };

  template<class KernelFunctor, class OpSignature> struct wrap_kernel_functor_unboxed_ final {};
  template<class KernelFunctor, class ReturnType, class... ParameterTypes> struct wrap_kernel_functor_unboxed_<KernelFunctor, ReturnType(ParameterTypes...)> final {
    static_assert(std::is_same<ReturnType, typename guts::infer_function_traits_t<KernelFunctor>::return_type>::value, "Return type mismatch");
    static_assert(std::is_same<guts::typelist::typelist<ParameterTypes...>, typename guts::infer_function_traits_t<KernelFunctor>::parameter_types>::value, "Parameter types mismatch");

    static ReturnType call(OperatorKernel* functor, ParameterTypes... args) {
      KernelFunctor* functor_ = static_cast<KernelFunctor*>(functor);
      return (*functor_)(std::forward<ParameterTypes>(args)...);
    }
  };
  template<class KernelFunctor> using wrap_kernel_functor_unboxed = wrap_kernel_functor_unboxed_<KernelFunctor, typename guts::infer_function_traits_t<KernelFunctor>::func_type>;

  template<class KernelFunctor, class... Args>
  class KernelFactory final {
    static_assert(std::is_constructible<KernelFunctor, Args...>::value, "Wrong argument types for constructor of kernel functor.");

  public:
    explicit constexpr KernelFactory(Args... args)
    : constructor_parameters_(std::move(args)...) {}

    std::unique_ptr<OperatorKernel> operator()() const {
      return guts::apply(
        [] (const Args&... params) -> std::unique_ptr<OperatorKernel> {return guts::make_unique_base<OperatorKernel, KernelFunctor>(params...); },
        constructor_parameters_);
    }

  private:
    std::tuple<Args...> constructor_parameters_;
  };

  template<class FuncType>
  std::unique_ptr<FunctionSchema> inferFunctionSchemaFlattenedReturns_() {
    return std::make_unique<FunctionSchema>(inferFunctionSchemaFlattenedReturns<FuncType>("", ""));
  }

  template<class KernelFunctor>
  class FunctionSchemaInferer final {
  public:
    using func_type = typename c10::guts::infer_function_traits_t<KernelFunctor>::func_type;
    std::unique_ptr<FunctionSchema> operator()() const {
      return inferFunctionSchemaFlattenedReturns_<func_type>();
    }
  };
}

}

namespace torch {
  using OperatorKernel = c10::OperatorKernel;
}
