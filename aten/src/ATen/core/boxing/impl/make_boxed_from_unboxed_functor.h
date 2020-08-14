#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <c10/util/Metaprogramming.h>

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

namespace impl {
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
    c10::QScheme,
    c10::ScalarType,
    c10::Device,
    c10::Layout,
    c10::MemoryFormat,
    at::Dimname
  >;

  // We have an unboxed functor in hand that takes C++ arguments, and
  // we're building a boxed functor wrapper for it that takes IValues.
  // So "outside" is boxed and "inside" is unboxed.
  //
  // So a valid input type is one that our boxed functor wrapper can
  // unbox from an IValue into a C++ value.
  //
  // Whereas a valid output type is one that our wrapper can recieve
  // as a C++ value from the unboxed functor, and box into an IValue.

  //
  // assert_is_valid_input_type
  // checks that T can be unboxed from an IValue into a C++ value.
  //

  template<class T, bool AllowDeprecatedTypes, class Enable = void>
  struct assert_is_valid_input_type {
    assert_is_valid_input_type() {
      guts::if_constexpr<guts::typelist::contains<supported_primitive_arg_types, T>::value>([] {
        /* everything is ok, this is a primitive type */
      }, /* else */ [] {
        /* otherwise this must be an instance of a valid custom class, since it can only
           have been created via IValue(x), which ensures this. */
      });
    }
  };

  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<c10::optional<T>, AllowDeprecatedTypes>
  : assert_is_valid_input_type<T, AllowDeprecatedTypes> {};

  template <bool AllowDeprecatedTypes, class... Args>
  struct TypeCheckHelper;

  template <bool AllowDeprecatedTypes>
  struct TypeCheckHelper<AllowDeprecatedTypes> {};

  template <bool AllowDeprecatedTypes, class Head, class... Rest>
  struct TypeCheckHelper<AllowDeprecatedTypes, Head, Rest...>
  : TypeCheckHelper<AllowDeprecatedTypes, Rest...> {
    assert_is_valid_input_type<Head, AllowDeprecatedTypes> check;
  };

  template<class... Contained, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<std::tuple<Contained...>, AllowDeprecatedTypes>
  : TypeCheckHelper<AllowDeprecatedTypes, Contained...> {};

  template<class Key, class Value, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<Dict<Key, Value>, AllowDeprecatedTypes>
  : assert_is_valid_input_type<Value, AllowDeprecatedTypes> {
    static_assert(guts::typelist::contains<impl::valid_dict_key_types, Key>::value,
      "You tried to register a kernel with an unsupported input type: Dict<Key, Value> where Key is invalid. We only support int64_t, double, bool, and string.");
  };

  template<class Key, class Value, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<std::unordered_map<Key, Value>, AllowDeprecatedTypes>
  : assert_is_valid_input_type<Value, AllowDeprecatedTypes> {
    static_assert(AllowDeprecatedTypes,
      "You tried to register a kernel with an unsupported input type: std::unordered_map<Key, Value>. Please use Dict<Key, Value> instead.");
    static_assert(guts::typelist::contains<impl::valid_dict_key_types, Key>::value,
      "You tried to register a kernel with an unsupported input type: std::unordered_map<Key, Value> where Key is invalid. We only support int64_t, double, bool, and string.");
  };

  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<List<T>, AllowDeprecatedTypes>
  : assert_is_valid_input_type<T, AllowDeprecatedTypes> {
    static_assert(!std::is_same<T, at::Scalar>::value,
      "You tried to register a kernel with an unsupported input type: List<Scalar>. Please use List<int64_t>, List<double> or Tensor instead.");
  };

  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<std::vector<T>, AllowDeprecatedTypes>
  : assert_is_valid_input_type<T, AllowDeprecatedTypes> {
    static_assert(!std::is_same<T, at::Scalar>::value,
      "You tried to register a kernel with an unsupported input type: std::vector<Scalar>. Please use List<int64_t>, List<double> or Tensor instead.");
    // TODO static_assert(AllowDeprecatedTypes, "You tried to register a kernel with an unsupported input type: std::vector<T>. Please use List<T> instead.");
  };

  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<c10::ArrayRef<T>, AllowDeprecatedTypes>
  : assert_is_valid_input_type<T, AllowDeprecatedTypes> {
    static_assert(!std::is_same<T, at::Scalar>::value,
      "You tried to register a kernel with an unsupported input type: ArrayRef<Scalar>. Please use List<int64_t>, List<double> or Tensor instead.");
  };

  template<class T, size_t N, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<std::array<T, N>, AllowDeprecatedTypes>
  : assert_is_valid_input_type<T, AllowDeprecatedTypes> {
    static_assert(!std::is_same<T, at::Scalar>::value,
      "You tried to register a kernel with an unsupported input type: std::array<Scalar, N>. Please use std::array<int64_t, N> instead.");
  };

  // The following specialisations of assert_is_valid_input_type are technically not
  // necessary since we would hit the base case and show an error message
  // there if they didn't exist, but we can show a better error message
  // in some common error scenarios.
  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_same<float, T>::value>> {
    // There is no reason to support float when we have double. Keep the API lean.
    static_assert(guts::false_t<T>::value,
      "You tried to register a kernel with an unsupported input type: float. Please use double instead.");
  };
  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_same<const char*, T>::value>> {
    static_assert(guts::false_t<T>::value,
      "You tried to register a kernel with an unsupported input type: const char*. Please use std::string instead.");
  };
  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_same<std::vector<bool>, T>::value>> {
    static_assert(guts::false_t<T>::value,
      "You tried to register a kernel with an unsupported input type: vector<bool>. Please use List<bool> instead.");
  };
  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_integral<T>::value && !guts::typelist::contains<supported_primitive_arg_types, T>::value>> {
    static_assert(guts::false_t<T>::value,
      "You tried to register a kernel with an unsupported integral input type. Please use int64_t instead.");
  };

  //
  // assert_is_valid_output_type
  //

  template<class T, bool AllowDeprecatedTypes, class Enable = void>
  struct assert_is_valid_output_type {
    assert_is_valid_output_type() {
      guts::if_constexpr<guts::typelist::contains<supported_primitive_arg_types, T>::value>([] {
        /* everything is ok, this is a primitive type */
      }, /* else */ [] {
        /* otherwise T is verified to be a registered custom class in the IValue
          constructor, so no benefit in double-checking here */
      });
    }
  };

  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_output_type<c10::optional<T>, AllowDeprecatedTypes>
  : assert_is_valid_output_type<T, AllowDeprecatedTypes> {};

  template<class Key, class Value, bool AllowDeprecatedTypes>
  struct assert_is_valid_output_type<Dict<Key, Value>, AllowDeprecatedTypes>
  : assert_is_valid_output_type<Value, AllowDeprecatedTypes> {
    static_assert(guts::typelist::contains<impl::valid_dict_key_types, Key>::value,
      "You tried to register a kernel with an unsupported output type: Dict<Key, Value> where Key is invalid. We only support int64_t, double, bool, and string.");
    static_assert(!std::is_same<Value, at::Scalar>::value,
      "You tried to register a kernel with an unsupported output type: Dict<Key, Scalar>. Please use Dict<Key, int64_t> or Dict<Key, double>.");
  };

  template<class Key, class Value, bool AllowDeprecatedTypes>
  struct assert_is_valid_output_type<std::unordered_map<Key, Value>, AllowDeprecatedTypes>
  : assert_is_valid_output_type<Value, AllowDeprecatedTypes> {
    static_assert(AllowDeprecatedTypes,
      "You tried to register a kernel with an unsupported output type: std::unordered_map<Key, Value>. Please use Dict<Key, Value> instead.");
    static_assert(guts::typelist::contains<impl::valid_dict_key_types, Key>::value,
      "You tried to register a kernel with an unsupported output type: std::unordered_map<Key, Value> where Key is invalid. We only support int64_t, double, bool, and string.");
    static_assert(!std::is_same<Value, at::Scalar>::value,
      "You tried to register a kernel with an unsupported output type: std::unordered_map<Key, Scalar>. Please use Dict<Key, int64_t> or Dict<Key, double>.");
  };

  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_output_type<List<T>, AllowDeprecatedTypes>
  : assert_is_valid_output_type<T, AllowDeprecatedTypes> {
    static_assert(!std::is_same<T, at::Scalar>::value,
      "You tried to register a kernel with an unsupported output type: List<Scalar>. Please use List<int64_t>, List<double> or Tensor instead.");
  };

  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_output_type<std::vector<T>, AllowDeprecatedTypes>
  : assert_is_valid_output_type<T, AllowDeprecatedTypes> {
    static_assert(!std::is_same<T, at::Scalar>::value,
      "You tried to register a kernel with an unsupported output type: std::vector<Scalar>. Please use List<int64_t>, List<double> or Tensor instead.");
    // TODO static_assert(AllowDeprecatedTypes, "You tried to register a kernel with an unsupported output type: std::vector<T>. Please use List<T> instead.");
  };

  template<class T, size_t N, bool AllowDeprecatedTypes>
  struct assert_is_valid_output_type<std::array<T, N>, AllowDeprecatedTypes>
  : assert_is_valid_output_type<T, AllowDeprecatedTypes> {
    static_assert(!std::is_same<T, at::Scalar>::value,
      "You tried to register a kernel with an unsupported output type: std::array<Scalar, N>. Please use std::array<int64_t, N> instead.");
  };

  // The following specialisations of assert_is_valid_output_type are technically not
  // necessary since we would hit the base case and show an error message
  // there if they didn't exist, but we can show a better error message
  // in some common error scenarios.
  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_output_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_same<float, T>::value>> {
    // There is no reason to support float when we have double. Keep the API lean.
    static_assert(guts::false_t<T>::value,
      "You tried to register a kernel with an unsupported output type: float. Please use double instead.");
  };
  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_output_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_same<const char*, T>::value>> {
    static_assert(guts::false_t<T>::value,
      "You tried to register a kernel with an unsupported output type: const char*. Please use std::string instead.");
  };
  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_output_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_same<std::vector<bool>, T>::value>> {
    static_assert(guts::false_t<T>::value,
      "You tried to register a kernel with an unsupported output type: vector<bool>. Please use List<bool> instead.");
  };
  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_output_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_integral<T>::value && !guts::typelist::contains<supported_primitive_arg_types, T>::value>> {
    static_assert(guts::false_t<T>::value,
      "You tried to register a kernel with an unsupported integral output type. Please use int64_t instead.");
  };

  // ivalue_to_arg

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
  template<bool AllowDeprecatedTypes>
  struct ivalue_to_arg<optional<ArrayRef<int64_t>>, AllowDeprecatedTypes> final {
    // If an argument is optional<ArrayRef<int64_t>>, convert the IValue to a optional<std::vector<int64_t>> and pass that
    // to the operator.
    static OptionalArray<int64_t> call(IValue&& v) {
      return std::move(v).toOptionalIntArray();
    }
  };
  template<bool AllowDeprecatedTypes>
  struct ivalue_to_arg<optional<ArrayRef<double>>, AllowDeprecatedTypes> final {
    // If an argument is optional<ArrayRef<T>>, convert the IValue to a optional<std::vector<T>> and pass that
    // to the operator.
    static OptionalArray<double> call(IValue&& v) {
      return std::move(v).toOptionalDoubleArray();
    }
  };

  // return_to_ivalue

  template<class T, bool AllowDeprecatedTypes>
  IValue return_to_ivalue(T&& v) {
    assert_is_valid_output_type<T, AllowDeprecatedTypes>();
    return c10::ivalue::from(std::forward<T>(v));
  }

  // Special case to allow kernels to return `Tensor&`.
  // TODO Delete this once kernels don't do that anymore
  template<>
  inline IValue return_to_ivalue<at::Tensor&, false>(at::Tensor& v) {
    return c10::ivalue::from(v);
  }

  // reference_cast allows casting references, e.g. T&& to T&:
  //    T make_t() {}
  //    T& v = reference_cast<T&>(make_t()); // make_t() returns a T&& which is cast to T&.
  // If the target is a non-reference value, then it gets moved:
  //    T make_t() {}
  //    T v = reference_cast<T>(make_t()); // no copies involved
  // The first example actually also shows why reference_cast is usually a very bad idea. v now is a lvalue
  // reference to a dead temporary. Use with caution!
  template<class T, class U>
  T reference_cast(U&& t) {
      return std::forward<T>(t);
  }

  template<class Functor, bool AllowDeprecatedTypes, size_t... ivalue_arg_indices>
  std::decay_t<typename guts::infer_function_traits_t<Functor>::return_type>
  call_functor_with_args_from_stack_(Functor* functor, Stack* stack, std::index_sequence<ivalue_arg_indices...>) {
    (void)(stack); // when sizeof...(ivalue_arg_indices) == 0, this argument would be unused and we have to silence the compiler warning.

    constexpr size_t num_ivalue_args = sizeof...(ivalue_arg_indices);

    /*
     * For ops that take "Tensor&" as an argument, ivalue_to_arg would still return a "Tensor" by value
     * and C++ doesn't allow us to call (*functor) with a temporary "Tensor" when it expects "Tensor&".
     * We use reference_cast to explicitly cast our temporary to a "Tensor&" and make it pass the compiler.
     * Even though usually dangerous, this is ok here because temporaries live until the end of the statement.
     * TODO We should remove reference_cast once kernels don't take "Tensor&" arguments anymore
     */
    using ArgTypes = typename guts::infer_function_traits_t<Functor>::parameter_types;
    return (*functor)(reference_cast<guts::typelist::element_t<ivalue_arg_indices, ArgTypes>>(
      ivalue_to_arg<std::decay_t<guts::typelist::element_t<ivalue_arg_indices, ArgTypes>>, AllowDeprecatedTypes>::call(
        std::move(torch::jit::peek(*stack, ivalue_arg_indices, num_ivalue_args))
    ))...);
  }

  template<class Functor, bool AllowDeprecatedTypes>
  std::decay_t<typename guts::infer_function_traits_t<Functor>::return_type>
  call_functor_with_args_from_stack(Functor* functor, Stack* stack) {
    constexpr size_t num_ivalue_args = guts::infer_function_traits_t<Functor>::number_of_parameters;
    return call_functor_with_args_from_stack_<Functor, AllowDeprecatedTypes>(functor, stack, std::make_index_sequence<num_ivalue_args>());
  }

  // push_outputs

  template<class OutputType, bool AllowDeprecatedTypes>
  struct push_outputs final {
    static void call(OutputType&& output, Stack* stack) {
      torch::jit::push(*stack, return_to_ivalue<OutputType, AllowDeprecatedTypes>(std::forward<OutputType>(output)));
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
  template<bool AllowDeprecatedTypes>
  struct push_outputs<void, AllowDeprecatedTypes> final {
    static void call(int /*dummy*/, Stack* /*stack*/) {
    }
  };

  // make_boxed_from_unboxed_functor

  template<class KernelFunctor, bool AllowDeprecatedTypes>
  struct make_boxed_from_unboxed_functor final {
    static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value,
      "Tried to register a kernel functor using the kernel<Functor>() API, but it doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");

    static void call(OperatorKernel* functor, const OperatorHandle&, Stack* stack) {
      constexpr size_t num_inputs = guts::infer_function_traits_t<KernelFunctor>::number_of_parameters;
      KernelFunctor* functor_ = static_cast<KernelFunctor*>(functor);

      using ReturnType = typename guts::infer_function_traits_t<KernelFunctor>::return_type;
      constexpr bool has_outputs = !std::is_same<void, ReturnType>::value;
      guts::if_constexpr<has_outputs>([&] (auto delay_check) {
        // Decay ReturnType to ReturnType_ so that if a reference gets returned, we actually store it by value
        // and don't get a dangling reference. This is only required because some kernels still return `Tensor&`.
        using ReturnType_ = std::decay_t<typename decltype(delay_check)::template type_identity<ReturnType>>;
        ReturnType_ output = call_functor_with_args_from_stack<KernelFunctor, AllowDeprecatedTypes>(functor_, delay_check(stack));
        torch::jit::drop(*stack, num_inputs);
        push_outputs<ReturnType_, AllowDeprecatedTypes>::call(std::move(output), stack);
      }, /* else */ [&] {
        call_functor_with_args_from_stack<KernelFunctor, AllowDeprecatedTypes>(functor_, stack);
        torch::jit::drop(*stack, num_inputs);
      });
    }
  };

  // wrap_kernel_functor_unboxed_

  template<class KernelFunctor, class OpSignature>
  struct wrap_kernel_functor_unboxed_ final {};

  template<class KernelFunctor, class ReturnType, class... ParameterTypes>
  struct wrap_kernel_functor_unboxed_<KernelFunctor, ReturnType(ParameterTypes...)> final {
    static_assert(std::is_same<ReturnType, typename guts::infer_function_traits_t<KernelFunctor>::return_type>::value,
      "Return type mismatch");
    static_assert(std::is_same<guts::typelist::typelist<ParameterTypes...>, typename guts::infer_function_traits_t<KernelFunctor>::parameter_types>::value,
      "Parameter types mismatch");

    static ReturnType call(OperatorKernel* functor, ParameterTypes... args) {
      KernelFunctor* functor_ = static_cast<KernelFunctor*>(functor);
      return (*functor_)(std::forward<ParameterTypes>(args)...);
    }
  };

  template<class KernelFunctor>
  using wrap_kernel_functor_unboxed = wrap_kernel_functor_unboxed_<KernelFunctor, typename guts::infer_function_traits_t<KernelFunctor>::func_type>;

} // namespace impl

} // namespace c10

namespace torch {
  using OperatorKernel = c10::OperatorKernel;
}
