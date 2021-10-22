#pragma once

#include <ATen/core/boxing/OperatorKernel.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/Metaprogramming.h>

namespace c10 {

using Stack = torch::jit::Stack; // TODO Instead of this, move torch::jit::Stack to the c10 namespace.
class OperatorHandle;

/*
 * [Note: Argument forwarding in the dispatcher]
 *
 * The dispatcher uses a somewhat unusual way to forward arguments through several layers of
 * wrapper functions. This can be confusing because an experienced C++ programmer would look at this
 * and think "oh this is supposed to be forwarding a universal reference but the && is missing. This is a bug.".
 * It is not a bug. The common way in C++ to forward arguments is to use universal references:
 *
 * > template<class T> void func(T&& arg) { func2(std::forward<T>(arg)); }
 *
 * but that relies on inferring the correct reference type (i.e. value vs & vs &&) from the argument.
 * In our case, we cannot rely on the argument as supplied by the caller, because that could infer a
 * different reference type than was used in the kernel function. The correct reference type
 * is dictated by the kernel signature and must be identical since we cast function pointers
 * through void* pointers and mismatches would be UB. So we need a forwarding pattern that determines
 * the reference type to use by looking at the explicitly supplied operator signature, not by looking at
 * the argument we're calling it with.
 *
 * What does std::forward do, exactly?
 * ------------------------------------
 * std::forward<T>(t) is a way to cast t to the reference type supplied in T.
 * Let's assume decay_t<T> == U and T is either U or some reference of U.
 *  - std::forward<T&>(t) will return U&, no matter what kind of reference t is.
 *  - std::forward<T&&>(t) will return U&&, no matter what kind of reference t is.
 *  - std::forward<T>(t) will return U&& (not U!), no matter what kind of reference t is.
 *
 * For universal references, that means that in the following function
 * > template<class T> void func(T&& arg) { func2(std::forward<T>(arg)); }
 *
 *  - when called with arg being a rvalue reference or non-reference value, T gets inferred to be
 *    a non-reference U, and std::forward<T>(t) will return U&&, correctly moving the argument.
 *  - when called with arg behind a lvalue reference, T gets inferred to be U& because that's the only
 *    way to match the signature (in C++, a type that is (T&)&& will collapse to T&).
 *    That means std::forward<T>(t) will return U& and the value will not be moved but passed on as
 *    a lvalue reference.
 *
 * How do we use that?
 * ------------------------------------
 * But std::forward can also be used outside of the common "universal forwarding" pattern to change
 * reference types. So instead of following the common C++ pattern, we notice what
 * std::forward<T>() actually does, and that is it takes a value and changes its reference to the
 * type of reference passed in as T. If we don't infer T but explicitly specify it, we can use this
 * to forward based on an explicitly specified reference type instead of the inferred argument type.
 *
 * This is why many of the dispatcher functions look like
 * > template<class T> func(T t) { func2<T>(std::forward<T>(t)); }
 * instead of the common
 * > template<class T> func(T&& t) { func2(std::forward<T>(t)); }
 *
 * and are expected to be called by explicitly specifying the template parameters in a way that matches
 * the expected operator signature at each call site.
 */

namespace impl {
  // supported_primitive_arg_types defines which primitive types we allow in
  // kernel functions as arguments or returns.
  // Additionally, we support lists, dicts and optionals containing these types.
  using supported_primitive_arg_types = guts::typelist::typelist<
    int64_t,
    double,
    bool,
    c10::string_view,
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
      "You tried to register a kernel with an unsupported input type: const char*. Please use c10::string_view instead.");
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
      "You tried to register a kernel with an unsupported output type: const char*. Please use c10::string_view instead.");
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

  template<class T>
  struct decay_if_not_tensor final {
    using type = std::decay_t<T>;
  };

  template<>
  struct decay_if_not_tensor<at::Tensor&> final {
    using type = at::Tensor&;
  };

  template<>
  struct decay_if_not_tensor<const at::Tensor&> final {
    using type = const at::Tensor&;
  };

  template<class T, bool AllowDeprecatedTypes>
  struct ivalue_to_arg final {
    static decltype(auto) call(IValue& v) {
      assert_is_valid_input_type<T, AllowDeprecatedTypes>();
      return std::move(v).to<T>();
    }
  };

  // The following two specializations take advantage of specialized
  // `toTensor()` overloads on IValue to avoid copying.
  template<bool AllowDeprecatedTypes>
  struct ivalue_to_arg<at::Tensor&, AllowDeprecatedTypes> final {
    // We cannot use the default implementation if they asked for a
    // `at::Tensor&` because it moves from the IValue, so it can't get
    // an lvalue reference.
    static at::Tensor& call(IValue& v) {
      // Tensor& is valid, don't bother asserting
      return v.toTensor();
    }
  };

  template<bool AllowDeprecatedTypes>
  struct ivalue_to_arg<const at::Tensor&, AllowDeprecatedTypes> final {
    // We should not use the default implementation if they asked for
    // a `const at::Tensor&` because it moves from the IValue and they
    // didn't ask for that.
    static const at::Tensor& call(IValue& v) {
      // const Tensor& is valid, don't bother asserting
      return v.toTensor();
    }
  };

  template<class T, bool AllowDeprecatedTypes>
  struct ivalue_to_arg<ArrayRef<T>, AllowDeprecatedTypes> final {
    // If an argument is ArrayRef<T>, convert the IValue to a std::vector<T> and pass that
    // to the operator. std::vector<T> is implicitly convertible to ArrayRef<T>.
    static std::vector<T> call(IValue& v) {
      return ivalue_to_arg<std::vector<T>, AllowDeprecatedTypes>::call(v);
    }
  };
  template<class T, bool AllowDeprecatedTypes>
  struct ivalue_to_arg<optional<ArrayRef<T>>, AllowDeprecatedTypes> final {
    // If an argument is optional<ArrayRef<T>>, convert the IValue to an optional<std::vector<T>> and pass that
    // to the operator. OptionalArray<T> is basically a optional<std::vector<T>> but impliticly convertible
    // to optional<ArrayRef<T>>.
    static OptionalArray<T> call(IValue& v) {
      return ivalue_to_arg<OptionalArray<T>, AllowDeprecatedTypes>::call(v);
    }
  };

  // return_to_ivalue
  template<class T, bool AllowDeprecatedTypes, class Enable = void>
  struct return_to_ivalue final {};

  template<class T, bool AllowDeprecatedTypes>
  struct return_to_ivalue<T, AllowDeprecatedTypes, std::enable_if_t<!std::is_same<at::Tensor&, T>::value>> final {
    static IValue call(T&& v) {
      assert_is_valid_output_type<T, AllowDeprecatedTypes>();
      return c10::ivalue::from(std::move(v));
    }
    static IValue copy(const T& v) {
      assert_is_valid_output_type<T, AllowDeprecatedTypes>();
      return IValue(v);
    }
  };

  // Special case to allow kernels to return `Tensor&`.
  // TODO Delete this once kernels don't do that anymore
  template<bool AllowDeprecatedTypes>
  struct return_to_ivalue<at::Tensor&, AllowDeprecatedTypes, void> final {
    static IValue call(at::Tensor& v) {
      return c10::ivalue::from(v);
    }
    static IValue copy(at::Tensor& v) {
      return IValue(v);
    }
  };

  // wrap_kernel_functor_unboxed_

  template<class KernelFunctor, class OpSignature>
  struct wrap_kernel_functor_unboxed_ final {};

  // This specialization is for kernels with a first argument that is NOT of type DispatchKeySet
  // This includes kernels with 0 arguments.
  template<class KernelFunctor, class ReturnType, class... ParameterTypes>
  struct wrap_kernel_functor_unboxed_<KernelFunctor, ReturnType(ParameterTypes...)> final {
    static_assert(std::is_same<ReturnType, typename guts::infer_function_traits_t<KernelFunctor>::return_type>::value,
      "Return type mismatch");
    static_assert(std::is_same<guts::typelist::typelist<ParameterTypes...>, typename guts::infer_function_traits_t<KernelFunctor>::parameter_types>::value,
      "Parameter types mismatch");

    // See [Note: Argument forwarding in the dispatcher] for why ParameterTypes doesn't use &&
    static ReturnType call(OperatorKernel* functor, DispatchKeySet, ParameterTypes... args) {
      KernelFunctor* functor_ = static_cast<KernelFunctor*>(functor);
      // Note [Plumbing Keys Through The Dispatcher 2]
      // See Note [Plumbing Keys Through The Dispatcher] for the background.
      // This functor explicitly takes in a dispatchKeySet and drops it on the floor- it does not forward it to the registered kernel.
      //
      // This is due to the calling convention within the dispatcher, which expects all registered kernels to have a first argument of type
      // DispatchKeySet.
      // This is not the case for pretty much all manually written kernels, however- this functor serves to separate the calling convention
      // of the dispatcher from the calling convention of manually written kernels.
      return (*functor_)(std::forward<ParameterTypes>(args)...);
    }
  };

  // This specialization is for kernels with a first argument of type DispatchKeySet
  template<class KernelFunctor, class ReturnType, class... ParameterTypes>
  struct wrap_kernel_functor_unboxed_<KernelFunctor, ReturnType(DispatchKeySet, ParameterTypes...)> final {
    static_assert(std::is_same<ReturnType, typename guts::infer_function_traits_t<KernelFunctor>::return_type>::value,
      "Return type mismatch");
    static_assert(std::is_same<guts::typelist::typelist<DispatchKeySet, ParameterTypes...>, typename guts::infer_function_traits_t<KernelFunctor>::parameter_types>::value,
      "Parameter types mismatch");

    // See [Note: Argument forwarding in the dispatcher] for why ParameterTypes doesn't use &&
    static ReturnType call(OperatorKernel* functor, DispatchKeySet dispatchKeySet, ParameterTypes... args) {
      KernelFunctor* functor_ = static_cast<KernelFunctor*>(functor);
      // We're explicitly taking in a dispatchKeySet and forwarding it to the registered kernel.
      // See Note [Plumbing Keys Through The Dispatcher 2] for details.
      return (*functor_)(dispatchKeySet, std::forward<ParameterTypes>(args)...);
    }
  };

  template<class KernelFunctor>
  using wrap_kernel_functor_unboxed = wrap_kernel_functor_unboxed_<KernelFunctor, typename guts::infer_function_traits_t<KernelFunctor>::func_type>;

  // call_functor_with_args_from_stack

  template<class Functor, bool AllowDeprecatedTypes, size_t... ivalue_arg_indices,  typename... ArgTypes>
  std::decay_t<typename guts::infer_function_traits_t<Functor>::return_type>
  call_functor_with_args_from_stack_(OperatorKernel* functor, DispatchKeySet dispatchKeySet, Stack* stack, std::index_sequence<ivalue_arg_indices...>, guts::typelist::typelist<ArgTypes...>*) {
    (void)(stack); // when sizeof...(ivalue_arg_indices) == 0, this argument would be unused and we have to silence the compiler warning.

    // We're explicitly filtering out DispatchKeySet from the argument list.
    // Some kernels take a DispatchKeySet as their first argument in order to plumb keys through the dispatcher.
    // We don't want to expose the DispatchKeySet type to jit, so we don't include this argument on the stack.
    // See Note [Plumbing Keys Through The Dispatcher] for the background.
    return wrap_kernel_functor_unboxed<Functor>::call(functor, dispatchKeySet,
      ivalue_to_arg<typename decay_if_not_tensor<ArgTypes>::type, AllowDeprecatedTypes>::call(
        torch::jit::peek(*stack, ivalue_arg_indices, sizeof...(ivalue_arg_indices))
    )...);
  }

  template<class Functor, bool AllowDeprecatedTypes>
  std::decay_t<typename guts::infer_function_traits_t<Functor>::return_type>
  call_functor_with_args_from_stack(OperatorKernel* functor, DispatchKeySet dispatchKeySet, Stack* stack) {
    // We're explicitly filtering out DispatchKeySet from the argument list.
    // Some kernels take a DispatchKeySet as their first argument in order to plumb keys through the dispatcher.
    // We don't want to expose the DispatchKeySet type to jit, so we don't include this argument on the stack.
    // See Note [Plumbing Keys Through The Dispatcher] for the background.
    using ArgTypes = typename c10::remove_DispatchKeySet_arg_from_func<Functor>::parameter_types;
    constexpr size_t num_ivalue_args = guts::typelist::size<ArgTypes>::value;
    return call_functor_with_args_from_stack_<Functor, AllowDeprecatedTypes>(functor, dispatchKeySet, stack, std::make_index_sequence<num_ivalue_args>(), static_cast<ArgTypes*>(nullptr));
  }

  // push_outputs

  template<class OutputType, bool AllowDeprecatedTypes>
  struct push_outputs final {
    // Contrary to [Note: Argument forwarding in the dispatcher], we use OutputType&& here
    // to avoid one extra call to the move constructor in this case. This is still not a
    // universal reference though because OutputType is an explicitly specified class
    // template parameter.
    static void call(OutputType&& output, Stack* stack) {
      torch::jit::push(*stack, return_to_ivalue<OutputType, AllowDeprecatedTypes>::call(std::forward<OutputType>(output)));
    }
    static void copy(const OutputType& output, Stack* stack) {
      torch::jit::push(*stack, return_to_ivalue<OutputType, AllowDeprecatedTypes>::copy(output));
    }
  };
  template<class... OutputTypes, bool AllowDeprecatedTypes>
  struct push_outputs<std::tuple<OutputTypes...>, AllowDeprecatedTypes> final {
    static void call(std::tuple<OutputTypes...>&& output, Stack* stack) {
      call_(std::move(output), stack, std::make_index_sequence<sizeof...(OutputTypes)>());
    }
    static void copy(const std::tuple<OutputTypes...>& output, Stack* stack) {
      copy_(output, stack, std::make_index_sequence<sizeof...(OutputTypes)>());
    }

  private:
    template<size_t... indices>
    static void call_(std::tuple<OutputTypes...>&& output, Stack* stack, std::index_sequence<indices...>) {
      torch::jit::push(*stack, return_to_ivalue<OutputTypes, AllowDeprecatedTypes>::call(std::forward<OutputTypes>(std::get<indices>(output)))...);
    }
    template<size_t... indices>
    static void copy_(const std::tuple<OutputTypes...>& output, Stack* stack, std::index_sequence<indices...>) {
      torch::jit::push(*stack, return_to_ivalue<OutputTypes, AllowDeprecatedTypes>::copy(std::get<indices>(output))...);
    }
  };
  template<bool AllowDeprecatedTypes>
  struct push_outputs<void, AllowDeprecatedTypes> final {
    static void call(int /*dummy*/, Stack* /*stack*/) {
    }
    static void copy(int /*dummy*/, Stack* /*stack*/) {
    }
  };

  // make_boxed_from_unboxed_functor

  template<class KernelFunctor, bool AllowDeprecatedTypes>
  struct make_boxed_from_unboxed_functor final {
    static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value,
      "Tried to register a kernel functor using the kernel<Functor>() API, but it doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");

    static void call(OperatorKernel* functor, const OperatorHandle&, DispatchKeySet dispatchKeySet, Stack* stack) {
      using ReturnType = typename guts::infer_function_traits_t<KernelFunctor>::return_type;
      // We're explicitly filtering out DispatchKeySet from the argument list.
      // Some kernels take a DispatchKeySet as their first argument in order to plumb keys through the dispatcher.
      // We don't want to expose the DispatchKeySet type to jit, so we don't include this argument on the stack.
      // See Note [Plumbing Keys Through The Dispatcher] for the background.
      using ArgTypes = typename c10::remove_DispatchKeySet_arg_from_func<KernelFunctor>::parameter_types;
      constexpr bool has_outputs = !std::is_same<void, ReturnType>::value;
      constexpr size_t num_inputs = guts::typelist::size<ArgTypes>::value;
#ifdef __cpp_if_constexpr
      if constexpr (has_outputs) {
#else
      guts::if_constexpr<has_outputs>([&] (auto delay_check) {
#endif
        // Decay ReturnType to ReturnType_ so that if a reference gets returned, we actually store it by value
        // and don't get a dangling reference. This is only required because some kernels still return `Tensor&`.
#ifdef __cpp_if_constexpr
        using ReturnType_ = std::decay_t<ReturnType>;
        ReturnType_ output = call_functor_with_args_from_stack<KernelFunctor, AllowDeprecatedTypes>(functor, dispatchKeySet, stack);
#else
        using ReturnType_ = std::decay_t<typename decltype(delay_check)::template type_identity<ReturnType>>;
        ReturnType_ output = call_functor_with_args_from_stack<KernelFunctor, AllowDeprecatedTypes>(functor, dispatchKeySet, delay_check(stack));
#endif
        torch::jit::drop(*stack, num_inputs);
        push_outputs<ReturnType_, AllowDeprecatedTypes>::call(std::move(output), stack);
#ifdef __cpp_if_constexpr
      } else {
#else
      }, /* else */ [&] {
#endif
        call_functor_with_args_from_stack<KernelFunctor, AllowDeprecatedTypes>(functor, dispatchKeySet, stack);
        torch::jit::drop(*stack, num_inputs);
#ifdef __cpp_if_constexpr
      }
#else
      });
#endif
    }
  };
} // namespace impl

} // namespace c10

namespace torch {
  using OperatorKernel = c10::OperatorKernel;
}
