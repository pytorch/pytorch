#pragma once

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
  // supported_primitive_arg_types defines which primitive types we allow in
  // kernel functions as arguments or returns.
  // Additionally, we support lists, dicts and optionals containing these types.
  using supported_primitive_arg_types = guts::typelist::typelist<
    int64_t,
    double,
    bool,
    std::string,
    at::Tensor,
    at::Scalar
  >;

  // ivalue_to_arg_type<T>: Take an IValue that is an argument to a kernel and
  // cast it to the type that should be passed to the kernel function.
  // Examples: If the IValue contains a plain type like an int, return that.
  //           If the IValue contains an IntList, return it as ArrayRef<int>.
  template<class T, class Enable = void> struct ivalue_to_arg_type {
    // This base case is hit whenever a type does not have a specialisation below.
    static_assert(guts::false_t<T>::value, "You tried to register a kernel with an unsupported argument type.");
  };
  template<class T>
  struct ivalue_to_arg_type<T, guts::enable_if_t<guts::typelist::contains<supported_primitive_arg_types, T>::value>> {
    static T call(IValue&& v) {
      return std::move(v).to<T>();
    }
  };
  template<class T>
  struct ivalue_to_arg_type<ArrayRef<T>> {
    static ArrayRef<T> call(const IValue& v) {
      // Note: This takes a `const IValue&` argument and not `IValue&&`, because the
      //        returned ArrayRef is non-owning, so the call site needs to keep ownership
      // TODO Do we want to support ArrayRef<optional<T>> ?
      static_assert(guts::typelist::contains<supported_primitive_arg_types, T>::value, "You tried to register a kernel with an unsupported argument type: c10::ArrayRef<T> and T is not one of the supported primitive types.");
      static_assert(!std::is_same<T, at::Scalar>::value, "You tried to register a kernel with an unsupported argument type: c10::ArrayRef<Scalar>. Please use c10::ArrayRef<int64_t>, c10::ArrayRef<double> or Tensor instead.");
      return v.to<intrusive_ptr<ivalue::List<T>>>()->elements();
    }
  };
  template<class T>
  struct ivalue_to_arg_type<optional<T>> {
    static optional<T> call(IValue&& v) {
      if (v.isNone()) {
        return nullopt;
      }
      return ivalue_to_arg_type<T>::call(std::move(v));
    }
  };
  template<class Key, class Value>
  struct ivalue_to_arg_type<Dict<Key, Value>> {
    static Dict<Key, Value> call(IValue&& v) {
      static_assert(guts::typelist::contains<supported_primitive_arg_types, Key>::value, "You tried to register a kernel with an unsupported argument type: c10::Dict<Key, Value> and Key is not one of the supported primitive types.");
      static_assert(guts::typelist::contains<supported_primitive_arg_types, Value>::value, "You tried to register a kernel with an unsupported argument type: c10::Dict<Key, Value> and Value is not one of the supported primitive types.");

      auto dict_ptr = std::move(v).toGenericDict();
      return impl::toTypedDict<Key, Value>(std::move(dict_ptr->elements()));
    }
  };
  // The following specialisations of ivalue_to_arg_type are technically not
  // necessary since we would hit the base case and show an error message
  // there if they didn't exist, but we can show a better error message
  // in some common error scenarios.
  template<class T>
  struct ivalue_to_arg_type<std::vector<T>> {
    // We don't support std::vector because that would prevent us from doing
    // internal optimization to how we represent lists (e.g. SmallVector).
    // Users should use ArrayRef instead.
    static_assert(guts::false_t<std::vector<T>>::value, "You tried to register a kernel with an unsupported argument type: std::vector<T>. Please use c10::ArrayRef<T> instead.");
  };
  template<class Key, class Value>
  struct ivalue_to_arg_type<std::unordered_map<Key, Value>> {
    // We don't support std::unordered_map because that would prevent us from doing
    // internal optimization to how we represent dicts.
    // Users should use Dict<Key, Value> instead.
    static_assert(guts::false_t<std::unordered_map<Key, Value>>::value, "You tried to register a kernel with an unsupported argument type: std::unordered_map<Key, Value>. Please use c10::Dict<Key, Value> instead.");
  };
  template<class T>
  struct ivalue_to_arg_type<T, guts::enable_if_t<std::is_same<float, T>::value>> {
    // There is no reason to support float when we have double. Keep the API lean.
    static_assert(guts::false_t<T>::value, "You tried to register a kernel with an unsupported argument type: float. Please use double instead.");
  };
  template<class T>
  struct ivalue_to_arg_type<T, guts::enable_if_t<std::is_same<const char*, T>::value>> {
    static_assert(guts::false_t<T>::value, "You tried to register a kernel with an unsupported argument type: const char*. Please use std::string instead.");
  };
  template<class T>
  struct ivalue_to_arg_type<T, guts::enable_if_t<std::is_integral<T>::value && !guts::typelist::contains<supported_primitive_arg_types, T>::value>> {
    static_assert(guts::false_t<T>::value, "You tried to register a kernel with an unsupported integral argument type. Please use int64_t instead.");
  };

  // legacy_ivalue_to_arg_type is like ivalue_to_arg_type but additionally
  // allows a few deprecated types like std::vector.
  template<class T, class Enable = void>
  struct legacy_ivalue_to_arg_type final {
    static auto call(IValue&& v) -> decltype(ivalue_to_arg_type<T>::call(std::move(v))) {
      return ivalue_to_arg_type<T>::call(std::move(v));
    }
  };
  template<class T>
  struct legacy_ivalue_to_arg_type<std::vector<T>, guts::enable_if_t<guts::typelist::contains<supported_primitive_arg_types, T>::value && !std::is_same<std::string, T>::value>> final {
    static std::vector<T> call(IValue&& v) {
      return std::move(*std::move(v).to<intrusive_ptr<ivalue::List<T>>>()).elements();
    }
  };
  template<class T>
  struct legacy_ivalue_to_arg_type<std::vector<T>, guts::enable_if_t<!guts::typelist::contains<supported_primitive_arg_types, T>::value || std::is_same<std::string, T>::value>> final {
    static std::vector<T> call(IValue&& v) {
      auto list = std::move(v).toGenericList();
      std::vector<T> result;
      result.reserve(list->elements().size());
      for (auto&& elem : std::move(list)->elements()) {
        result.push_back(legacy_ivalue_to_arg_type<T>::call(std::move(elem)));
      }
      return result;
    }
  };
  template<class Key, class Value>
  struct legacy_ivalue_to_arg_type<std::unordered_map<Key, Value>> final {
    static std::unordered_map<Key, Value> call(const IValue& v) {
      auto dict = std::move(v).toGenericDict();
      std::unordered_map<Key, Value> result;
      result.reserve(dict->elements().size());
      for (auto& element : dict->elements()) {
        result.emplace(legacy_ivalue_to_arg_type<Key>::call(element.key()), legacy_ivalue_to_arg_type<Value>::call(element.value()));
      }
      return result;
    }
  };

  // TODO Make nesting types work with new style API, e.g. Dicts of lists, lists of lists, and so on

  template<class T, class Enable = void>
  struct return_type_to_ivalue {
    static_assert(guts::false_t<T>::value, "You tried to register a kernel with an unsupported return type.");
  };
  template<class T>
  struct return_type_to_ivalue<T, guts::enable_if_t<guts::typelist::contains<supported_primitive_arg_types, T>::value>> {
    template<class T_>
    static IValue call(T_&& v) {
      return IValue(std::forward<T_>(v));
    }
  };
  template<class T>
  struct return_type_to_ivalue<optional<T>> {
    static IValue call(optional<T>&& v) {
      if (!v.has_value()) {
        return IValue();
      }
      return return_type_to_ivalue<T>::call(std::move(*v));
    }
  };
  template<class T>
  struct return_type_to_ivalue<std::vector<T>> {
    static IValue call(std::vector<T>&& v) {
      static_assert(guts::typelist::contains<supported_primitive_arg_types, T>::value, "You tried to register a kernel with an unsupported return type: vector<T> and T is not one of the supported primitive types.");
      static_assert(!std::is_same<T, at::Scalar>::value, "You tried to register a kernel with an unsupported return type: vector<Scalar>. Please use vector<int64_t>, vector<double> or Tensor instead.");
      return IValue(std::move(v));
    }
  };
  template<class Key, class Value>
  struct return_type_to_ivalue<c10::Dict<Key, Value>> {
    static IValue call(c10::Dict<Key, Value>&& v) {
      static_assert(guts::typelist::contains<supported_primitive_arg_types, Key>::value, "You tried to register a kernel with an unsupported return type: Dict<Key, Value> and Key is not one of the supported primitive types.");
      static_assert(guts::typelist::contains<supported_primitive_arg_types, Value>::value, "You tried to register a kernel with an unsupported return type: Dict<Key, Value> and Value is not one of the supported primitive types.");
      return IValue(impl::toGenericDict(std::move(v)));
    }
  };
  // The following specialisations of return_type_to_ivalue are technically not
  // necessary since we would hit the base case and show an error message
  // there if they didn't exist, but we can show a better error message
  // in some common error scenarios.
  template<class T>
  struct return_type_to_ivalue<c10::ArrayRef<T>> {
    static_assert(guts::false_t<c10::ArrayRef<T>>::value, "You tried to register a kernel with an unsupported return type: c10::ArrayRef<T>. Please use std::vector<T> instead.");
  };
  template<class Key, class Value>
  struct return_type_to_ivalue<std::unordered_map<Key, Value>> {
    static_assert(guts::false_t<std::unordered_map<Key, Value>>::value, "You tried to register a kernel with an unsupported return type: std::unordered_map<Key, Value>. Please use c10::Dict<Key, Value> instead.");
  };
  template<class T>
  struct return_type_to_ivalue<T, guts::enable_if_t<std::is_same<float, T>::value>> {
    static_assert(guts::false_t<T>::value, "You tried to register a kernel with an unsupported return type: float. Please use double instead.");
  };
  template<class T>
  struct return_type_to_ivalue<T, guts::enable_if_t<std::is_same<const char*, T>::value>> {
    static_assert(guts::false_t<T>::value, "You tried to register a kernel with an unsupported return type: const char*. Please use std::string instead.");
  };
  template<class T>
  struct return_type_to_ivalue<T, guts::enable_if_t<std::is_integral<T>::value && !guts::typelist::contains<supported_primitive_arg_types, T>::value>> {
    static_assert(guts::false_t<T>::value, "You tried to register a kernel with an unsupported integral return argument type. Please use int64_t instead.");
  };
  // legacy_return_type_to_ivalue is like return_type_to_ivalue but additionally
  // allows a few deprecated types like std::unordered_map.
  template<class T, class Enable = void>
  struct legacy_return_type_to_ivalue final {
    template<class T_>
    static IValue call(T_&& v) {
      return return_type_to_ivalue<T>::call(std::forward<T_>(v));
    }
  };
  template<class T>
  struct legacy_return_type_to_ivalue<std::vector<T>, guts::enable_if_t<guts::typelist::contains<supported_primitive_arg_types, T>::value && !std::is_same<std::string, T>::value>> {
    static IValue call(std::vector<T>&& v) {
      return return_type_to_ivalue<std::vector<T>>::call(std::move(v));
    }
  };
  template<class T>
  struct legacy_return_type_to_ivalue<std::vector<T>, guts::enable_if_t<!guts::typelist::contains<supported_primitive_arg_types, T>::value || std::is_same<std::string, T>::value>> {
    static IValue call(std::vector<T>&& v) {
      static_assert(!std::is_same<T, at::Scalar>::value, "You tried to register a kernel with an unsupported return type: vector<Scalar>. Please use vector<int64_t>, vector<double> or Tensor instead.");
      std::vector<IValue> result;
      result.reserve(v.size());
      for (auto& elem : v) {
        result.push_back(legacy_return_type_to_ivalue<T>::call(std::move(elem)));
      }
      return result;
    }
  };
  template<class Key, class Value>
  struct legacy_return_type_to_ivalue<std::unordered_map<Key, Value>> final {
    static IValue call(std::unordered_map<Key, Value>&& v) {
      c10::impl::GenericDict dict;
      dict.reserve(v.size());
      for (auto& element : v) {
        dict.insert(legacy_return_type_to_ivalue<Key>::call(Key{element.first}), legacy_return_type_to_ivalue<Value>::call(std::move(element.second)));
      }
      return dict;
    }
  };

  template<bool AllowDeprecatedTypes, class T> using ivalue_to_arg = guts::conditional_t<AllowDeprecatedTypes, legacy_ivalue_to_arg_type<T>, ivalue_to_arg_type<T>>;
  template<bool AllowDeprecatedTypes, class T> using return_to_ivalue = guts::conditional_t<AllowDeprecatedTypes, legacy_return_type_to_ivalue<guts::decay_t<T>>, return_type_to_ivalue<guts::decay_t<T>>>;

  template<class Functor, bool AllowDeprecatedTypes, size_t... ivalue_arg_indices>
  typename guts::infer_function_traits_t<Functor>::return_type call_functor_with_args_from_stack_(Functor* functor, Stack* stack, guts::index_sequence<ivalue_arg_indices...>) {
    (void)(stack); // when sizeof...(ivalue_arg_indices) == 0, this argument would be unused and we have to silence the compiler warning.

    constexpr size_t num_ivalue_args = sizeof...(ivalue_arg_indices);

    using IValueArgTypes = typename guts::infer_function_traits_t<Functor>::parameter_types;
    return (*functor)(ivalue_to_arg<AllowDeprecatedTypes, guts::remove_cv_t<guts::remove_reference_t<guts::typelist::element_t<ivalue_arg_indices, IValueArgTypes>>>>::call(
      std::move(torch::jit::peek(*stack, ivalue_arg_indices, num_ivalue_args))
    )...);
  }

  template<class Functor, bool AllowDeprecatedTypes>
  typename guts::infer_function_traits_t<Functor>::return_type call_functor_with_args_from_stack(Functor* functor, Stack* stack) {
    constexpr size_t num_ivalue_args = guts::infer_function_traits_t<Functor>::number_of_parameters;
    return call_functor_with_args_from_stack_<Functor, AllowDeprecatedTypes>(functor, stack, guts::make_index_sequence<num_ivalue_args>());
  }

  template<class OutputType, bool AllowDeprecatedTypes>
  struct push_outputs final {
    static void call(OutputType&& output, Stack* stack) {
      torch::jit::push(*stack, return_to_ivalue<AllowDeprecatedTypes, OutputType>::call(std::move(output)));
    }
  };
  template<class... OutputTypes, bool AllowDeprecatedTypes>
  struct push_outputs<std::tuple<OutputTypes...>, AllowDeprecatedTypes> final {
    static void call(std::tuple<OutputTypes...>&& output, Stack* stack) {
      call_(std::move(output), stack, guts::make_index_sequence<sizeof...(OutputTypes)>());
    }

  private:
    template<size_t... indices>
    static void call_(std::tuple<OutputTypes...>&& output, Stack* stack, guts::index_sequence<indices...>) {
      torch::jit::push(*stack, return_to_ivalue<AllowDeprecatedTypes, OutputTypes>::call(std::move(std::get<indices>(output)))...);
    }
  };

  template<class KernelFunctor, bool AllowDeprecatedTypes, class Enable = void> struct wrap_kernel_functor final {};

  // SFINAE version for kernels that return an output
  template<class KernelFunctor, bool AllowDeprecatedTypes>
  struct wrap_kernel_functor<KernelFunctor, AllowDeprecatedTypes, guts::enable_if_t<!std::is_same<void, typename guts::infer_function_traits_t<KernelFunctor>::return_type>::value>> final {
    static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to register a kernel functor using the kernel<Functor>() API, but it doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");

    static void call(Stack* stack, KernelCache* cache) {
      constexpr size_t num_inputs = guts::infer_function_traits_t<KernelFunctor>::number_of_parameters;
      KernelFunctor* functor = static_cast<KernelFunctor*>(cache);
      auto output = call_functor_with_args_from_stack<KernelFunctor, AllowDeprecatedTypes>(functor, stack);
      torch::jit::drop(*stack, num_inputs);
      push_outputs<typename guts::infer_function_traits_t<KernelFunctor>::return_type, AllowDeprecatedTypes>::call(std::move(output), stack);
    }
  };

  // SFINAE version for kernels that don't return an output
  template<class KernelFunctor, bool AllowDeprecatedTypes>
  struct wrap_kernel_functor<KernelFunctor, AllowDeprecatedTypes, guts::enable_if_t<std::is_same<void, typename guts::infer_function_traits_t<KernelFunctor>::return_type>::value>> final {
    static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to register a kernel functor using the kernel<Functor>() API, but it doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");

    static void call(Stack* stack, KernelCache* cache) {
      constexpr size_t num_inputs = guts::infer_function_traits_t<KernelFunctor>::number_of_parameters;
      KernelFunctor* functor = static_cast<KernelFunctor*>(cache);
      call_functor_with_args_from_stack<KernelFunctor, AllowDeprecatedTypes>(functor, stack);
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
}

}
