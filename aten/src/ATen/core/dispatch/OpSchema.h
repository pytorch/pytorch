#pragma once

#include <ATen/core/ivalue.h>
#include <c10/util/Array.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/TypeList.h>
#include <c10/core/DeviceType.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/stack.h>
#include <ATen/core/dispatch/KernelFunction.h>
#include <ATen/core/function_schema.h>

namespace c10 {

namespace details {

/**
 * If Arg is a Tensor or reference to a Tensor, provide the member constant value equal to true.  Otherwise
 * return false.
 */
template <class Arg>
using is_tensor_arg = std::
    is_same<at::Tensor, guts::remove_cv_t<guts::remove_reference_t<Arg>>>;

/**
 * Extract the type ids of all tensors in a variadic list of arguments
 *
 * @tparam Args Inferred variadic list of argument types
 * @param args List of arguments to get type ids from
 * @return guts::array<TensorParameterDispatchKey, n>, where n is the number of tensor arguments (is_tensor_arg) in the class
 */
template<class OpSchemaDef>
TensorTypeId getDispatchKey_(ArrayRef<IValue> args) {
  using ParameterTypes = typename guts::function_traits<typename OpSchemaDef::Signature>::parameter_types;
  static constexpr size_t index_of_first_tensor_arg = guts::typelist::find_if<ParameterTypes, is_tensor_arg>::value;
  return args[index_of_first_tensor_arg].toTensor().type_id(); // TODO Possible without copying the Tensor holder?
}

/**
 * If T is a struct with a type field Signature, provides the member constant
 * @tparam T
 */
template<class T, typename = void>
struct has_signature_defined : std::false_type {};
template<class T>
struct has_signature_defined<T, guts::void_t<
  typename T::Signature
>> : std::true_type {};

// TODO Test has_signature_defined

template<class T, typename = void>
struct has_parameter_names_defined : std::false_type {};
template<class T>
struct has_parameter_names_defined<T, guts::void_t<
  decltype(T::parameter_names)
>> : std::true_type {};

// TODO Test has_parameter_names_defined

template<class T, typename = void>
struct has_name_defined : std::false_type {};
template<class T>
struct has_name_defined<T, guts::void_t<
        decltype(T::name)
>> : std::true_type {};

// TODO Test has_name_defined

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

template<class FuncType, class... ExtraArgs, size_t... ivalue_arg_indices>
typename guts::function_traits<FuncType>::return_type call_with_ivalue_args_(FuncType* func, ArrayRef<IValue> ivalue_args, guts::index_sequence<ivalue_arg_indices...>, ExtraArgs&&... extra_args) {
  using IValueArgTypes = typename guts::function_traits<FuncType>::parameter_types;
  return (*func)(ivalue_to_arg_type<guts::remove_cv_t<guts::remove_reference_t<guts::typelist::element_t<ivalue_arg_indices, IValueArgTypes>>>>::call(ivalue_args[ivalue_arg_indices])..., std::forward<ExtraArgs>(extra_args)...);
}

template<class FuncType, class... ExtraArgs>
typename guts::function_traits<FuncType>::return_type call_with_ivalue_args(FuncType* func, ArrayRef<IValue> ivalue_args, ExtraArgs&&... extra_args) {
  constexpr size_t num_ivalue_args = guts::function_traits<FuncType>::number_of_parameters - sizeof...(ExtraArgs);
  return call_with_ivalue_args_<FuncType>(func, ivalue_args, guts::make_index_sequence<num_ivalue_args>(), std::forward<ExtraArgs>(extra_args)...);
}

template<class OutputType>
struct write_outputs final {
  static void call(OutputType&& output, ArrayRef<IValue> outputs) {
    write_outputs<std::tuple<OutputType>>(std::tuple<OutputType>(std::move(output)), outputs);
  }
};
template<class... OutputTypes>
struct write_outputs<std::tuple<OutputTypes...>> final {
  static void call(std::tuple<OutputTypes...>&& output, ArrayRef<IValue> outputs) {
    AT_ASSERT(outputs.size() == sizeof...(OutputTypes)); // Mismatch in number of returns between kernel function and operator schema.
    for (size_t i = 0; i < sizeof...(OutputTypes); ++i) {
      outputs[i] = return_type_to_ivalue(std::move(output));
    }
  }
};


// SFINAE over (1) does the operator kernel have a cache and (2) does it return a value or void
template<class CacheTypeOrVoid, class FuncType, class Enable = void> struct call_kernel_with_ivalue_args {};
// SFINAE version for kernels with output and with cache
template<class CacheTypeOrVoid, class FuncType>
struct call_kernel_with_ivalue_args<CacheTypeOrVoid, FuncType, guts::enable_if_t<!std::is_same<void, CacheTypeOrVoid>::value && !std::is_same<void, typename guts::function_traits<FuncType>::return_type>::value>> final {
  static typename guts::function_traits<FuncType>::return_type call(FuncType* func, ArrayRef<IValue> ivalue_args, ArrayRef<IValue> outputs, c10::KernelCache* cache) {
    auto output = call_with_ivalue_args(func, ivalue_args, static_cast<CacheTypeOrVoid*>(cache));
    write_outputs<typename guts::function_traits<FuncType>::return_type>(std::move(output), outputs);
  }
};
// SFINAE version for kernels with output and without a cache
template<class CacheTypeOrVoid, class FuncType>
struct call_kernel_with_ivalue_args<CacheTypeOrVoid, FuncType, guts::enable_if_t<std::is_same<void, CacheTypeOrVoid>::value && !std::is_same<void, typename guts::function_traits<FuncType>::return_type>::value>> final {
  static typename guts::function_traits<FuncType>::return_type call(FuncType* func, ArrayRef<IValue> ivalue_args, ArrayRef<IValue> outputs, c10::KernelCache* /*cache*/) {
    auto output = call_with_ivalue_args(func, ivalue_args);
    write_outputs<typename guts::function_traits<FuncType>::return_type>(std::move(output), outputs);
  }
};
// SFINAE version for kernels without output and with a cache
template<class CacheTypeOrVoid, class FuncType>
struct call_kernel_with_ivalue_args<CacheTypeOrVoid, FuncType, guts::enable_if_t<!std::is_same<void, CacheTypeOrVoid>::value && std::is_same<void, typename guts::function_traits<FuncType>::return_type>::value>> final {
  static typename guts::function_traits<FuncType>::return_type call(FuncType* func, ArrayRef<IValue> ivalue_args, ArrayRef<IValue> outputs, c10::KernelCache* cache) {
    call_with_ivalue_args(func, ivalue_args, static_cast<CacheTypeOrVoid*>(cache));
  }
};
// SFINAE version for kernels without output and without a cache
template<class CacheTypeOrVoid, class FuncType>
struct call_kernel_with_ivalue_args<CacheTypeOrVoid, FuncType, guts::enable_if_t<std::is_same<void, CacheTypeOrVoid>::value && std::is_same<void, typename guts::function_traits<FuncType>::return_type>::value>> final {
  static typename guts::function_traits<FuncType>::return_type call(FuncType* func, ArrayRef<IValue> ivalue_args, ArrayRef<IValue> outputs, c10::KernelCache* /*cache*/) {
    call_with_ivalue_args(func, ivalue_args);
  }
};

template<class FuncType, class AddedParameter, class Enable = void> struct add_ptr_parameter_if_not_void final {};
template<class Return, class... Parameters, class AddedParameter>
struct add_ptr_parameter_if_not_void<Return(Parameters...), AddedParameter, guts::enable_if_t<!std::is_same<void, AddedParameter>::value>> final {
  using type = Return(Parameters..., AddedParameter*);
};
template<class FuncType> struct add_ptr_parameter_if_not_void<FuncType, void, void> final {
  using type = FuncType;
};

template<class ReturnType> struct parse_return_types_ final {
  using type = guts::typelist::typelist<ReturnType>;
};

template<class... ReturnTypes> struct parse_return_types_<std::tuple<ReturnTypes...>> final {
  using type = guts::typelist::typelist<ReturnTypes...>;
};

template<> struct parse_return_types_<void> final {
  using type = guts::typelist::typelist<>;
};

/**
 * Wrapper class around a user-provided schema definition some useful information about the schema.
 *
 * @tparam OpSchemaDef Operator schema definition.  See OpSchema for more details.
 */
template<class OpSchemaDef> class OpSignatureSchema final {
  static_assert(details::has_signature_defined<OpSchemaDef>::value, "Operator schema doesn't define a valid Signature member type.");
  static_assert(guts::is_function_type<typename OpSchemaDef::Signature>::value, "Signature member of operator schema must be a function type.");

  using signature_traits = guts::function_traits<typename OpSchemaDef::Signature>;
public:
  /**
   * The function type OpSchemaDef::Signature
   */
  using func_type = typename signature_traits::func_type;
  /**
   * The return type of the function OpSchemaDef::Signature
   */
  using return_type = typename signature_traits::return_type;
  /**
   * A type list of the parameter types of OpSchemaDef::Signature
   */
  using parameter_types = typename signature_traits::parameter_types;

  using return_types = typename parse_return_types_<return_type>::type;

  /**
   * The number of arguments of OpSchemaDef::Signature
   */
  static constexpr size_t num_args = guts::typelist::size<parameter_types>::value;
  /**
   * The number of tensor arguments (as per is_tensor_arg) in OpSchemaDef::Signature
   */
  static constexpr size_t num_tensor_args = guts::typelist::count_if<details::is_tensor_arg, parameter_types>::value;

  static constexpr size_t num_outputs = OpSchemaDef::num_outputs();

  template<class CacheTypeOrVoid> using func_type_with_cache = typename add_ptr_parameter_if_not_void<func_type, CacheTypeOrVoid>::type;

  template<class CacheTypeOrVoid, func_type_with_cache<CacheTypeOrVoid>* kernel>
  static void wrap_kernel(Stack* stack, KernelCache* cache) {
    constexpr size_t num_inputs = guts::typelist::size<parameter_types>::value;
    constexpr size_t num_outputs = 1; // TODO allow multiple outputs if it's a tuple

    ArrayRef<IValue> inputs = torch::jit::peekSlice(*stack, 0, num_inputs + num_outputs, num_inputs);
    ArrayRef<IValue> outputs = torch::jit::peekSlice(*stack, 0, num_outputs, num_outputs);

    call_kernel_with_ivalue_args<CacheTypeOrVoid, func_type_with_cache<CacheTypeOrVoid>>::call(kernel, inputs, outputs, cache);
  }

private:
  static_assert(details::has_parameter_names_defined<OpSchemaDef>::value, "Operator schema doesn't define parameter_names member.");
  // TODO Allow simpler definition of parameter_names without having to spell out the guts::array type in the schema def.
  static_assert(std::is_same<guts::array<const char*, num_args>, decltype(OpSchemaDef::parameter_names())>::value, "Operator schema defines parameter_names member, but it isn't the correct type. Must be a static constexpr function returning guts::array of const char* with one entry for each parameter.");

public:
  /**
   * The names of the parameters (as per OpSchemaDef::parameter_names)
   * @return Array
   */
  static constexpr const guts::array<const char*, num_args> parameter_names() {
    return OpSchemaDef::parameter_names();
  }
};

/**
 * Wrapper class around a user-defined schema definition providing a way of computing a dispatch key
 * from arguments matching the signature of that schema.
 *
 * @tparam OpSchemaDef Operator schema definition.  See OpSchema for more details.
 * @tparam Enable Inferred, used to control specialization
 */
template<class OpSchemaDef, class Enable = void> class OpDispatchKeySchema final {};

template<class OpSchemaDef>
class OpDispatchKeySchema<OpSchemaDef> final {
  using signature = OpSignatureSchema<OpSchemaDef>;

public:
  static inline TensorTypeId dispatch_key(const Stack* stack) {
    /* TODO Should we make this a runtime assert now?
    using guts::typelist::map_t;
    using guts::typelist::typelist;
    static_assert(std::is_same<
      map_t<guts::remove_cv_t, map_t<guts::remove_reference_t, typelist<Args...>>>,
      map_t<guts::remove_cv_t, map_t<guts::remove_reference_t, typename signature::parameter_types>>
      >::value, "Invalid argument types passed to OpSchema::dispatch_key()");*/
    return details::getDispatchKey_<OpSchemaDef>(torch::jit::last(*stack, signature::num_args));
  }
};

template<class OpSchemaDef>
class OpMetadataSchema final {
private:
    static_assert(has_name_defined<OpSchemaDef>::value, "The operator schema has to define a 'static constexpr const char* name = ...' member to specify the operator name.");
    static_assert(std::is_same<const char* const, decltype(OpSchemaDef::name)>::value, "The 'name' member of the operator schema must have type 'static constexpr const char*'");

public:
    static constexpr const char* name() {
        return OpSchemaDef::name;
    }
};

}  // namespace details

/**
 * Wrapper class for user-defined OpSchemaDef, providing functionality for determining
 * information about the signature and dispatching on that signature.  This is the
 * "public" facing class.
 *
 * @tparam OpSchemaDef User-defined OpSchemaDef.
 *   This struct is expected to define:
 *      - a function type Signature
 *      - a constexpr array<const char*, n_args> parameter_names field (where n_args is
 *        the number of arguments in Signature)
 */
template <class OpSchemaDef>
class CAFFE2_API OpSchema final {
  // TODO static_assert OpSchemaDef isn't an instanciation of OpSchema. If yes, the caller probably passed an OpSchema somewhere where an OpSchemaDef was expected and wants a good error message.
public:
  using metadata = details::OpMetadataSchema<OpSchemaDef>;
  /**
   * Information about the signature
   */
  using signature = details::OpSignatureSchema<OpSchemaDef>;
  /**
   * Functionality for dispatching on that signature
   */
  using dispatch = details::OpDispatchKeySchema<OpSchemaDef>;


  static FunctionSchema create_function_schema() {
    return FunctionSchema(
      metadata::name(),
      _create_jit_types<typename signature::parameter_types>(),
      _create_jit_types<typename signature::return_types>()
    );
  }

private:
  template<class TypeList>
  static std::vector<Argument> _create_jit_types() {
    return __create_jit_types<TypeList>(guts::make_index_sequence<guts::typelist::size<TypeList>::value>());
  }
  template<class TypeList, size_t... arg_index>
  static std::vector<Argument> __create_jit_types(guts::index_sequence<arg_index...>) {
    return { Argument(
      signature::parameter_names()[arg_index],
      getTypePtr<decayed_element_t_<arg_index, TypeList>>(),
      /* N = */ c10::nullopt,
      // If it is an optional argument, set the default value to IValue() (i.e. None). If it is not optional, pass in nullopt, which makes the argument mandatory.
      /* default_value = */
      (guts::is_instantiation_of<c10::optional, decayed_element_t_<arg_index, TypeList>>::value)
        ? c10::optional<IValue>(IValue()) : c10::nullopt
    )... };
  }

  template<size_t index, class TypeList> using decayed_element_t_ =
      guts::decay_t<guts::typelist::element_t<index, TypeList>>;

};

// TODO test OpSchema::dispatch stuff
}  // namespace c10
