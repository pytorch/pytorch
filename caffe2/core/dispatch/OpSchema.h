#pragma once

#include "caffe2/core/dispatch/DispatchKey.h"
#include "caffe2/proto/caffe2_pb.h"
#include <c10/util/Array.h>
#include <c10/util/Metaprogramming.h>

namespace caffe2 {
class Tensor;
}  // namespace caffe2

namespace c10 {

namespace details {

/**
 * If Arg is a Tensor or reference to a Tensor, provide the member constant value equal to true.  Otherwise
 * return false.
 */
template <class Arg>
using is_tensor_arg = std::
    is_same<caffe2::Tensor, guts::remove_cv_t<guts::remove_reference_t<Arg>>>;

inline DeviceTypeId to_device_type_id(caffe2::DeviceType device_type) {
  switch (device_type) {
    case caffe2::CPU:
      return DeviceTypeId::CPU;
    case caffe2::CUDA:
      return DeviceTypeId::CUDA;
    default:
      return DeviceTypeId::UNDEFINED;
  }
}

// TODO get rid of tensor_to_dispatch_key once c2::Tensor is de-templatized. This then fits into a template lambda instead of a functor.
struct tensor_to_dispatch_key final {
    template<class TensorType>
    TensorParameterDispatchKey operator()(const TensorType& tensor) const {
      return TensorParameterDispatchKey{
          to_device_type_id(tensor.GetDeviceType()),
          LayoutId(0),
          tensor.dtype().id()};
    }
};

/**
 * Extract the type ids of all tensors in a variadic list of arguments
 *
 * @tparam Args Inferred variadic list of argument types
 * @param args List of arguments to get type ids from
 * @return guts::array<TensorParameterDispatchKey, n>, where n is the number of tensor arguments (is_tensor_arg) in the class
 */
template<class... Args> auto getTensorTypeIds_(const Args&... args)
-> guts::array<TensorParameterDispatchKey, guts::typelist::count_if<is_tensor_arg, guts::typelist::typelist<Args...>>::value> {
  return guts::filter_map<TensorParameterDispatchKey, is_tensor_arg>(tensor_to_dispatch_key(), args...);
}

// TODO Test getTensorTypeIds_

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

  /**
   * The number of arguments of OpSchemaDef::Signature
   */
  static constexpr size_t num_args = guts::typelist::size<parameter_types>::value;
  /**
   * The number of tensor arguments (as per is_tensor_arg) in OpSchemaDef::Signature
   */
  static constexpr size_t num_tensor_args = guts::typelist::count_if<details::is_tensor_arg, parameter_types>::value;

private:
  static_assert(details::has_parameter_names_defined<OpSchemaDef>::value, "Operator schema doesn't define parameter_names member.");
  // TODO Allow simpler definition of parameter_names without having to spell out the guts::array type in the schema def.
  static_assert(std::is_same<const guts::array<const char*, num_args>, decltype(OpSchemaDef::parameter_names)>::value, "Operator schema defines parameter_names member, but it isn't the correct type. Must be a static constexpr guts::array of const char* with one entry for each parameter.");

public:
  /**
   * The names of the parameters (as per OpSchemaDef::parameter_names)
   * @return Array
   */
  static constexpr const guts::array<const char*, num_args>& parameter_names() {
    return OpSchemaDef::parameter_names;
  }
};

/**
 * If T has a method dispatch_key, provide a member constant value equal to true.  Otherwise return false.
 * @tparam T
 */
template<class T, typename = void>
struct has_function_dispatch_key_defined : std::false_type {};
template<class T>
struct has_function_dispatch_key_defined<T, guts::void_t<
  decltype(&T::dispatch_key)
>> : std::true_type {};

/**
 * Wrapper class around a user-defined schema definition providing a way of computing a dispatch key
 * from arguments matching the signature of that schema.
 *
 * @tparam OpSchemaDef Operator schema definition.  See OpSchema for more details.
 * @tparam Enable Inferred, used to control specialization
 */
template<class OpSchemaDef, class Enable = void> class OpDispatchKeySchema final {};

// General case. Operator doesn't overwrite DispatchKey generation. Use default.
template<class OpSchemaDef>
class OpDispatchKeySchema<OpSchemaDef, guts::enable_if_t<!has_function_dispatch_key_defined<OpSchemaDef>::value>> final {
  using signature = OpSignatureSchema<OpSchemaDef>;

  // TODO Static assert that dispatch_key_type has operator<<(ostream, _) defined for debug output.
  // TODO Use an ADL-based debugString(DispatchKey) function instead of operator<< for debug printing.

public:
  using dispatch_key_type = DispatchKey<signature::num_tensor_args>;

  template<class... Args>
  static inline dispatch_key_type dispatch_key(const Args&... args) {
    using guts::typelist::map_t;
    using guts::typelist::typelist;
    static_assert(std::is_same<
      map_t<guts::remove_cv_t, map_t<guts::remove_reference_t, typelist<Args...>>>,
      map_t<guts::remove_cv_t, map_t<guts::remove_reference_t, typename signature::parameter_types>>
      >::value, "Invalid argument types passed to OpSchema::dispatch_key()");
    return dispatch_key_type {
      details::getTensorTypeIds_(args...)
    };
  }
};

// Special case. Operator overwrites DispatchKey generation. Use that.
template<class OpSchemaDef>
class OpDispatchKeySchema<OpSchemaDef, guts::enable_if_t<has_function_dispatch_key_defined<OpSchemaDef>::value>> final {
  using signature = OpSignatureSchema<OpSchemaDef>;

  static_assert(guts::is_function_type<decltype(OpSchemaDef::dispatch_key)>::value, "Operator schema defines dispatch_key member, but it isn't a function.");

  using dispatch_key_traits = guts::function_traits<decltype(OpSchemaDef::dispatch_key)>;

public:
  using dispatch_key_type = typename dispatch_key_traits::return_type;

private:

  static_assert(guts::is_equality_comparable<dispatch_key_type>::value, "Operator schema specified custom dispatch_key() derivation function, but the returned dispatch key type doesn't have the equality operator defined. Please define it.");
  static_assert(guts::is_hashable<dispatch_key_type>::value, "Operator schema specified custom dispatch_key() derivation function, but the returned dispatch key type doesn't have an overload for std::hash. Please define it.");

  static_assert(std::is_same<
    guts::typelist::map_t<guts::remove_cv_t, guts::typelist::map_t<guts::remove_reference_t, typename dispatch_key_traits::parameter_types>>,
    guts::typelist::map_t<guts::remove_cv_t, guts::typelist::map_t<guts::remove_reference_t, typename signature::parameter_types>>
    >::value, "Operator schema defines custom dispatch_key() derivation function, but the arguments don't match the operator signature.");

public:

  template<class... Args>
  static inline dispatch_key_type dispatch_key(const Args&... args) {
    using guts::typelist::map_t;
    using guts::typelist::typelist;
    static_assert(std::is_same<
      map_t<guts::remove_cv_t, map_t<guts::remove_reference_t, typelist<Args...>>>,
      map_t<guts::remove_cv_t, map_t<guts::remove_reference_t, typename signature::parameter_types>>
      >::value, "Invalid argument types passed to OpSchema::dispatch_key()");
    return OpSchemaDef::dispatch_key(args...);
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
 *      - a constexpr guts<const char*, n_args> parameter_names field (where n_args is
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
};

// TODO test OpSchema::dispatch stuff
}  // namespace c10
