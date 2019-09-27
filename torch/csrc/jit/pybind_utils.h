#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/QScheme.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/python_tracer.h>
#include <torch/csrc/jit/resource_guard.h>
#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/jit/script/schema_matching.h>
#include <torch/csrc/jit/tracer.h>
#include <torch/csrc/utils/auto_gil.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/six.h>
#include <ATen/core/EnableNamedTensor.h>

#include <ATen/core/function_schema.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

// The visibility attribute is to avoid a warning about storing a field in the
// struct that has a different visibility (from pybind) than the struct.
#ifdef _WIN32
#define VISIBILITY_HIDDEN
#else
#define VISIBILITY_HIDDEN __attribute__((visibility("hidden")))
#endif

namespace torch {
namespace jit {

// error reporting: when reporting user-caused errors, these functions should
// not use AT_ERROR macros, since these macros add stack trace information
// that is confusing to display to the end user since it always reports
// locations in libtorch code rather than user code.

using tracer::TypedStack;

inline std::shared_ptr<script::CompilationUnit> get_python_cu() {
  return py::module::import("torch.jit")
      .attr("_python_cu")
      .cast<std::shared_ptr<script::CompilationUnit>>();
}

struct TypedIValue : public std::pair<IValue, TypePtr> {
  using pair::pair;

  IValue& ivalue() {
    return this->first;
  }
  TypePtr& type() {
    return this->second;
  }
};

inline TypedIValue toDictKeyIValue(py::handle key) {
  if (py::isinstance<py::str>(key)) {
    return TypedIValue(
        ConstantString::create(py::cast<std::string>(key)),
        StringType::create());
  } else if (py::isinstance<py::int_>(key)) {
    return TypedIValue(py::cast<int64_t>(key), IntType::create());
  } else if (py::isinstance<py::float_>(key)) {
    return TypedIValue(py::cast<double>(key), FloatType::create());
  } else {
    AT_ERROR("Dictionary inputs may only have string, int, or float keys");
  }
}

inline c10::optional<TypePtr> unifyOrInitializeType(
    TypePtr accum,
    TypePtr unify) {
  if (!accum) {
    return unify;
  }
  return unifyTypes(accum, unify);
}

struct InferredType {
  InferredType(TypePtr type) : type_(std::move(type)) {}
  InferredType(std::string reason)
      : type_(nullptr), reason_(std::move(reason)) {}
  TypePtr type() const {
    TORCH_INTERNAL_ASSERT(type_);
    return type_;
  }
  bool success() const {
    return type_ != nullptr;
  }
  const std::string& reason() const {
    TORCH_INTERNAL_ASSERT(!type_);
    return reason_;
  }

 private:
  TypePtr type_;
  std::string reason_;
};

InferredType tryToInferContainerType(py::handle input);

// Try to infer the type of a Python object
// The type cannot be inferred if:
//   input is a None
//   input is an empty container (list, dict)
//   input is an list with element types that cannot be unified
//   input is an dict with key or value types that cannot be unified
inline InferredType tryToInferType(py::handle input) {
  // Try tensor types
  if (THPVariable_Check(input.ptr())) {
    auto tensor = py::cast<at::Tensor>(input);
    return InferredType(TensorType::create(tensor));
  }

  if (input.is(py::none())) {
    return InferredType("Cannot infer type of a None value");
  }

  // Try basic types first
  if (py::isinstance<py::bool_>(input)) {
    return InferredType(BoolType::get());
  } else if (py::isinstance<py::int_>(input)) {
    return InferredType(IntType::get());
  } else if (py::isinstance<py::float_>(input)) {
    return InferredType(FloatType::get());
  } else if (py::isinstance<py::str>(input)) {
    return InferredType(StringType::get());
  } else if (THPLayout_Check(input.ptr())) {
    return InferredType(IntType::get());
  } else if (THPDevice_Check(input.ptr())) {
    return InferredType(DeviceObjType::get());
  } else if (THPDtype_Check(input.ptr())) {
    return InferredType(IntType::get());
  } else if (THPQScheme_Check(input.ptr())) {
    return InferredType(IntType::get());
  }

  // Try container types
  return tryToInferContainerType(input);
}

inline InferredType tryToInferContainerType(py::handle input) {
  if (six::isTuple(input)) {
    py::tuple tuple = py::cast<py::tuple>(input);
    std::vector<TypePtr> element_types;
    element_types.reserve(tuple.size());

    for (py::handle elem : tuple) {
      auto type_match = tryToInferType(elem);
      if (type_match.success()) {
        element_types.push_back(type_match.type());
      } else {
        // Forward error message along
        return type_match.reason();
      }
    }
    return InferredType(TupleType::create(element_types));
  } else if (PyDict_Check(input.ptr())) {
    // Check to make sure we can generate useful input/output types
    auto dict = py::cast<py::dict>(input);
    size_t len = py::len(dict);
    if (!len) {
      return InferredType("Dictionary inputs must have entries");
    }

    TypePtr key_type = nullptr;
    TypePtr value_type = nullptr;

    for (auto entry : dict) {
      // Try to infer the key type and unify it with the existing one
      auto entry_key_type_match = tryToInferType(entry.first);
      if (!entry_key_type_match.success()) {
        return entry_key_type_match.reason();
      }
      auto unified_key =
          unifyOrInitializeType(key_type, entry_key_type_match.type());
      if (!unified_key) {
        return InferredType(c10::str(
            "Dictionary inputs to traced functions must have consistent type. Found ",
            key_type->python_str(),
            " and ",
            (entry_key_type_match.type())->python_str()));
      }

      // Try to infer the value type and unify it with the existing one
      auto entry_value_type_match = tryToInferType(entry.second);
      if (!entry_value_type_match.success()) {
        return entry_value_type_match.reason();
      }
      auto unified_value =
          unifyOrInitializeType(value_type, entry_value_type_match.type());
      if (!unified_value) {
        return InferredType(c10::str(
            "Dictionary inputs to traced functions must have consistent type. Found ",
            value_type->python_str(),
            " and ",
            (entry_value_type_match.type())->python_str()));
      }

      key_type = *unified_key;
      value_type = *unified_value;
    }
    return InferredType(DictType::create(key_type, value_type));
  } else if (PyList_Check(input.ptr())) {
    auto list = py::cast<py::list>(input);
    size_t len = py::len(list);
    if (!len) {
      return InferredType("List trace inputs must have elements");
    }

    TypePtr element_type = nullptr;
    for (auto elem : list) {
      auto element_type_match = tryToInferType(elem);
      if (!element_type_match.success()) {
        return InferredType(c10::str(
            "Could not infer type of list element: ",
            element_type_match.reason()));
      }
      auto unified_type =
          unifyOrInitializeType(element_type, element_type_match.type());
      if (!unified_type) {
        return InferredType(c10::str(
            "List inputs to traced functions must have consistent element type. Found ",
            element_type->python_str(),
            " and ",
            (element_type_match.type())->python_str()));
      }
      element_type = *unified_type;
    }
    return InferredType(ListType::create(element_type));
  } else {
    return InferredType(c10::str(
        "Only tensors and (possibly nested) tuples of tensors, lists, or dicts",
        "are supported ",
        "as inputs or outputs of traced functions",
        ", but instead got value of type ",
        py::str(input.get_type().attr("__name__")),
        ".",
        "\nValue: ",
        py::repr(input)));
  }
}

inline IValue toIValue(
    py::handle obj,
    const TypePtr& type,
    c10::optional<int32_t> N = c10::nullopt);

inline bool isTraceableType(TypePtr type) {
  if (type->isSubtypeOf(TensorType::get())) {
    return true;
  }

  if (auto list_type = type->cast<ListType>()) {
    return isTraceableType(list_type->getElementType());
  }

  if (auto tuple_type = type->cast<TupleType>()) {
    return std::all_of(
        tuple_type->elements().begin(),
        tuple_type->elements().end(),
        [](TypePtr element_type) { return isTraceableType(element_type); });
  }

  if (auto dict_type = type->cast<DictType>()) {
    return isTraceableType(dict_type->getValueType());
  }

  return false;
}

inline TypedIValue toTraceableIValue(py::handle input) {
  auto match = tryToInferType(input);
  if (!match.success()) {
    AT_ERROR(
        "Tracer cannot infer type of ", py::str(input), "\n:", match.reason());
  }
  auto type = match.type();

  if (isTraceableType(type)) {
    return TypedIValue(toIValue(input, type), type);
  }

  AT_ERROR(
      "Type '",
      type->python_str(),
      "' cannot be traced. Only Tensors and (possibly nested) Lists, Dicts, and"
      " Tuples of Tensors can be traced");
}

inline IValue toIValue(py::handle input) {
  return toTraceableIValue(input).ivalue();
}

inline Stack toStack(const py::tuple& inputs) {
  return toIValue(inputs).toTuple()->elements();
}

inline TypedStack toTypedStack(const py::tuple& inputs) {
  auto info = toTraceableIValue(inputs);
  return TypedStack(
      info.ivalue().toTuple()->elements(), info.type()->expect<TupleType>());
}

inline IValue createGenericList(py::handle obj, const TypePtr& elem_type) {
  auto elems = c10::impl::GenericList(elem_type);
  for (auto elem : obj) {
    elems.push_back(toIValue(std::move(elem), elem_type));
  }
  return IValue(std::move(elems));
}

inline IValue createGenericDict(
    py::handle obj,
    const TypePtr& key_type,
    const TypePtr& value_type) {
  c10::impl::GenericDict elems(key_type, value_type);
  elems.reserve(py::len(obj));
  for (auto key : obj) {
    elems.insert(
        toIValue(key, key_type), toIValue(obj[key], value_type));
  }
  return IValue(std::move(elems));
}

template <class T>
inline void guardAgainstNamedTensor(const T& var) {
#ifdef BUILD_NAMEDTENSOR
  TORCH_CHECK(!var.has_names(),
      "NYI: Named tensors are currently unsupported in TorchScript.");
#endif
}

inline IValue toIValue(
    py::handle obj,
    const TypePtr& type,
    c10::optional<int32_t> N) {
  switch (type->kind()) {
    case TypeKind::TensorType: {
      auto var = py::cast<autograd::Variable>(obj);
      if (var.is_sparse()) {
        AT_WARN(
            "Using sparse tensors in TorchScript is experimental. Many optimization "
            "pathways have not been thoroughly tested with sparse tensors. Please "
            "include the fact that the network is running sparse tensors in any bug "
            "reports submitted.");
      }
      guardAgainstNamedTensor<autograd::Variable>(var);
      return var;
    }
    case TypeKind::FloatType:
      return py::cast<double>(obj);
    case TypeKind::IntType:
      if (THPDtype_Check(obj.ptr())) {
        auto dtype = reinterpret_cast<THPDtype*>(obj.ptr());
        return static_cast<int64_t>(dtype->scalar_type);
      }
      if (THPQScheme_Check(obj.ptr())) {
        auto qscheme = reinterpret_cast<THPQScheme*>(obj.ptr());
        return static_cast<uint8_t>(qscheme->qscheme);
      }
      return py::cast<int64_t>(obj);
    case TypeKind::NoneType:
      if (!obj.is_none()) {
        throw py::cast_error(
            c10::str("Cannot cast ", py::str(obj), " to None"));
      }
      return {};
    case TypeKind::BoolType:
      return py::cast<bool>(obj);
    case TypeKind::TupleType: {
      py::tuple tuple = py::cast<py::tuple>(obj);
      size_t tuple_size = tuple.size();
      auto tuple_type = type->cast<TupleType>();
      const auto& elem_types = tuple_type->elements();
      if (elem_types.size() != tuple_size) {
        throw py::cast_error(c10::str(
            "Object ",
            py::str(obj),
            " had a different number of elements than type ",
            type->python_str()));
      }
      std::vector<IValue> values;
      values.reserve(tuple_size);
      for (size_t i = 0; i < tuple_size; ++i) {
        values.push_back(toIValue(tuple[i], elem_types[i]));
      }
      return tuple_type->name()
          ? c10::ivalue::Tuple::createNamed(std::move(values), tuple_type)
          : c10::ivalue::Tuple::create(std::move(values));
    }
    case TypeKind::StringType:
      return ConstantString::create(py::cast<std::string>(obj));
    case TypeKind::DeviceObjType: {
      auto device = reinterpret_cast<THPDevice*>(obj.ptr());
      return device->device;
    }
    case TypeKind::ListType: {
      const auto& elem_type = type->expect<ListType>()->getElementType();
      switch (elem_type->kind()) {
        // allows single int/float to be broadcasted to a fixed size list
        case TypeKind::IntType:
          if (!N || !py::isinstance<py::int_>(obj)) {
            return c10::impl::toList(py::cast<std::vector<int64_t>>(obj));
          } else {
            double value = py::cast<int64_t>(obj);
            c10::List<double> repeated;
            repeated.reserve(*N);
            for (int i = 0; i < *N; ++i) {
              repeated.push_back(value);
            }
            return repeated;
          }
        case TypeKind::FloatType:
          if (!N || !py::isinstance<py::float_>(obj)) {
            return c10::impl::toList(py::cast<std::vector<double>>(obj));
          } else {
            double value = py::cast<double>(obj);
            c10::List<double> repeated;
            repeated.reserve(*N);
            for (int i = 0; i < *N; ++i) {
              repeated.push_back(value);
            }
            return repeated;
          }
        case TypeKind::BoolType:
          return c10::impl::toList(py::cast<std::vector<bool>>(obj));
        case TypeKind::TensorType:
          return c10::impl::toList(py::cast<std::vector<at::Tensor>>(obj));
        default:
          return createGenericList(obj, elem_type);
      }
    }
    case TypeKind::DictType: {
      const auto& dict_type = type->expect<DictType>();
      return createGenericDict(
          obj, dict_type->getKeyType(), dict_type->getValueType());
    }
    case TypeKind::OptionalType: {
      // check if it's a none obj since optional accepts NoneType
      if (obj.is_none()) {
        // check if it's a none obj since optional accepts NoneType
        // return an IValue() to denote a NoneType
        return {};
      }
      return toIValue(obj, type->expect<OptionalType>()->getElementType());
    }
    case TypeKind::ClassType: {
      auto classType = type->expect<ClassType>();
      // 1. create a bare ivalue
      const size_t numAttrs = classType->numAttributes();
      auto cu = classType->compilation_unit();
      auto userObj = c10::ivalue::Object::create(
          c10::StrongTypePtr(cu, classType), numAttrs);

      // 2. copy all the contained types
      for (size_t slot = 0; slot < numAttrs; slot++) {
        const auto& attrType = classType->getAttribute(slot);
        const auto& attrName = classType->getAttributeName(slot);

        const auto& contained = py::getattr(obj, attrName.c_str());
        userObj->setSlot(slot, toIValue(contained, attrType));
      }
      return userObj;
    }
    case TypeKind::NumberType: {
      if (THPDtype_Check(obj.ptr())) {
        auto dtype = reinterpret_cast<THPDtype*>(obj.ptr());
        return static_cast<int64_t>(dtype->scalar_type);
      }
      if (THPQScheme_Check(obj.ptr())) {
        auto qscheme = reinterpret_cast<THPQScheme*>(obj.ptr());
        return static_cast<uint8_t>(qscheme->qscheme);
      }
      if (py::isinstance<py::int_>(obj)) {
        return py::cast<int64_t>(obj);
      } else if (py::isinstance<py::float_>(obj)) {
        return py::cast<double>(obj);
      }
    }
    case TypeKind::GeneratorType:
    case TypeKind::VarType:
    case TypeKind::FutureType:
    case TypeKind::InterfaceType:
      break;
    case TypeKind::FunctionType:
      AT_ERROR("Function Values aren't yet supported");
    case TypeKind::CapsuleType:
      AT_ERROR("Capsule Values aren't supported");
    case TypeKind::AnyType:
      AT_ERROR("AnyType Values aren't supported");
  }
  AT_ERROR(
      "Missing cases in toIValue for type: ",
      type->str(),
      "! File a bug report.");
}

// Small wrapper around getting the type name string from Python to make
// types easier to interpret, e.g. give the structural type for a NamedTuple
inline std::string friendlyTypeName(py::handle obj) {
  if (py::isinstance<py::tuple>(obj) && py::hasattr(obj, "_fields")) {
    auto field_names =
        py::cast<std::vector<std::string>>(py::getattr(obj, "_fields"));
    std::stringstream ss;
    ss << py::str(obj.get_type().attr("__name__"));
    ss << " (aka NamedTuple(";
    bool first = true;
    for (auto& field_name : field_names) {
      if (!first) {
        ss << ", ";
      }
      ss << field_name;
      first = false;
    }
    ss << "))";
    return ss.str();
  } else {
    return py::str(obj.get_type().attr("__name__"));
  }
}

inline IValue argumentToIValue(
    const FunctionSchema& schema,
    size_t argumentPosition,
    py::handle object) {
  const auto& argument = schema.arguments().at(argumentPosition);
  try {
    return toIValue(object, argument.type(), argument.N());
  } catch (const py::cast_error& error) {
    throw std::runtime_error(schema.formatTypeMismatchMsg(
        argument,
        friendlyTypeName(object),
        argumentPosition,
        py::repr(object)));
  }
}

inline IValue returnToIValue(const TypePtr& type, py::handle object) {
  try {
    return toIValue(object, type);
  } catch (const py::cast_error& error) {
    throw std::runtime_error(c10::str(
        " expected value of type ",
        type->str(),
        " for return value but instead got value of type ",
        py::str(object.get_type().attr("__name__")),
        ".",
        "\nValue: ",
        py::repr(object)));
  }
}

inline c10::optional<py::object> tryToConvertToCustomClass(
    const c10::intrusive_ptr<c10::ivalue::Object>& obj) {
  if (obj->name().find("__torch__.torch.classes") == 0) {
    auto objPtr = (void*)obj->getSlot(0).toCapsule().release();
    auto classConverter = c10::getClassConverter()[obj->name()];
    py::handle rawPyObj = classConverter(objPtr);
    auto o = py::reinterpret_steal<py::object>(rawPyObj);
    return o;
  }
  return c10::nullopt;
}
inline py::object toPyObject(IValue&& ivalue) {
  if (ivalue.isNone()) {
    return py::none();
  } else if (ivalue.isTensor()) {
    auto tensor = std::move(ivalue).toTensor();
    if (tensor.is_sparse()) {
      AT_WARN(
          "Using sparse tensors in TorchScript is experimental. Many optimization "
          "pathways have not been thoroughly tested with sparse tensors. Please "
          "include the fact that the network is running sparse tensors in any bug "
          "reports submitted.");
    }
    guardAgainstNamedTensor<at::Tensor>(tensor);
    return py::cast(autograd::Variable(std::move(tensor)));
  } else if (ivalue.isDouble()) {
    return py::cast(std::move(ivalue).toDouble());
  } else if (ivalue.isInt()) {
    return py::cast(std::move(ivalue).toInt());
  } else if (ivalue.isBool()) {
    return py::cast(std::move(ivalue).toBool());
  } else if (ivalue.isString()) {
    return py::cast(std::move(ivalue).toStringRef());
  } else if (ivalue.isIntList()) {
    return py::cast(c10::impl::toVector(std::move(ivalue).toIntList()));
  } else if (ivalue.isDoubleList()) {
    return py::cast(c10::impl::toVector(std::move(ivalue).toDoubleList()));
  } else if (ivalue.isBoolList()) {
    return py::cast(c10::impl::toVector(std::move(ivalue).toBoolList()));
  } else if (ivalue.isTensorList()) {
    return py::cast(c10::impl::toVector(std::move(ivalue).toTensorList()));
  } else if (ivalue.isGenericList()) {
    auto list = std::move(ivalue).toGenericList();
    py::list t{list.size()};
    for (size_t i = 0; i < list.size(); ++i) {
      t[i] = toPyObject(IValue{list.get(i)});
    }
    return std::move(t);
  } else if (ivalue.isTuple()) {
    auto tuple = std::move(ivalue).toTuple();
    const auto& elements = tuple->elements();
    py::tuple t{elements.size()};
    for (size_t i = 0; i < elements.size(); ++i) {
      t[i] = toPyObject(IValue{elements.at(i)});
    }
    if (tuple->type() && tuple->type()->schema() &&
        tuple->type()->schema()->name() != "") {
      auto unqualName = tuple->type()->name()->name();
      auto fieldNames = fmap(
          tuple->type()->schema()->arguments(),
          [](const Argument& arg) { return arg.name(); });
      return py::module::import("torch.jit")
          .attr("_create_named_tuple")(
              t, unqualName, fieldNames);
    } else {
      return std::move(t);
    }
  } else if (ivalue.isDevice()) {
    return py::cast<py::object>(THPDevice_New(std::move(ivalue).toDevice()));
  } else if (ivalue.isGenericDict()) {
    auto dict = std::move(ivalue).toGenericDict();
    py::dict py_dict;
    for (auto& pair : dict) {
      py_dict[toPyObject(IValue{pair.key()})] = toPyObject(IValue{pair.value()});
    }
    return std::move(py_dict);
  } else if (ivalue.isObject()) {
    const auto obj = std::move(ivalue).toObject();
    auto pyCu = get_python_cu();
    auto res = tryToConvertToCustomClass(obj);
    if (res.has_value()) {
      return res.value();
    }
    const auto classType = pyCu->get_class(c10::QualifiedName(obj->name()));
    AT_ASSERT(classType);
    auto pyClass =
        py::module::import("torch.jit").attr("_get_script_class")(obj->name());
    auto pyObj = pyClass.attr("__new__")(pyClass);

    const auto numAttrs = classType->numAttributes();

    for (size_t slot = 0; slot < numAttrs; slot++) {
      const auto& attrName = classType->getAttributeName(slot);
      IValue v = obj->getSlot(slot);
      py::setattr(pyObj, attrName.c_str(), toPyObject(std::move(v)));
    }
    return pyObj;
  } else {
    AT_ERROR(
        "Missing cases in 'toPyObject'! Can't convert ",
        ivalue.tagKind(),
        " to a Python object");
  }
}

struct VISIBILITY_HIDDEN tuple_slice {
  /*implicit*/ tuple_slice(py::tuple tup_)
      : tup(std::move(tup_)), b(0), e(tup.size()) {}
  tuple_slice(py::tuple tup_, int64_t b_)
      : tup(std::move(tup_)), b(b_), e(tup.size()) {}
  tuple_slice(py::tuple tup_, int64_t b_, int64_t e_)
      : tup(std::move(tup_)), b(b_), e(e_) {}
  py::detail::tuple_iterator begin() const {
    return {tup, static_cast<pybind11::ssize_t>(b)};
  }
  py::detail::tuple_iterator end() const {
    return {tup, static_cast<pybind11::ssize_t>(e)};
  }
  size_t size() const {
    return e - b;
  }
  py::detail::tuple_accessor operator[](size_t index) const {
    return {tup, static_cast<size_t>(b + index)};
  }

 private:
  py::tuple tup;
  int64_t b;
  int64_t e;
};

inline Stack createStackForSchema(
    const FunctionSchema& schema,
    const tuple_slice& args,
    const py::kwargs& kwargs,
    c10::optional<IValue> self) {
  size_t all_arguments = (self ? 1 : 0) + args.size() + kwargs.size();
  if (all_arguments > schema.arguments().size()) {
    throw std::runtime_error(c10::str(
        schema.name(),
        "() expected at most ",
        schema.arguments().size(),
        " argument(s) but received ",
        all_arguments,
        " argument(s). Declaration: ",
        schema));
  }
  Stack stack;
  stack.reserve(schema.arguments().size());

  if (self) {
    push(stack, std::move(*self));
  }
  // First push all positional args.
  for (size_t i = 0; i < args.size(); ++i) {
    // Use the type information from the schema to convert the PyObject.
    push(stack, argumentToIValue(schema, stack.size(), args[i]));
  }

  // Now for every remaining non-positional argument in the schema, look for it
  // in the kwargs dict and push it if found, or use its default value if it
  // has one.
  size_t consumed_kwargs = 0;
  for (size_t i = stack.size(); i < schema.arguments().size(); ++i) {
    const auto& arg = schema.arguments()[i];
    if (kwargs.contains(arg.name().c_str())) {
      push(stack, argumentToIValue(schema, i, kwargs[arg.name().c_str()]));
      consumed_kwargs += 1;
    } else if (arg.default_value()) {
      push(stack, *arg.default_value());
    } else {
      throw std::runtime_error(c10::str(
          schema.name(),
          "() is missing value for argument '",
          arg.name(),
          "'. Declaration: ",
          schema));
    }
  }

  if (consumed_kwargs != kwargs.size()) {
    std::vector<std::string> names;
    for (const auto& kwarg : kwargs) {
      names.emplace_back(py::cast<std::string>(kwarg.first));
    }
    schema.findErrorInKwargs(names);
  }

  return stack;
}

inline py::object createPyObjectForStack(Stack&& stack) {
  if (stack.empty()) {
    return py::none();
  }

  // Return a simple value and not a single-element tuple if there is only one
  // return value.
  if (stack.size() == 1) {
    return toPyObject(std::move(stack[0]));
  }

  // If there is more than one return value, pop them into a py::tuple.
  py::tuple return_values(stack.size());
  for (size_t ret = 0; ret < return_values.size(); ++ret) {
    return_values[ret] = toPyObject(std::move(stack[ret]));
  }

  return std::move(return_values);
}

// TODO: Remove once we clean up the GraphExecutor usage.
inline Stack evilDeprecatedBadCreateStackDoNotUse(
    const py::tuple& tuple,
    at::ArrayRef<Value*> inputs,
    size_t reserve_extra_space = 0) {
  if (tuple.size() != inputs.size()) {
    AT_ERROR(
        "expected " + std::to_string(inputs.size()) + " inputs, but got " +
        std::to_string(tuple.size()));
  }
  Stack result;
  result.reserve(tuple.size() + reserve_extra_space);
  for (size_t i = 0; i < inputs.size(); ++i) {
    result.push_back(toIValue(std::move(tuple[i]), inputs[i]->type()));
  }
  return result;
}

// Run `callee`, potentially inserting a CallFunction/CallMethod node into the
// tracing graph.
inline py::object runAndInsertCall(
    Function& callee,
    tuple_slice args,
    py::kwargs kwargs,
    c10::optional<IValue> self,
    // Lambda that tells this function how to insert `callee` into the graph if
    // we're tracing.
    std::function<Value*(Graph&, const script::MatchedSchema& match)>
        callInserter) {
  auto stack = createStackForSchema(
      callee.getSchema(), std::move(args), std::move(kwargs), std::move(self));
  auto tracing_state = tracer::getTracingState();
  if (!tracing_state) {
    AutoNoGIL no_gil_guard;
    // If we're not tracing, just run the callee as normal.
    callee.run(stack);
  } else {
    // If we are tracing, insert the appropriate CallFunction or CallMethod node
    // and then run the callee with tracing disabled.

    // Get the graph `Value`s that represent the input IValues
    auto inputs = last(stack, callee.graph()->inputs().size());
    auto input_values =
        fmap(inputs, [](const IValue& v) { return tracer::getValueTrace(v); });
    TORCH_INTERNAL_ASSERT(callee.getSchema().returns().size() == 1)
    auto return_type = callee.getSchema().returns().at(0).type();
    auto graph = tracing_state->graph;
    std::vector<NamedValue> named_values;
    for (Value* v : input_values) {
      named_values.emplace_back(v);
    }

    // Add a call node.
    script::MatchedSchema match = script::matchSchema(
        callee.getSchema(),
        tracer::getPythonInterpreterSourceRange(),
        *graph,
        named_values,
        {});
    auto output_value = callInserter(*graph, match);

    // Actually run the callee. Pause the tracer so that we don't double-add the
    // callee nodes.
    {
      AutoNoGIL no_gil_guard;
      ResourceGuard guard(tracer::pauseTracing());
      callee.run(stack);
    }

    // Associate the output IValues with the output `Value`s in the graph
    tracer::setValueTrace(stack.back(), output_value);
  }

  TORCH_CHECK(
      stack.size() > 0,
      "Expected values in the stack after execution but found none");
  return toPyObject(std::move(stack.back()));
}

inline py::object invokeScriptFunctionFromPython(
    Function& callee,
    tuple_slice args,
    py::kwargs kwargs) {
  return runAndInsertCall(
      callee,
      args,
      kwargs,
      /*self=*/c10::nullopt,
      [&](Graph& graph, const script::MatchedSchema& match) {
        return graph.insertFunctionCall(&callee, match);
      });
}

inline py::object invokeScriptMethodFromPython(
    script::Method& callee,
    tuple_slice args,
    py::kwargs kwargs) {
  auto self = callee.owner().module_object();
  return runAndInsertCall(
      callee.function(),
      args,
      kwargs,
      self,
      [&](Graph& graph, const script::MatchedSchema& match) {
        return graph.insertMethodCall(callee.name(), match);
      });
}

inline py::object invokeOperatorFromPython(
    const Operator& op,
    py::args args,
    py::kwargs kwargs) {
  // Create a stack full of the arguments and keyword arguments.
  auto stack = createStackForSchema(
      op.schema(), std::move(args), std::move(kwargs), c10::nullopt);

  // Invoke the operation, which puts the return values onto the stack.
  op.getOperation()(stack);

  return createPyObjectForStack(std::move(stack));
}

} // namespace jit
} // namespace torch
