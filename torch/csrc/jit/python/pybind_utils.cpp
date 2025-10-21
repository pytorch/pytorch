#include <torch/csrc/jit/ir/graph_utils.h>
#include <torch/csrc/jit/python/module_python.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/python/python_dict.h>
#include <torch/csrc/jit/python/python_ivalue.h>
#include <torch/csrc/jit/python/python_list.h>
#include <torch/csrc/jit/python/utf8_decoding_ignore.h>

#include <ATen/ScalarOps.h>

#include <c10/core/QScheme.h>
#include <c10/util/irange.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include <limits>
#include <optional>
#include <utility>

namespace torch::jit {

static thread_local bool allow_numbers_as_tensors = false;

ToIValueAllowNumbersAsTensors::ToIValueAllowNumbersAsTensors(bool enable)
    : old_(allow_numbers_as_tensors) {
  allow_numbers_as_tensors = enable;
}

ToIValueAllowNumbersAsTensors::~ToIValueAllowNumbersAsTensors() {
  allow_numbers_as_tensors = old_;
}

// This is a hack to remove instances deleted in C++ from the PyBind cache
// C++->Python. We need this because otherwise we may get the old Python object
// if C++ creates a new object at the memory location of the deleted object.
void clear_registered_instances(void* ptr) {
#if IS_PYBIND_2_13_PLUS
  py::detail::with_instance_map(
      ptr, [&](py::detail::instance_map& registered_instances) {
        auto range = registered_instances.equal_range(ptr);
        for (auto it = range.first; it != range.second; ++it) {
          auto vh = it->second->get_value_and_holder();
          vh.set_instance_registered(false);
        }
        registered_instances.erase(ptr);
      });
#else
  auto& registered_instances =
      pybind11::detail::get_internals().registered_instances;
  auto range = registered_instances.equal_range(ptr);
  for (auto it = range.first; it != range.second; ++it) {
    auto vh = it->second->get_value_and_holder();
    vh.set_instance_registered(false);
  }
  registered_instances.erase(ptr);
#endif
}

// WARNING: Precondition for this function is that, e.g., you have tested if a
// SymIntList is in fact only ints, and if so, you called this with T=int64_t.
// This precondition is NOT checked at runtime.
template <typename T>
static IValue listToIValue(py::handle obj) {
  c10::List<T> rs;
  for (auto it = obj.begin(); it != obj.end(); it++) {
    auto elm = *it;
    rs.push_back(py::cast<T>(elm));
  }
  // Promises that we have decayed the list appropriately
  return c10::impl::toList<T>(rs);
}

IValue toIValue(py::handle obj, const TypePtr& type, std::optional<int32_t> N) {
  switch (type->kind()) {
    case TypeKind::TensorType: {
      if (obj.ptr() == Py_None) {
        // None gets converted to undefined Tensors
        return autograd::Variable();
      }
      if (THPVariable_Check(obj.ptr())) {
        auto var = py::cast<autograd::Variable>(obj);
        guardAgainstNamedTensor<autograd::Variable>(var);
        return var;
      } else {
        if (!allow_numbers_as_tensors) {
          throw py::cast_error(
              c10::str("Unable to cast ", py::str(obj), " to Tensor"));
        }
        bool save_symint = false;
        at::Scalar scalar;
        if (PyBool_Check(obj.ptr())) {
          scalar = at::Scalar(THPUtils_unpackBool(obj.ptr()));
        } else if (THPUtils_checkLong(obj.ptr())) {
          scalar = THPUtils_unpackInteger<at::Scalar>(obj.ptr());
        } else if (PyComplex_Check(obj.ptr())) {
          scalar = at::Scalar(THPUtils_unpackComplexDouble(obj.ptr()));
        } else if (THPUtils_checkDouble(obj.ptr())) {
          scalar = at::Scalar(THPUtils_unpackDouble(obj.ptr()));
        } else if (torch::is_symint(py::handle(obj))) {
          save_symint = true;
          scalar = at::Scalar(7777777);
        } else if (torch::is_symfloat(py::handle(obj))) {
          save_symint = true;
          scalar = at::Scalar(std::numeric_limits<double>::quiet_NaN());
        } else if (torch::is_symbool(py::handle(obj))) {
          save_symint = true;
          scalar = at::Scalar(true);
        } else {
          throw py::cast_error(
              c10::str("Unable to cast ", py::str(obj), " to Tensor"));
        }
        at::Tensor tensor = at::scalar_to_tensor(scalar);
        tensor.unsafeGetTensorImpl()->set_wrapped_number(true);

        if (save_symint) {
          auto py_tensor = py::cast(tensor);
          if (PyObject_SetAttrString(
                  py_tensor.ptr(), "_wrapped_number", obj.ptr()) < 0) {
            throw python_error();
          }
        }

        return tensor;
      }
    }
    case TypeKind::StorageType:
      return py::cast<at::Storage>(obj);
    case TypeKind::FloatType:
      if (torch::is_symfloat(py::handle(obj))) {
        return py::cast<c10::SymFloat>(obj).guard_float(__FILE__, __LINE__);
      }
      if (THPVariable_Check(obj.ptr())) {
        auto var = py::cast<autograd::Variable>(obj);
        // NB: We carefully test if the storage is meta, because that is
        // always accurate even if you have a fake tensor (which is the
        // primary case we are trying to detect here)
        if (var.storage().device_type() == c10::kMeta) {
          throw py::cast_error(
              "cannot extract float from tensor with meta storage");
        }
      }
      return py::cast<double>(obj);
    case TypeKind::ComplexType: {
      auto c_obj = py::cast<std::complex<double>>(obj.ptr());
      return static_cast<c10::complex<double>>(c_obj);
    }
    case TypeKind::IntType:
      // TODO: Properly fake this type
      if (THPQScheme_Check(obj.ptr())) {
        auto qscheme = reinterpret_cast<THPQScheme*>(obj.ptr());
        return static_cast<uint8_t>(qscheme->qscheme);
      }
      // For backwards compatibility
      if (THPDtype_Check(obj.ptr())) {
        auto dtype = reinterpret_cast<THPDtype*>(obj.ptr());
        return static_cast<int64_t>(dtype->scalar_type);
      }
      if (THPQScheme_Check(obj.ptr())) {
        auto qscheme = reinterpret_cast<THPQScheme*>(obj.ptr());
        return static_cast<uint8_t>(qscheme->qscheme);
      }
      if (THPLayout_Check(obj.ptr())) {
        auto layout = reinterpret_cast<THPLayout*>(obj.ptr());
        return static_cast<int8_t>(layout->layout);
      }
      if (THPMemoryFormat_Check(obj.ptr())) {
        auto memory_format = reinterpret_cast<THPMemoryFormat*>(obj.ptr());
        return static_cast<int8_t>(memory_format->memory_format);
      }
      if (torch::is_symint(py::handle(obj))) {
        return py::cast<c10::SymInt>(obj).guard_int(__FILE__, __LINE__);
      }
      if (THPVariable_Check(obj.ptr())) {
        auto var = py::cast<autograd::Variable>(obj);
        if (var.storage().device_type() == c10::kMeta) {
          throw py::cast_error(
              "cannot extract int from tensor with meta storage");
        }
      }
      return py::cast<int64_t>(obj);
    case TypeKind::LayoutType: {
      if (THPLayout_Check(obj.ptr())) {
        auto layout = reinterpret_cast<THPLayout*>(obj.ptr());
        return static_cast<int8_t>(layout->layout);
      }
      // For backwards compatibility
      return py::cast<int64_t>(obj);
    }
    case TypeKind::ScalarTypeType: {
      if (THPDtype_Check(obj.ptr())) {
        auto dtype = reinterpret_cast<THPDtype*>(obj.ptr());
        return static_cast<int64_t>(dtype->scalar_type);
      }
      // For backwards compatibility
      return py::cast<int64_t>(obj);
    }
    case TypeKind::MemoryFormatType: {
      if (THPMemoryFormat_Check(obj.ptr())) {
        auto memory_format = reinterpret_cast<THPMemoryFormat*>(obj.ptr());
        return static_cast<int8_t>(memory_format->memory_format);
      }
      // For backwards compatibility
      return py::cast<int64_t>(obj);
    }
    case TypeKind::SymIntType:
      if (torch::is_symint(obj.ptr())) {
        return py::cast<c10::SymInt>(obj);
      }
      return py::cast<int64_t>(obj);
    case TypeKind::SymFloatType:
      if (torch::is_symfloat(obj.ptr())) {
        return py::cast<c10::SymFloat>(obj);
      }
      return py::cast<double>(obj);
    case TypeKind::SymBoolType:
      if (torch::is_symbool(obj.ptr())) {
        return py::cast<c10::SymBool>(obj);
      }
      return py::cast<bool>(obj);
    case TypeKind::NoneType:
      if (!obj.is_none()) {
        throw py::cast_error(
            c10::str("Cannot cast ", py::str(obj), " to None"));
      }
      return {};
    case TypeKind::BoolType:
      if (torch::is_symbool(obj.ptr())) {
        return py::cast<c10::SymBool>(obj).guard_bool(__FILE__, __LINE__);
      }
      if (THPVariable_Check(obj.ptr())) {
        auto var = py::cast<autograd::Variable>(obj);
        if (var.storage().device_type() == c10::kMeta) {
          throw py::cast_error(
              "cannot extract bool from tensor with meta storage");
        }
      }
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
            type->repr_str()));
      }
      std::vector<IValue> values;
      values.reserve(tuple_size);
      for (const auto i : c10::irange(tuple_size)) {
        values.push_back(toIValue(tuple[i], elem_types[i]));
      }
      return tuple_type->name()
          ? c10::ivalue::Tuple::createNamed(std::move(values), tuple_type)
          : c10::ivalue::Tuple::create(std::move(values));
    }
    case TypeKind::UnionType: {
      auto actual_type = toTypeInferredIValue(obj);
      auto actual_type_ptr = actual_type.type();
      auto union_type = type->expect<UnionType>();
      if (!actual_type_ptr->isSubtypeOf(union_type)) {
        throw py::cast_error(c10::str(
            "Expected a member of ",
            union_type->annotation_str(),
            " but instead found type ",
            actual_type.type()->annotation_str()));
      }
      return actual_type;
    }
    case TypeKind::StringType:
      return ConstantString::create(py::cast<std::string>(obj));
    case TypeKind::DeviceObjType: {
      if (THPDevice_Check(obj.ptr())) {
        auto device = reinterpret_cast<THPDevice*>(obj.ptr());
        return device->device;
      }
      return c10::Device(py::cast<std::string>(obj.ptr()));
    }
    case TypeKind::StreamObjType: {
      auto thp_stream = reinterpret_cast<THPStream*>(obj.ptr());
      auto stream = c10::Stream::unpack3(
          thp_stream->stream_id,
          static_cast<c10::DeviceIndex>(thp_stream->device_index),
          static_cast<c10::DeviceType>(thp_stream->device_type));
      return stream;
    }
    case TypeKind::ListType: {
      // If the object is a ScriptList, retrieve the c10::List
      // instance inside it.
      if (py::isinstance<ScriptList>(obj)) {
        return py::cast<ScriptList>(obj).list_;
      }

      // If not (i.e. it is a regular Python list), make a new
      // c10::List.
      const auto& elem_type = type->expectRef<ListType>().getElementType();
      switch (elem_type->kind()) {
        // allows single int/float to be broadcasted to a fixed size list
        case TypeKind::IntType:
          if (!N || !py::isinstance<py::int_>(obj)) {
            return IValue(py::cast<std::vector<int64_t>>(obj));
          } else {
            int64_t value = py::cast<int64_t>(obj);
            c10::List<int64_t> repeated;
            repeated.reserve(*N);
            for (int i = 0; i < *N; ++i) {
              repeated.push_back(value);
            }
            return repeated;
          }
        case TypeKind::SymIntType: {
          bool is_symbolic = false;
          for (auto it = obj.begin(); it != obj.end(); it++) {
            auto elm = *it;
            if (torch::is_symint(elm) || THPVariable_Check(elm.ptr())) {
              is_symbolic = true;
              break;
            }
          }
          if (is_symbolic) {
            return listToIValue<c10::SymInt>(obj);
          } else {
            return listToIValue<int64_t>(obj);
          }
        }
        case TypeKind::SymFloatType: {
          bool is_symbolic = false;
          for (auto it = obj.begin(); it != obj.end(); it++) {
            auto elm = *it;
            // TODO: what about SymInt conversion to SymFloat?
            if (torch::is_symfloat(elm)) {
              is_symbolic = true;
              break;
            }
          }
          if (is_symbolic) {
            return listToIValue<c10::SymFloat>(obj);
          } else {
            return listToIValue<double>(obj);
          }
        }
        case TypeKind::SymBoolType: {
          bool is_symbolic = false;
          for (auto it = obj.begin(); it != obj.end(); it++) {
            auto elm = *it;
            if (torch::is_symbool(elm)) {
              is_symbolic = true;
              break;
            }
          }
          if (is_symbolic) {
            return listToIValue<c10::SymBool>(obj);
          } else {
            return listToIValue<bool>(obj);
          }
        }
        case TypeKind::FloatType:
          if (!N || !py::isinstance<py::float_>(obj)) {
            return IValue(py::cast<std::vector<double>>(obj));
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
          return IValue(py::cast<std::vector<bool>>(obj));
        case TypeKind::TensorType:
          return IValue(py::cast<std::vector<at::Tensor>>(obj));
        default:
          return createGenericList(obj, elem_type);
      }
    }
    case TypeKind::DictType: {
      const auto& dict_type = type->expect<DictType>();

      // If the object is a ScriptDict, retrieve the c10::Dict
      // instance inside it.
      try {
        auto script_dict = py::cast<ScriptDict>(obj);
        return script_dict.dict_;
      } catch (py::cast_error& e) {
      }

      // If not (i.e. it is a regular Python dictionary), make a new
      // c10::Dict.
      return createGenericDict(
          py::cast<py::dict>(obj),
          dict_type->getKeyType(),
          dict_type->getValueType());
    }
    case TypeKind::OptionalType: {
      // check if it's a none obj since optional accepts NoneType
      if (obj.is_none()) {
        // check if it's a none obj since optional accepts NoneType
        // return an IValue() to denote a NoneType
        return {};
      }
      return toIValue(obj, type->expectRef<OptionalType>().getElementType(), N);
    }
    case TypeKind::ClassType: {
      auto classType = type->expect<ClassType>();
      auto object = py::cast<py::object>(obj);
      if (auto mod = as_module(object)) {
        // if obj is already a ScriptModule, just return its ivalue
        return mod.value()._ivalue();
      }

      // Check if the obj is a ScriptObject.
      if (auto script_obj = as_object(object)) {
        return script_obj.value()._ivalue();
      }

      // otherwise is a normal class object, we create a fresh
      // ivalue::Object to use from the py object.
      // 1. create a bare ivalue
      const size_t numAttrs = classType->numAttributes();
      auto cu = classType->compilation_unit();
      auto userObj = c10::ivalue::Object::create(
          c10::StrongTypePtr(cu, classType), numAttrs);

      // 2. copy all the contained types
      for (const auto slot : c10::irange(numAttrs)) {
        const auto& attrType = classType->getAttribute(slot);
        const auto& attrName = classType->getAttributeName(slot);

        if (!py::hasattr(obj, attrName.c_str())) {
          throw py::cast_error(c10::str(
              "Tried to cast object to type ",
              type->repr_str(),
              " but object",
              " was missing attribute ",
              attrName));
        }

        try {
          const auto& contained = py::getattr(obj, attrName.c_str());
          userObj->setSlot(slot, toIValue(contained, attrType));
        } catch (std::exception& e) {
          throw py::cast_error(c10::str(
              "Could not cast attribute '",
              attrName,
              "' to type ",
              attrType->repr_str(),
              ": ",
              e.what()));
        }
      }
      return userObj;
    }
    case TypeKind::InterfaceType: {
      auto interfaceType = type->expect<InterfaceType>();
      // When converting an pyobj to an interface, we check if rhs
      // is module or normal torchscript class, get the type and ivalue
      // from them correspondingly.
      c10::ClassTypePtr classType = nullptr;
      IValue res;
      if (auto mod = as_module(py::cast<py::object>(obj))) {
        classType = mod.value().type();
        res = mod.value()._ivalue();
      } else if (auto object = as_object(py::cast<py::object>(obj))) {
        classType = object.value().type();
        res = object.value()._ivalue();
      } else {
        // We inspect the value to found the compiled TorchScript class
        // and then create a ivalue::Object from that class type.
        py::str qualified_name =
            py::module::import("torch._jit_internal")
                .attr("_qualified_name")(py::type::handle_of(obj));
        auto pyCu = get_python_cu();
        classType = pyCu->get_class(c10::QualifiedName(qualified_name));
        if (!classType) {
          throw std::runtime_error(c10::str(
              "Assigning the object ",
              py::str(obj),
              " to an interface fails because the value is not "
              "a TorchScript compatible type, did you forget to",
              "turn it into a user defined TorchScript class?"));
        }
        res = toIValue(obj, classType);
      }
      // check if the classType conform with the interface or not
      std::stringstream why_not;
      if (!classType->isSubtypeOfExt(*interfaceType, &why_not)) {
        throw py::cast_error(c10::str(
            "Object of type ",
            classType->repr_str(),
            " is not compatible with interface ",
            interfaceType->repr_str(),
            "\n",
            why_not.str()));
      }
      return res;
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
      if (THPLayout_Check(obj.ptr())) {
        auto layout = reinterpret_cast<THPLayout*>(obj.ptr());
        return static_cast<int8_t>(layout->layout);
      }
      if (py::isinstance<py::bool_>(obj)) {
        return py::cast<bool>(obj);
      } else if (py::isinstance<py::int_>(obj)) {
        return THPUtils_unpackInteger<IValue>(obj.ptr());
      } else if (py::isinstance<py::float_>(obj)) {
        return py::cast<double>(obj);
      } else if (PyComplex_CheckExact(obj.ptr())) {
        auto c_obj = py::cast<std::complex<double>>(obj.ptr());
        return static_cast<c10::complex<double>>(c_obj);
      } else if (torch::is_symint(obj)) {
        return py::cast<c10::SymInt>(obj);
      } else if (torch::is_symfloat(obj)) {
        return py::cast<c10::SymFloat>(obj);
      } else if (torch::is_symbool(obj)) {
        return py::cast<c10::SymBool>(obj);
      } else {
        throw py::cast_error(
            c10::str("Cannot cast ", py::str(obj), " to ", type->repr_str()));
      }
    }
    case TypeKind::RRefType: {
#ifdef USE_RPC
      return obj.cast<torch::distributed::rpc::PyRRef>().toIValue();
#else
      TORCH_CHECK(false, "RRef is only supported with the distributed package");
#endif
    } break;
    case TypeKind::PyObjectType: {
      return c10::ivalue::ConcretePyObjectHolder::create(obj);
    }
    case TypeKind::CapsuleType: {
      return IValue::make_capsule(py::cast<c10::Capsule>(obj).obj_ptr);
    }
    case TypeKind::FutureType: {
      return obj.cast<std::shared_ptr<PythonFutureWrapper>>()->fut;
    }
    case TypeKind::AwaitType: {
      return obj.cast<std::shared_ptr<PythonAwaitWrapper>>()->aw_;
    }
    case TypeKind::AnyType:
      return toTypeInferredIValue(obj);
    case TypeKind::QSchemeType: {
      if (py::isinstance<py::int_>(obj)) {
        return static_cast<at::QScheme>(py::cast<int64_t>(obj));
      }
      throw py::cast_error(
          c10::str("Cannot cast ", py::str(obj), " to ", type->repr_str()));
    }
    case TypeKind::GeneratorType:
      return py::cast<at::Generator>(obj);
    case TypeKind::DynamicType:
    case TypeKind::FunctionType:
    case TypeKind::QuantizerType:
    case TypeKind::VarType:
    case TypeKind::AnyListType:
    case TypeKind::AnyTupleType:
    case TypeKind::AnyClassType:
    case TypeKind::AnyEnumType:
      break;
    case TypeKind::EnumType:
      EnumTypePtr enum_type = type->expect<EnumType>();
      py::object py_obj = py::reinterpret_borrow<py::object>(obj);
      std::string name = py::cast<std::string>(obj.attr("name"));
      IValue value = toIValue(obj.attr("value"), enum_type->getValueType(), {});
      auto enum_holder =
          c10::make_intrusive<c10::ivalue::EnumHolder>(enum_type, name, value);
      return IValue(enum_holder);
  }
  throw py::cast_error(c10::str(
      "toIValue() cannot handle converting to type: ", type->repr_str()));
}

py::object toPyObject(IValue ivalue) {
  if (ivalue.isNone()) {
    return py::none();
  } else if (ivalue.isTensor()) {
    auto tensor = std::move(ivalue).toTensor();
    if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
      TORCH_INTERNAL_ASSERT(tensor.device().is_cpu());
      auto py_tensor = py::cast(tensor);
      if (PyObject_HasAttrString(py_tensor.ptr(), "_wrapped_number")) {
        return py_tensor.attr("_wrapped_number");
      }
      auto scalar_type = tensor.scalar_type();
      switch (scalar_type) {
        case at::ScalarType::Bool:
          return py::cast(*tensor.const_data_ptr<bool>());
        case at::ScalarType::Long:
          return py::cast(*tensor.const_data_ptr<int64_t>());
        case at::ScalarType::UInt64:
          return py::cast(*tensor.const_data_ptr<uint64_t>());
        case at::ScalarType::Double:
          return py::cast(*tensor.const_data_ptr<double>());
        case at::ScalarType::ComplexDouble:
          // TODO: https://github.com/pytorch/pytorch/issues/77134
          return py::cast(static_cast<std::complex<double>>(
              *tensor.const_data_ptr<c10::complex<double>>()));
        default:
          TORCH_CHECK(
              false,
              "Missing cases in 'toPyObject' wrapped number handling! Can't convert ",
              scalar_type,
              " to a Python object");
      }
    } else {
      guardAgainstNamedTensor<at::Tensor>(tensor);
      return py::cast(std::move(tensor));
    }
  } else if (ivalue.isStorage()) {
    return py::cast(std::move(ivalue).toStorage());
  } else if (ivalue.isGenerator()) {
    return py::cast(std::move(ivalue).toGenerator());
  } else if (ivalue.isDouble()) {
    return py::cast(std::move(ivalue).toDouble());
  } else if (ivalue.isComplexDouble()) {
    return py::cast(
        static_cast<std::complex<double>>(std::move(ivalue).toComplexDouble()));
  } else if (ivalue.isInt()) {
    return py::cast(std::move(ivalue).toInt());
  } else if (ivalue.isBool()) {
    return py::cast(std::move(ivalue).toBool());
  } else if (ivalue.isString()) {
    if (getUTF8DecodingIgnore()) {
      std::string s = std::move(ivalue).toStringRef();
      PyObject* pyObj = PyUnicode_DecodeUTF8(s.data(), s.length(), "ignore");
      return py::reinterpret_steal<py::object>(pyObj);
    } else {
      return py::cast(std::move(ivalue).toStringRef());
    }
  } else if (ivalue.isList()) {
    auto list = std::move(ivalue).toList();
    py::list t{list.size()};
    for (const auto i : c10::irange(list.size())) {
      t[i] = toPyObject(IValue{list.get(i)});
    }
#if C10_RETURN_MOVE_IF_OLD_COMPILER
    return std::move(t);
#else
    return t;
#endif
  } else if (ivalue.isTuple()) {
    auto tuple = std::move(ivalue).toTuple();
    const auto& elements = tuple->elements();

    py::tuple t{elements.size()};
    for (const auto i : c10::irange(elements.size())) {
      t[i] = toPyObject(IValue{elements.at(i)});
    }

    // If we have a NamedTuple
    if (tuple->type() && tuple->type()->schema() &&
        !tuple->type()->schema()->name().empty()) {
      auto unqualName = tuple->type()->name()->name();

      std::vector<Argument> tuple_args = tuple->type()->schema()->arguments();

      std::vector<pybind11::object> defaults;
      auto it = std::find_if(
          tuple_args.begin(), tuple_args.end(), [](const Argument& arg) {
            return arg.default_value().has_value();
          });
      std::transform(
          it,
          tuple_args.end(),
          std::back_inserter(defaults),
          [](const Argument& arg) { return toPyObject(*arg.default_value()); });

      std::vector<std::string> fieldNames =
          fmap(tuple_args, [](const Argument& arg) { return arg.name(); });

      return py::module::import("torch._jit_internal")
          .attr("_create_named_tuple")(
              t, unqualName, fieldNames, py::make_tuple(defaults));
    } else {
#if C10_RETURN_MOVE_IF_OLD_COMPILER
      return std::move(t);
#else
      return t;
#endif
    }
  } else if (ivalue.isDevice()) {
    return py::cast(std::move(ivalue).toDevice());
  } else if (ivalue.isStream()) {
    return py::cast(std::move(ivalue).toStream());
  } else if (ivalue.isGenericDict()) {
    auto dict = std::move(ivalue).toGenericDict();
    py::dict py_dict;
    for (auto& pair : dict) {
      py_dict[toPyObject(IValue{pair.key()})] =
          toPyObject(IValue{pair.value()});
    }
#if C10_RETURN_MOVE_IF_OLD_COMPILER
    return std::move(py_dict);
#else
    return py_dict;
#endif
  } else if (ivalue.isRRef()) {
#ifdef USE_RPC
    auto RRefPtr =
        c10::dynamic_intrusive_pointer_cast<torch::distributed::rpc::RRef>(
            std::move(ivalue).toRRef());
    return py::cast(torch::distributed::rpc::PyRRef(RRefPtr));
#else
    TORCH_CHECK(false, "RRef is only supported with the distributed package");
#endif
  } else if (ivalue.isObject()) {
    const auto obj = std::move(ivalue).toObject();
    if (obj->type()->is_module()) {
      return py::cast(Module(obj));
    }

    auto pyCu = get_python_cu();
    if (obj->name().find("__torch__.torch.classes") == 0) {
      return py::cast(Object(obj));
    }
    const auto classType = pyCu->get_class(c10::QualifiedName(obj->name()));
    AT_ASSERT(classType, c10::str(obj->name(), " is not found."));
    auto pyClass = getScriptedClassOrError(obj->type());
    auto pyObj = pyClass.attr("__new__")(pyClass);

    const auto numAttrs = classType->numAttributes();

    for (const auto slot : c10::irange(numAttrs)) {
      const auto& attrName = classType->getAttributeName(slot);
      IValue v = obj->getSlot(slot);
      py::setattr(pyObj, attrName.c_str(), toPyObject(std::move(v)));
    }
    return pyObj;
  } else if (ivalue.isPyObject()) {
    // return borrowed reference to ensure it correctly incref the underlying
    // PyObject
    return py::reinterpret_borrow<py::object>(ivalue.toPyObject());
  } else if (ivalue.isCapsule()) {
    return py::cast(c10::Capsule(ivalue.toCapsule()));
  } else if (ivalue.isFuture()) {
    return py::cast(std::make_shared<PythonFutureWrapper>(ivalue.toFuture()));
  } else if (ivalue.isAwait()) {
    return py::cast(std::make_shared<PythonAwaitWrapper>(ivalue.toAwait()));
  } else if (ivalue.isEnum()) {
    auto enum_holder = ivalue.toEnumHolder();
    auto py_class = getScriptedClassOrError(enum_holder->type());
    return py_class.attr(enum_holder->name().c_str());
  } else if (ivalue.isRRef()) {
#ifdef USE_RPC
    return py::cast(torch::distributed::rpc::PyRRef(
        c10::static_intrusive_pointer_cast<distributed::rpc::RRef>(
            ivalue.toRRef())));
#else
    TORCH_CHECK(false, "RRef is only supported with the distributed package");
#endif
  } else if (ivalue.isSymInt()) {
    return py::cast(std::move(ivalue).toSymInt());
  } else if (ivalue.isSymFloat()) {
    return py::cast(std::move(ivalue).toSymFloat());
  } else if (ivalue.isSymBool()) {
    return py::cast(std::move(ivalue).toSymBool());
  } else if (ivalue.isUnsigned()) {
    return py::cast(std::move(ivalue).toUInt());
  } else {
    TORCH_CHECK(
        false,
        "Missing cases in 'toPyObject'! Can't convert ",
        ivalue.tagKind(),
        " to a Python object");
  }
}

std::pair<std::shared_ptr<Operator>, Stack> getOpWithStack(
    const std::vector<std::shared_ptr<Operator>>& operations,
    const py::args& args,
    const py::kwargs& kwargs) {
  return getOpWithStack(
      c10::ArrayRef<std::shared_ptr<Operator>>(operations), args, kwargs);
}

std::pair<std::shared_ptr<Operator>, Stack> getOpWithStack(
    c10::ArrayRef<std::shared_ptr<Operator>> operations,
    const py::args& args,
    const py::kwargs& kwargs) {
  Stack stack;
  if (operations.size() == 1) {
    std::shared_ptr<Operator> op = operations[0];
    // Create a stack full of the arguments and keyword arguments.
    stack = createStackForSchema(op->schema(), args, kwargs, std::nullopt);

    return std::make_pair(std::move(op), std::move(stack));
  } else {
    std::vector<schema_match_error> errors;
    std::shared_ptr<Operator> found_op = nullptr;
    for (const auto& op : operations) {
      try {
        stack = createStackForSchema(op->schema(), args, kwargs, std::nullopt);
        found_op = op;
        break;
      } catch (schema_match_error& error) {
        errors.push_back(std::move(error));
      }
    }
    if (!found_op) {
      std::stringstream ss;
      ss << "Overloaded torch operator invoked from Python failed to match any schema:\n";
      for (const auto& err : errors) {
        ss << err.what() << "\n\n";
      }
      throw std::runtime_error(ss.str());
    }

    return std::make_pair(std::move(found_op), std::move(stack));
  }
}

// This function is used to check if the schema is valid for the given args and
// kwargs. It checks script object by checking whether the FakeScriptObject is
// an instance of the corresponding fake class for the actual class used in
// schema.
bool checkSchemaAllowFakeScriptObject(
    const FunctionSchema& schema,
    const py::args& args,
    const py::kwargs& kwargs) {
  bool match = false;
  try {
    match = matchSchemaAllowFakeScriptObject(schema, args, kwargs);
  } catch (schema_match_error& error) {
    throw std::runtime_error(error.what());
  }
  return match;
}

py::object invokeOperatorFromPython(
    const std::vector<std::shared_ptr<Operator>>& operations,
    const py::args& args,
    const py::kwargs& kwargs,
    std::optional<c10::DispatchKey> dk) {
  return invokeOperatorFromPython(
      c10::ArrayRef<std::shared_ptr<Operator>>(operations), args, kwargs, dk);
}

py::object invokeOperatorFromPython(
    c10::ArrayRef<std::shared_ptr<Operator>> operations,
    const py::args& args,
    const py::kwargs& kwargs,
    std::optional<c10::DispatchKey> dk) {
  auto [found_op, stack] = getOpWithStack(operations, args, kwargs);
  {
    pybind11::gil_scoped_release no_gil_guard;
    if (dk) {
      found_op->getOperationForDispatchKey (*dk)(stack);
    } else {
      found_op->getOperation()(stack);
    }
  }

  return createPyObjectForStack(std::move(stack));
}

std::optional<py::object> _maybe_handle_torch_function(
    const std::string& ns,
    const std::string& method_name,
    const std::string& overload_name,
    bool is_overload,
    const py::args& args,
    const py::kwargs& kwargs) {
  std::vector<PyObject*> overloaded_args;
  const auto args_size = args.size();
  size_t total_arg_num = args_size + kwargs.size();
  for (const auto i : c10::irange(args_size)) {
    is_tensor_and_append_overloaded(args[i].ptr(), &overloaded_args);
    is_tensor_list_and_append_overloaded(
        args[i].ptr(),
        &overloaded_args,
        static_cast<int>(total_arg_num),
        false /* throw_error */);
  }
  // NB: for kwargs, we cannot guarantee the order of appending
  // is the same as the argument order in operator's schema.
  // This is suboptimal, but should be fine. Later when we have
  // better schema matching and argument parsing, we could
  // match the operator in `operations` first, then the order will
  // be guaranteed.
  for (auto item : kwargs) {
    is_tensor_and_append_overloaded(item.second.ptr(), &overloaded_args);
    is_tensor_list_and_append_overloaded(
        item.second.ptr(),
        &overloaded_args,
        total_arg_num,
        false /* throw_error */);
  }
  if (!overloaded_args.empty() || at::impl::torch_function_mode_enabled()) {
    auto self_func = py::module::import("torch")
                         .attr("ops")
                         .attr(ns.c_str())
                         .attr(method_name.c_str());
    if (is_overload) {
      if (overload_name.empty()) {
        self_func = self_func.attr("default");
      } else {
        self_func = self_func.attr(overload_name.c_str());
      }
    }
    std::string module_name("torch.ops");
    module_name.append(ns);
    return {pybind11::reinterpret_steal<py::object>(
        handle_torch_function_no_python_arg_parser(
            overloaded_args,
            args.ptr(),
            kwargs.ptr(),
            method_name.c_str(),
            self_func.ptr(),
            module_name.c_str()))};
  }
  return std::nullopt;
}

py::object _get_operation_for_overload_or_packet(
    const std::vector<std::shared_ptr<Operator>>& operations,
    Symbol symbol,
    const py::args& args,
    const py::kwargs& kwargs,
    bool is_overload,
    std::optional<c10::DispatchKey> dk) {
  return _get_operation_for_overload_or_packet(
      c10::ArrayRef(operations), symbol, args, kwargs, is_overload, dk);
}

py::object _get_operation_for_overload_or_packet(
    c10::ArrayRef<std::shared_ptr<Operator>> operations,
    Symbol symbol,
    const py::args& args,
    const py::kwargs& kwargs,
    bool is_overload,
    std::optional<c10::DispatchKey> dk) {
  std::string ns = symbol.ns().toUnqualString();
  std::string method_name = symbol.toUnqualString();
  std::string overload_name = operations[0]->schema().overload_name();
  auto res = _maybe_handle_torch_function(
      ns, method_name, overload_name, is_overload, args, kwargs);
  auto torch_function_called = res.has_value();
  return torch_function_called
      ? *res
      : invokeOperatorFromPython(operations, args, kwargs, dk);
}

} // namespace torch::jit
