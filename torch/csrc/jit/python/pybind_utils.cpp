#include <torch/csrc/jit/python/module_python.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/python/python_dict.h>
#include <torch/csrc/jit/python/python_ivalue.h>
#include <torch/csrc/jit/python/python_list.h>

#include <ATen/ScalarOps.h>

#include <c10/core/QScheme.h>
#include <c10/util/irange.h>
#include <torch/csrc/utils/python_arg_parser.h>

namespace torch {
namespace jit {

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
  auto& registered_instances =
      pybind11::detail::get_internals().registered_instances;
  auto range = registered_instances.equal_range(ptr);
  for (auto it = range.first; it != range.second; ++it) {
    auto vh = it->second->get_value_and_holder();
    vh.set_instance_registered(false);
  }
  registered_instances.erase(ptr);
}

IValue toIValue(py::handle obj, const TypePtr& type, c10::optional<int32_t> N) {
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
        at::Scalar scalar;
        if (PyBool_Check(obj.ptr())) {
          scalar = at::Scalar(THPUtils_unpackBool(obj.ptr()));
        } else if (THPUtils_checkLong(obj.ptr())) {
          scalar = at::Scalar(THPUtils_unpackLong(obj.ptr()));
        } else if (PyComplex_Check(obj.ptr())) {
          scalar = at::Scalar(THPUtils_unpackComplexDouble(obj.ptr()));
        } else if (THPUtils_checkDouble(obj.ptr())) {
          scalar = at::Scalar(THPUtils_unpackDouble(obj.ptr()));
        } else {
          throw py::cast_error(
              c10::str("Unable to cast ", py::str(obj), " to Tensor"));
        }
        at::Tensor tensor = at::scalar_to_tensor(scalar);
        tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
        return tensor;
      }
    }
    case TypeKind::StorageType:
      return py::cast<at::Storage>(obj);
    case TypeKind::FloatType:
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
      if (torch::is_symint_node(obj.ptr())) {
        return py::cast<c10::SymInt>(obj);
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
      auto stream = reinterpret_cast<THPStream*>(obj.ptr());
      return static_cast<int64_t>(stream->cdata);
    }
    case TypeKind::ListType: {
      // If the object is a ScriptList, retrieve the c10::List
      // instance inside it.
      try {
        auto script_list = py::cast<ScriptList>(obj);
        return script_list.list_;
      } catch (...) {
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
          c10::List<c10::SymInt> symints;
          for (auto it = obj.begin(); it != obj.end(); it++) {
            auto elm = *it;
            auto si = py::cast<c10::SymInt>(elm);
            symints.push_back(si);
          }
          return symints;
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
      return toIValue(obj, type->expectRef<OptionalType>().getElementType());
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
        py::str qualified_name = py::module::import("torch._jit_internal")
                                     .attr("_qualified_name")(obj.get_type());
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
      if (py::isinstance<py::int_>(obj)) {
        return py::cast<int64_t>(obj);
      } else if (py::isinstance<py::float_>(obj)) {
        return py::cast<double>(obj);
      } else if (PyComplex_CheckExact(obj.ptr())) {
        auto c_obj = py::cast<std::complex<double>>(obj.ptr());
        return static_cast<c10::complex<double>>(c_obj);
      } else {
        throw py::cast_error(
            c10::str("Cannot cast ", py::str(obj), " to ", type->repr_str()));
      }
    }
    case TypeKind::RRefType: {
#ifdef USE_RPC
      return obj.cast<torch::distributed::rpc::PyRRef>().toIValue();
#else
      AT_ERROR("RRef is only supported with the distributed package");
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

} // namespace jit
} // namespace torch
