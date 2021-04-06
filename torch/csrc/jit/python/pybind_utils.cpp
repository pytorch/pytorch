#include <torch/csrc/jit/python/pybind_utils.h>

#include <torch/csrc/jit/python/python_ivalue.h>

namespace torch {
namespace jit {

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
      auto var = py::cast<autograd::Variable>(obj);
      if (var.is_sparse()) {
        TORCH_WARN_ONCE(
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
    case TypeKind::ComplexType: {
      auto c_obj = py::cast<std::complex<double>>(obj.ptr());
      return static_cast<c10::complex<double>>(c_obj);
    }
    case TypeKind::IntType:
    // TODO(xintchen): Handling LayoutType and ScalarTypeType correctly.
    case TypeKind::LayoutType:
    case TypeKind::ScalarTypeType:
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
      TupleTypePtr tuple_type = type->cast<TupleType>();
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
      if (auto mod = as_module(py::cast<py::object>(obj))) {
        // if obj is already a ScriptModule, just return its ivalue
        return mod.value()._ivalue();
      }
      // otherwise is a normal class object, we create a fresh
      // ivalue::Object to use from the py object.
      // 1. create a bare ivalue
      const size_t numAttrs = classType->numAttributes();
      auto cu = classType->compilation_unit();
      auto userObj = c10::ivalue::Object::create(
          c10::StrongTypePtr(cu, classType), numAttrs);

      // 2. copy all the contained types
      for (size_t slot = 0; slot < numAttrs; slot++) {
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
      if (!classType->isSubtypeOfExt(interfaceType, &why_not)) {
        throw py::cast_error(c10::str(
            "Object ",
            py::str(obj),
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
    case TypeKind::FunctionType:
    case TypeKind::GeneratorType:
    case TypeKind::StorageType:
    case TypeKind::QuantizerType:
    case TypeKind::VarType:
    case TypeKind::QSchemeType:
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
