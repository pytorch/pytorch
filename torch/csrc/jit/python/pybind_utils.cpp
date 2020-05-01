#include <torch/csrc/jit/python/pybind_utils.h>

namespace torch {
namespace jit {
InferredType tryToInferType(py::handle input) {
  // Try tensor types
  if (THPVariable_Check(input.ptr())) {
    auto tensor = py::cast<at::Tensor>(input);
    return InferredType(TensorType::create(tensor));
  }

  if (input.is(py::none())) {
    return InferredType(NoneType::get());
  }

  if (py::isinstance<StrongFunctionPtr>(input)) {
    auto fn = py::cast<StrongFunctionPtr>(input).function_;
    return InferredType(FunctionType::create(fn));
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
  } else if (THPLayout_Check(input.ptr())) {
    return InferredType(IntType::get());
  }

  py::bool_ isClass =
      py::module::import("inspect").attr("isclass")(input.get_type());
  if (py::cast<bool>(isClass)) {
    py::str qualifiedName = py::module::import("torch.jit")
                                .attr("_qualified_name")(input.get_type());
    auto pyClass = py::module::import("torch.jit")
                       .attr("_get_script_class")(qualifiedName);
    if (!pyClass.is_none()) {
      auto cu = get_python_cu();
      const auto classname =
          c10::QualifiedName(py::cast<std::string>(qualifiedName));
      auto class_type = cu->get_class(classname);
      TORCH_INTERNAL_ASSERT(class_type);
      return InferredType(class_type);
    }
  }

  if (py::isinstance<Object>(input)) {
    auto object = py::cast<Object>(input);
    return InferredType(object.type());
#ifdef USE_DISTRIBUTED
  } else if (py::isinstance<torch::distributed::rpc::PyRRef>(input)) {
    auto rref_ivalue = input.cast<torch::distributed::rpc::PyRRef>().toIValue();
    return InferredType(rref_ivalue.type());
#endif
  }

  // Try container types
  return tryToInferContainerType(input);
}

py::object toPyObject(IValue ivalue) {
  if (ivalue.isNone()) {
    return py::none();
  } else if (ivalue.isTensor()) {
    auto tensor = std::move(ivalue).toTensor();
    if (tensor.is_sparse()) {
      TORCH_WARN_ONCE(
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
  } else if (ivalue.isList()) {
    auto list = std::move(ivalue).toList();
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
          .attr("_create_named_tuple")(t, unqualName, fieldNames);
    } else {
      return std::move(t);
    }
  } else if (ivalue.isDevice()) {
    return py::cast<py::object>(THPDevice_New(std::move(ivalue).toDevice()));
  } else if (ivalue.isGenericDict()) {
    auto dict = std::move(ivalue).toGenericDict();
    py::dict py_dict;
    for (auto& pair : dict) {
      py_dict[toPyObject(IValue{pair.key()})] =
          toPyObject(IValue{pair.value()});
    }
    return std::move(py_dict);
  } else if (ivalue.isRRef()) {
#ifdef USE_DISTRIBUTED
    auto RRefPtr =
        c10::dynamic_intrusive_pointer_cast<torch::distributed::rpc::RRef>(
            std::move(ivalue).toRRef());
    return py::cast(torch::distributed::rpc::PyRRef(RRefPtr));
#else
    AT_ERROR("RRef is only supported with the distributed package");
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
    AT_ASSERT(classType);
    auto pyClass =
        py::module::import("torch.jit").attr("_get_script_class")(obj->name());
    if (pyClass.is_none()) {
      std::stringstream err;
      err << "Unknown reference to ScriptClass ";
      err << obj->name();
      err << ". Did you forget to import it?)";
      throw std::runtime_error(err.str());
    }
    auto pyObj = pyClass.attr("__new__")(pyClass);

    const auto numAttrs = classType->numAttributes();

    for (size_t slot = 0; slot < numAttrs; slot++) {
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
    return py::cast(ivalue.toCapsule());
  } else if (ivalue.isFuture()) {
    return py::cast(std::make_shared<PythonFutureWrapper>(ivalue.toFuture()));
  } else if (ivalue.isRRef()) {
#ifdef USE_DISTRIBUTED
    return py::cast(torch::distributed::rpc::PyRRef(
        c10::static_intrusive_pointer_cast<distributed::rpc::RRef>(
            ivalue.toRRef())));
#else
    TORCH_CHECK(false, "RRef is only supported with the distributed package");
#endif
  } else {
    AT_ERROR(
        "Missing cases in 'toPyObject'! Can't convert ",
        ivalue.tagKind(),
        " to a Python object");
  }
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
            return IValue(py::cast<std::vector<int64_t>>(obj));
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
      return toIValue(obj, type->expect<OptionalType>()->getElementType());
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

        const auto& contained = py::getattr(obj, attrName.c_str());
        userObj->setSlot(slot, toIValue(contained, attrType));
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
        py::str qualified_name = py::module::import("torch.jit")
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
        res = toIValue(std::move(obj), classType);
      }
      // check if the classType conform with the interface or not
      std::stringstream why_not;
      if (!classType->isSubtypeOfExt(interfaceType, &why_not)) {
        throw py::cast_error(c10::str(
            "Object ",
            py::str(obj),
            " is not compatible with interface ",
            interfaceType->python_str(),
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
      } else {
        throw py::cast_error(
            c10::str("Cannot cast ", py::str(obj), " to ", type->python_str()));
      }
    }
    case TypeKind::RRefType: {
#ifdef USE_DISTRIBUTED
      return obj.cast<torch::distributed::rpc::PyRRef>().toIValue();
#else
      AT_ERROR("RRef is only supported with the distributed package");
#endif
    } break;
    case TypeKind::PyObjectType:
      // convert a py::handle to the IValue that holds the py::object
      return c10::ivalue::ConcretePyObjectHolder::create(
          obj.cast<py::object>());

    case TypeKind::CapsuleType: {
      return IValue::make_capsule(
          py::cast<c10::intrusive_ptr<CustomClassHolder>>(obj));
    }
    case TypeKind::FutureType: {
      return obj.cast<std::shared_ptr<PythonFutureWrapper>>()->fut;
    }
    case TypeKind::AnyType:
      return toTypeInferredIValue(obj);
    case TypeKind::FunctionType:
    case TypeKind::GeneratorType:
    case TypeKind::VarType:
    case TypeKind::QSchemeType:
    case TypeKind::AnyListType:
    case TypeKind::AnyTupleType:
    case TypeKind::AnyClassType:
      break;
  }
  throw py::cast_error(c10::str(
      "toIValue() cannot handle converting to type: ", type->python_str()));
}

} // namespace jit
} // namespace torch
