#include <torch/csrc/jit/script/python_sugared_value.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/jit/script/module_python.h>
#include <torch/csrc/jit/script/schema_matching.h>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace torch {
namespace jit {
namespace script {

std::string typeString(py::handle h) {
  return py::str(h.get_type().attr("__name__"));
}

std::shared_ptr<Function> as_function(const py::object& obj) {
  if (py::isinstance<Function>(obj)) {
    return py::cast<std::shared_ptr<Function>>(obj);
  }
  return nullptr;
}

FunctionSchema PythonValue::getSchema(
    const size_t n_args,
    const size_t n_binders) {
  auto annotations = py::module::import("torch.jit.annotations");
  auto signature = annotations.attr("get_signature")(self);
  std::vector<Argument> args, rets;
  // We may mutate this if we can determine the number of args from Python
  // introspection.
  size_t actual_n_args = n_args;
  if (!signature.is_none()) {
    std::vector<TypePtr> arg_types;
    TypePtr ret_type;
    std::tie(arg_types, ret_type) =
        py::cast<std::pair<std::vector<TypePtr>, TypePtr>>(signature);
    args.reserve(arg_types.size());
    size_t idx = 0; // Fake argument names by putting in the index
    for (auto& arg_type : arg_types) {
      args.push_back(
          Argument(std::to_string(idx++), std::move(arg_type), {}, {}, false));
    }
    rets.push_back(Argument("0", std::move(ret_type), {}, {}, false));
  } else {
    // Create a default signature using what information we have

    // First see if we can introspect the number of function parameters
    // irrespective of the presence of explicit type annotations
    auto num_params = annotations.attr("get_num_params")(self);
    if (!num_params.is_none()) {
      // Return a signature with the correct number of params according to the
      // Python function. The error handling in call() will catch any mismatch
      // later.
      actual_n_args = py::cast<size_t>(num_params);
    }
    // Construct the default signature: all arguments and returns will be
    // DynamicType
    args.reserve(actual_n_args);
    for (size_t i = 0; i < actual_n_args; ++i) {
      args.push_back(
          Argument(std::to_string(i), TensorType::get(), {}, {}, false));
    }
    TypePtr ret_type = TensorType::get();
    if (n_binders == 0) {
      ret_type = NoneType::get();
    } else if (n_binders > 1) {
      std::vector<TypePtr> tuple_values(n_binders, ret_type);
      ret_type = TupleType::create(std::move(tuple_values));
    }
    rets.push_back(Argument("0", ret_type, {}, {}, false));
  }
  return FunctionSchema("", "", std::move(args), std::move(rets));
}

std::shared_ptr<SugaredValue> PythonValue::call(
    const SourceRange& loc,
    Function& m,
    at::ArrayRef<NamedValue> inputs_,
    at::ArrayRef<NamedValue> attributes,
    size_t n_binders) {
  auto inputs = toValues(*m.graph(), inputs_);
  auto schema = getSchema(inputs.size(), n_binders);

  std::stringstream failure_messages;
  c10::optional<MatchedSchema> matched_schema = tryMatchSchema(
      schema,
      loc,
      *m.graph(),
      c10::nullopt,
      inputs_,
      attributes,
      failure_messages,
      /*conv_tensor_to_num*/ true);
  if (!matched_schema)
    throw ErrorReport(loc) << failure_messages.str();

  // Release the function object so we can wrap it in a PythonOp
  py::object func = self;
  std::string cconv(inputs.size(), 'd');
  Node* new_node = m.graph()->insertNode(
      m.graph()->createPythonOp(THPObjectPtr(func.release().ptr()), cconv, {}));

  // Mark if function is ignored on export
  if (py::cast<bool>(
          py::module::import("torch.jit").attr("_try_get_ignored_op")(self))) {
    auto python_op = static_cast<PythonOp*>(new_node);
    python_op->ignore_on_export = true;
  }
  new_node->setSourceLocation(std::make_shared<SourceRange>(loc));
  for (auto& i : matched_schema->inputs)
    new_node->addInput(i);

  Value* output =
      new_node->addOutput()->setType(matched_schema->return_types.at(0));
  return std::make_shared<SimpleValue>(output);
}

std::string PythonValue::kind() const {
  std::stringstream ss;
  ss << "python value of type '" << typeString(self) << "'";
  return ss.str();
}

std::vector<std::shared_ptr<SugaredValue>> PythonValue::asTuple(
    const SourceRange& loc,
    Function& m,
    const c10::optional<size_t>& size_hint) {
  const std::string type_str = typeString(self);
  std::stringstream ss;
  ss << kind() << " cannot be used as a tuple";
  checkForAddToConstantsError(ss);
  throw ErrorReport(loc) << ss.str();
}

std::shared_ptr<SugaredValue> PythonValue::attr(
    const SourceRange& loc,
    Function& m,
    const std::string& field) {
  const std::string type_str = typeString(self);
  std::stringstream ss;
  ss << "attribute lookup is not defined on " << kind();
  checkForAddToConstantsError(ss);
  throw ErrorReport(loc) << ss.str();
}

py::object PythonValue::getattr(
    const SourceRange& loc,
    const std::string& name) {
  try {
    return py::getattr(self, name.c_str());
  } catch (py::error_already_set& e) {
    throw ErrorReport(loc) << "object has no attribute " << name;
  }
}

void PythonValue::checkForAddToConstantsError(std::stringstream& ss) {
  auto nn = py::module::import("torch.nn");
  if (py::isinstance(self, nn.attr("ModuleList")) ||
      py::isinstance(self, nn.attr("Sequential"))) {
    ss << ". Did you forget to add it to __constants__? ";
  }
}

std::shared_ptr<SugaredValue> PythonModuleValue::attr(
    const SourceRange& loc,
    Function& m,
    const std::string& field) {
  py::object member = getattr(loc, field);
  // note: is_constant = true because we consider that global properties
  // on modules like math.pi or torch.float to be constants
  // eventhough it is possible, though rare, for someone to mutate them
  return toSugaredValue(member, m, loc, /*is_constant=*/true);
}

std::vector<std::shared_ptr<SugaredValue>> ConstantPythonTupleValue::asTuple(
    const SourceRange& loc,
    Function& m,
    const c10::optional<size_t>& size_hint) {
  py::tuple tup = self;
  std::vector<std::shared_ptr<SugaredValue>> result;
  result.reserve(tup.size());
  for (py::handle t : tup) {
    py::object obj = py::reinterpret_borrow<py::object>(t);
    result.push_back(toSugaredValue(obj, m, loc, true));
  }
  return result;
}

Value* ConstantPythonTupleValue::asValue(const SourceRange& loc, Function& m) {
  std::vector<Value*> values;
  for (const auto& sugared_item : asTuple(loc, m)) {
    values.push_back(sugared_item->asValue(loc, m));
  }
  auto node = m.graph()->createTuple(values);
  return m.graph()->insertNode(node)->output();
}

std::shared_ptr<SugaredValue> OverloadedFunctionValue::call(
    const SourceRange& loc,
    Function& caller,
    at::ArrayRef<NamedValue> inputs,
    at::ArrayRef<NamedValue> attributes,
    size_t n_binders) {
  std::stringstream err;
  std::vector<NamedValue> new_inputs = inputs.vec();
  new_inputs.insert(new_inputs.begin(), module_);

  for (const std::string& method_name : method_names_) {
    auto cls = module_->type()->expect<ClassType>();
    std::shared_ptr<Function> fn = cls->getMethod(method_name);
    auto match = tryMatchSchema(
        fn->getSchema(),
        loc,
        *caller.graph().get(),
        c10::nullopt,
        new_inputs,
        attributes,
        err,
        true);
    if (match) {
      return MethodValue(module_, fn)
          .call(loc, caller, inputs, attributes, n_binders);
    }
  }
  throw ErrorReport(loc) << "Could not find any matching overloads\n"
                         << err.str();
}

std::shared_ptr<SugaredValue> ModuleValue::attr(
    const SourceRange& loc,
    Function& m,
    const std::string& field) {
  // workaround to make self.training work
  // it adds a buffer 'training' to the model if one doesn't exist
  // and then loads that parameter, casting it to bool
  if (field == "training") {
    Slot* v = module_->find_buffer(field);
    if (!v) {
      bool training = py::cast<bool>(py::getattr(py_module_, "training"));
      auto t =
          autograd::make_variable(at::full({}, training ? 1 : 0, at::kLong));
      module_->register_buffer("training", std::move(t));
      v = module_->find_buffer(field);
    }
    Value* the_tensor = m.graph()->insertGetAttr(self_, "training");
    Value* the_bool = m.graph()->insert(prim::Bool, {the_tensor});
    return std::make_shared<SimpleValue>(the_bool);
  }

  if (std::shared_ptr<Module> v = module_->find_module(field)) {
    return std::make_shared<ModuleValue>(
        m.graph()->insertGetAttr(self_, field),
        v,
        py_module_.attr(field.c_str()));
  } else if (auto kind = module_->kind_of(field)) {
    // methods, parameters, attributes, and buffers are all first class
    return SimpleValue(self_).attr(loc, m, field);
  }

  // This can also be a call to a non-script module, or a plain
  // python method. If so return this as a python value.

  py::object overloads =
      py_module_.attr("_overloads").attr("get")(field, py::none());
  if (!overloads.is_none()) {
    return std::make_shared<OverloadedFunctionValue>(
        self_, py::cast<std::vector<std::string>>(overloads));
  }

  if (py::object attr = py::getattr(py_module_, field.c_str(), py::none())) {
    if (py::isinstance<py::function>(attr) &&
        py::hasattr(attr, "_parameter_names_fn")) {
      // Fetch the names of the parameters in the list so they're in the
      // right order
      auto fn_self = py::getattr(attr, "__self__");
      auto param_names = py::getattr(attr, "_parameter_names_fn")(fn_self);

      Graph& g = *m.graph();
      // Add all module parameters as inputs to the graph
      std::vector<Value*> params;
      for (auto name : param_names) {
        params.emplace_back(g.insertGetAttr(self_, py::str(name)));
      }
      auto list = g.insertNode(g.createTuple(params))->output();
      return std::make_shared<ConstantParameterList>(list);
    }
    if (py::isinstance<py::function>(attr) ||
        py::isinstance(attr, py::module::import("torch.nn").attr("Module")) ||
        py_module_.attr("_constants_set").contains(field.c_str())) {
      return toSugaredValue(attr, m, loc, true);
    } else {
      std::string hint = "did you forget to add it __constants__?";
      if (py::isinstance(attr, py::module::import("torch").attr("Tensor"))) {
        hint = "Tensors must be added to a module as a buffer or parameter";
      }
      throw ErrorReport(loc)
          << "attribute '" << field << "' of type '" << typeString(attr)
          << "' is not usable in a script method (" << hint << ")";
    }
  }
  throw ErrorReport(loc) << "module has no attribute '" << field << "'";
}

std::vector<std::shared_ptr<SugaredValue>> ModuleValue::asTuple(
    const SourceRange& loc,
    Function& m,
    const c10::optional<size_t>& size_hint) {
  if (!py::isinstance(
          py_module_, py::module::import("torch.jit").attr("_ConstModuleList")))
    return SugaredValue::asTuple(loc, m, size_hint);
  std::vector<std::shared_ptr<SugaredValue>> result;
  for (py::handle py_submodule : py_module_) {
    py::object obj = py::reinterpret_borrow<py::object>(py_submodule);
    if (auto sub_module = as_module(obj)) {
      Value* module_v = m.graph()->insertGetAttr(self_, sub_module->name());
      result.emplace_back(
          std::make_shared<ModuleValue>(module_v, sub_module, obj));
    } else {
      result.push_back(toSugaredValue(
          obj,
          m,
          loc,
          /*is_constant =*/false));
    }
  }
  return result;
}

std::shared_ptr<SugaredValue> BooleanDispatchValue::call(
    const SourceRange& loc,
    Function& caller,
    at::ArrayRef<NamedValue> inputs,
    at::ArrayRef<NamedValue> attributes,
    size_t n_binders) {
  c10::optional<bool> result;
  Graph& graph = *(caller.graph());

  auto index = py::cast<size_t>(dispatched_fn_["index"]);
  auto arg_name = py::str(dispatched_fn_["arg_name"]);

  if (index < inputs.size()) {
    // Dispatch flag is in arg list
    result = constant_as<bool>(inputs.at(index).value(graph));
  } else if (auto i = findInputWithName(arg_name, attributes)) {
    // Dispatch flag is in kwargs
    result = constant_as<bool>(attributes[*i].value(graph));
  } else {
    // Didn't find dispatch flag, so use default value
    result = py::cast<bool>(dispatched_fn_["default"]);
  }

  if (!result) {
    throw ErrorReport(loc) << "value for boolean dispatch was not constant";
  }

  std::shared_ptr<SugaredValue> value;
  if (*result) {
    value = toSugaredValue(dispatched_fn_["if_true"], caller, loc);
  } else {
    value = toSugaredValue(dispatched_fn_["if_false"], caller, loc);
  }
  return value->call(loc, caller, inputs, attributes, n_binders);
}
std::shared_ptr<SugaredValue> toSugaredValue(
    py::object obj,
    Function& m,
    SourceRange loc,
    bool is_constant) {
  // directly create SimpleValues when possible, because they are first-class
  // and can be re-assigned. Otherwise, this would be invalid:
  // f = python_constant
  // while ...
  //   f = f + 1
  auto& g = *m.graph();
  if (is_constant) {
    if (py::isinstance<py::bool_>(obj)) {
      return toSimple(g.insertConstant(py::cast<bool>(obj), nullptr, loc));
    } else if (py::isinstance<py::int_>(obj)) {
      return toSimple(g.insertConstant(py::cast<int64_t>(obj), nullptr, loc));
    } else if (py::isinstance<py::float_>(obj)) {
      return toSimple(g.insertConstant(py::cast<double>(obj), nullptr, loc));
    } else if (py::isinstance<py::str>(obj)) {
      return toSimple(
          g.insertConstant(py::cast<std::string>(obj), nullptr, loc));
    } else if (obj.is(py::none())) {
      return toSimple(g.insertConstant(IValue(), nullptr, loc));
    } else if (THPDevice_Check(obj.ptr())) {
      auto device = reinterpret_cast<THPDevice*>(obj.ptr());
      return toSimple(g.insertConstant(device->device));
    } else if (THPLayout_Check(obj.ptr())) {
      auto layout = reinterpret_cast<THPLayout*>(obj.ptr());
      const auto v = static_cast<int64_t>(layout->layout);
      return toSimple(g.insertConstant(v, nullptr, loc));
    } else if (THPDtype_Check(obj.ptr())) {
      auto dtype = reinterpret_cast<THPDtype*>(obj.ptr());
      const auto v = static_cast<int64_t>(dtype->scalar_type);
      return toSimple(g.insertConstant(v, nullptr, loc));
    } else if (py::isinstance<py::tuple>(obj)) {
      return std::make_shared<ConstantPythonTupleValue>(obj);
    }
  }

  auto weak_obj =
      py::module::import("torch.jit").attr("_try_get_weak_module")(obj);
  if (!weak_obj.is_none()) {
    obj = weak_obj;
  }
  if (auto callee = as_function(obj)) {
    return std::make_shared<MethodValue>(c10::nullopt, callee);
  } else if (py::isinstance<py::module>(obj)) {
    return std::make_shared<PythonModuleValue>(obj);
  } else if (obj.ptr() == py::module::import("torch.jit").attr("_fork").ptr()) {
    return std::make_shared<ForkValue>();
  } else if (
      obj.ptr() == py::module::import("torch.jit").attr("annotate").ptr()) {
    return std::make_shared<AnnotateValue>();
  } else if (auto callee = as_module(obj)) {
    throw ErrorReport(loc) << "Cannot call a ScriptModule that is not"
                           << " a submodule of the caller";
  }

  py::object builtin_name =
      py::module::import("torch.jit").attr("_find_builtin")(obj);
  if (!builtin_name.is_none()) {
    return std::make_shared<BuiltinFunction>(
        Symbol::fromQualString(py::str(builtin_name)), c10::nullopt);
  }

  if (py::isinstance<py::function>(obj)) {
    auto compiled_fn =
        py::module::import("torch.jit").attr("_try_compile_weak_script")(obj);
    if (auto callee = as_function(compiled_fn)) {
      return std::make_shared<MethodValue>(c10::nullopt, callee);
    }
  }

  py::object dispatched_fn =
      py::module::import("torch.jit").attr("_try_get_dispatched_fn")(obj);
  if (!dispatched_fn.is_none()) {
    return std::make_shared<BooleanDispatchValue>(std::move(dispatched_fn));
  }

  py::bool_ isClass = py::module::import("inspect").attr("isclass")(obj);
  if (py::cast<bool>(isClass)) {
    py::str qualifiedName =
        py::module::import("torch.jit").attr("_qualified_name")(obj);
    if (auto classType = ClassType::get(c10::QualifiedName(qualifiedName))) {
      return std::make_shared<ClassValue>(classType);
    }
  }

  return std::make_shared<PythonValue>(obj);
}

} // namespace script
} // namespace jit
} // namespace torch
