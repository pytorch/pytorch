#include <torch/csrc/jit/script/init.h>

#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/jit/import.h>
#include <torch/csrc/jit/script/compiler.h>
#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/jit/script/schema_matching.h>
#include <torch/csrc/jit/script/sugared_value.h>
#include <torch/csrc/jit/testing/file_check.h>

#include <torch/csrc/jit/constants.h>
#include <torch/csrc/jit/hooks_for_testing.h>
#include <torch/csrc/jit/import_source.h>
#include <torch/csrc/jit/irparser.h>
#include <torch/csrc/jit/passes/python_print.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/jit/python_tracer.h>
#include <torch/csrc/jit/script/logging.h>
#include <torch/csrc/jit/script/parser.h>
#include <torch/csrc/jit/tracer.h>

#include <torch/csrc/api/include/torch/ordered_dict.h>

#include <ATen/ATen.h>
#include <ATen/core/function_schema.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <chrono>
#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

PYBIND11_MAKE_OPAQUE(torch::jit::script::ExtraFilesMap);

namespace torch {
namespace jit {
namespace script {

using ::c10::Argument;
using ::c10::FunctionSchema;

using ResolutionCallback = std::function<py::function(std::string)>;
using FunctionDefaults = std::unordered_map<std::string, py::object>;

static std::string typeString(py::handle h) {
  return py::str(h.get_type().attr("__name__"));
}

inline std::shared_ptr<SugaredValue> toSimple(Value* v) {
  return std::make_shared<SimpleValue>(v);
}

// NB: This should be the single entry-point for instantiating a SugaredValue
// from a Python object. If you are adding support for converting a new Python
// type, *add it in this function's implementation*.
std::shared_ptr<SugaredValue> toSugaredValue(
    py::object obj,
    Function& m,
    SourceRange loc,
    bool is_constant = false);

struct VISIBILITY_HIDDEN PythonValue : public SugaredValue {
  PythonValue(py::object self) : self(std::move(self)) {}

  FunctionSchema getSchema(const size_t n_args, const size_t n_binders) {
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
        args.push_back(Argument(
            std::to_string(idx++), std::move(arg_type), {}, {}, false));
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

  // call it like a function, e.g. `outputs = this(inputs)`
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& m,
      at::ArrayRef<NamedValue> inputs_,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override {
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
    Node* new_node = m.graph()->insertNode(m.graph()->createPythonOp(
        THPObjectPtr(func.release().ptr()), cconv, {}));

    // Mark if function is ignored on export
    if (py::cast<bool>(py::module::import("torch.jit")
                           .attr("_try_get_ignored_op")(self))) {
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

  std::string kind() const override {
    std::stringstream ss;
    ss << "python value of type '" << typeString(self) << "'";
    return ss.str();
  }

  void checkForAddToConstantsError(std::stringstream& ss) {
    auto nn = py::module::import("torch.nn");
    if (py::isinstance(self, nn.attr("ModuleList")) ||
        py::isinstance(self, nn.attr("Sequential"))) {
      ss << ". Did you forget to add it to __constants__? ";
    }
  }

  std::vector<std::shared_ptr<SugaredValue>> asTuple(
      const SourceRange& loc,
      Function& m,
      const c10::optional<size_t>& size_hint = {}) override {
    const std::string type_str = typeString(self);
    std::stringstream ss;
    ss << kind() << " cannot be used as a tuple";
    checkForAddToConstantsError(ss);
    throw ErrorReport(loc) << ss.str();
  }

  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      Function& m,
      const std::string& field) override {
    const std::string type_str = typeString(self);
    std::stringstream ss;
    ss << "attribute lookup is not defined on " << kind();
    checkForAddToConstantsError(ss);
    throw ErrorReport(loc) << ss.str();
  }

 protected:
  py::object getattr(const SourceRange& loc, const std::string& name) {
    try {
      return py::getattr(self, name.c_str());
    } catch (py::error_already_set& e) {
      throw ErrorReport(loc) << "object has no attribute " << name;
    }
  }

  py::object self;
};

struct VISIBILITY_HIDDEN PythonModuleValue : public PythonValue {
  explicit PythonModuleValue(py::object mod) : PythonValue(std::move(mod)) {}

  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      Function& m,
      const std::string& field) override {
    py::object member = getattr(loc, field);
    // note: is_constant = true because we consider that global properties
    // on modules like math.pi or torch.float to be constants
    // eventhough it is possible, though rare, for someone to mutate them
    return toSugaredValue(member, m, loc, /*is_constant=*/true);
  }
};

struct VISIBILITY_HIDDEN ConstantPythonTupleValue : public PythonValue {
  explicit ConstantPythonTupleValue(py::object tup)
      : PythonValue(std::move(tup)) {}
  std::vector<std::shared_ptr<SugaredValue>> asTuple(
      const SourceRange& loc,
      Function& m,
      const c10::optional<size_t>& size_hint = {}) override {
    py::tuple tup = self;
    std::vector<std::shared_ptr<SugaredValue>> result;
    result.reserve(tup.size());
    for (py::handle t : tup) {
      py::object obj = py::reinterpret_borrow<py::object>(t);
      result.push_back(toSugaredValue(obj, m, loc, true));
    }
    return result;
  }

  Value* asValue(const SourceRange& loc, Function& m) override {
    std::vector<Value*> values;
    for (const auto& sugared_item : asTuple(loc, m)) {
      values.push_back(sugared_item->asValue(loc, m));
    }
    auto node = m.graph()->createTuple(values);
    return m.graph()->insertNode(node)->output();
  }
};

// Represents all the parameters of a module as a List[Tensor]
struct VISIBILITY_HIDDEN ConstantParameterList : public SugaredValue {
  ConstantParameterList(Value* the_list) : the_list_(the_list) {}
  std::string kind() const override {
    return "constant parameter list";
  }
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& caller,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override {
    return toSimple(the_list_);
  }

 private:
  Value* the_list_;
};

struct VISIBILITY_HIDDEN OverloadedFunctionValue : public SugaredValue {
  OverloadedFunctionValue(Value* module, std::vector<std::string> method_names)
      : module_(module), method_names_(std::move(method_names)) {}

  std::string kind() const override {
    return "overloaded function";
  }

  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& caller,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override {
    std::stringstream err;
    std::vector<NamedValue> new_inputs = inputs.vec();
    new_inputs.insert(new_inputs.begin(), module_);

    for (const std::string& method_name : method_names_) {
      auto cls = module_->type()->expect<ClassType>();
      Function* fn = cls->getMethod(method_name);
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
        return MethodValue(module_, *fn)
            .call(loc, caller, inputs, attributes, n_binders);
      }
    }
    throw ErrorReport(loc) << "Could not find any matching overloads\n"
                           << err.str();
  }

 private:
  Value* module_;
  std::vector<std::string> method_names_;
};

// defines how modules/methods behave inside the script subset.
// for now this does not have any interaction with python.
// in the future, we will add the ability to resolve `self.foo` to python
// {functions, modules, contants} so this SugaredValue is defined here
// anticipating we will eventually need to replace Module with a py::object
// holding the actual nn.Module class.

struct ModuleValue : public SugaredValue {
  ModuleValue(Value* self, std::shared_ptr<Module> module)
      : self_(self), module_(std::move(module)) {}

  std::string kind() const override {
    return "module";
  }

  // select an attribute on it, e.g. `this.field`
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      Function& m,
      const std::string& field) override {
    // workaround to make self.training work
    // it adds a buffer 'training' to the model if one doesn't exist
    // and then loads that parameter, casting it to bool
    if (field == "training") {
      Slot* v = module_->find_buffer(field);
      if (!v) {
        py::object py_module = py::cast(module_);
        bool training = py::cast<bool>(py::getattr(py_module, "training"));
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
          m.graph()->insertGetAttr(self_, field), v);
    } else if (auto kind = module_->kind_of(field)) {
      // methods, parameters, attributes, and buffers are all first class
      return SimpleValue(self_).attr(loc, m, field);
    }

    // This can also be a call to a non-script module, or a plain
    // python method. If so return this as a python value.
    py::object py_module = py::cast(module_);

    py::object overloads =
        py_module.attr("_overloads").attr("get")(field, py::none());
    if (!overloads.is_none()) {
      return std::make_shared<OverloadedFunctionValue>(
          self_, py::cast<std::vector<std::string>>(overloads));
    }

    if (py::object attr = py::getattr(py_module, field.c_str(), py::none())) {
      if (py::isinstance<py::function>(attr) &&
          py::hasattr(attr, "_is_parameter_list") &&
          py::cast<bool>(py::getattr(attr, "_is_parameter_list"))) {
        Graph& g = *m.graph();
        // Add all module parameters as inputs to the graph
        std::vector<Value*> params;
        const auto& param_list = module_->get_parameters();
        for (auto it = param_list.rbegin(); it != param_list.rend(); ++it) {
          auto& param = *it;
          params.emplace_back(g.insertGetAttr(self_, param.name()));
        }
        auto list =
            g.insertNode(g.createList(TensorType::get(), params))->output();
        return std::make_shared<ConstantParameterList>(list);
      }
      if (py::isinstance<py::function>(attr) ||
          py::isinstance(attr, py::module::import("torch.nn").attr("Module")) ||
          py_module.attr("_constants_set").contains(field.c_str())) {
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

  // call module.forward
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& caller,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override {
    return attr(loc, caller, "forward")
        ->call(loc, caller, inputs, attributes, n_binders);
  }

  std::vector<std::shared_ptr<SugaredValue>> asTuple(
      const SourceRange& loc,
      Function& m,
      const c10::optional<size_t>& size_hint = {}) override {
    py::object py_module = py::cast(module_);
    if (!py::isinstance(
            py_module,
            py::module::import("torch.jit").attr("_ConstModuleList")))
      return SugaredValue::asTuple(loc, m, size_hint);
    std::vector<std::shared_ptr<SugaredValue>> result;
    for (py::handle py_submodule : py_module) {
      py::object obj = py::reinterpret_borrow<py::object>(py_submodule);
      if (py::isinstance<Module>(obj)) {
        auto sub_module = py::cast<std::shared_ptr<Module>>(obj);
        Value* module_v = m.graph()->insertGetAttr(self_, sub_module->name());
        result.emplace_back(
            std::make_shared<ModuleValue>(module_v, sub_module));
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

 private:
  Value* self_;
  std::shared_ptr<Module> module_;
};

struct VISIBILITY_HIDDEN BooleanDispatchValue : public SugaredValue {
  BooleanDispatchValue(py::dict dispatched_fn)
      : dispatched_fn_(std::move(dispatched_fn)) {}

  std::string kind() const override {
    return "boolean dispatch";
  }

  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& caller,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override {
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

 private:
  py::dict dispatched_fn_;
};

std::shared_ptr<MethodValue> moduleToMethod(
    const std::shared_ptr<Module>& mod) {
  // this path only supports calling raw script functions
  // but because they are not distinguished from models, we have to check
  // that they are function-like here. They must not have state, and they
  // must have a forward method. When we expose functions to python
  //  this will be replaced with a direct py::isinstance<Function> call.

  if (mod->get_parameters().size() != 0) {
    throw ErrorReport()
        << "Attempted to inline a Module with parameters. "
           "Stateful modules to be inlined must be submodules of the callee.";
  }
  Method* forward = mod->find_method("forward");
  if (!forward) {
    throw ErrorReport() << " expected this module to have a forward function.";
  }
  return std::make_shared<MethodValue>(at::nullopt, forward->function());
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
  if (py::isinstance<Module>(obj)) {
    // In the case that this Python object is not a submodule, inline *ONLY
    // PURE* ScriptModules. This allows us to call arbitrary @script functions
    // within a scripting context while still enforcing that parameters from
    // stateful submodules are properly accounted for.
    auto mod = py::cast<std::shared_ptr<Module>>(obj);
    return moduleToMethod(mod);
  } else if (py::isinstance<py::module>(obj)) {
    return std::make_shared<PythonModuleValue>(obj);
  } else if (obj.ptr() == py::module::import("torch.jit").attr("_fork").ptr()) {
    return std::make_shared<ForkValue>();
  } else if (
      obj.ptr() == py::module::import("torch.jit").attr("annotate").ptr()) {
    return std::make_shared<AnnotateValue>();
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
    if (!compiled_fn.is(py::none())) {
      auto mod = py::cast<std::shared_ptr<Module>>(compiled_fn);
      return moduleToMethod(mod);
    }
  }

  py::object dispatched_fn =
      py::module::import("torch.jit").attr("_try_get_dispatched_fn")(obj);
  if (!dispatched_fn.is_none()) {
    return std::make_shared<BooleanDispatchValue>(std::move(dispatched_fn));
  }

  return std::make_shared<PythonValue>(obj);
}

py::object unpackVariableTensorList(std::vector<at::Tensor> outputs) {
  // if we don't tell pybind these are variables it chokes on the
  // conversion.
  // TODO: fix conversions to be sane and make sure this works.
  if (outputs.size() == 0) {
    return py::none();
  } else if (outputs.size() == 1) {
    return py::cast(autograd::as_variable_ref(outputs[0]));
  } else {
    py::tuple tuple(outputs.size());
    for (size_t i = 0; i < outputs.size(); i++) {
      tuple[i] = py::cast(autograd::as_variable_ref(outputs[i]));
    }
    return std::move(tuple);
  }
}

static void gatherParametersAndBuffers(
    std::vector<Slot>& values,
    const Module& m) {
  for (auto& param : m.get_parameters()) {
    values.push_back(param);
  }
  for (auto& param : m.get_attributes()) {
    if (param.type()->isSubtypeOf(TensorType::get())) {
      values.push_back(param);
    }
  }
  for (const auto& sub : m.get_modules()) {
    gatherParametersAndBuffers(values, *sub);
  }
}

namespace {

Resolver pythonResolver(const ResolutionCallback& rcb) {
  return [rcb](const std::string& name, Function& m, const SourceRange& loc)
             -> std::shared_ptr<SugaredValue> {
    AutoGIL ag;
    py::object obj = rcb(name);
    if (obj.is(py::none())) {
      return nullptr;
    }
    return toSugaredValue(obj, m, loc);
  };
}
} // namespace

FunctionSchema getSchemaWithNameAndDefaults(
    const SourceRange& range,
    const FunctionSchema& schema,
    const at::optional<std::string>& new_name,
    const FunctionDefaults& default_args) {
  std::vector<Argument> new_args;
  for (auto& arg : schema.arguments()) {
    auto it = default_args.find(arg.name());
    if (it != default_args.end()) {
      try {
        IValue value;
        auto n = arg.N();
        auto list_type = arg.type()->cast<ListType>();
        if (n && *n > 0 && list_type) {
          // BroadcastingList, allow default values T for arg types List[T]
          value = toIValue(it->second, list_type->getElementType());
        } else {
          value = toIValue(it->second, arg.type());
        }
        new_args.emplace_back(
            arg.name(), arg.type(), arg.N(), value, arg.kwarg_only());
      } catch (py::cast_error& e) {
        throw ErrorReport(range)
            << "Expected a default value of type " << arg.type()->str()
            << " on parameter \"" << arg.name() << "\"";
      }
    } else {
      new_args.push_back(arg);
    }
  }

  return FunctionSchema(
      new_name.value_or(schema.name()),
      schema.overload_name(),
      new_args,
      schema.returns(),
      schema.is_vararg(),
      schema.is_varret());
}

static Self moduleSelf(const std::shared_ptr<Module>& m) {
  return [m](Value* v) {
    v->setType(m->module_object()->type());
    return std::make_shared<ModuleValue>(v, m);
  };
}
static Self simpleSelf(const TypePtr& typ) {
  return [typ](Value* v) {
    v->setType(typ);
    return std::make_shared<SimpleValue>(v);
  };
}

void initJitScriptBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // STL containers are not mutable by default and hence we need to bind as
  // follows.
  py::bind_map<ExtraFilesMap>(m, "ExtraFilesMap");

  // torch.jit.ScriptModule is a subclass of this C++ object.
  // Methods here are prefixed with _ since they should not be
  // public.
  py::class_<Module, std::shared_ptr<Module>>(m, "ScriptModule")
      .def(py::init<>())
      .def(
          "save",
          [](std::shared_ptr<Module> m,
             const std::string& filename,
             const ExtraFilesMap& _extra_files = ExtraFilesMap()) {
            m->save(filename, _extra_files);
          },
          py::arg("filename"),
          py::arg("_extra_files") = ExtraFilesMap())
      .def(
          "save_to_buffer",
          [](std::shared_ptr<Module> m,
             const ExtraFilesMap& _extra_files = ExtraFilesMap()) {
            std::ostringstream buf;
            m->save(buf, _extra_files);
            return py::bytes(buf.str());
          },
          py::arg("_extra_files") = ExtraFilesMap())
      .def("_set_optimized", &Module::set_optimized)
      .def(
          "_define",
          [](std::shared_ptr<Module> m,
             const std::string& script,
             ResolutionCallback rcb,
             bool has_self) {
            c10::optional<Self> self;
            if (has_self) {
              m->class_compilation_unit().define(
                  script, pythonResolver(rcb), moduleSelf(m));
            } else {
              m->_define_lowered(script, pythonResolver(rcb));
            }
            didFinishEmitModule(m);
          })
      .def(
          "_create_methods",
          [](std::shared_ptr<Module> m,
             const std::vector<Def>& defs,
             const std::vector<ResolutionCallback>& rcbs,
             const std::vector<FunctionDefaults>& defaults) {
            std::vector<Resolver> resolvers;
            resolvers.reserve(rcbs.size());
            for (auto& callback : rcbs) {
              resolvers.push_back(pythonResolver(callback));
            }
            m->class_compilation_unit().define(defs, resolvers, moduleSelf(m));
            // Stitch in default arguments for each Def if provided
            auto defaults_it = defaults.begin();
            auto defs_it = defs.begin();
            while (defs_it != defs.end()) {
              auto& method = m->class_compilation_unit().get_function(
                  (*defs_it).name().name());
              method.setSchema(getSchemaWithNameAndDefaults(
                  defs_it->range(),
                  method.getSchema(),
                  at::nullopt,
                  *defaults_it));
              ++defs_it;
              ++defaults_it;
            }
            didFinishEmitModule(m);
          })
      .def(
          "_get_method",
          [](Module& self, const std::string& name) -> const Method& {
            return self.get_method(name);
          },
          py::return_value_policy::reference_internal)
      .def("_register_parameter", &Module::register_parameter)
      .def(
          "_register_attribute",
          [](Module& self, std::string name, TypePtr type, py::object value) {
            self.register_attribute(name, type, toIValue(value, type));
          })
      .def("_register_module", &Module::register_module)
      .def("_register_buffer", &Module::register_buffer)
      .def("_set_parameter", &Module::set_parameter)
      .def("_get_parameter", &Module::get_parameter)
      .def("_get_buffer", &Module::get_buffer)
      .def("_get_attribute", &Module::get_attribute)
      .def("_get_module", &Module::get_module)
      .def(
          "_get_modules",
          [](Module& self) -> py::tuple {
            auto modules = self.get_modules();
            py::tuple result(modules.size());
            for (size_t i = 0; i < modules.size(); ++i) {
              auto& item = modules[i];
              result[i] = std::make_pair(item->name(), item);
            }
            return result;
          })
      .def(
          "_get_parameters",
          [](Module& self) -> py::tuple {
            auto parameters = self.get_parameters();
            py::tuple result(parameters.size());
            for (size_t i = 0; i < parameters.size(); ++i) {
              auto& p = parameters[i];
              py::tuple r(2);
              result[i] = std::make_tuple(
                  p.name(), autograd::as_variable_ref(p.value().toTensor()));
            }
            return result;
          })
      .def(
          "_get_attributes",
          [](Module& self) -> py::tuple {
            auto attributes = self.get_attributes();
            py::tuple result(attributes.size());
            for (size_t i = 0; i < attributes.size(); ++i) {
              auto& buffer = attributes[i];
              py::tuple r(3);
              IValue v = buffer.value();
              result[i] = std::make_tuple(
                  buffer.name(), buffer.type(), toPyObject(std::move(v)));
            }
            return result;
          })
      .def(
          "_has_attribute",
          [](Module& self, const std::string& name) -> bool {
            return self.find_attribute(name);
          })
      .def(
          "_has_parameter",
          [](Module& self, const std::string& name) -> bool {
            return self.find_parameter(name);
          })
      .def(
          "_has_buffer",
          [](Module& self, const std::string& name) -> bool {
            return self.find_buffer(name);
          })
      .def(
          "_has_module",
          [](Module& self, const std::string& name) {
            return bool(self.find_module(name));
          })
      .def(
          "_has_method",
          [](Module& self, const std::string& name) {
            return bool(self.find_method(name));
          })
      .def(
          "_method_names",
          [](Module& self) {
            return fmap(
                self.get_methods(), [](const std::unique_ptr<Method>& method) {
                  return method->name();
                });
          })
      .def(
          "_create_method_from_graph",
          [](Module& self,
             const std::string& name,
             std::shared_ptr<Graph> graph) {
            self._define_lowered(name, std::move(graph), {});
          })
      .def(
          "_create_method_from_trace",
          [](std::shared_ptr<Module> self,
             const std::string& name,
             py::function func,
             py::tuple input_tuple,
             py::function var_lookup_fn,
             bool force_outplace) {
            // prereq: Module's buffers and parameters are unique
            // this was ensured in python before calling this function
            std::vector<Slot> parameters;
            gatherParametersAndBuffers(parameters, *self);
            Stack inputs = toStack(input_tuple);
            for (const Slot& param : parameters) {
              inputs.emplace_back(param.value());
            }
            auto graph = tracer::createGraphByTracing(
                func,
                inputs,
                var_lookup_fn,
                force_outplace,
                input_tuple.size());
            self->_define_lowered(
                name, std::move(graph), std::move(parameters));
            didFinishEmitModule(self);
          })
      .def(
          "graph_for",
          [](py::args args, py::kwargs kwargs) {
            // [pybind11 varargs] note: old version of pybind11 have a bug that
            // leaks memory when py::args is mixed with positional arguments
            // https://github.com/pybind/pybind11/pull/1216
            // we work around this by not mixing positional arguments with
            // varargs
            Module& self = py::cast<Module&>(args[0]);
            if (self.find_method("forward")) {
              Method& m = self.get_method("forward");
              return m.graph_for(createStackForSchema(
                  m.getSchema(), tuple_slice(std::move(args), 1), kwargs));
            }
            throw std::runtime_error(
                "Attempted to call graph_for on a Module without a compiled forward()");
          })
      .def(
          "get_debug_state",
          [](Module& self) {
            if (self.find_method("forward")) {
              Method& m = self.get_method("forward");
              return m.get_executor().getDebugState();
            }
            throw std::runtime_error(
                "Attempted to call get_debug_state on a Module without a compiled forward()");
          })
      .def(
          "debug_disable_autodiff_subgraph_inlining",
          [](Module& self) {
            if (self.find_method("forward")) {
              Method& m = self.get_method("forward");
              m.get_executor().debugDisableAutodiffSubgraphInlining();
            }
          })
      .def(
          "forward",
          [](py::args args, py::kwargs kwargs) {
            // We implement this in C++ to avoid incurring the pybind11 dispatch
            // overhead twice: once to call into the method lookup for "forward"
            // and once to actually invoke the method.
            //
            // There is a thin wrapper on top of this method in the C++ version
            // of ScriptModule.

            // see: [pybind11 varargs]
            Module& self = py::cast<Module&>(args[0]);
            return invokeScriptMethodFromPython(
                self.get_method("forward"),
                tuple_slice(std::move(args), 1),
                std::move(kwargs));
          })
      .def(
          "_python_print",
          [](Module& self) {
            std::ostringstream ss;
            std::vector<at::Tensor> tensors;
            std::vector<ClassTypePtr> classes;
            PythonPrint(ss, self, tensors, classes, true);
            return std::make_pair(ss.str(), tensors);
          })
      .def_property_readonly(
          "code",
          [](Module& self) {
            std::ostringstream ss;
            std::vector<at::Tensor> tensors;
            std::vector<ClassTypePtr> classes;
            PythonPrint(ss, self, tensors, classes, false);
            return ss.str();
          })
      .def("apply", &Module::apply)
      .def("_copy_into", &Module::copy_into)
      .def(
          "_copy_method",
          [](std::shared_ptr<Module> m,
             std::string name,
             std::vector<std::tuple<std::shared_ptr<Module>, std::string>>
                 params,
             std::shared_ptr<Module> orig) {
            std::vector<Slot> member_inputs;
            for (auto& p : params) {
              Slot* np = std::get<0>(p)->find_parameter(std::get<1>(p));
              if (np == nullptr) {
                np = std::get<0>(p)->find_buffer(std::get<1>(p));
              }
              AT_ASSERT(np != nullptr);
              member_inputs.push_back(*np);
            }

            Method* orig_method = orig->find_method(name);
            m->_define_lowered(
                name, orig_method->graph()->copy(), std::move(member_inputs));
          });

  py::class_<Method>(m, "ScriptMethod", py::dynamic_attr())
      .def("graph", [&](Method& self) { return self.graph(); })
      .def(
          "__call__",
          [](py::args args, py::kwargs kwargs) {
            // see: [pybind11 varargs]
            Method& method = py::cast<Method&>(args[0]);
            return invokeScriptMethodFromPython(
                method, tuple_slice(std::move(args), 1), std::move(kwargs));
          })
      .def_property_readonly("graph", [](Method& m) { return m.graph(); })
      .def(
          "propagate_shapes",
          [](Method& m, const std::vector<at::Tensor>& inputs, bool with_grad) {
            return propagate_shapes(
                *m.graph(), inputs, m.initial_ivalues(), with_grad);
          })
      .def(
          "propagate_and_assign_input_and_output_shapes",
          [](Method& m,
             const std::vector<at::Tensor>& inputs,
             std::vector<at::Tensor> outputs,
             bool with_grad,
             bool propagate) {
            return propagate_and_assign_input_and_output_shapes(
                *m.graph(),
                inputs,
                m.initial_ivalues(),
                outputs,
                with_grad,
                propagate);
          })
      .def(
          "initial_ivalues",
          [](Method& m) {
            std::vector<at::Tensor> tensors;
            for (auto& t : m.initial_ivalues()) {
              tensors.push_back(t.value().toTensor());
            }
            return tensors;
          })
      .def(
          "graph_for",
          [](py::args args, py::kwargs kwargs) {
            // see: [pybind11 varargs]
            Method& self = py::cast<Method&>(args[0]);
            return self.graph_for(createStackForSchema(
                self.getSchema(), tuple_slice(std::move(args), 1), kwargs));
          })
      .def(
          "debug_disable_autodiff_subgraph_inlining",
          [](Method& m) {
            return m.get_executor().debugDisableAutodiffSubgraphInlining();
          })
      .def("schema", &Method::getSchema)
      .def(
          "pretty_print_schema",
          [](Method& m) {
            const FunctionSchema& schema = m.getSchema();
            std::stringstream ss;
            ss << schema;
            return ss.str();
          })
      .def(
          "python_print",
          [](Method& m) {
            std::ostringstream oss;
            std::vector<at::Tensor> constants;
            std::vector<ClassTypePtr> classes;
            PythonPrint(oss, m, constants, classes, true);
            return std::make_pair(oss.str(), std::move(constants));
          })
      .def_property_readonly("code", [](Method& self) {
        std::ostringstream ss;
        std::vector<at::Tensor> tensors;
        std::vector<ClassTypePtr> classes;
        PythonPrint(ss, self, tensors, classes, false);
        return ss.str();
      });

  m.def(
      "_jit_script_compile",
      [](std::shared_ptr<Module> mod,
         const Def& def,
         ResolutionCallback rcb,
         FunctionDefaults defaults) {
        auto def_f = def.withName("forward");

        mod->_define_lowered({def_f}, {pythonResolver(rcb)});
        auto& func = mod->lowered_methods().get_function("forward");
        func.setSchema(getSchemaWithNameAndDefaults(
            def.range(), func.getSchema(), def.name().name(), defaults));
        auto& func2 = mod->class_compilation_unit().get_function("forward");
        func2.setSchema(getSchemaWithNameAndDefaults(
            def.range(), func2.getSchema(), def.name().name(), defaults));
        didFinishEmitModule(mod);
        return mod;
      });

  m.def(
      "_jit_script_class_compile",
      [](const ClassDef& classDef, ResolutionCallback rcb) {
        auto cu = std::make_shared<CompilationUnit>();
        auto classType = ClassType::create(classDef.name().name(), cu);
        std::vector<Resolver> rcbs;
        std::vector<Def> methodDefs;
        for (const auto& def : classDef.defs()) {
          methodDefs.push_back(def);
          rcbs.push_back(pythonResolver(rcb));
        }
        cu->define(methodDefs, rcbs, simpleSelf(classType));
      });

  m.def("parse_type_comment", [](const std::string& comment) {
    Parser p(comment);
    return Decl(p.parseTypeComment());
  });

  m.def("merge_type_from_type_comment", &mergeTypesFromTypeComment);
  m.def(
      "import_ir_module",
      [](ModuleLookup module_lookup,
         const std::string& filename,
         py::object map_location,
         ExtraFilesMap& extra_files) {
        c10::optional<at::Device> optional_device;
        if (!map_location.is(py::none())) {
          AT_ASSERT(THPDevice_Check(map_location.ptr()));
          optional_device =
              reinterpret_cast<THPDevice*>(map_location.ptr())->device;
        }
        import_ir_module(module_lookup, filename, optional_device, extra_files);
      });
  m.def(
      "import_ir_module_from_buffer",
      [](ModuleLookup module_lookup,
         const std::string& buffer,
         py::object map_location,
         ExtraFilesMap& extra_files) {
        std::istringstream in(buffer);
        c10::optional<at::Device> optional_device;
        if (!map_location.is(py::none())) {
          AT_ASSERT(THPDevice_Check(map_location.ptr()));
          optional_device =
              reinterpret_cast<THPDevice*>(map_location.ptr())->device;
        }
        import_ir_module(module_lookup, in, optional_device, extra_files);
      });
  m.def("_jit_import_methods", import_methods);
  m.def("_jit_set_emit_module_hook", setEmitModuleHook);
  m.def("_jit_clear_class_registry", ClassType::clearRegistry);

  py::class_<testing::FileCheck>(m, "FileCheck")
      .def(py::init<>())
      .def("check", &testing::FileCheck::check)
      .def("check_not", &testing::FileCheck::check_not)
      .def("check_same", &testing::FileCheck::check_same)
      .def("check_next", &testing::FileCheck::check_next)
      .def("check_count", &testing::FileCheck::check_count)
      .def("check_dag", &testing::FileCheck::check_dag)
      .def("check_count", &testing::FileCheck::check_count)
      .def(
          "check_count",
          [](testing::FileCheck& f,
             const std::string& str,
             size_t count,
             bool exactly) { return f.check_count(str, count, exactly); },
          "Check Count",
          py::arg("str"),
          py::arg("count"),
          py::arg("exactly") = false)
      .def(
          "run",
          [](testing::FileCheck& f, const std::string& str) {
            return f.run(str);
          })
      .def(
          "run", [](testing::FileCheck& f, const Graph& g) { return f.run(g); })
      .def(
          "run",
          [](testing::FileCheck& f,
             const std::string& input,
             const std::string& output) { return f.run(input, output); },
          "Run",
          py::arg("checks_file"),
          py::arg("test_file"))
      .def(
          "run",
          [](testing::FileCheck& f, const std::string& input, const Graph& g) {
            return f.run(input, g);
          },
          "Run",
          py::arg("checks_file"),
          py::arg("graph"));

  m.def(
      "_logging_set_logger",
      [](logging::LoggerBase* logger) { return logging::setLogger(logger); },
      py::return_value_policy::reference);
  py::class_<logging::LoggerBase, std::shared_ptr<logging::LoggerBase>>(
      m, "LoggerBase");
  py::enum_<logging::LockingLogger::AggregationType>(m, "AggregationType")
      .value("SUM", logging::LockingLogger::AggregationType::SUM)
      .value("AVG", logging::LockingLogger::AggregationType::AVG)
      .export_values();
  py::class_<
      logging::LockingLogger,
      logging::LoggerBase,
      std::shared_ptr<logging::LockingLogger>>(m, "LockingLogger")
      .def(py::init<>())
      .def("set_aggregation_type", &logging::LockingLogger::setAggregationType)
      .def("get_counter_val", &logging::LockingLogger::getCounterValue);
  py::class_<
      logging::NoopLogger,
      logging::LoggerBase,
      std::shared_ptr<logging::NoopLogger>>(m, "NoopLogger")
      .def(py::init<>());
}
} // namespace script
} // namespace jit
} // namespace torch
