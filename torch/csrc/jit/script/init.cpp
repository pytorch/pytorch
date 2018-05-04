#include "torch/csrc/jit/script/init.h"
#include "torch/csrc/jit/script/compiler.h"

namespace torch {
namespace jit {
namespace script {

using ResolutionCallback = std::function<py::function(std::string)>;

// The visibility attribute is to avoid a warning about storing a field in the
// struct that has a different visibility (from pybind) than the struct.
#ifdef _WIN32
#define VISIBILITY_HIDDEN
#else
#define VISIBILITY_HIDDEN __attribute__((visibility("hidden")))
#endif

static std::string typeString(py::handle h) {
  return py::str(h.get_type().attr("__name__"));
}

struct VISIBILITY_HIDDEN PythonValue : public SugaredValue {
  PythonValue(py::object self)
  : self(std::move(self)) {}

  std::pair<std::vector<TypePtr>, TypePtr> getFunctionType(size_t n_args, size_t n_binders) {
    auto annotations = py::module::import("torch.jit.annotations");
    return py::cast<std::pair<std::vector<TypePtr>, TypePtr>>(annotations.attr("get_signature")(self, n_args, n_binders));
  }

  // call it like a function, e.g. `outputs = this(inputs)`
  virtual std::shared_ptr<SugaredValue> call(SourceRange loc, Method & m, at::ArrayRef<Value*> inputs, at::ArrayRef<NamedValue> attributes, size_t n_binders) override {
    std::vector<TypePtr> arg_types;
    TypePtr ret_type;
    std::tie(arg_types, ret_type) = getFunctionType(inputs.size(), n_binders);

    if (arg_types.size() != inputs.size())
      throw ErrorReport(loc) << "calling a Python function with an incorrect number "
                             << "of arguments: expected " << arg_types.size() << ", but got "
                             << inputs.size();
    for (size_t i = 0; i < arg_types.size(); ++i) {
      if (!inputs[i]->type()->isSubtypeOf(*arg_types[i]))
        throw ErrorReport(loc) << "type mismatch at argument " << i << ": expected "
                               << arg_types[i]->name() << ", but got " << inputs[i]->type()->name();
    }
    // We have to do this check here, because implementation of this function is tightly
    // coupled with the impl for PythonOp in the interpreter. Right now it assumes that
    // all inputs taken from the stack are Tensors, so that's what we have to do.
    ensureTensors(loc, inputs);

    if (attributes.size() > 0)
      throw ErrorReport(loc) << "keyword arguments in Python calls aren't supported";
    Graph& g = *m.graph();

    // this python object might be a @trace or @script stand-alone function
    // if so, inline the graph rather than calling the python
    if(py::isinstance<GraphExecutor>(self)) {
      GraphExecutor& ge = py::cast<GraphExecutor&>(self);
      ensureSizeMatches(loc, ge.graph()->inputs().size(), inputs.size(), "arguments");
      return packOutputs(*m.graph(),inlineCallTo(*m.graph(), *ge.graph(), inputs));
    }

    // Release the function object so we can wrap it in a PythonOp
    py::object func = self;
    std::string cconv(inputs.size(), 't');
    Node* new_node = g.insertNode(g.createPythonOp(
      THPObjectPtr(func.release().ptr()), cconv, {}, {}, false));
    new_node->setSourceLocation(std::make_shared<SourceRange>(loc));
    for(auto i : inputs)
      new_node->addInput(i);

    // This is really dumb, but relaxing the constraints on return types would
    // require us to change the implementation of PythonOps in the interpreter.
    // Note that this effectively makes the return type of Tuple[Tensor] and Tensor
    // equivalent, but the PythonOp impl ends with an optional tuple unpack, so we need
    // to do it.
    std::shared_ptr<TupleType> ret_tuple_type;
    if (ret_type->kind() != TypeKind::TupleType) {
      ret_tuple_type = std::make_shared<TupleType>(std::vector<TypePtr>{ret_type});
    } else {
      ret_tuple_type = std::static_pointer_cast<TupleType>(ret_type);
    }
    for (auto & ret_type_elem : ret_tuple_type->elements()) {
      if (!ret_type_elem->isSubtypeOf(*DynamicType::get())) {
        throw ErrorReport(loc) << "Python functions can currently only return Tensors";
      }
    }

    std::vector<Value*> outputs;
    for(size_t i = 0; i < ret_tuple_type->elements().size(); ++i)
      outputs.push_back(new_node->addOutput());
    return packOutputs(*m.graph(), outputs);
  }

  virtual std::shared_ptr<SugaredValue> attr(SourceRange loc, Method & m, const std::string& field) override {
    // We generally don't want to allow traversing arbitrary Python objects, but we
    // make an exception for traversing modules because we want to be access
    // torch, torch.nn.functional, and the functions they expose.
    py::object member = getattr(loc, field);
    if (isBuiltinModule() && py::isinstance<py::function>(member)) {
      return std::make_shared<BuiltinFunction>(field);
    }
    if (py::isinstance<py::module>(self) && py::isinstance<py::module>(member)) {
      return std::make_shared<PythonValue>(member);
    }
    throw ErrorReport(loc) << "unsupported attribute lookup on " << py::repr(self) << ".";
  }

  virtual std::string kind() const override {
    std::stringstream ss;
    ss << "python value of type '" << typeString(self) << "'";
    return ss.str();
  }

protected:
  bool isBuiltinModule() {
    // XXX: these can't be static, or they will be destructed after the Python interpreter
    // exits and that generally sounds like a bad idea
    py::object torch = py::module::import("torch");
    py::object functional = py::module::import("torch.nn.functional");
    return self.is(torch) || self.is(functional);
  }

  py::object getattr(SourceRange loc, const std::string& name) {
    try {
      return py::getattr(self, name.c_str());
    } catch (py::error_already_set& e) {
      throw ErrorReport(loc) << "object has no attribute " << name;
    }
  }

  py::object self;
};

// by using torch.jit.Const, a user can mark a python value constant
// we then make that value immutable.
// once marked constant, we enable additional behavior such as
// 1. conversion via asValue to a constant Tensor
// 2. unrolling of for loops
struct VISIBILITY_HIDDEN ConstantPythonValue : public PythonValue {
  using PythonValue::PythonValue;
  virtual Value * asValue(SourceRange loc, Method & m) override {

    return PythonValue::asValue(loc, m);
  }
  virtual std::vector<std::shared_ptr<SugaredValue>> asTuple(SourceRange loc, Method& m) override {
    if(!py::isinstance<py::tuple>(self))
      return PythonValue::asTuple(loc, m);

    py::tuple tup = self;
    std::vector<std::shared_ptr<SugaredValue>> result;
    for(size_t i = 0; i < tup.size(); ++i) {
      result.push_back(create(loc, m, tup[i]));
    }
    return result;
  }
  static std::shared_ptr<SugaredValue> create(SourceRange loc, Method& m, py::object self) {
    // directly create SimpleValues when possible, because they are first-class
    // and can be re-assigned. Otherwise, this would be invalid:
    // f = python_constant
    // while ...
    //   f = f + 1
    if(py::isinstance<py::int_>(self)) {
      return createConstant(loc, m, at::CPU(at::kLong).scalarTensor(py::cast<int64_t>(self)));
    } else if(py::isinstance<py::float_>(self)) {
      return createConstant(loc, m, at::CPU(at::kFloat).scalarTensor(py::cast<float>(self)));
    } else if(py::isinstance<py::bool_>(self)) {
      return createConstant(loc, m, at::CPU(at::kByte).scalarTensor(py::cast<bool>(self)));
    }
    return std::make_shared<ConstantPythonValue>(self);
  }
private:
  static std::shared_ptr<SugaredValue> createConstant(SourceRange loc, Method& m, const at::Tensor& val) {
    auto n = m.graph()->createConstant(val);
    n->setSourceLocation(std::make_shared<SourceRange>(loc));
    return std::make_shared<SimpleValue>(m.graph()->insertNode(n)->output());
  }
};

Resolver pythonResolver(ResolutionCallback rcb) {
  return [=](const std::string& name) -> std::shared_ptr<SugaredValue> {
      AutoGIL ag;
      py::object obj = rcb(name);
      if(obj.is(py::none())) {
        return nullptr;
      }
      return std::make_shared<PythonValue>(obj);
  };
}

// defines how modules/methods behave inside the script subset.
// for now this does not have any interaction with python.
// in the future, we will add the ability to resolve `self.foo` to python
// {functions, modules, contants} so this SugaredValue is defined here
// anticipating we will eventually need to replace Module with a py::object
// holding the actual nn.Module class.

// defines how a method obtained from a module behaves in script
struct MethodValue : public SugaredValue {
  MethodValue(std::shared_ptr<Module> module, Method& method)
  : module(std::move(module)) //insurance that method stays alive
  , method(method) {}
  std::string kind() const override {
    return "method";
  }
  virtual std::shared_ptr<SugaredValue> call(SourceRange loc, Method & caller, at::ArrayRef<Value*> inputs, at::ArrayRef<NamedValue> attributes, size_t n_binders) override {
    if(attributes.size() != 0) {
      throw ErrorReport(loc) << "not yet implemented - calls to script methods using keyword arguments";
    }
    return packOutputs(*caller.graph(), caller.emit_call_to(loc, method, inputs));
  }
private:
  std::shared_ptr<Module> module;
  Method& method;

};


struct ModuleValue : public SugaredValue {
  ModuleValue(std::shared_ptr<Module> module)
  : module(std::move(module)) {}

  virtual std::string kind() const override {
    return "module";
  }

  // select an attribute on it, e.g. `this.field`
  virtual std::shared_ptr<SugaredValue> attr(SourceRange loc, Method & m, const std::string& field) override {
    if(at::optional<NamedModule&> v = module->find_module(field)) {
      return std::make_shared<ModuleValue>(v->module);
    } else if(at::optional<Method&> v = module->find_method(field)) {
      return std::make_shared<MethodValue>(module, *v);
    } else if(at::optional<NamedParameter&> v = module->find_parameter(field)) {
      return std::make_shared<SimpleValue>(m.get_or_add_parameter(v->slot()));
    }
    // This can also be a call to a non-script module, or a plain
    // python method. If so return this as a python value.
    py::object py_module = py::cast(module);
    if(py::object attr = py::getattr(py_module, field.c_str(), py::none())) {
      if(py::isinstance<py::function>(attr) ||
         py::isinstance(attr, py::module::import("torch.nn").attr("Module"))) {
        return std::make_shared<PythonValue>(attr);
      } else if(py_module.attr("_constants_set").contains(field.c_str())) {
        return ConstantPythonValue::create(loc, m, attr);
      } else {
        throw ErrorReport(loc) << "attribute '" << field << "' of type '" << typeString(attr) << "' is not usable in a script method (did you forget to add it __constants__?)";
      }
    }
    throw ErrorReport(loc) << "module has no attribute '" << field << "'";
  }
  // call module.forward
  virtual std::shared_ptr<SugaredValue> call(SourceRange loc, Method & caller, at::ArrayRef<Value*> inputs, at::ArrayRef<NamedValue> attributes, size_t n_binders) override {
    return attr(loc, caller, "forward")->call(loc, caller, inputs, attributes, n_binders);
  }

  virtual std::vector<std::shared_ptr<SugaredValue>> asTuple(SourceRange loc, Method& m) override {
    py::object py_module = py::cast(module);
    if(!py::isinstance(py_module, py::module::import("torch.jit").attr("_ConstModuleList")))
      return SugaredValue::asTuple(loc, m);
    std::vector<std::shared_ptr<SugaredValue>> result;
    for(py::handle module : py_module) {
      py::object obj = py::reinterpret_borrow<py::object>(module);
      if(py::isinstance<Module>(obj)) {
        auto r = py::cast<std::shared_ptr<Module>>(obj);
        result.push_back(std::make_shared<ModuleValue>(r));
      } else {
        result.push_back(ConstantPythonValue::create(loc, m, obj));
      }
    }
    return result;
  }

private:
  std::shared_ptr<Module> module;
};


// TODO: dedup with other init

// we cannot use the default py:cast<autograd::Variable> because it currently
// unwraps the data tensor in the conversion process

variable_tensor_list createVariableTensorList(py::tuple tuple, size_t reserve_extra_space = 0) {
  variable_tensor_list result;
  result.reserve(tuple.size() + reserve_extra_space);
  for(auto e : tuple) {
    result.push_back(py::cast<autograd::Variable>(e));
  }
  return result;
}

py::object unpackVariableTensorList(std::vector<at::Tensor> outputs) {
  // if we don't tell pybind these are variables it chokes on the
  // conversion.
  // TODO: fix conversions to be sane and make sure this works.
  if (outputs.size() == 0) {
    return py::none();
  } else if (outputs.size() == 1) {
    return py::cast(static_cast<autograd::Variable&>(outputs[0]));
  } else {
    py::tuple tuple(outputs.size());
    for(size_t i = 0; i < outputs.size(); i++) {
      tuple[i] = py::cast(static_cast<autograd::Variable&>(outputs[i]));
    }
    return tuple;
  }
}

static void gatherParametersAndBuffers(std::vector<at::Tensor*> & values, const Module & m) {
  for(auto & params : m.get_parameters()) {
    values.push_back(params.slot());
  }
  for(const auto & sub : m.get_modules()) {
    gatherParametersAndBuffers(values, *sub.module);
  }
}

void initJitScriptBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  // torch.jit.ScriptModule is a subclass of this C++ object.
  // Methods here are prefixed with _ since they should not be
  // public.
  py::class_<Module, std::shared_ptr<Module>>(m, "ScriptModule")
      .def(py::init<>())
      .def("_set_optimized", &Module::set_optimized)
      .def(
          "_define",
          [](Module& m,
             const std::string& script,
             ResolutionCallback rcb, bool has_self) {
            auto self = has_self ? std::make_shared<ModuleValue>(m.shared_from_this()) : nullptr;
            return defineMethodsInModule(m, script, pythonResolver(rcb), self);
          })
      .def("_create_methods", [](Module& m, const std::vector<Def>& defs, const std::vector<ResolutionCallback>& rcbs) {
        std::vector<Resolver> resolvers;
        for(auto & callback : rcbs) {
          resolvers.push_back(pythonResolver(callback));
        }
        defineMethodsInModule(
          m,
          defs,
          resolvers,
          std::make_shared<ModuleValue>(m.shared_from_this()));
      })
      .def("_get_method",
      [](Module& self, const std::string& name) -> const Method& {
        return self.get_method(name);
      }, py::return_value_policy::reference_internal)
      .def("_register_parameter", &Module::register_parameter)
      .def("_register_module", &Module::register_module)
      .def("_set_parameter", &Module::set_parameter)
      .def("_get_parameter", &Module::get_parameter)
      .def("_get_module", &Module::get_module)
      .def("_get_modules", [](Module& self) -> py::tuple {
        auto & modules = self.get_modules();
        py::tuple result(modules.size());
        for(size_t i = 0; i < modules.size(); ++i) {
          auto & nm = modules[i];
          result[i] = std::make_pair(nm.name, nm.module);
        }
        return result;
      })
      .def("_get_parameters", [](Module& self) -> py::tuple {
        auto & parameters = self.get_parameters();
        py::tuple result(parameters.size());
        for(size_t i = 0; i < parameters.size(); ++i) {
          auto & p = parameters[i];
          py::tuple r(3);
          result[i] = std::make_tuple(
            p.name,
            static_cast<const autograd::Variable&>(*p.slot()),
            p.is_buffer);

        }
        return result;
      })
      .def("_has_parameter", [](Module& self, const std::string& name) {
        if(auto r = self.find_parameter(name)) {
          return !r->is_buffer;
        }
        return false;
      })
      .def("_has_buffer", [](Module& self, const std::string& name) {
        if(auto r = self.find_parameter(name)) {
          return r->is_buffer;
        }
        return false;
      })
      .def("_has_module", [](Module& self, const std::string& name) {
        return bool(self.find_module(name));
      })
      .def("_has_method", [](Module& self, const std::string& name) {
        return bool(self.find_method(name));
      })
      .def("_method_names", [](Module& self) {
        return fmap(self.get_methods(), [](const std::unique_ptr<Method> & m) {
          return m->name();
        });
      })
      .def("_create_method_from_trace", [](
        Module& self,
        const std::string& name,
        py::function func,
        tracer::variable_list inputs) {
          size_t num_inputs = inputs.size();
          // prereq: Module's buffers and parameters are unique
          // this was ensured in python before calling this function
          std::vector<at::Tensor*> parameters;
          gatherParametersAndBuffers(parameters, self);
          for(at::Tensor* param : parameters) {
            inputs.push_back(static_cast<autograd::Variable&>(*param));
          }
          auto graph = tracer::createGraphByTracing(func, std::move(inputs), num_inputs);
          self.create_method(name, std::move(graph), std::move(parameters));
      });

  py::class_<Method>(m, "ScriptMethod")
    .def("graph", [&](Method& self) {
      return self.graph();
    })
    .def("__call__", [](Method& m, py::args args) -> py::object {
      auto inputs = createVariableTensorList(args);
      auto outputs = m.run(std::move(inputs));
      return unpackVariableTensorList(std::move(outputs));
    })
    .def_property_readonly("graph", [](Method& m) {
      return m.graph();
    })
    .def("propagate_shapes", &Method::propagate_shapes)
    .def("propagate_and_assign_input_and_output_shapes", &Method::propagate_and_assign_input_and_output_shapes)
    .def("params", &Method::params);

  m.def("_jit_script_compile", [](Def def, ResolutionCallback rcb) {
    return compileFunction(def, pythonResolver(rcb));
  });

}

} // namespace script
} // namespace jit
} // namespace torch
