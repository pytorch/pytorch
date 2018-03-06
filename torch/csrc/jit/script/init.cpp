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


struct PythonValue : public SugaredValue {
  PythonValue(py::object self)
  : self(std::move(self)) {}

  // call it like a function, e.g. `outputs = this(inputs)`
  virtual std::vector<Value*> call(SourceRange loc, Graph& g, at::ArrayRef<Value*> inputs, size_t n_outputs) override {
    // Release the function object so we can wrap it in a PythonOp
    py::object func = self;
    std::string cconv(inputs.size(), 't');
    Node* new_node = g.insertNode(g.createPythonOp(
      THPObjectPtr(func.release().ptr()), cconv, false, {}, {}, false));
    for(auto i : inputs)
      new_node->addInput(i);
    std::vector<Value*> outputs;
    for(size_t i = 0; i < n_outputs; ++i)
      outputs.push_back(new_node->addOutput());
    return outputs;
  }

  virtual std::string kind() const override {
    return py::repr(self);
  }
private:
  py::object self;
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
  if(outputs.size() == 1) {
    return py::cast(static_cast<autograd::Variable&>(outputs[0]));
  } else {
    py::tuple tuple(outputs.size());
    for(size_t i = 0; i < outputs.size(); i++) {
      tuple[i] = py::cast(static_cast<autograd::Variable&>(outputs[i]));
    }
    return tuple;
  }
}

void initJitScriptBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  py::class_<Module, std::shared_ptr<Module>>(m, "ScriptModule")
      .def(py::init<>())
      .def(
          "define",
          [](Module& self,
             const std::string& script,
             ResolutionCallback rcb) {
            return defineMethodsInModule(self, script, pythonResolver(rcb));
          })
      .def("get_method",
      [](Module& self, const std::string& name) {
        return self.get_method(name);
      }, py::return_value_policy::reference_internal);

  py::class_<Method>(m, "ScriptMethod")
    .def("graph", [&](Method& self) {
      return self.graph();
    })
    .def("__call__", [](Method& m, py::args args) -> py::object {
      auto inputs = createVariableTensorList(args);
      auto outputs = m.run(std::move(inputs));
      return unpackVariableTensorList(std::move(outputs));
    });

  m.def("_jit_script_compile", [](Def def, ResolutionCallback rcb) {
    return defineFunction(def, pythonResolver(rcb));
  });
}

} // namespace script
} // namespace jit
} // namespace torch
