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

struct VISIBILITY_HIDDEN PythonResolver : public Resolver {
  PythonResolver(ResolutionCallback rcb) : rcb(rcb) {}

  Node* resolveCall(SourceRange location, Node* n) const override {
    AutoGIL ag;
    py::function func;
    func = rcb(n->kind().toString());
    auto* py_func = func.ptr();
    if (py_func == Py_None) {
      throw ErrorReport(location)
          << "Unknown function " << n->kind().toString();
    }
    // Release the function object so we can wrap it in a PythonOp
    auto fn_ptr = THPObjectPtr(func.release().ptr());
    std::string cconv(n->inputs().size(), 't');
    Node* new_node = n->owningGraph()->createPythonOp(
        std::move(fn_ptr), cconv, false, {}, {}, false);
    return new_node;
  }

  ResolutionCallback rcb;
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
            return defineMethodsInModule(self, script, PythonResolver(rcb));
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
    return defineFunction(def, PythonResolver(rcb));
  });
}

} // namespace script
} // namespace jit
} // namespace torch
