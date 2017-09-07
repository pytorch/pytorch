#include <Python.h>

#include <pybind11/pybind11.h>
// DO NOT REMOVE, this enables std containers to be recognized
// with pybind11, removing the include disables the support
#include <pybind11/stl.h>
namespace py = pybind11;

#include "torch/csrc/jit/python_tracer.h"
#include "torch/csrc/jit/tracer.h"
#include "torch/csrc/jit/assert.h"
#include "torch/csrc/jit/export.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/THP.h"
#include "torch/csrc/DynamicTypes.h"

#include <sstream>

using namespace torch::autograd;
using namespace torch::jit;
using namespace torch::jit::tracer;

namespace pybind11 { namespace detail {
  template<> struct type_caster<TraceInput> {
  public:
    PYBIND11_TYPE_CASTER(TraceInput, _("torch::jit::tracer::TraceInput"));
    bool load(handle src, bool) {
      PyObject *source = src.ptr();
      if (THPVariable_Check(source)) {
        value = TraceInput(((THPVariable*)source)->cdata);
        return true;
      } else if (THPModule_isTensor(source)) {
        value = TraceInput(torch::createTensor(source));
        return true;
      } else {
        return false;
      }
    }
    static handle cast(TraceInput src, return_value_policy /* policy */, handle /* parent */) {
      if (src.variable.defined()) {
        return handle(THPVariable_Wrap(src.variable));
      } else {
        return handle(torch::createPyObject(src.buffer));
      }
    }
  };

  template<> struct type_caster<Variable> {
  public:
    PYBIND11_TYPE_CASTER(Variable, _("torch::autograd::Variable"));
    bool load(handle src, bool) {
      PyObject *source = src.ptr();
      if (THPVariable_Check(source)) {
        value = ((THPVariable*)source)->cdata;
        return true;
      } else {
        return false;
      }
    }
    static handle cast(Variable src, return_value_policy /* policy */, handle /* parent */) {
      return handle(THPVariable_Wrap(src));
    }
  };
}}

namespace torch { namespace jit {

#define ASSERT_UNEXPIRED(METHOD_NAME) if (s.is_expired()) throw std::runtime_error("calling " METHOD_NAME " on an expired trace")

void initPythonTracerBindings(PyObject* module_) {
  auto m = py::handle(module_).cast<py::module>();
  py::class_<TracingState,std::shared_ptr<TracingState>>(m, "TracingState")
    // NB: no constructor; you have to get it from C++ code
    .def("__repr__", [](const TracingState& s) {
      std::ostringstream ss;
      ss << "<TracingState " << (const void*)&s << ">";
      return ss.str();
    })
    .def("__str__", [](const TracingState& s) -> std::string {
      if (s.is_expired()) return "<expired TracingState>";
      std::ostringstream ss;
      ss << *s.graph;
      return ss.str();
    })
    .def("export", [](TracingState& s) {
      ASSERT_UNEXPIRED("export");
      return py::bytes(ExportGraph(s.graph, {}));
    })
    .def("export", [](TracingState& s, const std::vector<at::Tensor>& initializers) {
      ASSERT_UNEXPIRED("export");
      return py::bytes(ExportGraph(s.graph, initializers));
    })
    .def("graph", [](TracingState& s) {
      return s.graph;
    })
    .def_property_readonly("is_expired", [](TracingState& s) {
      return s.is_expired();
    })
    .def_property_readonly("is_complete", [](TracingState& s) {
      return s.is_complete();
    });

  m.def("_tracer_enter", [](std::vector<TraceInput> trace_inputs, std::size_t num_backwards) {
    return enter(std::move(trace_inputs), num_backwards + 1);
  });
  m.def("_tracer_exit", [](variable_list var_outputs) {
    tracer::exit(var_outputs);
  });
}

}} // namespace torch::jit
