#include <Python.h>

#include "torch/csrc/jit/python_tracer.h"
#include "torch/csrc/jit/tracer.h"
#include "torch/csrc/assertions.h"
#include "torch/csrc/jit/export.h"
#include "torch/csrc/jit/pybind.h"
#include "torch/csrc/utils/python_strings.h"

#include <sstream>

using namespace torch::autograd;
using namespace torch::jit;
using namespace torch::jit::tracer;

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
