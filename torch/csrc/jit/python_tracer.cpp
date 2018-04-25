#include "torch/csrc/python_headers.h"

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
  py::class_<TracingState,std::shared_ptr<TracingState>>(m, "TracingState", py::dynamic_attr())
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
    .def("push_scope", [](TracingState& s, const std::string& scope_name) {
      ASSERT_UNEXPIRED("push_scope");
      s.push_scope(scope_name);
    })
    .def("pop_scope", [](TracingState& s) {
      ASSERT_UNEXPIRED("pop_scope");
      s.pop_scope();
    })
    .def("set_graph", [](TracingState& s, std::shared_ptr<Graph> g) {
      s.graph = g;
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

  m.def("_tracer_enter", [](variable_list trace_inputs, std::size_t num_backwards) {
    return tracer::enter(std::move(trace_inputs), num_backwards + 1, true);
  });
  m.def("_tracer_exit", [](variable_list var_outputs) {
    tracer::exit(var_outputs);
  });
  m.def("_get_tracing_state", [](const variable_list& vars) {
    return getTracingState(vars);
  });
  m.def("_get_value_trace", [](std::shared_ptr<TracingState>& state, const Variable& var) {
    return getValueTrace(state, var);
  });
  m.def("_set_value_trace", [](std::shared_ptr<TracingState>& state, const Variable& var, Value* value) {
    return setValueTrace(state, var, value);
  });
  m.def("_is_tracing", [](const variable_list& vars) {
    return isTracingVar(vars);
  });
}

}} // namespace torch::jit
