#include <torch/csrc/python_headers.h>

#include <torch/csrc/jit/export.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/pybind.h>
#include <torch/csrc/jit/python_tracer.h>
#include <torch/csrc/jit/tracer.h>
#include <torch/csrc/utils/python_strings.h>

#include <c10/util/Exception.h>

#include <sstream>

using namespace torch::autograd;
using namespace torch::jit;
using namespace torch::jit::tracer;

namespace torch {
namespace jit {
namespace tracer {

// Python interpreter retrieval routine adapted from
// https://stackoverflow.com/a/8706144
SourceRange getPythonInterpreterSourceRange() {
  c10::optional<std::string> source_filename;
  size_t source_line = 0;
  std::stringstream stack_trace;

  AutoGIL gil;
  PyFrameObject* frame = PyEval_GetFrame();

  while (nullptr != frame) {
    int line = PyCode_Addr2Line(frame->f_code, frame->f_lasti);
    std::string filename = THPUtils_unpackString(frame->f_code->co_filename);
    std::string funcname = THPUtils_unpackString(frame->f_code->co_name);
    stack_trace << filename << "(" << line << "): " << funcname << "\n";
    if (!source_filename) {
      source_filename = filename;
      source_line = line;
    }
    frame = frame->f_back;
  }

  auto stack_trace_text = stack_trace.str();
  auto source =
      std::make_shared<Source>(stack_trace_text, source_filename, source_line);
  return SourceRange(source, 0, stack_trace_text.size());
}

std::shared_ptr<torch::jit::Graph> createGraphByTracing(
    const py::function& func,
    Stack trace_inputs,
    const py::function& var_name_lookup_fn,
    bool force_outplace,
    script::Module* self) {
  C10_LOG_API_USAGE_ONCE("torch.tracer");

  try {
    auto enter_info = tracer::enter(std::move(trace_inputs), self);
    auto graph = enter_info.first->graph;

    getTracingState()->lookup_var_name_fn =
        [var_name_lookup_fn](const Variable& var) -> std::string {
      AutoGIL ag;
      return py::cast<std::string>(var_name_lookup_fn(var));
    };
    getTracingState()->force_outplace = force_outplace;
    size_t num_func_inputs = enter_info.second.size();
    py::tuple py_inputs(num_func_inputs);
    for (size_t i = 0; i < num_func_inputs; ++i) {
      py_inputs[i] = py::cast(enter_info.second[i]);
    }
    auto out = func(*py_inputs);
    if (out.ptr() == Py_None) {
      AT_ERROR(
          "The traced function didn't return any values! Side-effects are not "
          "captured in traces, so it would be a no-op.");
    }
    tracer::exit({toTypeInferredIValue(out)});
    if (script::getInlineEverythingMode()) {
      Inline(*graph);
    }
    LowerSimpleTuples(graph);
    EliminateDeadCode(graph);
    return graph;
  } catch (...) {
    tracer::abandon();
    throw;
  }
}

Node* preRecordPythonTrace(
    THPObjectPtr pyobj,
    const std::string& arg_types,
    at::ArrayRef<Variable> inputs,
    pyobj_list scalar_args) {
  THPObjectPtr apply(PyObject_GetAttrString(pyobj.get(), "apply"));
  if (!apply) {
    throw python_error();
  }

  auto& graph = getTracingState()->graph;

  Node* n = graph->createPythonOp(
      std::move(apply), arg_types, std::move(scalar_args));
  recordSourceLocation(n);

  for (const Variable& input : inputs) {
    n->addInput(getValueTrace(input));
  }

  // NB: Order matters. This must append after inputs but before outputs.
  graph->appendNode(n);

  return n;
}

void pythonRecordSourceLocation(Node* n) {
  n->setSourceRange(getPythonInterpreterSourceRange());
}

void pythonWarn(const std::string& reason) {
  AutoGIL gil;
  auto warn_class = py::module::import("torch.jit").attr("TracerWarning");
  PyErr_WarnEx(warn_class.ptr(), reason.c_str(), 1);
}

void initPythonTracerBindings(PyObject* module) {
  setRecordSourceLocation(pythonRecordSourceLocation);

  auto m = py::handle(module).cast<py::module>();
  py::class_<TracingState, std::shared_ptr<TracingState>>(
      m, "TracingState", py::dynamic_attr())
      // NB: no constructor; you have to get it from C++ code
      .def(
          "__repr__",
          [](const TracingState& s) {
            std::ostringstream ss;
            ss << "<TracingState " << (const void*)&s << ">";
            return ss.str();
          })
      .def(
          "__str__",
          [](const TracingState& s) -> std::string {
            std::ostringstream ss;
            ss << *s.graph;
            return ss.str();
          })
      .def(
          "push_scope",
          [](TracingState& s, const std::string& scope_name) {
            s.graph->push_scope(scope_name);
          })
      .def("pop_scope", [](TracingState& s) { s.graph->pop_scope(); })
      .def(
          "set_graph",
          [](TracingState& s, std::shared_ptr<Graph> g) { s.graph = g; })
      .def("graph", [](TracingState& s) { return s.graph; });

  m.def("_tracer_warn_use_python", []() { tracer::setWarn(pythonWarn); });
  m.def("_tracer_enter", [](py::args trace_inputs) {
    return tracer::enter(toTraceableStack(trace_inputs));
  });
  m.def("_tracer_exit", [](py::tuple var_outputs) {
    tracer::exit(toTraceableStack(var_outputs));
  });
  m.def("_tracer_abandon", []() { tracer::abandon(); });
  m.def("_get_tracing_state", []() { return getTracingState(); });
  m.def("_set_tracing_state", [](std::shared_ptr<TracingState> state) {
    return setTracingState(state);
  });
  m.def("_get_value_trace", [](const Variable& var) {
    return getValueTrace(var);
  });
  m.def("_set_value_trace", [](const Variable& var, Value* value) {
    return setValueTrace(var, value);
  });
  m.def("_tracer_set_get_unique_name_fn", [](py::function func) {
    const auto& tracing_state = getTracingState();
    AT_ASSERT(tracing_state);
    tracing_state->lookup_var_name_fn =
        [func](const Variable& var) -> std::string {
      AutoGIL ag;
      return py::cast<std::string>(func(var));
    };
  });
  m.def("_tracer_set_force_outplace", [](bool force_outplace) {
    const auto& tracing_state = getTracingState();
    AT_ASSERT(tracing_state);
    tracing_state->force_outplace = force_outplace;
  });
}

} // namespace tracer
} // namespace jit
} // namespace torch
