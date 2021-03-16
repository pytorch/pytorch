#include <torch/csrc/python_headers.h>

#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/jit/python/python_tracer.h>
#include <torch/csrc/jit/serialization/export.h>
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
std::vector<StackEntry> _pythonCallstack() {
  pybind11::gil_scoped_acquire gil;
  PyFrameObject* frame = PyEval_GetFrame();
  std::vector<StackEntry> entries;

  while (nullptr != frame) {
    size_t line = PyCode_Addr2Line(frame->f_code, frame->f_lasti);
    std::string filename = THPUtils_unpackString(frame->f_code->co_filename);
    std::string funcname = THPUtils_unpackString(frame->f_code->co_name);
    auto source = std::make_shared<Source>(funcname, filename, line);
    entries.emplace_back(
        StackEntry{funcname, SourceRange(source, 0, funcname.size())});
    frame = frame->f_back;
  }
  return entries;
}

SourceRange getPythonInterpreterSourceRange() {
  auto cs = pythonCallstack();
  c10::optional<std::string> source_filename;
  size_t source_line = 0;
  std::stringstream stack_trace;
  for (const auto& entry : cs) {
    auto& range = entry.range;
    if (range.source()) {
      auto& src = range.source();
      if (src && src->filename()) {
        auto line =
            src->starting_line_no() + src->lineno_for_offset(range.start());
        stack_trace << *(src->filename()) << "(" << line
                    << "): " << entry.filename << "\n";
        if (!source_filename) {
          source_filename = *(src->filename());
          source_line = line;
        }
      }
    }
  }

  auto stack_trace_text = stack_trace.str();
  auto source =
      std::make_shared<Source>(stack_trace_text, source_filename, source_line);
  return SourceRange(source, 0, stack_trace_text.size());
}

std::pair<std::shared_ptr<Graph>, Stack> createGraphByTracing(
    const py::function& func,
    Stack trace_inputs,
    const py::function& var_name_lookup_fn,
    bool strict,
    bool force_outplace,
    Module* self,
    const std::vector<std::string>& argument_names) {
  C10_LOG_API_USAGE_ONCE("torch.tracer");

  auto lookup_fn_adapter =
      [var_name_lookup_fn](const Variable& var) -> std::string {
    pybind11::gil_scoped_acquire ag;
    return py::cast<std::string>(var_name_lookup_fn(var));
  };

  auto outs = tracer::trace(
      std::move(trace_inputs),
      [&func](Stack inputs) -> Stack {
        size_t num_func_inputs = inputs.size();
        py::tuple py_inputs(num_func_inputs);
        for (size_t i = 0; i < num_func_inputs; ++i) {
          py_inputs[i] = py::cast(inputs[i]);
        }
        auto out = func(*py_inputs);
        if (out.ptr() == Py_None) {
          AT_ERROR(
              "The traced function didn't return any values! Side-effects are not "
              "captured in traces, so it would be a no-op.");
        }
        return {toTypeInferredIValue(out)};
      },
      lookup_fn_adapter,
      strict,
      force_outplace,
      self,
      argument_names);
  return std::make_pair(std::get<0>(outs)->graph, std::get<1>(outs));
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

  graph->insertNode(n);

  return n;
}

void pythonRecordSourceLocation(Node* n) {
  n->setSourceRange(getPythonInterpreterSourceRange());
}

void pythonWarn(const std::string& reason) {
  pybind11::gil_scoped_acquire gil;
  auto warn_class = py::module::import("torch.jit").attr("TracerWarning");
  PyErr_WarnEx(warn_class.ptr(), reason.c_str(), 1);
}

void initPythonTracerBindings(PyObject* module) {
  setPythonCallstack(_pythonCallstack);
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
          "current_scope",
          [](TracingState& s) {
            return s.graph->current_scope()->name().toUnqualString();
          })
      .def(
          "set_graph",
          [](TracingState& s, std::shared_ptr<Graph> g) {
            s.graph = std::move(g);
          })
      .def("graph", [](TracingState& s) { return s.graph; });

  m.def("_tracer_warn_use_python", []() { tracer::setWarn(pythonWarn); });
  m.def(
      "_create_graph_by_tracing",
      createGraphByTracing,
      py::arg("func"),
      py::arg("inputs"),
      py::arg("var_name_lookup_fn"),
      py::arg("strict"),
      py::arg("force_outplace"),
      py::arg("self") = nullptr,
      py::arg("argument_names") = std::vector<std::string>());
  m.def("_get_tracing_state", []() { return getTracingState(); });
  m.def("_set_tracing_state", [](std::shared_ptr<TracingState> state) {
    return setTracingState(std::move(state));
  });
  m.def("_get_value_trace", [](const Variable& var) {
    return getValueTrace(var);
  });
  m.def("_set_value_trace", [](const Variable& var, Value* value) {
    return setValueTrace(var, value);
  });
  m.def("_tracer_set_get_unique_name_fn", [](const py::function& func) {
    const auto& tracing_state = getTracingState();
    AT_ASSERT(tracing_state);
    tracing_state->lookup_var_name_fn =
        [func](const Variable& var) -> std::string {
      pybind11::gil_scoped_acquire ag;
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
