#include "torch/csrc/python_headers.h"

#include "python_compiled_function.h"

#include "torch/csrc/jit/pybind.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/tracer.h"
#include "torch/csrc/jit/tracer_state.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/peephole.h"
#include "torch/csrc/jit/passes/graph_fuser.h"
#include "torch/csrc/jit/passes/inplace_check.h"
#include "torch/csrc/jit/passes/batch_mm.h"
#include "torch/csrc/jit/python_arg_flatten.h"
#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/jit/interpreter_autograd_function.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>
#include <vector>

namespace torch { namespace jit { namespace python {

using namespace torch::autograd;
using namespace torch::jit::tracer;

namespace {

// pybind casts are really verbose...
py::object steal(py::handle x) {
  return py::reinterpret_steal<py::object>(x);
}

} // anonymous namespace

// Lifecycle of a CompiledFunction:
//
// - It is given an underlying function, which knows how to actually
//   execute the code that we want to compile.
// - When we encounter an input configuration for which we don't
//   have an optimized trace, we run the underlying function, tracing its
//   result.  The trace is not done yet, so we save it into our set of pending
//   traces for that configuration.
// - When we encounter an input configuration whose trace is "ready"
//   (that is, we've seen all of the passes, so the trace contains
//   forwards/backwards/etc), we compile it, and then register this
//   as the compiled trace.
// - When we encounter an input configuration whose trace is compiled,
//   we just directly run the compiled trace.
struct CompiledFunction {

  struct TraceForKey {
    TraceForKey(CompiledFunction& fn, bool grad_enabled)
      : fn_(fn)
      , grad_enabled_(grad_enabled) {}

    bool ready() {
      if (is_ready_) return true;

      // Remove expired traces
      traces_.erase(std::remove_if(traces_.begin(),
                                   traces_.end(),
                                   [](const std::shared_ptr<TracingState>& state) {
                                     return state->is_expired();
                                   }),
                    traces_.end());

      // Check if any trace is complete
      auto complete_it = std::find_if(traces_.begin(),
                                   traces_.end(),
                                   [](const std::shared_ptr<TracingState>& state) {
                                     return state->is_complete();
                                   });
      if (complete_it == traces_.end())
        return false;

      auto complete_trace = *complete_it; // NOTE: copy, because we clear right after
      traces_.clear();

      // Now, we have a complete trace. Compile it.
      EliminateDeadCode(complete_trace->graph);
      CheckInplace(complete_trace->graph);
      if (fn_.optimize_) {
        PeepholeOptimize(complete_trace->graph);
        BatchMM(complete_trace->graph);
        FuseGraph(complete_trace->graph);
        EliminateCommonSubexpression(complete_trace->graph);
      }
      factory_ = std::make_shared<InterpreterFunctionFactory>(complete_trace.get());
      graph_ = complete_trace->graph;
      is_ready_ = true;
      return true;
    }

    variable_list run(variable_list inputs) {
      JIT_ASSERT(is_ready_);
      AutoNoGIL _gil_guard;
      auto fn = factory_->construct();
      fn->will_release_variables(); // forward pass is never reused, so it is safe to release anything it can
      return fn->apply(inputs);
    }

    PyObject* add_trace(PyObject *args, ParsedArgs input_info) {
      JIT_ASSERT(!is_ready_);
      // Start tracing
      AutoGradMode grad_mode(grad_enabled_);
      auto num_stages = grad_enabled_ ? fn_.nderivs_ + 1 : 1;
      auto enter_info = tracer::enter(input_info.vars, num_stages, true);
      auto & trace = enter_info.first;
      auto & new_vars = enter_info.second;

      // Enter returns us a new list of Variables to use as inputs, so handle that.
      std::size_t num_all_inputs = input_info.vars.size();
      std::size_t num_captured = fn_.captured_vars_.size();
      // Check that no captured Variables were replaced by enter. It's hard to handle that.
      for (std::size_t i = num_all_inputs - num_captured; i < num_all_inputs; ++i) {
        TORCH_EXPECTM(input_info.vars[i].is_same(new_vars[i]),
                      "Some of the Variables captured by the JIT are repeated");
      }
      // Now only arguments to this function could have changed. Slice their vars out, and
      // re-create the structure of args, but using new Variables.
      variable_list new_inputs(new_vars.begin(),
                               new_vars.end() - num_captured);
      THPObjectPtr new_args { unflatten(new_inputs, input_info.desc) };

      // Call back into Python function
      auto out = PyObject_CallObject(fn_.function_.get(), new_args.get());
      if (!out) throw py::error_already_set();

      // Flatten outputs and update fields
      auto out_info = flatten(out);
      if (out_desc_.structure.empty()) {
        out_desc_ = std::move(out_info.desc);
      } else {
        // TODO: assert matches but only in debug mode
      }

      // Finish tracing and save the current trace
      tracer::exit(out_info.vars);
      traces_.emplace_back(std::move(trace));
      return out;
    }

    CompiledFunction& fn_;
    IODescriptor out_desc_;
    std::vector<std::shared_ptr<TracingState>> traces_;
    bool grad_enabled_ = false;
    bool is_ready_ = false;

    std::shared_ptr<InterpreterFunctionFactory> factory_;
    std::shared_ptr<jit::Graph> graph_;
  };

  TraceForKey& getTrace(ParsedArgs& args) {
    auto it = ktraces_.find(args.desc);
    if (it == ktraces_.end()) {
      bool grad_enabled = args.desc.grad_enabled;
      std::tie(it, std::ignore) = ktraces_.emplace(args.desc,
                                                   TraceForKey(*this, grad_enabled));
    }
    return it->second;
  }

  ParsedArgs flattenArgs(py::handle pyargs) {
    auto args = flatten(pyargs);
    // We need to take captured_var types into account when choosing the trace
    args.extend(captured_vars_);
    return args;
  }

  py::object fallback(py::handle pyargs) {
    return steal(PyObject_CallObject(function_.get(), pyargs.ptr()));
  }

  py::object call(py::handle pyargs) {
    if (!enabled_) {
      return fallback(pyargs);
    }
    auto args = flattenArgs(pyargs);

    if(isTracingVar(args.vars)) {
      // Some outer compiled function has called another compiled function.
      // In this case we just fall back to the original python function,
      // allowing the inner trace to be inlined into the outer.
      // This scenario occurs when blocking an lstm loop.
      return fallback(pyargs);
    }

    auto& ktrace = getTrace(args);

    variable_list out_vars;
    if (ktrace.ready()) {
      hits_++;
      return steal(unflatten(ktrace.run(std::move(args.vars)), ktrace.out_desc_));
    } else {
      misses_++;
      return steal(ktrace.add_trace(pyargs.ptr(), std::move(args)));
    }
  }

  void clearCache() {
    ktraces_.clear();
  }

  CompiledFunction(int nderivs, bool optimize, bool enabled, py::object function,
                   std::string name)
    : nderivs_(nderivs)
    , optimize_(optimize)
    , enabled_(enabled)
    , hits_(0)
    , misses_(0)
    , function_(function.release().ptr())
    , name_(std::move(name))
    , captured_vars_()
    , ktraces_() {}

  int nderivs_;
  bool optimize_;
  bool enabled_;
  std::atomic<uint64_t> hits_;
  std::atomic<uint64_t> misses_;
  THPObjectPtr function_;
  std::string name_;
  variable_list captured_vars_;
  std::unordered_map<IODescriptor, TraceForKey, torch::hash<IODescriptor>> ktraces_;
};


std::ostream& operator<<(std::ostream& out, const CompiledFunction::TraceForKey & trace) {
  if(!const_cast<CompiledFunction::TraceForKey&>(trace).ready()) {
      out << "<trace has been started but has not been completed>";
      return out;
  }
  out << *trace.graph_ << "\n";
  return out;
}

std::ostream& operator<<(std::ostream& out, const CompiledFunction & cf) {
  out << "CompiledFunction: " << cf.name_ << "(nderivs=" << cf.nderivs_ << ", optimized=" << cf.optimize_ << ", enabled=" << cf.enabled_ << "):\n";
  out << "trace cache hits: " << cf.hits_ << "\n";
  out << "trace cache misses: " << cf.misses_ << "\n";
  std::vector<std::string> trace_info;
  for(auto & v : cf.ktraces_) {
    std::stringstream ss;
    ss << v.first << v.second <<  "\n\n";
    trace_info.push_back(ss.str());
  }
  // unordered map, so sort to make this deterministic, the IODescriptors will
  // be different so comparison won't read most of the string.
  std::sort(trace_info.begin(), trace_info.end());
  out << trace_info.size() << " traces found.\n";

  for(size_t i = 0; i < trace_info.size(); ++i) {
    out << "Trace " << i << " for input with layout " << trace_info[i];
  }
  return out;
}


namespace {

CompiledFunction::TraceForKey* getTraceFor(CompiledFunction& fn,
                                           py::handle pyargs) {
  auto args = fn.flattenArgs(pyargs);
  auto it = fn.ktraces_.find(args.desc);
  if (it == fn.ktraces_.end())
    return nullptr;
  return it->second.ready() ? &it->second : nullptr;
}

} // anonymous namespace

void initCompilerMixin(PyObject *module) {
  auto m = py::handle(module).cast<py::module>();
  py::class_<CompiledFunction>(m, "CompiledFunction", py::dynamic_attr())
    .def(py::init<int, bool, bool, py::object, std::string>())
    .def("__call__", [](py::args args_) -> py::object {
      auto fn = py::cast<CompiledFunction*>(args_[0]);
      auto args = tuple_tail(args_);
      return fn->call(args);
    })
    .def("has_trace_for", [](py::args args_) -> bool {
      auto fn = py::cast<CompiledFunction*>(args_[0]);
      auto args = tuple_tail(args_);
      return getTraceFor(*fn, args) != nullptr;
    })
    .def("graph_for", [](py::args args_) -> py::object {
      auto fn = py::cast<CompiledFunction*>(args_[0]);
      auto args = tuple_tail(args_);
      auto trace = getTraceFor(*fn, args);
      return trace ? py::cast(trace->graph_) : py::none();
    })
    .def("clear_cache", [](CompiledFunction& fn) {
      fn.clearCache();
    })
    .def("set_captured_vars", [](CompiledFunction& fn, variable_list vars) {
      fn.captured_vars_ = std::move(vars);
    })
    .def("jit_debug_info", [](const CompiledFunction& s) -> std::string {
      std::ostringstream ss;
      ss << s;
      return ss.str();
    })
    .def_property_readonly("hits", [](CompiledFunction& fn) {
      return fn.hits_.load();
    })
    .def_property_readonly("misses", [](CompiledFunction& fn) {
      return fn.misses_.load();
    })
    .def_readwrite("enabled", &CompiledFunction::enabled_);
}

}}} // namespace torch::jit::python
