#include "python_compiled_function.h"

#include "torch/csrc/jit/pybind.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/functions/jit_closure.h"
#include "torch/csrc/jit/tracer.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/peephole.h"
#include "torch/csrc/jit/passes/graph_fuser.h"
#include "torch/csrc/jit/passes/inplace_check.h"
#include "torch/csrc/jit/python_arg_flatten.h"

#include <algorithm>
#include <functional>
#include <atomic>

namespace torch { namespace jit { namespace python {

using namespace torch::autograd;
using namespace torch::jit::tracer;

namespace {

// pybind casts are really verobse...
py::object steal(py::handle x) {
  return py::reinterpret_steal<py::object>(x);
}

py::object borrow(py::handle x) {
  return py::reinterpret_borrow<py::object>(x);
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
    TraceForKey(CompiledFunction& fn, bool is_volatile)
      : fn_(fn)
      , is_volatile_(is_volatile) {}

    bool ready() {
      if (closure_) return true;

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
        FuseGraph(complete_trace->graph);
      }

      closure_ = std::make_shared<AutogradClosureFactory>(complete_trace.get());
      return true;
    }

    variable_list run(const variable_list& in_vars) {
      JIT_ASSERT(closure_);
      AutoNoGIL _gil_guard;
      auto fn = closure_->construct();
      return (*fn)(in_vars);
    }

    PyObject* add_trace(PyObject *args, const variable_list& in_vars) {
      JIT_ASSERT(!closure_);
      // Start tracing
      auto trace = tracer::enter(fmap<TraceInput>(in_vars), is_volatile_ ? 1 : (fn_.nderivs_ + 1));

      // Call back into Python function
      auto out = PyObject_CallObject(fn_.function_.get(), args);
      if (!out) throw py::error_already_set();

      // Flatten outputs and update fields
      auto out_info = flatten(out);
      if (out_desc_.empty()) {
        out_desc_ = out_info.desc;
      } else {
        // TODO: assert matches but only in debug mode
      }

      // Finish tracing and save the current trace
      tracer::exit(out_info.vars);
      traces_.emplace_back(std::move(trace));
      return out;
    }

    CompiledFunction& fn_;
    std::string out_desc_;
    std::shared_ptr<AutogradClosureFactory> closure_;
    std::vector<std::shared_ptr<TracingState>> traces_;
    bool is_volatile_;
  };

  TraceForKey& getTrace(ParsedArgs& args) {
    auto it = ktraces_.find(args.desc);
    if (it == ktraces_.end()) {
      std::tie(it, std::ignore) = ktraces_.emplace(args.desc,
                                                   TraceForKey(*this, args.is_volatile));
    }
    return it->second;
  }

  py::object makeTuple(py::handle a, py::handle b) {
    py::tuple result(2);
    result[0] = borrow(a);
    result[1] = borrow(b);
    return result;
  }

  py::object call(py::handle pyargs, py::handle pyparams) {
    if (!enabled_) {
      return steal(PyObject_CallObject(function_.get(), pyargs.ptr()));
    }
    auto all_pyargs = makeTuple(pyargs, pyparams);
    auto args = flatten(all_pyargs);
    auto& ktrace = getTrace(args);

    variable_list out_vars;
    if (ktrace.ready()) {
      hits_++;
      return steal(unflatten(ktrace.run(args.vars), ktrace.out_desc_));
    } else {
      misses_++;
      return steal(ktrace.add_trace(pyargs.ptr(), args.vars));
    }
  }

  bool hasTraceFor(py::handle pyargs, py::handle pyparams) {
    auto all_pyargs = makeTuple(pyargs, pyparams);
    auto args = flatten(all_pyargs);
    auto it = ktraces_.find(args.desc);
    if (it == ktraces_.end())
      return false;
    return it->second.ready();
  }

  void clearCache() {
    ktraces_.clear();
  }

  CompiledFunction(int nderivs, bool optimize, py::object function,
                   std::string name)
    : nderivs_(nderivs)
    , optimize_(optimize)
    , enabled_(true)
    , hits_(0)
    , misses_(0)
    , function_(function.release().ptr())
    , name_(std::move(name))
    , ktraces_() {}

  int nderivs_;
  bool optimize_;
  bool enabled_;
  std::atomic<uint64_t> hits_;
  std::atomic<uint64_t> misses_;
  THPObjectPtr function_;
  std::string name_;
  std::unordered_map<std::string, TraceForKey> ktraces_;
};


void initCompilerMixin(PyObject *module) {
  auto m = py::handle(module).cast<py::module>();
  py::class_<CompiledFunction>(m, "CompiledFunction")
    .def(py::init<int, bool, py::object, std::string>())
    .def("__call__", [](CompiledFunction& fn, py::handle args, py::handle parameters) -> py::object {
      return fn.call(args, parameters);
    })
    .def("has_trace_for", [](CompiledFunction& fn, py::handle args, py::handle parameters) -> bool {
      return fn.hasTraceFor(args, parameters);
    })
    .def("clear_cache", [](CompiledFunction& fn) {
        fn.clearCache();
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
