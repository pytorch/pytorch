#include <torch/csrc/autograd/profiler_python.h>

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

#include <Python.h>
#include <frameobject.h>

#include <c10/util/flat_hash_map.h>
#include <c10/util/variant.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;


namespace torch { namespace autograd { namespace profiler { namespace python_tracer {

// ============================================================================
// == Global state tracking ===================================================
// ============================================================================
//
// We only expect a single instance of the Python tracer to be active at once.
// We leverage this fact and use various global caches (along with the guarantee
// that writes originating from our Py_tracefunc will be guarded by the GIL) to
// keep the hot path lightweight and low distortion.
namespace {

bool active_ = false;

// Set during init. Pointer to `torch.nn.Module.__call__.__code__` which lets
// us identify calls to nn Module's forward method.
PyObject* module_call_code_;

// std::hash doesn't have a specialization for pairs so we have to define one.
// A simple XOR is good enough for our purposes.
struct hash_pair {
    template <class T1, class T2>
    size_t operator() (const std::pair<T1, T2>& pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

// The size of our metadata variant is the largest of the constituents (plus
// the tag bytes). Moreover, walking over an arbitrarily large filename string
// as part of the hot path is not desirable. To ameliorate these issues we move
// the string storage to a separate struct, and cache the result. In particular,
// once the model is warmed up cache hit rate should be high and yield low
// distortion.
using CodeKey = std::pair</*f_code=*/PyCodeObject*, /*f_lasti=*/int>;
struct PyCodeDescription {
    int line_no_;
    std::string filename_;
    std::string funcname_;

    std::string name() {
        std::stringstream loc;
        loc << filename_ << "(" << line_no_ << "): " << funcname_;
        return loc.str();
    }
};
ska::flat_hash_map<CodeKey, PyCodeDescription, hash_pair> py_code_cache_;

void maybe_populate_description(PyCodeObject* f_code, const int f_lasti) {
    const CodeKey key { f_code, f_lasti };
    if (py_code_cache_.find(key) == py_code_cache_.end()) {
        PyCodeDescription d {
            /*line_no_=*/PyCode_Addr2Line(/*f_code=*/key.first, /*f_lasti=*/key.second),
            /*filename_=*/THPUtils_unpackString(key.first->co_filename),
            /*funcname_=*/THPUtils_unpackString(key.first->co_name)
        };
        py_code_cache_.insert({ key, d });
    }
}

std::string lookup_description(PyCodeObject* f_code, int f_lasti) {
    const CodeKey key { f_code, f_lasti };
    if (py_code_cache_.find(key) == py_code_cache_.end()) {
        return "Python: ???";
    } else {
        return py_code_cache_.at(key).name();
    }
}

// We have to track the calling frame in case multiple threads call the same
// bound C function.
using CCallFrame = std::pair<PyFrameObject*, PyObject*>;

// We have to track when we enter a frame so we can report the span to Kineto.
// For frames already on the interpreter stack when we begin profiling, we
// simply use the time when we started profiling.
int64_t profile_start_time_;
ska::flat_hash_map<PyFrameObject*, int64_t> frame_call_time_;
ska::flat_hash_map<CCallFrame, int64_t, hash_pair> ccall_time_;

int64_t pop_call_time(PyFrameObject* frame) {
    const auto& it = frame_call_time_.find(frame);
    if (it == frame_call_time_.end()) {
        return profile_start_time_;
    }
    frame_call_time_.erase(it);
    return it->second;
}

int64_t pop_call_time(PyFrameObject* frame, PyObject* fn) {
    const CCallFrame key { frame, fn };
    const auto& it = ccall_time_.find(key);
    if (it == ccall_time_.end()) {
        return profile_start_time_;
    }
    ccall_time_.erase(it);
    return it->second;
}

}  // namespace


struct PyCallMetadata {
    PyCallMetadata(PyFrameObject* frame) : f_code_(frame->f_code), f_lasti_(frame->f_lasti) {
        // The first time we see an entry we populate the cache while the
        // objects are guaranteed to be live.
        maybe_populate_description(f_code_, f_lasti_);
    }

    // This is a non-owning reference. We rely on the cache in the ctor, and only
    // use the code pointer so we can lookup the cached value.
    PyCodeObject* f_code_;
    int f_lasti_;
};

struct PyModuleCallMetadata {
    PyModuleCallMetadata(PyObject* self)
        : self_(py::reinterpret_borrow<py::object>(self)) {};

    std::string name() {
        std::stringstream loc;
        loc << "nn.Module: " << py::repr(self_.attr("__class__").attr("__name__"));
        return loc.str();
    }

    // We borrow a reference to the module. This can extend the lifetime of
    // the module; however, unlike arbitrary Python objects we can reasonably
    // assume a module surviving until the end of a profile session will not
    // have any adverse effect.
    py::object self_;
};

struct CCallMetadata {
    CCallMetadata(PyObject* fn) : fn_(fn) {}
    PyObject* fn_;  // Non-owning reference.
};

using Metadata = c10::variant<
    PyCallMetadata,
    PyModuleCallMetadata,
    CCallMetadata>;

struct nameVisitor {
    std::string operator()(PyCallMetadata& m) {
        return lookup_description(m.f_code_, m.f_lasti_);
    }
    std::string operator()(PyModuleCallMetadata& m) {
        return m.name();
    }
    std::string operator()(CCallMetadata& m) {
        return py::repr(m.fn_);
    }
};

struct CompletedCall {

    // PyTrace_CALL
    CompletedCall(int64_t t0, int64_t t1, PyFrameObject* frame)
        : t0_(t0), duration_(t1 - t0), metadata_(select_py_metadata(frame)) {}

    // PyTrace_C_CALL
    CompletedCall(int64_t t0, int64_t t1, PyObject* fn)
        : t0_(t0),
          duration_(t1 - t0),
          metadata_(CCallMetadata(fn)) {}

    static Metadata select_py_metadata(PyFrameObject* frame) {
        if ((PyObject*)(frame->f_code) == module_call_code_) {
            // By default `f_locals` is NULL as a performance optimization. (so the
            // interpreter can use an interal data type rather that a dict) We have
            // to tell it to materialize the dict, and then return to the fast path
            // when we're done.
            PyFrame_FastToLocals(frame);
            PyObject* self = PyDict_GetItemString(frame->f_locals, "self");
            PyFrame_LocalsToFast(frame, 0);
            return PyModuleCallMetadata(self);
        } else {
            return PyCallMetadata(frame);
        }
    }

    std::string name() {
        return c10::visit(nameVisitor(), metadata_);
    }

    int64_t t0_;
    int64_t duration_;
    Metadata metadata_;
};

namespace {
std::vector<CompletedCall> completed_calls_;
}  // namespace

void clear_state() {
    TORCH_INTERNAL_ASSERT(!active_, "Cannot clear state while tracer is running.");
    profile_start_time_ = 0;
    py_code_cache_.clear();
    frame_call_time_.clear();
    ccall_time_.clear();
    completed_calls_.clear();
}

int py_profile_fn(
    PyObject* obj,
    PyFrameObject* frame,
    int what,
    PyObject* arg) {
  switch (what) {
    case PyTrace_CALL:
        frame_call_time_.insert({ frame, now() });
        break;

    case PyTrace_C_CALL:
        ccall_time_.insert({ { frame, arg }, now() });
        break;

    case PyTrace_RETURN:
    case PyTrace_EXCEPTION:
        completed_calls_.emplace_back(
            /*t0=*/pop_call_time(frame),
            /*t1=*/now(),
            /*frame=*/frame
         );
         break;

    case PyTrace_C_RETURN:
    case PyTrace_C_EXCEPTION:
        completed_calls_.emplace_back(
            /*t0=*/pop_call_time(frame, arg),
            /*t1=*/now(),
            /*fn=*/arg
         );
         break;

    default:
        break;
  }
  return 0;
}

void start_profiling() {
    if (module_call_code_ == nullptr) {
        module_call_code_ = py::module::import("torch.nn")
            .attr("Module")
            .attr("__call__")
            .attr("__code__")
            .ptr();
    }

    clear_state();
    profile_start_time_ = now();

    // TODO: respect other profilers.
    PyEval_SetProfile(py_profile_fn, NULL);
    active_ = true;
}

void stop_profiling() {
    TORCH_INTERNAL_ASSERT(active_, "Tracer is running.")

    // TODO: respect other profilers.
    PyEval_SetProfile(NULL, NULL);
    active_ = false;
}

std::vector<PyTraceEvent> get_events() {
    std::vector<PyTraceEvent> out;
    for (auto& c : completed_calls_) {
        PyTraceEvent e { c.t0_, c.t0_ + c.duration_, c.name() };
        out.push_back(e);
    }
    return out;
}

void init() {
    registerFunctions(
        /*start=*/&start_profiling,
        /*stop=*/&stop_profiling,
        /*clear=*/&clear_state,
        /*get_events=*/&get_events
    );
}

}}}} // namespace torch::autograd::profiler::python_tracer
