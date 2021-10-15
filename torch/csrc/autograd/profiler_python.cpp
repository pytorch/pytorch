#include <torch/csrc/autograd/profiler_python.h>

#include <algorithm>
#include <iostream>
#include <regex>
#include <string>
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
int64_t profile_start_time_;

std::string torch_path_;

// Set during init. Pointer to `torch.nn.Module.__call__.__code__` which lets
// us identify calls to nn Module's forward method.
PyObject* module_call_code_;
}  // namespace

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
class PyCodeDescription {
 public:
    static std::string lookup(PyCodeObject* f_code, int f_lasti) {
        auto& c = cache();
        const Key key { f_code, f_lasti };
        if (c.find(key) == c.end()) {
            return "Python: ???";
        } else {
            return c.at(key).name();
        }
    }

    static void populate(PyCodeObject* f_code, const int f_lasti) {
        auto& c = cache();
        const Key key { f_code, f_lasti };
        if (c.find(key) == c.end()) {
            PyCodeDescription desc {
                /*line_no_=*/PyCode_Addr2Line(/*f_code=*/key.first, /*f_lasti=*/key.second),
                /*filename_=*/THPUtils_unpackString(key.first->co_filename),
                /*funcname_=*/THPUtils_unpackString(key.first->co_name)
            };
            c.insert({ key, desc });
        }
    }

    static void clear_cache() {
        cache().clear();
    }

    std::string name() {
        std::stringstream loc;
        loc << filename_ << "(" << line_no_ << "): " << funcname_;
        std::regex e (torch_path_);
        return std::regex_replace (loc.str(), e, "torch");
    }

 private:
    PyCodeDescription(int line_no, std::string filename, std::string funcname)
        : line_no_(line_no), filename_(filename), funcname_(funcname) {}

    int line_no_;
    std::string filename_;
    std::string funcname_;

    using Key = std::pair</*f_code=*/PyCodeObject*, /*f_lasti=*/int>;
    using Cache = ska::flat_hash_map<Key, PyCodeDescription, hash_pair>;
    static Cache& cache() {
        static Cache cache_;
        return cache_;
    }
};

// We have to track when we enter a frame so we can report the span to Kineto.
// For frames already on the interpreter stack when we begin profiling, we
// simply use the time when we started profiling.
class CallTime {
 public:
    static void clear_cache() {
        py_cache().clear();
        c_cache().clear();
    }

    static void insert(PyFrameObject* frame, int64_t t) {
        py_cache().insert({ frame, t });
    }

    static void insert(PyFrameObject* frame, PyObject* fn, int64_t t) {
        c_cache().insert({ { frame, fn }, t });
    }

    static int64_t pop(PyFrameObject* frame) {
        auto& c = py_cache();
        const auto& it = c.find(frame);
        if (it == c.end()) {
            return profile_start_time_;
        }
        c.erase(it);
        return it->second;
    }

    static int64_t pop(PyFrameObject* frame, PyObject* fn) {
        auto& c = c_cache();
        const CCallFrame key { frame, fn };
        const auto& it = c.find(key);
        if (it == c.end()) {
            return profile_start_time_;
        }
        c.erase(it);
        return it->second;
    }

 private:
    // We have to track the calling frame in case multiple threads call the
    // same bound C function.
    using PyCache = ska::flat_hash_map<PyFrameObject*, int64_t>;
    using CCallFrame = std::pair<PyFrameObject*, PyObject*>;
    using CCache = ska::flat_hash_map<CCallFrame, int64_t, hash_pair>;

    static PyCache& py_cache() {
        static PyCache cache_;
        return cache_;
    }

    static CCache& c_cache() {
        static CCache cache_;
        return cache_;
    }
};

// ============================================================================
// == Per-call tracking =======================================================
// ============================================================================
struct PyCallMetadata {
    PyCallMetadata(PyFrameObject* frame) : f_code_(frame->f_code), f_lasti_(frame->f_lasti) {
        // Give PyCodeDescription a chance to populate the string cache while
        // the objects are live. This way we can get the descriptions without
        // extending the lifetimes of arbitrary Python objects. (Which could
        // significantly alter the runtime characteristics of the program.)
        PyCodeDescription::populate(f_code_, f_lasti_);
    }

    std::string name() {
        return PyCodeDescription::lookup(f_code_, f_lasti_);
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
        loc << "nn.Module: " << py::str(self_.attr("__class__").attr("__name__"));
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

    std::string name() {
        return py::str(fn_);
    }

    // Non-owning reference. We do not expect extensions to be unbound while
    // we are profiling.
    PyObject* fn_;
};

using Metadata = c10::variant<
    PyCallMetadata,
    PyModuleCallMetadata,
    CCallMetadata>;

struct typeVisitor {
    CallType operator()(PyCallMetadata&) { return CallType::kPyCall; }
    CallType operator()(PyModuleCallMetadata&) { return CallType::kPyModuleCall; }
    CallType operator()(CCallMetadata&) { return CallType::kCCall; }
    static CallType visit(Metadata& m) {
        return c10::visit(typeVisitor(), m);
    }
};

struct nameVisitor {
    std::string operator()(PyCallMetadata& m) { return m.name(); }
    std::string operator()(PyModuleCallMetadata& m) { return m.name(); }
    std::string operator()(CCallMetadata& m) { return m.name(); }
    static std::string visit(Metadata& m) {
        return c10::visit(nameVisitor(), m);
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

    static std::vector<CompletedCall>& history() {
        static std::vector<CompletedCall> history_;
        return history_;
    }

    CallType call_type() { return typeVisitor::visit(metadata_); }
    std::string name() { return nameVisitor::visit(metadata_); }

    int64_t t0_;
    int64_t duration_;
    Metadata metadata_;
};

// ============================================================================
// == CPython interpreter integration =========================================
// ============================================================================
int py_profile_fn(
    PyObject* obj,
    PyFrameObject* frame,
    int what,
    PyObject* arg) {
  switch (what) {
    case PyTrace_CALL:
        CallTime::insert(frame, now());
        break;

    case PyTrace_C_CALL:
        CallTime::insert(frame, arg, now());
        break;

    case PyTrace_RETURN:
    case PyTrace_EXCEPTION:
        CompletedCall::history().emplace_back(
            /*t0=*/CallTime::pop(frame),
            /*t1=*/now(),
            /*frame=*/frame
         );
         break;

    case PyTrace_C_RETURN:
    case PyTrace_C_EXCEPTION:
        CompletedCall::history().emplace_back(
            /*t0=*/CallTime::pop(frame, arg),
            /*t1=*/now(),
            /*fn=*/arg
         );
         break;

    default:
        break;
  }
  return 0;
}

void clear_state() {
    TORCH_INTERNAL_ASSERT(!active_, "Cannot clear state while tracer is running.");
    profile_start_time_ = 0;
    PyCodeDescription::clear_cache();
    CallTime::clear_cache();
    CompletedCall::history().clear();
}

void start_profiling() {
    // We have to lazily initialize because some symbols are not yet available
    // when `init()` is called.
    if (module_call_code_ == nullptr) {
        module_call_code_ = py::module::import("torch.nn")
            .attr("Module")
            .attr("__call__")
            .attr("__code__")
            .ptr();

        // TODO: use os.path.split
        torch_path_ = py::str(py::module::import("torch").attr("__file__"));
        std::regex e ("/__init__.py");
        torch_path_ = std::regex_replace(torch_path_, e, "");
    }

    clear_state();
    profile_start_time_ = now();

    // Note:
    //   This profile will not compose with other profilers, and cannot be
    //   round tripped via `sys.settrace(sys.gettrace())`
    PyEval_SetProfile(py_profile_fn, NULL);
    active_ = true;
}

void stop_profiling() {
    TORCH_INTERNAL_ASSERT(active_, "Tracer is running.")

    // Note:
    //   This will clear ALL active profilers. This tracer should not be used
    //   with any other profiler.
    PyEval_SetProfile(NULL, NULL);
    active_ = false;
}

std::vector<PyTraceEvent> get_events() {
    std::vector<PyTraceEvent> out;
    for (auto& c : CompletedCall::history()) {
        PyTraceEvent e { c.t0_, c.t0_ + c.duration_, c.name(), c.call_type() };
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
