#include <torch/csrc/autograd/profiler_python.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <regex>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include <Python.h>
#include <frameobject.h>

#include <c10/macros/Macros.h>
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
bool init_finished_ = false;

// Strings to trim from the beginning of filenames.
std::string path_prefixes_;

// torch.nn.Module.__call__.__code__
PyObject* module_call_code_;

bool active_ = false;
int64_t profile_start_time_;
}  // namespace


void lazy_init() {
    if (!init_finished_) {
        path_prefixes_ = py::module::import("torch.profiler.python_tracer")
            .attr("_prefix_regex")().cast<std::string>();

        module_call_code_ = py::module::import("torch.nn")
            .attr("Module")
            .attr("__call__")
            .attr("__code__")
            .ptr();

        init_finished_ = true;
    }
}

static_assert(PyTrace_CALL >= 0 && PyTrace_CALL <= 255);
static_assert(PyTrace_EXCEPTION >= 0 && PyTrace_EXCEPTION <= 255);
static_assert(PyTrace_LINE >= 0 && PyTrace_LINE <= 255);
static_assert(PyTrace_RETURN >= 0 && PyTrace_RETURN <= 255);
static_assert(PyTrace_C_CALL >= 0 && PyTrace_C_CALL <= 255);
static_assert(PyTrace_C_EXCEPTION >= 0 && PyTrace_C_EXCEPTION <= 255);
static_assert(PyTrace_C_RETURN >= 0 && PyTrace_C_RETURN <= 255);
static_assert(PyTrace_OPCODE >= 0 && PyTrace_OPCODE <= 255);

int64_t pack_tag_and_time(int what) {
    constexpr int64_t mask = ((int64_t)1 << 8) - 1;
    int64_t t = now();
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(t == (t << 8 >> 8));
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(what >= 0 && what <= 255);
    return (t << 8) | ((int64_t)what & mask);
}

int extract_tag(int64_t packed) {
    constexpr int64_t mask = ((int64_t)1 << 8) - 1;
    return (int)(mask & packed);
}

int64_t extract_time(int64_t packed) {
    return packed >> 8;
}

// std::hash doesn't have a specialization for pairs so we have to define one.
// A simple XOR is good enough for our purposes.
struct hash_pair {
    template <class T1, class T2>
    size_t operator() (const std::pair<T1, T2>& pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

class PyStateCache {
 public:
    static int py_profile_fn(
            PyObject* obj,
            PyFrameObject* frame,
            int what,
            PyObject* arg) {
        return PyStateCache::singleton()._py_profile_fn(obj, frame, what, arg);
    };

    static void replay_stack() {
        PyStateCache::singleton()._replay_stack();
    };

    static void reset() {
        PyStateCache::singleton()._reset();
    };

 private:
    PyStateCache() = default;
    static PyStateCache& singleton();

    int _py_profile_fn(
        PyObject* obj,
        PyFrameObject* frame,
        int what,
        PyObject* arg);
    void _replay_stack();
    void _reset();

    size_t store_description(PyFrameObject* frame);
    void track_module(PyFrameObject* frame);

    union TracePayload {
        // PyTrace_CALL
        size_t code_description_index;

        // PyTrace_EXCEPTION
        // PyTrace_RETURN
        // ** Unused (placeholder) **
        void* null;

        // PyTrace_C_CALL
        // PyTrace_C_EXCEPTION
        // PyTrace_C_RETURN
        PyObject* arg;
    };

    struct ProfileEvent {
        ProfileEvent(int what) : tag_and_time_(pack_tag_and_time(what)) {}
        int64_t tag_and_time_;
        TracePayload breadcrumbs_;

        int tag() { return extract_tag(tag_and_time_); }
        int64_t time() { return extract_time(tag_and_time_); }
    };

    struct CodeDescription {
        CodeDescription(int line_no, std::string filename, std::string funcname)
            : line_no_(line_no), filename_(filename), funcname_(funcname) {}
        int line_no_;
        std::string filename_;
        std::string funcname_;

        std::string name() {
            // std::regex e(path_prefixes_);
            std::stringstream loc;
            // loc << std::regex_replace(filename_, e, "") << "(" << line_no_ << "): " << funcname_;
            loc << filename_ << "(" << line_no_ << "): " << funcname_;
            return loc.str();
        }
    };

    struct ModuleForward {
        ModuleForward(size_t event_index, PyObject* self)
            : event_index_(event_index), self_(self) {}
        size_t event_index_;

        // NB: This is a non-owning reference. We Py_INCREF in `track_module`,
        //     and `reset` is responsible for calling Py_DECREF when clearing
        //     `module_calls_`.
        PyObject* self_;
    };

    struct ReplayFrame;

    std::vector<ProfileEvent> events_;
    std::vector<ModuleForward> module_calls_;

    using DescriptionKey = std::pair</*f_code=*/PyCodeObject*, /*f_lasti=*/int>;
    std::vector<CodeDescription> code_descriptions_;
    ska::flat_hash_map<DescriptionKey, size_t, hash_pair> code_description_positions_;
};

PyStateCache& PyStateCache::singleton() {
    static PyStateCache singleton_;
    return singleton_;
}

size_t PyStateCache::store_description(PyFrameObject* frame) {
    const auto& it = code_description_positions_.find({ frame->f_code, frame->f_lasti });
    if C10_UNLIKELY(it == code_description_positions_.end()) {
        size_t index = code_descriptions_.size();
        code_descriptions_.emplace_back(
            /*line_no_=*/PyCode_Addr2Line(frame->f_code, frame->f_lasti),
            /*filename_=*/THPUtils_unpackString(frame->f_code->co_filename),
            /*funcname_=*/THPUtils_unpackString(frame->f_code->co_name)
        );
        code_description_positions_.insert({ { frame->f_code, frame->f_lasti }, index });
        return index;
    }
    return it->second;
}

void PyStateCache::track_module(PyFrameObject* frame) {
    if ((PyObject*)(frame->f_code) == module_call_code_) {
        // By default, CPython stores locals in a "fast" format, with an array
        // of names and an array of values. Consequently, frame->f_locals is
        // NULL since the interpreter has no need to populate it.
        //
        // If these arrays were part of the public API then we could very
        // quickly access `self`. Unfortunately they are not, and moreover are
        // not stable across versions. As a result, we are forced to call
        // `PyFrame_FastToLocals` which forces the interpreter to materialize
        // the full dict of locals.
        PyFrame_FastToLocals(frame);
        auto self = PyDict_GetItemString(frame->f_locals, "self");
        Py_INCREF(self);
        module_calls_.emplace_back(
            /*event_index=*/events_.size() - 1,
            /*self=*/self
        );
        PyFrame_LocalsToFast(frame, 0);
    }
};

void PyStateCache::_reset() {
    events_.clear();
    code_descriptions_.clear();
    code_description_positions_.clear();
    for (auto& i : module_calls_) {
        Py_DECREF(i.self_);
    }
    module_calls_.clear();
}

struct PyStateCache::ReplayFrame {
    ReplayFrame(int64_t t0, std::string name, CallType call_type, size_t id, size_t parent_id)
        : t0_(t0),
          t1_(-1),
          name_(name),
          call_type_(call_type),
          id_(id),
          parent_id_(parent_id) {}
    int64_t t0_;
    int64_t t1_;
    std::string name_;
    CallType call_type_;
    size_t id_;
    size_t parent_id_;
};

std::string stringify(int tag) {
    switch (tag) {
        case PyTrace_CALL:
            return "PyTrace_CALL       ";
        case PyTrace_C_CALL:
            return "PyTrace_C_CALL     ";
        case PyTrace_C_RETURN:
            return "PyTrace_C_RETURN   ";
        case PyTrace_C_EXCEPTION:
            return "PyTrace_C_EXCEPTION";
        case PyTrace_RETURN:
            return "PyTrace_RETURN     ";
        case PyTrace_EXCEPTION:
            return "PyTrace_EXCEPTION  ";
        default:
            return "???";
    }
}

void print_frame(PyTraceEvent* e, int depth) {
    if (e == nullptr || depth >= 10) {
        return;
    }
    print_frame(e->parent_, depth + 1);
    std::cout << e->name_ << std::endl;
}

void PyStateCache::_replay_stack() {
    ska::flat_hash_map<size_t, PyObject*> module_map;
    for (const auto& call : module_calls_) {
        module_map.insert({ call.event_index_, call.self_ });
    }

    size_t id_counter = 0;
    std::vector<ReplayFrame> stack;
    std::vector<ReplayFrame> results;

    for (size_t i = 0; i < events_.size(); i++) {
        auto raw_event = events_[i];
        const auto tag = raw_event.tag();

        switch (tag) {
            case PyTrace_CALL:
            case PyTrace_C_CALL:
                {
                    std::string name;
                    CallType call_type;
                    const auto& it = module_map.find(i);
                    if (tag == PyTrace_C_CALL) {
                        name = py::repr(raw_event.breadcrumbs_.arg);
                        call_type = CallType::kCCall;
                    } else if (it != module_map.end()) {
                        std::stringstream loc;
                        loc << "nn.Module: " << py::str(py::handle(it->second).attr("__class__").attr("__name__"));
                        name = loc.str();
                        call_type = CallType::kPyModuleCall;
                    } else {
                        name = code_descriptions_[raw_event.breadcrumbs_.code_description_index].name();
                        call_type = CallType::kPyCall;
                    }

                    stack.emplace_back(
                        raw_event.time(),
                        name,
                        call_type,
                        id_counter++,
                        stack.size() ? stack.back().id_ : 0
                    );

                }
                break;

            case PyTrace_RETURN:
            case PyTrace_EXCEPTION:
            case PyTrace_C_RETURN:
            case PyTrace_C_EXCEPTION:
                // We begin profiling midway through execution, so we may return
                // to a frame above the one where we started profiling.
                if (stack.size()) {
                    stack.back().t1_ = raw_event.time();
                    results.push_back(std::move(stack.back()));
                    stack.pop_back();
                }
                break;

            default:
                break;
        }
    }

    ska::flat_hash_map<size_t, PyTraceEvent*> parent_map {{0, nullptr}};
    std::vector<std::unique_ptr<PyTraceEvent>> out;
    for (auto& r : results) {
        out.push_back(std::make_unique<PyTraceEvent>(
            r.t0_,
            r.t1_,
            r.name_,
            nullptr,
            r.call_type_
        ));
        parent_map.insert({r.id_, out.back().get()});
        // std::cout << r.t0_ << "  " << r.t1_ << "  " << (int)(r.call_type_) << "      "
        //           << r.id_ << "  " << r.parent_id_ << "      " << r.name_ << std::endl;
    }
    for (int i = 0; i < results.size(); i++) {
        out[i]->parent_ = parent_map[results[i].parent_id_];
    }

    for (auto& e : out) {
        print_frame(e.get(), 0);
        std::cout << e.get() << "  " << e->parent_ << std::endl << std::endl;
    }
}

int PyStateCache::_py_profile_fn(
        PyObject* obj,
        PyFrameObject* frame,
        int what,
        PyObject* arg) {

    //uint64_t start_cycle = __builtin_ia32_rdtsc();
    events_.emplace_back(what);
    switch (what) {
        case PyTrace_CALL:
            events_.back().breadcrumbs_.code_description_index = store_description(frame);
            track_module(frame);
            break;

        case PyTrace_C_CALL:
        case PyTrace_C_RETURN:
        case PyTrace_C_EXCEPTION:
            events_.back().breadcrumbs_.arg = arg;
            break;

        case PyTrace_RETURN:
        case PyTrace_EXCEPTION:
            break;

        default:
            break;
    }
    // uint64_t end_cycle = __builtin_ia32_rdtsc();
    // int64_t delta = now() - extract_time(events_.back().tag_and_time_);
    // std::cout << what << "      " << end_cycle - start_cycle << "    " << delta << std::endl;
    return 0;
}












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
                /*line_no_=*/PyCode_Addr2Line(/*f_code=*/f_code, /*f_lasti=*/f_lasti),
                /*filename_=*/THPUtils_unpackString(f_code->co_filename),
                /*funcname_=*/THPUtils_unpackString(f_code->co_name)
            };
            c.insert({ key, desc });
        }
    }

    static void clear_cache() {
        cache().clear();
    }

    std::string name() {
        std::regex e(path_prefixes_);
        std::stringstream loc;
        loc << std::regex_replace(filename_, e, "") << "(" << line_no_ << "): " << funcname_;
        return loc.str();
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
int _py_profile_fn(
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
    PyStateCache::reset();


    PyCodeDescription::clear_cache();
    CallTime::clear_cache();
    CompletedCall::history().clear();
}

void start_profiling() {
    // We have to lazily initialize because some symbols are not yet available
    // when `init()` is called.
    lazy_init();

    clear_state();
    profile_start_time_ = now();

    // Note:
    //   This profile will not compose with other profilers, and cannot be
    //   round tripped via `sys.settrace(sys.gettrace())`
    PyEval_SetProfile(PyStateCache::py_profile_fn, NULL);
    active_ = true;
}

void stop_profiling() {
    TORCH_INTERNAL_ASSERT(active_, "Tracer is running.")

    // Note:
    //   This will clear ALL active profilers. This tracer should not be used
    //   with any other profiler.
    PyEval_SetProfile(NULL, NULL);
    active_ = false;

    PyStateCache::replay_stack();
}

std::vector<PyTraceEvent> get_events() {
    std::vector<PyTraceEvent> out;
    // for (auto& c : CompletedCall::history()) {
    //     PyTraceEvent e { c.t0_, c.t0_ + c.duration_, c.name(), c.call_type() };
    //     out.push_back(e);
    // }
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
