#include <torch/csrc/autograd/profiler_python.h>

#include <iostream>
#include <limits>
#include <memory>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include <Python.h>
#include <frameobject.h>

#include <c10/macros/Macros.h>
#include <c10/util/flat_hash_map.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;


namespace torch { namespace autograd { namespace profiler { namespace python_tracer {


struct TraceContext {
    PyObject_HEAD
    uint8_t thread_id_;
    PyThreadState* thread_state_;
    int64_t initial_us_;
    // TODO: Incremental TSC to skip clock calls.
};

static PyTypeObject TraceContextType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "TraceContext",
    .tp_basicsize = sizeof(TraceContext),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "",
    .tp_new = PyType_GenericNew,
};


enum TraceTag {
    kPy_Call = 0,
    kPy_Return,
    kC_Call,
    kC_Return
};


struct RawEvent {
    RawEvent(TraceTag tag, int lasti, TraceContext* ctx)
            : tag_(static_cast<uint8_t>(tag)),
              thread_id_(ctx->thread_id_),
              lasti_((uint16_t)lasti) {
        int64_t t = now() - ctx->initial_us_;
        t_ = static_cast<uint32_t>(t);

        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(lasti <= std::numeric_limits<uint16_t>::max());
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(t <= std::numeric_limits<uint32_t>::max());
    }

    union Misc {
        // TraceTag::kPy_Call
        PyCodeObject* f_code_;

        // TraceTag::kC_Call
        PyObject* arg_;

        // TraceTag::kPy_Return
        // TraceTag::kC_Return
        // ** Unused (placeholder) **
        void* null;
    };

    uint8_t tag_;
    uint8_t thread_id_;
    uint16_t lasti_;
    uint32_t t_;
    Misc misc_;

    TraceTag tag() const {
        return static_cast<TraceTag>(tag_);
    }

    int lasti() const {
        // f_lasti is positive, with one exception: CPython intializes frames
        // with `f_lasti = -1`. We don't want to give up half of the range by
        // switching to int16_t. So instead we do the fast (underflowing) cast
        // in the ctor, and rectify the value in this accessor which should
        // only be called during trace post processing.
        return lasti_ == std::numeric_limits<uint16_t>::max()
            ? (int)(-1)
            : (int)lasti_;
    }
};

static_assert(sizeof(RawEvent) == 16);


// std::hash doesn't have a specialization for pairs so we have to define one.
// A simple XOR is good enough for our purposes.
struct hash_pair {
    template <class T1, class T2>
    size_t operator() (const std::pair<T1, T2>& pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};


class PythonTracer {
 public:
    static void call(Command c);

    static int py_profile_fn(
        PyObject* obj,
        PyFrameObject* frame,
        int what,
        PyObject* arg);

    static std::vector<std::unique_ptr<PyTraceEvent>> replay_stack() {
        return PythonTracer::singleton()._replay_stack();
    };

 private:
    PythonTracer();
    static PythonTracer& singleton();

    void _start();
    void _stop();
    void _clear();

    int _record_event(
        TraceContext* ctx,
        PyFrameObject* frame,
        TraceTag tag,
        PyObject* arg);

    std::vector<std::unique_ptr<PyTraceEvent>> _replay_stack();

    void store_description(PyFrameObject* frame);
    void track_module(PyFrameObject* frame);

    struct CodeDescription {
        CodeDescription(int line_no, std::string filename, std::string funcname)
            : line_no_(line_no), filename_(filename), funcname_(funcname) {}
        int line_no_;
        std::string filename_;
        std::string funcname_;
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

    bool active_;
    PyObject* module_call_code_;
    std::string path_prefixes_;
    std::vector<TraceContext*> trace_contexts_;

    std::vector<RawEvent> events_;
    std::vector<ModuleForward> module_calls_;

    using DescriptionKey = std::pair</*f_code=*/PyCodeObject*, /*f_lasti=*/int>;
    ska::flat_hash_map<DescriptionKey, CodeDescription, hash_pair> code_descriptions_;
};


int PythonTracer::_record_event(
        TraceContext* ctx,
        PyFrameObject* frame,
        TraceTag tag,
        PyObject* arg) {
    events_.emplace_back(tag, frame->f_lasti, ctx);
    switch (tag) {
        case TraceTag::kPy_Call:
            events_.back().misc_.f_code_ = frame->f_code;
            store_description(frame);
            track_module(frame);
            break;

        case TraceTag::kC_Call:
            events_.back().misc_.arg_ = arg;
            break;

        case TraceTag::kPy_Return:
        case TraceTag::kC_Return:
            break;
    }
    return 0;
}


void PythonTracer::store_description(PyFrameObject* frame) {
    const auto& it = code_descriptions_.find({ frame->f_code, frame->f_lasti });
    if C10_UNLIKELY(it == code_descriptions_.end()) {
        code_descriptions_.insert({
            { frame->f_code, frame->f_lasti },
            {
                /*line_no_=*/PyCode_Addr2Line(frame->f_code, frame->f_lasti),
                /*filename_=*/THPUtils_unpackString(frame->f_code->co_filename),
                /*funcname_=*/THPUtils_unpackString(frame->f_code->co_name)
            }
        });
    }
}


void PythonTracer::track_module(PyFrameObject* frame) {
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


struct PythonTracer::ReplayFrame {
    int64_t t0_;
    int64_t t1_;
    std::string name_;
    CallType call_type_;
    size_t id_;
    size_t parent_id_;
    uint64_t thread_id_;
    int32_t t0_epsilon_;
    int32_t t1_epsilon_;
};


std::vector<std::unique_ptr<PyTraceEvent>> PythonTracer::_replay_stack() {
    ska::flat_hash_map<size_t, std::string> module_name_map;
    for (const auto& call : module_calls_) {
        std::stringstream name_stream;
        auto py_class_name = py::handle(call.self_)
            .attr("__class__")
            .attr("__name__");
        name_stream << "nn.Module: " << py::str(py_class_name);
        module_name_map.insert({ call.event_index_, name_stream.str() });
    }

    // Pruning the path prefix is somewhat expensive, so we cache it.
    std::regex filename_prune(path_prefixes_);
    ska::flat_hash_map<std::string, std::string> filename_map;

    auto py_name = [&](const RawEvent& e) {
        const auto& desc_it = code_descriptions_.find({e.misc_.f_code_, e.lasti()});
        if (desc_it != code_descriptions_.end()) {
            if (filename_map.find(desc_it->second.filename_) == filename_map.end()) {
                auto s = std::regex_replace(desc_it->second.filename_, filename_prune, "");
                filename_map[desc_it->second.filename_] = s;
            }
            std::stringstream name_stream;
            name_stream << filename_map.at(desc_it->second.filename_) << "("
                        << desc_it->second.line_no_ << "): " << desc_it->second.funcname_;
            return name_stream.str();
        }
        return std::string("Python: ???");
    };

    size_t id_counter = 0;
    std::vector<std::vector<ReplayFrame>> stacks(trace_contexts_.size());
    std::vector<ReplayFrame> results;

    int32_t epsilon = 0;
    int64_t t_prior = 0;
    for (size_t i = 0; i < events_.size(); i++) {
        auto raw_event = events_[i];
        auto& stack = stacks[raw_event.thread_id_];
        auto ctx = trace_contexts_[raw_event.thread_id_];
        auto t = static_cast<int64_t>(raw_event.t_) + ctx->initial_us_;

        // In theory t0_ and t1_ should be enough to convert back to a stream
        // of events. Unfortunately we only record microseconds, so events
        // which round to the same timestamp cannot be properly ordered when
        // sorting events. To address this issue we also include an epsilon
        // counter to disambiguate the order of events with the same timestamp.
        // This will be obsolete if we track `__rdtsc`.
        epsilon = (t == t_prior ? epsilon + 1 : 0);
        t_prior = t;

        auto push_frame = [&](std::string name, CallType call_type) {
            ReplayFrame frame {
                .t0_ = t,
                .t1_ = -1,  // Placeholder
                .name_ = name,
                .call_type_ = call_type,
                .id_ = id_counter++,
                .parent_id_ = stack.size() ? stack.back().id_ : 0,
                .thread_id_ = raw_event.thread_id_,
                .t0_epsilon_ = epsilon,
                .t1_epsilon_ = 0
            };
            stack.push_back(frame);
        };

        switch (raw_event.tag()) {
            case TraceTag::kPy_Call:
                if (module_name_map.find(i) != module_name_map.end()) {
                    push_frame(module_name_map.at(i), CallType::kPyModuleCall);
                } else {
                    push_frame(py_name(raw_event), CallType::kPyCall);
                }
                break;

            case TraceTag::kC_Call:
                push_frame(py::repr(raw_event.misc_.arg_), CallType::kCCall);
                break;

            case TraceTag::kPy_Return:
            case TraceTag::kC_Return:
                TORCH_INTERNAL_ASSERT(stack.size(), "Python replay stack is empty.")
                stack.back().t1_ = t;
                stack.back().t1_epsilon_ = epsilon;
                results.push_back(std::move(stack.back()));
                stack.pop_back();
                break;
        }
    }

    ska::flat_hash_map<size_t, PyTraceEvent*> event_id_map {{0, nullptr}};
    std::vector<std::unique_ptr<PyTraceEvent>> out;
    for (auto& r : results) {
        out.push_back(std::unique_ptr<PyTraceEvent>(
            new PyTraceEvent {
                .t0_ = r.t0_,
                .t1_ = r.t1_,
                .name_ = r.name_,
                .thread_id_ = r.thread_id_,
                .parent_ = nullptr,
                .call_type_ = r.call_type_,
                .t0_epsilon_ = r.t0_epsilon_,
                .t1_epsilon_ = r.t1_epsilon_
            }
        ));
        event_id_map.insert({r.id_, out.back().get()});
    }

    for (int i = 0; i < results.size(); i++) {
        out[i]->parent_ = event_id_map[results[i].parent_id_];
    }

    return out;
}


// ============================================================================
// == API and interfaces ======================================================
// ============================================================================

PythonTracer& PythonTracer::singleton() {
    static PythonTracer singleton_;
    return singleton_;
}


PythonTracer::PythonTracer() {
    path_prefixes_ = py::module::import("torch.profiler.python_tracer")
        .attr("_prefix_regex")().cast<std::string>();

    module_call_code_ = py::module::import("torch.nn")
        .attr("Module")
        .attr("__call__")
        .attr("__code__")
        .ptr();
}


void PythonTracer::_start() {
    TORCH_CHECK(!active_, "PythonTracer is already active")
    TORCH_CHECK(!trace_contexts_.size(), "PythonTracer should not have active contexts");

    pybind11::gil_scoped_acquire gil;
    auto t0 = now();

    // Loop over all interpreters and all threads within each interpreter.
    // We will need to register a trace function with each thread. We set the
    // current thread to position zero to ensure that it is traced, and so we
    // can restore the thread state after registration.
    std::vector<PyThreadState*> thread_states { PyThreadState_Get() };
    PyInterpreterState* interpreter_state = PyInterpreterState_Head();
    while (interpreter_state != nullptr) {
        PyThreadState* thread_state = PyInterpreterState_ThreadHead(interpreter_state);
        while (thread_state != nullptr) {
            if (thread_state != thread_states[0]) {
                thread_states.push_back(thread_state);
            }
            thread_state = PyThreadState_Next(thread_state);
        }
        interpreter_state = PyInterpreterState_Next(interpreter_state);
    }

    // Check that we don't overrun the (quite generous) limit of 255 threads.
    constexpr size_t max_tid = std::numeric_limits<uint8_t>::max();
    if (thread_states.size() > max_tid) {
        std::cout << "Warning: can only trace " << max_tid + 1 << " threads. "
                  << thread_states.size() << " are currently active." << std::endl;
        thread_states.resize(max_tid);
    }

    // Register the tracer in each thread.
    for (size_t i = 0; i < thread_states.size(); i++) {
        PyThreadState* thread_state = thread_states[i];
        auto ctx = (TraceContext*) TraceContextType.tp_alloc(&TraceContextType, 0);
        ctx->thread_id_ = (uint8_t)i;
        ctx->thread_state_ = thread_state;
        ctx->initial_us_ = t0;
        trace_contexts_.push_back(ctx);

        PyThreadState_Swap(thread_state);

        // When we begin profiling there are already frames on the Python
        // interpreter stack. To ensure a complete trace, we must push calls
        // to all the prior frames onto our event stack. (We stop at depth=128)
        std::vector<PyFrameObject*> current_stack;
        auto frame = PyEval_GetFrame();
        size_t depth = 0;
        while (frame != nullptr && depth <= 128) {
            current_stack.push_back(frame);
            frame = frame->f_back;
            depth++;
        }
        for (auto it = current_stack.rbegin(); it != current_stack.rend(); it++) {
            _record_event(ctx, *it, TraceTag::kPy_Call, /*arg=*/Py_None);
        }

        // Note:
        //   This profile will not compose with other profilers, and cannot be
        //   round tripped via `sys.settrace(sys.gettrace())`
        PyEval_SetProfile(PythonTracer::py_profile_fn, (PyObject*)ctx);
    }

    // Restore the thread state to its initial value.
    PyThreadState_Swap(thread_states[0]);

    active_ = true;
};


void PythonTracer::_stop() {
    TORCH_INTERNAL_ASSERT(active_, "PythonTracer is not running.")

    pybind11::gil_scoped_acquire gil;

    PyThreadState* initial_thread_state = PyThreadState_Get();
    for (const auto i : trace_contexts_) {
        PyThreadState_Swap(i->thread_state_);
        PyEval_SetProfile(NULL, NULL);
    }
    PyThreadState_Swap(initial_thread_state);
    active_ = false;
}


void PythonTracer::_clear() {
    TORCH_CHECK(!active_, "Cannot clear state while PythonTracer is active.");
    for (auto i : trace_contexts_) {
        Py_DECREF((PyObject*) i);
    }
    trace_contexts_.clear();
    events_.clear();
    code_descriptions_.clear();
    for (auto& i : module_calls_) {
        Py_DECREF(i.self_);
    }
    module_calls_.clear();
}


int PythonTracer::py_profile_fn(
        PyObject* obj,
        PyFrameObject* frame,
        int what,
        PyObject* arg) {
    switch (what) {
        case PyTrace_CALL:
            return PythonTracer::singleton()._record_event(
                (TraceContext*) obj, frame, TraceTag::kPy_Call, arg);

        case PyTrace_EXCEPTION:
        case PyTrace_RETURN:
            return PythonTracer::singleton()._record_event(
                (TraceContext*) obj, frame, TraceTag::kPy_Return, arg);

        case PyTrace_C_CALL:
            return PythonTracer::singleton()._record_event(
                (TraceContext*) obj, frame, TraceTag::kC_Call, arg);

        case PyTrace_C_EXCEPTION:
        case PyTrace_C_RETURN:
            return PythonTracer::singleton()._record_event(
                (TraceContext*) obj, frame, TraceTag::kC_Return, arg);

        default:
            return 0;
    }
}


void PythonTracer::call(Command c) {
    switch (c) {
        case Command::kStart:
            PythonTracer::singleton()._start();
            break;

        case Command::kStop:
            PythonTracer::singleton()._stop();
            break;

        case Command::kClear:
        default:
            break;
    }
};


void init() {
    TORCH_CHECK(PyType_Ready(&TraceContextType) == 0);

    registerFunctions(
        /*call=*/&PythonTracer::call,
        /*get_events=*/&PythonTracer::replay_stack
    );
}

}}}} // namespace torch::autograd::profiler::python_tracer
