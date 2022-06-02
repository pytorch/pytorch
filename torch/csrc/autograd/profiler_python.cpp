#include <torch/csrc/autograd/profiler_python.h>

#include <cstdint>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <Python.h>
#include <frameobject.h>

#include <c10/macros/Macros.h>
#include <c10/util/C++17.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <c10/util/strong_type.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/containers.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

namespace torch {
namespace autograd {
namespace profiler {
namespace {
enum CallType { PyCall = 0, PyModuleCall, PyCCall };
static constexpr size_t CallTypeSize = 3;

using torch::profiler::impl::AppendOnlyList;
using torch::profiler::impl::python_tracer::PythonTracerBase;
using torch::profiler::impl::python_tracer::PyTraceEvent;

// ============================================================================
// == Miscellaneous structs and utils =========================================
// ============================================================================
struct PyFrameState {
  int line_no_;
  at::StringView filename_;
  at::StringView funcname_;
};

template <typename T, typename Tag>
using strong_t = strong::
    type<T, Tag, strong::regular, strong::convertible_to<T>, strong::hashable>;

using PyModuleSelf = strong_t<PyObject*, struct PyModuleSelf_>;
using PyModuleCls = strong_t<PyObject*, struct PyModuleCls_>;
using PyCFunction = strong_t<PyObject*, struct PyCFunction_>;

struct CodeLocation {
  CodeLocation() = default;
  explicit CodeLocation(const PyFrameObject* frame)
      : code_{frame->f_code}, lasti_{frame->f_lasti} {}

  bool operator==(const CodeLocation& other) const {
    return code_ == other.code_ && lasti_ == other.lasti_;
  }

  PyCodeObject* code_{nullptr};
  int lasti_{0};
};

// Temporary struct. This will be replaced by ExtraFields<EventType>.
struct FrameArgs {
  std::string name_;
  CallType call_type_;
  c10::optional<std::pair<PyModuleSelf, PyModuleCls>> module_;
  c10::optional<size_t> module_id_;
};

PyObject* nnModuleCode() {
  static auto module_call_code = py::module::import("torch.nn")
      .attr("Module")
      .attr("__call__")
      .attr("__code__")
      .ptr();
  return module_call_code;
}

} // namespace
} // namespace profiler
} // namespace autograd
} // namespace torch

template <>
struct std::hash<torch::autograd::profiler::CodeLocation> {
  size_t operator()(const torch::autograd::profiler::CodeLocation& x) {
    return c10::get_hash(x.code_, x.lasti_);
  }
};

namespace torch {
namespace autograd {
namespace profiler {
namespace python_tracer {
namespace {

// ============================================================================
// == CallTypeHelper: Tools for generic programming on specializations. =======
// ============================================================================
template <template <CallType> class ClassT>
class CallTypeHelper final {
 private:
  static_assert(
      CallType::PyCall == 0,
      "CallTypeHelper uses integer math which depends on a zero start.");
  static constexpr size_t End = CallTypeSize;

  template <size_t... I>
  static constexpr std::tuple<ClassT<(CallType)I>...> make_tuple_impl(
      std::index_sequence<I...>);

  template <size_t C, typename T, typename FunctorT, typename... Args>
  static void map(T& t, FunctorT& f, Args... args) {
    f(std::get<C>(t), args...);
    c10::guts::if_constexpr<C + 1 < End>(
        [&](auto _) { map<C + 1>(_(t), f, std::forward<Args>(args)...); },
        [&](auto _) {});
  }

 public:
  using tuple_type = decltype(make_tuple_impl(std::make_index_sequence<End>{}));

  template <typename FunctorT, typename... Args>
  static void map(tuple_type& t, FunctorT& f, Args... args) {
    map<0>(t, f, std::forward<Args>(args)...);
  }
};

// ============================================================================
// == Event type definitions. =================================================
// ============================================================================
// When we are tracing a Python program, the general procedure is to record
// every time we enter or exit a function and later replay these events during
// post processing. Thus, during the profiling phase we want to do the MINIMAL
// amount of work to capture all of the information that we need; otherwise we
// will distort the profile. (While we don't wish to be terribly inefficient
// during post processing, we are willing to do extra fixup work in post if it
// reduces overhead in the profiling phase.)
//
// When the tracer first enters a frame, it constructs a CallKey for that
// location. The contents of the key vary by context. For a python function
// the key is the (PyCodeObject*, int) pair that defines the bytecode of the
// function. For an `nn.Module` the key is a (non-owning) pointer to `self`.
// For a bound C function it is a (non-owning) pointer to the bound function.
// A CallKey should be small, inexpensive, and POD.
//
// We then collect a CallKey<CallType::PyCall> for the calling frame for better
// source tracking. This pair is a `Callsite`, and serves as a first level key
// during tracing. We lookup the Callsite in a thread local cache which maps
// Callsite to a unique integer `TraceKey`. On a cache hit, we simply store the
// TraceKey and return. On a cache miss, we use a global value cache to store
// whatever fields we need from the two CallKeys, generate a new TraceKey, and
// update the local cache.
//
// During post processing we:
//   1) Determine the type represented by a TraceKey by checking which
//      sub-cache it appears in in the thread local cache.
//   2) Look up the pair of CallKeys from the thread local cache.
//   3) Look up the expanded values of each CallKey from the global value cache.
//
// To add a new event type to the cache:
//   1) Add an entry to the `CallType` enum.
//   2) Add a specialization of Config which defined key_t and cache_t.
//   3) Add a specialization of ValueCache::store and ValueCache::load.

template<CallType>
struct Config;

template<>
struct Config<CallType::PyCall> {
  using key_t = CodeLocation;
  using cache_t = ska::flat_hash_map<key_t, PyFrameState>;
};

template <>
struct Config<CallType::PyModuleCall> {
  using key_t = PyModuleSelf;
  struct cache_t {
    c10::optional<CodeLocation> module_forward_;
    ska::flat_hash_map<PyModuleSelf, PyModuleCls> modules_;
    ska::flat_hash_map<PyModuleCls, at::StringView> module_cls_names_;
  };
};

template<>
struct Config<CallType::PyCCall> {
  using key_t = PyCFunction;
  using cache_t = ska::flat_hash_map<key_t, at::StringView>;
};

// ============================================================================
// == Callsite & ValueCache: Storage during profiling =========================
// ============================================================================
template <CallType C>
class Callsite {
 public:
  static constexpr CallType call_type = C;
  using key_t = typename Config<C>::key_t;

  static_assert(
      std::is_trivially_copyable<key_t>::value,
      "Key should be trivial, as it is passed by value.");

  template <typename U>
  Callsite(U value, const PyFrameObject* f_back)
      : value_(value), caller_(f_back) {}

  bool operator==(const Callsite<C>& other) const {
    return value_ == other.value_ && caller_ == other.caller_;
  }

  key_t value_;
  Config<CallType::PyCall>::key_t caller_;
};

class ValueCache {
 public:
  template <CallType C>
  void store(const typename Config<C>::key_t);

  template <CallType C>
  auto load(Callsite<C> callsite) {
    // NB: For now caller is dropped. It will be used in the next PR.
    return load<C>(callsite.value_);
  }

  void trimPrefixes();

 private:
  template <CallType C>
  FrameArgs load(const typename Config<C>::key_t) const;

  template <CallType C>
  using State = typename Config<C>::cache_t;

  CallTypeHelper<State>::tuple_type state_;
};

// ============================================================================
// == Type specific store and load implementations. ===========================
// ============================================================================
using PyCallKey = Config<CallType::PyCall>::key_t;
using PyModuleCallKey = Config<CallType::PyModuleCall>::key_t;
using PyCCallKey = Config<CallType::PyCCall>::key_t;

template <>
void ValueCache::store<CallType::PyCall>(const PyCallKey key) {
  auto& locations = std::get<CallType::PyCall>(state_);
  if (C10_UNLIKELY(locations.find(key) == locations.end())) {
    TORCH_INTERNAL_ASSERT(key.code_ != nullptr);
    locations[key] = {
        PyCode_Addr2Line(key.code_, key.lasti_),
        at::StringView(THPUtils_unpackString(key.code_->co_filename)),
        at::StringView(THPUtils_unpackString(key.code_->co_name))};
  }
}

template <>
FrameArgs ValueCache::load<CallType::PyCall>(const PyCallKey key) const {
  auto frame_state = std::get<CallType::PyCall>(state_).at(key);
  return {
      fmt::format(
          "{}({}): {}",
          frame_state.filename_.str(),
          frame_state.line_no_,
          frame_state.funcname_.str()),
      CallType::PyCall};
}

template <>
void ValueCache::store<CallType::PyModuleCall>(const PyModuleCallKey key) {
  auto& cache = std::get<CallType::PyModuleCall>(state_);
  if (C10_UNLIKELY(cache.modules_.find(key) == cache.modules_.end())) {
    if (C10_UNLIKELY(!cache.module_forward_.has_value())) {
      auto frame = PyEval_GetFrame();
      TORCH_INTERNAL_ASSERT((PyObject*)(frame->f_code) == nnModuleCode());
      cache.module_forward_ = PyCallKey(frame);
      store<CallType::PyCall>(*cache.module_forward_);
    }
    auto cls_handle = py::handle((PyObject*)key).attr("__class__");
    auto cls = PyModuleCls(cls_handle.ptr());
    cache.modules_[key] = cls;

    if (cache.module_cls_names_.find(cls) == cache.module_cls_names_.end()) {
      cache.module_cls_names_[cls] =
          at::StringView(py::str(cls_handle.attr("__name__")));
    }
  }
}

template <>
FrameArgs ValueCache::load<CallType::PyModuleCall>(
    const PyModuleCallKey key) const {
  auto& cache = std::get<CallType::PyModuleCall>(state_);
  auto cls = cache.modules_.at(key);

  // NB: For now fwd is not used.
  // TORCH_INTERNAL_ASSERT(cache.module_forward_.has_value());
  // auto fwd = std::get<CallType::PyCall>(state_).at(*cache.module_forward_);

  return {
      fmt::format("nn.Module: {}", cache.module_cls_names_.at(cls).str()),
      CallType::PyModuleCall,
      std::make_pair(key, cls)};
}

template <>
void ValueCache::store<CallType::PyCCall>(PyCCallKey key) {
  auto& names = std::get<CallType::PyCCall>(state_);
  if (C10_UNLIKELY(names.find(key) == names.end())) {
    names[key] = at::StringView(py::repr((PyObject*)key));
  }
}

template <>
FrameArgs ValueCache::load<CallType::PyCCall>(const PyCCallKey key) const {
  return {std::get<CallType::PyCCall>(state_).at(key).str(), CallType::PyCCall};
}

// TODO: Use re2.
void ValueCache::trimPrefixes() {
  static auto prefixes = py::module::import("torch.profiler.python_tracer")
    .attr("_prefix_regex")().cast<std::vector<std::string>>();

  for (auto& it : std::get<CallType::PyCall>(state_)) {
    std::string filename = it.second.filename_.str();
    for (const auto& p : prefixes) {
      if (filename.compare(0, p.size(), p) == 0) {
        filename.erase(0, p.size());
        it.second.filename_ = at::StringView(filename);
        break;
      }
    }
  }
}

// ============================================================================
// == TraceKey cache ==========================================================
// ============================================================================
using TraceKey =
    strong::type<uint64_t, struct TraceKey_, strong::regular, strong::hashable>;

TraceKey nextKey() {
  static std::atomic<uint64_t> key{0};
  return TraceKey{++key};
}

template <CallType C>
struct TraceKeyCacheState {
  struct Hash {
    size_t operator()(const Callsite<C>& key) {
      return c10::get_hash(key.value_, key.caller_);
    }
  };

  TraceKey intern(Callsite<C> callsite, ValueCache& value_cache) {
    auto it = state_.find(callsite);
    if (C10_UNLIKELY(it == state_.end())) {
      value_cache.store<C>(callsite.value_);
      value_cache.store<CallType::PyCall>(callsite.caller_);
      it = state_.insert({callsite, nextKey()}).first;

    }
    return it->second;
  }

  auto lookup(Callsite<C>& callsite, ValueCache& value_cache) const {
    return std::make_pair(
        value_cache.load<C>(callsite.value_),
        value_cache.load<CallType::PyCall>(callsite.caller_));
  }

  ska::flat_hash_map<Callsite<C>, TraceKey, Hash> state_;
};

// ============================================================================
// == Core CPython data types =================================================
// ============================================================================
// PyObject that allows different threads to record events without colliding.
// It is passed as the second argument when enabling tracing via
// `PyEval_SetProfile`.
struct ThreadLocalResults;
struct TraceContext {
  PyObject_HEAD
  ThreadLocalResults* thread_local_results_;
};

// CPython boilerplate to define `TraceContext` as a proper python object.
static PyTypeObject TraceContextType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "TraceContext",             /* tp_name */
  sizeof(TraceContext),       /* tp_basicsize */
  0,                          /* tp_itemsize */
  nullptr,                    /* tp_dealloc */
  0,                          /* tp_vectorcall_offset */  // NOLINT: modernize-use-nullptr
  nullptr,                    /* tp_getattr */
  nullptr,                    /* tp_setattr */
  nullptr,                    /* tp_reserved */
  nullptr,                    /* tp_repr */
  nullptr,                    /* tp_as_number */
  nullptr,                    /* tp_as_sequence */
  nullptr,                    /* tp_as_mapping */
  nullptr,                    /* tp_hash  */
  nullptr,                    /* tp_call */
  nullptr,                    /* tp_str */
  nullptr,                    /* tp_getattro */
  nullptr,                    /* tp_setattro */
  nullptr,                    /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,         /* tp_flags */
  "Python tracer TLS",        /* tp_doc */
  nullptr,                    /* tp_traverse */
  nullptr,                    /* tp_clear */
  nullptr,                    /* tp_richcompare */
  0,                          /* tp_weaklistoffset */
  nullptr,                    /* tp_iter */
  nullptr,                    /* tp_iternext */
  nullptr,                    /* tp_methods */
  nullptr,                    /* tp_members */
  nullptr,                    /* tp_getset */
  nullptr,                    /* tp_base */
  nullptr,                    /* tp_dict */
  nullptr,                    /* tp_descr_get */
  nullptr,                    /* tp_descr_set */
  0,                          /* tp_dictoffset */
  nullptr,                    /* tp_init */
  nullptr,                    /* tp_alloc */
  PyType_GenericNew,          /* tp_new */
  nullptr                     /* tp_free */
};

// ============================================================================
// == Thread local cache ======================================================
// ============================================================================
struct ThreadLocalResults {
  ThreadLocalResults(PyThreadState* thread_state, ValueCache* value_cache)
      : thread_state_{thread_state},
        ctx_{(TraceContext*)TraceContextType.tp_alloc(&TraceContextType, 0)},
        value_cache_{value_cache} {
    ctx_->thread_local_results_ = this;
  }

  ThreadLocalResults() = delete;
  ThreadLocalResults(const ThreadLocalResults&) = delete;
  ThreadLocalResults(ThreadLocalResults&&) = delete;
  ThreadLocalResults& operator=(const ThreadLocalResults&) = delete;

  ~ThreadLocalResults() {
    Py_DECREF((PyObject*)ctx_);
  }

  template <CallType C, typename... Args>
  TraceKey intern(Args... args) {
    return std::get<C>(trace_keys_)
        .intern(Callsite<C>(std::forward<Args>(args)...), *value_cache_);
  }

  static constexpr size_t BLOCK_SIZE = 1024;

  PyThreadState* thread_state_;
  TraceContext* ctx_;
  ValueCache* value_cache_;
  CallTypeHelper<TraceKeyCacheState>::tuple_type trace_keys_;
  AppendOnlyList<std::pair<TraceKey, int64_t>, BLOCK_SIZE> enters_;
  AppendOnlyList<int64_t, BLOCK_SIZE> exit_times_;
  AppendOnlyList<int64_t, BLOCK_SIZE> c_exit_times_;
};

// ============================================================================
// == Tracing implementation ==================================================
// ============================================================================
class PythonTracer final : public PythonTracerBase {
 public:
  static int pyProfileFn(
      PyObject* obj,
      PyFrameObject* frame,
      int what,
      PyObject* arg);

  static PythonTracer& singleton();
  void start() override;
  void stop() override;
  std::vector<std::unique_ptr<PyTraceEvent>> getEvents() override;
  void clear() override;

 private:
  PythonTracer();
  friend class PyTraceReplay;

  void recordPyCall(ThreadLocalResults& tls, PyFrameObject* frame);
  void recordCCall(ThreadLocalResults& tls, PyFrameObject* frame, PyObject* arg);

  bool active_;
  PyObject* module_call_code_;

  // TODO: Move to RecordQueue
  std::deque<ThreadLocalResults> thread_local_results_;
  ValueCache value_cache_;
};

PythonTracer& PythonTracer::singleton() {
  static PythonTracer singleton_;
  return singleton_;
}

PythonTracer::PythonTracer()
    : active_(false), module_call_code_(nnModuleCode()) {}

void PythonTracer::start() {
  TORCH_CHECK(!active_, "PythonTracer is already active")
  TORCH_CHECK(
      !thread_local_results_.size(),
      "PythonTracer should not have active contexts");

  pybind11::gil_scoped_acquire gil;
  auto t0 = now();

  // Loop over all threads within the current interpreter. We will need to
  // register a trace function with each thread. We set the current thread to
  // position zero to ensure that it is traced, and so we can restore the
  // thread state after registration.
  std::vector<PyThreadState*> thread_states{PyThreadState_Get()};
  /*
  if (all_threads) {
    auto thread_state = thread_states[0];
    while (thread_state != nullptr) {
      if (thread_state != thread_states[0]) {
        thread_states.push_back(thread_state);
      }
      thread_state = PyThreadState_Next(thread_state);
    }
  }
  */

  // Register the tracer in each thread.
  for (const auto i : c10::irange(thread_states.size())) {
    PyThreadState* thread_state = thread_states[i];
    PyThreadState_Swap(thread_state);

    thread_local_results_.emplace_back(thread_state, &value_cache_);
    auto* ctx = thread_local_results_.back().ctx_;

    // When we begin profiling there are already frames on the Python
    // interpreter stack. To ensure a complete trace, we must push calls
    // to all the prior frames onto our event stack. (We stop at depth=128)
    std::vector<PyFrameObject*> current_stack;
    auto frame = PyEval_GetFrame();
    size_t depth = 0; // Make sure we can't infinite loop.
    while (frame != nullptr && depth <= 128) {
      current_stack.push_back(frame);
      frame = frame->f_back;
      depth++;
    }
    for (auto it = current_stack.rbegin(); it != current_stack.rend(); it++) {
      recordPyCall(thread_local_results_.back(), *it);
    }

    // Note:
    //   This profile will not compose with other CPython profilers, and
    //   cannot be round tripped via `sys.settrace(sys.gettrace())`
    PyEval_SetProfile(PythonTracer::pyProfileFn, (PyObject*)ctx);
  }

  // Restore the thread state to its initial value.
  PyThreadState_Swap(thread_states[0]);

  active_ = true;
};

void PythonTracer::stop() {
  TORCH_INTERNAL_ASSERT(active_, "PythonTracer is not running.")

  pybind11::gil_scoped_acquire gil;

  PyThreadState* initial_thread_state = PyThreadState_Get();
  for (const auto& i : thread_local_results_) {
    PyThreadState_Swap(i.thread_state_);
    PyEval_SetProfile(nullptr, nullptr);
  }
  PyThreadState_Swap(initial_thread_state);
  active_ = false;
}

void PythonTracer::clear() {
  TORCH_CHECK(!active_, "Cannot clear state while PythonTracer is active.");
  thread_local_results_.clear();
  value_cache_ = ValueCache();
}

void PythonTracer::recordPyCall(ThreadLocalResults& tls, PyFrameObject* frame) {
  auto get_key = [&]() -> TraceKey {
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
      PyFrame_LocalsToFast(frame, 0);
      TORCH_INTERNAL_ASSERT(frame->f_back != nullptr);
      return tls.intern<CallType::PyModuleCall>(self, frame->f_back);

    } else {
      auto f_back = frame->f_back != nullptr ? frame->f_back : frame;
      return tls.intern<CallType::PyCall>(frame, f_back);
    }
  };
  tls.enters_.emplace_back(get_key(), now());
}

void PythonTracer::recordCCall(
    ThreadLocalResults& tls,
    PyFrameObject* frame,
    PyObject* arg) {
  // NB: For C calls a new frame is not created, so we use `frame` rather than
  //     `frame->f_back`.
  tls.enters_.emplace_back(tls.intern<CallType::PyCCall>(arg, frame), now());
}

// ============================================================================
// == Post processing =========================================================
// ============================================================================

class PyTraceReplay {
 public:
  static std::vector<std::unique_ptr<PyTraceEvent>> getEvents() {
    return PyTraceReplay().replayStack();
  }

 private:
  PyTraceReplay();
  std::vector<std::unique_ptr<PyTraceEvent>> replayStack() const;

  struct RawEvent {
    int64_t t_;
    size_t thread_id_;
    TraceKey key_;
    int what_;
  };

  struct ReplayFrame {
    std::unique_ptr<PyTraceEvent> event_;
    size_t id_;
    size_t parent_id_;
  };

  ska::flat_hash_map<TraceKey, FrameArgs> frame_args_;
  std::vector<RawEvent> raw_events_;
};

PyTraceReplay::PyTraceReplay() {
  auto& tracer = PythonTracer::singleton();
  tracer.value_cache_.trimPrefixes();

  ska::flat_hash_map<PyModuleCallKey, size_t> self_to_id;
  ska::flat_hash_map<PyModuleCls, size_t> cls_id_counter;

  for (auto& local_results : tracer.thread_local_results_) {
    auto f = [&](auto& cache) {
      for (const auto& it : cache.state_) {
        auto frame = tracer.value_cache_.load(it.first);
        if (frame.module_.has_value()) {
          auto it = self_to_id.find(frame.module_->first);
          if (it == self_to_id.end()) {
            auto id = cls_id_counter[frame.module_->second]++;
            it = self_to_id.insert({frame.module_->first, id}).first;
          }
          frame.module_id_ = it->second;
        }
        auto inserted = frame_args_.insert({it.second, frame});
        TORCH_INTERNAL_ASSERT(inserted.second);
      }
    };
    CallTypeHelper<TraceKeyCacheState>::map(local_results.trace_keys_, f);
  }

  for (const auto py_tid : c10::irange(tracer.thread_local_results_.size())) {
    auto& local_results = tracer.thread_local_results_[py_tid];
    for (const auto& i : local_results.exit_times_) {
      raw_events_.push_back({i, py_tid, TraceKey(), PyTrace_RETURN});
    }
    for (const auto& i : local_results.c_exit_times_) {
      raw_events_.push_back({i, py_tid, TraceKey(), PyTrace_C_RETURN});
    }

    for (const auto& it : local_results.enters_) {
      auto call_type = frame_args_.at(it.first).call_type_;
      auto what =
          call_type == CallType::PyCCall ? PyTrace_C_CALL : PyTrace_CALL;
      raw_events_.push_back({it.second, py_tid, it.first, what});
    }
  }
  std::stable_sort(
      raw_events_.begin(), raw_events_.end(), [](const auto& a, const auto& b) {
        return a.t_ < b.t_;
      });
}

std::vector<std::unique_ptr<PyTraceEvent>> PyTraceReplay::replayStack() const {
  auto& tracer = PythonTracer::singleton();
  size_t id_counter = 0;
  std::vector<std::vector<ReplayFrame>> stacks(tracer.thread_local_results_.size());
  std::vector<ReplayFrame> results;

  // Match calls and returns.
  size_t event_idx = 0;
  for (auto& raw_event : raw_events_) {
    auto& stack = stacks[raw_event.thread_id_];
    auto push_frame =
        [&]() {
          auto& args = frame_args_.at(raw_event.key_);
          stack.push_back(ReplayFrame{
              /*event_=*/std::make_unique<PyTraceEvent>(PyTraceEvent{
                  /*startTime_=*/raw_event.t_,
                  /*endTime_=*/-1, // Placeholder
                  /*name_=*/args.name_,
                  /*thread_id_=*/raw_event.thread_id_,
                  /*parent_=*/nullptr, // Placeholder
                  /*module_id_=*/args.module_id_,
                  /*call_idx_=*/event_idx,
                  /*return_idx_=*/0 // Placeholder
              }),
              /*id_=*/id_counter++,
              /*parent_id_=*/stack.size() ? stack.back().id_ : 0,
          });
        };

    switch (raw_event.what_) {
      case PyTrace_CALL:
      case PyTrace_C_CALL:
        push_frame();
        break;

      case PyTrace_RETURN:
      case PyTrace_C_RETURN:
        TORCH_INTERNAL_ASSERT(stack.size(), "Python replay stack is empty.")
        stack.back().event_->endTime_ = raw_event.t_;
        stack.back().event_->return_idx_ = event_idx;
        results.push_back(std::move(stack.back()));
        stack.pop_back();
        break;
    }
    event_idx++;
  }

  // Cleanup by feining return to close out the stack. This is needed so
  // frames above the one that called the profiler still appear in the trace.
  const auto t_final = now();
  for (auto& stack : stacks) {
    while (stack.size()) {
      stack.back().event_->endTime_ = t_final;
      stack.back().event_->return_idx_ = event_idx;
      results.push_back(std::move(stack.back()));
      stack.pop_back();
      event_idx++;
    }
  }

  // Convert to `PyTraceEvent`, and map id to pointer.
  ska::flat_hash_map<size_t, PyTraceEvent*> event_id_map{{0, nullptr}};
  std::vector<std::unique_ptr<PyTraceEvent>> out;
  for (auto& r : results) {
    out.push_back(std::move(r.event_));
    event_id_map.insert({r.id_, out.back().get()});
  }

  // Link parents to children.
  for (const auto i : c10::irange(results.size())) {
    out[i]->parent_ = event_id_map.at(results[i].parent_id_);
  }
  return out;
}

std::vector<std::unique_ptr<PyTraceEvent>> PythonTracer::getEvents() {
  return PyTraceReplay::getEvents();
}

// ============================================================================
// == API =====================================================================
// ============================================================================
int PythonTracer::pyProfileFn(
    PyObject* obj,
    PyFrameObject* frame,
    int what,
    PyObject* arg) {
  auto& local_results =
      *reinterpret_cast<TraceContext*>(obj)->thread_local_results_;
  switch (what) {
    case PyTrace_CALL:
      PythonTracer::singleton().recordPyCall(local_results, frame);
      break;

    case PyTrace_C_CALL:
      PythonTracer::singleton().recordCCall(local_results, frame, arg);
      break;

    case PyTrace_EXCEPTION:
    case PyTrace_RETURN:
      local_results.exit_times_.emplace_back(now());
      break;

    case PyTrace_C_EXCEPTION:
    case PyTrace_C_RETURN:
      local_results.c_exit_times_.emplace_back(now());
      break;
  }
  return 0;
}

PythonTracerBase& getTracer() {
  return PythonTracer::singleton();
}
} // namespace

void init() {
  pybind11::gil_scoped_acquire gil;
  TORCH_CHECK(PyType_Ready(&TraceContextType) == 0);
  torch::profiler::impl::python_tracer::registerTracer(&getTracer);
}
} // namespace python_tracer
} // namespace profiler
} // namespace autograd
} // namespace torch
