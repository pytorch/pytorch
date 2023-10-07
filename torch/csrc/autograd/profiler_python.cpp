#include <torch/csrc/autograd/profiler_python.h>

#include <atomic>
#include <cstdint>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include <Python.h>
#include <frameobject.h>

#include <ATen/core/TensorBase.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <c10/util/Optional.h>
#include <c10/util/StringUtil.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/containers.h>
#include <torch/csrc/profiler/orchestration/python_tracer.h>
#include <torch/csrc/profiler/util.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_compat.h>
#include <torch/csrc/utils/python_strings.h>

namespace py = pybind11;

namespace torch {
namespace profiler {
namespace impl {
namespace {
enum CallType { PyCall = 0, PyModuleCall, PyCCall, PyOptimizerCall };
static constexpr size_t CallTypeSize = 4;
using no_ephemeral_t = std::tuple<>;
static constexpr uint64_t NoTID = std::numeric_limits<uint64_t>::max();

// ============================================================================
// == Miscellaneous structs and utils =========================================
// ============================================================================
struct CodeLocation {
  CodeLocation() = default;
  explicit CodeLocation(PyFrameObject* frame)
      : line_number_{PyFrame_GetLineNumber(frame)} {
    auto code = THPCodeObjectPtr(PyFrame_GetCode(frame));
    filename_ = THPUtils_unpackStringView(code->co_filename).data();
    name_ = THPUtils_unpackStringView(code->co_name).data();
  }

  bool operator==(const CodeLocation& other) const {
    return filename_ == other.filename_ && name_ == other.name_ &&
        line_number_ == other.line_number_;
  }

  const char* filename_{nullptr};
  const char* name_{nullptr};
  int line_number_{0};
};

template <CallType C>
PyCodeObject* getCode();

template <>
PyCodeObject* getCode<CallType::PyModuleCall>() {
  static auto module_call_code = []() {
    pybind11::gil_scoped_acquire gil;
    auto res = py::module::import("torch.nn")
                   .attr("Module")
                   .attr("__call__")
                   .attr("__code__")
                   .ptr();
    TORCH_INTERNAL_ASSERT(PyCode_Check(res));
    return (PyCodeObject*)res;
  }();
  return module_call_code;
};

template <>
PyCodeObject* getCode<CallType::PyOptimizerCall>() {
  static auto optimizer_step_code = []() {
    pybind11::gil_scoped_acquire gil;
    auto res = py::module::import("torch.optim")
                   .attr("Optimizer")
                   .attr("_optimizer_step_code")
                   .attr("__code__")
                   .ptr();
    TORCH_INTERNAL_ASSERT(PyCode_Check(res));
    return (PyCodeObject*)res;
  }();
  return optimizer_step_code;
};

} // namespace
} // namespace impl
} // namespace profiler
} // namespace torch

template <>
struct std::hash<torch::profiler::impl::CodeLocation> {
  size_t operator()(const torch::profiler::impl::CodeLocation& x) {
    return c10::get_hash(x.filename_, x.name_, x.line_number_);
  }
};

namespace torch {
namespace profiler {
namespace impl {
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
  static void map(T& t, FunctorT& f, Args&&... args) {
    f(std::get<C>(t), args...);
    if constexpr (C + 1 < End) {
      map<C + 1>(t, f, std::forward<Args>(args)...);
    }
  }

 public:
  using tuple_type = decltype(make_tuple_impl(std::make_index_sequence<End>{}));

  template <typename FunctorT, typename... Args>
  static void map(tuple_type& t, FunctorT& f, Args&&... args) {
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
//   2) Add a specialization of Config which defined key_t, ephemeral_t and
//      cache_t.
//   3) Add a specialization of ValueCache::store and ValueCache::load.
//
// -------------------------
// -- Ephemeral arguments --
// -------------------------
// The value cache mechanism assumes that `key_t` is enough to specify the
// correct value. However it may not be possible to materialize a value using
// only an instance of `key_t`. As a result, the cache also accepts "ephemeral"
// inputs which can be used to populate the value cache. Ephemeral inputs come
// with two caveats:
//  1) They are NOT safe to save, and cannot be used after `ValueCache::store`.
//  2) They should be used to access data that is not expect to change from
//     call to call, such as the name of a function.

template <CallType>
struct Config;

template <>
struct Config<CallType::PyCall> {
  using key_t = CodeLocation;
  using ephemeral_t = no_ephemeral_t;
  using cache_t = ska::flat_hash_map<key_t, PyFrameState>;
  static constexpr EventType event_type = EventType::PyCall;
};

template <typename Key, typename Cls, typename ParameterInfo>
struct ExtendedPyCallConfig {
  using key_t = Key;
  using cls_t = Cls;
  using ephemeral_t = PyFrameObject*;

  struct ClsAndParameters {
    cls_t cls_;
    std::vector<ParameterInfo> parameters_;
  };

  struct Cache {
    // `nn.Module.forward` or `optim.Optimizer._optimizer_step_code`
    c10::optional<CodeLocation> location_;
    ska::flat_hash_map<key_t, ClsAndParameters> cls_and_parameters_;
    ska::flat_hash_map<cls_t, at::StringView> cls_names_;
  };
  using cache_t = Cache;

  static constexpr EventType event_type = EventType::PyCall;
};

template <>
struct Config<CallType::PyModuleCall> : ExtendedPyCallConfig<
                                            PyModuleSelf,
                                            PyModuleCls,
                                            NNModuleInfo::ParameterInfo> {};

template <>
struct Config<CallType::PyOptimizerCall> : ExtendedPyCallConfig<
                                               PyOptimizerSelf,
                                               PyOptimizerCls,
                                               OptimizerInfo::ParameterInfo> {};

template <>
struct Config<CallType::PyCCall> {
  using key_t = PyMethod;
  using ephemeral_t = PyObject*;
  using cache_t = ska::flat_hash_map<key_t, at::StringView>;
  static constexpr EventType event_type = EventType::PyCCall;
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
  Callsite(U value, PyFrameObject* f_back) : value_(value), caller_(f_back) {}

  bool operator==(const Callsite<C>& other) const {
    return value_ == other.value_ && caller_ == other.caller_;
  }

  key_t value_;
  Config<CallType::PyCall>::key_t caller_;
};

// ============================================================================
// == Type specific store and load implementations. ===========================
// ============================================================================
using PyCallKey = Config<CallType::PyCall>::key_t;
using PyModuleCallKey = Config<CallType::PyModuleCall>::key_t;
using PyCCallKey = Config<CallType::PyCCall>::key_t;
using PyOptimizerCallKey = Config<CallType::PyOptimizerCall>::key_t;

class ValueCache {
 public:
  ValueCache() = default;
  ValueCache(const ValueCache&) = delete;

  template <CallType C>
  void store(const typename Config<C>::key_t&, typename Config<C>::ephemeral_t);

  template <CallType C>
  auto load(const Callsite<C>& callsite, size_t python_tid) const {
    auto caller = load<CallType::PyCall>(callsite.caller_);
    TORCH_INTERNAL_ASSERT(!caller.module_info_.has_value());
    return ExtraFields<Config<C>::event_type>{
        /*end_time_ns=*/std::numeric_limits<time_t>::min(),
        python_tid,
        caller.frame_state_,
        load<C>(callsite.value_)};
  }

  c10::optional<TensorMetadata> recordIfTensor(py::handle p);
  std::vector<std::pair<std::string, TensorMetadata>> unpackTensorMap(
      const py::dict& tensor_map);
  void trimPrefixes();

 private:
  template <CallType C>
  typename ExtraFields<Config<C>::event_type>::args_t load(
      const typename Config<C>::key_t&) const;

  template <CallType C>
  using State = typename Config<C>::cache_t;

  CallTypeHelper<State>::tuple_type state_;
};

template <CallType C>
typename Config<C>::cls_t set_class(
    ValueCache* value_cache,
    typename Config<C>::cache_t& cache,
    const typename Config<C>::key_t& key,
    const typename Config<C>::ephemeral_t& frame) {
  if (C10_UNLIKELY(!cache.location_.has_value())) {
    auto code = THPCodeObjectPtr(PyFrame_GetCode(frame));
    TORCH_INTERNAL_ASSERT(code.get() == getCode<C>());
    cache.location_ = PyCallKey(frame);
    value_cache->store<CallType::PyCall>(*cache.location_, no_ephemeral_t());
  }

  auto cls_handle = py::handle((PyObject*)key).attr("__class__");
  auto cls = typename Config<C>::cls_t(cls_handle.ptr());
  if (cache.cls_names_.find(cls) == cache.cls_names_.end()) {
    cache.cls_names_[cls] =
        at::StringView(py::str(cls_handle.attr("__name__")));
  }
  return cls;
}

TensorMetadata toTensorMetadata(PyObject* self) {
  TORCH_INTERNAL_ASSERT(THPVariable_CheckExact(self));
  const auto& t = THPVariable_Unpack(self);
  RawTensorMetadata m{t};
  return TensorMetadata{
      m,
      t.sizes().vec(),
      m.layout_ == at::kStrided ? t.strides().vec() : std::vector<int64_t>()};
}

c10::optional<TensorMetadata> ValueCache::recordIfTensor(py::handle p) {
  return THPVariable_CheckExact(p.ptr())
      ? c10::optional<TensorMetadata>{toTensorMetadata(p.ptr())}
      : c10::nullopt;
}

std::vector<std::pair<std::string, TensorMetadata>> ValueCache::unpackTensorMap(
    const py::dict& tensor_map) {
  std::vector<std::pair<std::string, TensorMetadata>> out;
  for (auto& it : tensor_map) {
    auto* value = it.second.ptr();
    if (py::isinstance<py::str>(it.first) && THPVariable_CheckExact(value)) {
      out.emplace_back(
          py::cast<std::string>(it.first), toTensorMetadata(value));
    }
  }
  return out;
}

template <>
void ValueCache::store<CallType::PyCall>(const PyCallKey& key, no_ephemeral_t) {
  auto& locations = std::get<CallType::PyCall>(state_);
  if (C10_UNLIKELY(locations.find(key) == locations.end())) {
    locations[key] = {
        key.line_number_,
        at::StringView(key.filename_),
        at::StringView(key.name_)};
  }
}

template <>
ExtraFields<EventType::PyCall>::args_t ValueCache::load<CallType::PyCall>(
    const PyCallKey& key) const {
  return {std::get<CallType::PyCall>(state_).at(key), c10::nullopt};
}

template <>
void ValueCache::store<CallType::PyModuleCall>(
    const PyModuleCallKey& key,
    Config<CallType::PyModuleCall>::ephemeral_t frame) {
  auto& cache = std::get<CallType::PyModuleCall>(state_);
  if (C10_UNLIKELY(
          cache.cls_and_parameters_.find(key) ==
          cache.cls_and_parameters_.end())) {
    auto cls = set_class<CallType::PyModuleCall>(this, cache, key, frame);

    py::dict params = py::handle((PyObject*)key).attr("_parameters");
    std::vector<NNModuleInfo::ParameterInfo> params_;
    for (auto& it : params) {
      auto* p = it.second.ptr();
      if (py::isinstance<py::str>(it.first) && THPVariable_CheckExact(p)) {
        params_.push_back(
            {it.first.cast<std::string>(),
             toTensorMetadata(p),
             recordIfTensor(py::getattr(it.second, "grad", py::none()))});
      }
    }
    cache.cls_and_parameters_[key] = {cls, std::move(params_)};
  }
}

template <>
ExtraFields<EventType::PyCall>::args_t ValueCache::load<CallType::PyModuleCall>(
    const PyModuleCallKey& key) const {
  auto& cache = std::get<CallType::PyModuleCall>(state_);
  TORCH_INTERNAL_ASSERT(cache.location_.has_value());
  const auto& cls_and_parameters = cache.cls_and_parameters_.at(key);
  const auto& cls = cls_and_parameters.cls_;
  NNModuleInfo info{
      key, cls, cache.cls_names_.at(cls), cls_and_parameters.parameters_};
  return {
      /*frame_state_=*/std::get<CallType::PyCall>(state_).at(*cache.location_),
      /*module_info_=*/std::move(info),
      /*optimizer_info_=*/c10::nullopt};
}

template <>
void ValueCache::store<CallType::PyOptimizerCall>(
    const PyOptimizerCallKey& key,
    Config<CallType::PyOptimizerCall>::ephemeral_t frame) {
  auto& cache = std::get<CallType::PyOptimizerCall>(state_);
  if (C10_UNLIKELY(
          cache.cls_and_parameters_.find(key) ==
          cache.cls_and_parameters_.end())) {
    auto cls = set_class<CallType::PyOptimizerCall>(this, cache, key, frame);
    const py::handle self{(PyObject*)key};
    std::vector<OptimizerInfo::ParameterInfo> params;

    for (const auto& i : (py::list)self.attr("param_groups")) {
      for (auto& param : py::cast<py::dict>(i).attr("get")("params")) {
        if (THPVariable_CheckExact(param.ptr())) {
          // While `self.state` is permitted to store data in an arbitrary way,
          // all generic optimizers (SGD, Adam, etc) use param as the key since
          // the state in question is tied to particular parameters. We can
          // relax this assumption if the need arises.
          params.push_back(
              {toTensorMetadata(param.ptr()),
               recordIfTensor(py::getattr(param, "grad", py::none())),
               unpackTensorMap(py::cast<py::dict>(self.attr("state"))
                                   .attr("get")(param, py::dict()))});
        }
      }
    }

    cache.cls_and_parameters_[key] = {cls, std::move(params)};
  }
}

template <>
ExtraFields<EventType::PyCall>::args_t ValueCache::load<
    CallType::PyOptimizerCall>(const PyOptimizerCallKey& key) const {
  auto& cache = std::get<CallType::PyOptimizerCall>(state_);
  const auto& cls_and_parameters = cache.cls_and_parameters_.at(key);
  auto cls = cls_and_parameters.cls_;
  OptimizerInfo info{
      key, cls, cache.cls_names_.at(cls), cls_and_parameters.parameters_};
  return {
      /*frame_state_=*/std::get<CallType::PyCall>(state_).at(*cache.location_),
      /*module_info_=*/c10::nullopt,
      /*optimizer_info_=*/std::move(info)};
}

template <>
void ValueCache::store<CallType::PyCCall>(
    const PyCCallKey& key,
    Config<CallType::PyCCall>::ephemeral_t arg) {
  auto& names = std::get<CallType::PyCCall>(state_);
  if (C10_UNLIKELY(names.find(key) == names.end())) {
    names[key] = at::StringView(py::repr(arg));
  }
}

template <>
ExtraFields<EventType::PyCCall>::args_t ValueCache::load<CallType::PyCCall>(
    const PyCCallKey& key) const {
  return std::get<CallType::PyCCall>(state_).at(key);
}

// TODO: Use re2.
void ValueCache::trimPrefixes() {
  static const auto prefixes = []() {
    pybind11::gil_scoped_acquire gil;
    return py::module::import("torch.profiler.python_tracer")
        .attr("_prefix_regex")()
        .cast<std::vector<std::string>>();
  }();

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
using python_tracer::TraceKey;

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

  TraceKey intern(
      Callsite<C> callsite,
      typename Config<C>::ephemeral_t ephemeral,
      ValueCache& value_cache) {
    auto it = state_.find(callsite);
    if (C10_UNLIKELY(it == state_.end())) {
      value_cache.store<C>(callsite.value_, ephemeral);
      value_cache.store<CallType::PyCall>(callsite.caller_, no_ephemeral_t());
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
  PyObject_HEAD;
  ThreadLocalResults* thread_local_results_;
};

// CPython boilerplate to define `TraceContext` as a proper python object.
static PyTypeObject TraceContextType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "TraceContext", /* tp_name */
    sizeof(TraceContext), /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    0,
    /* tp_vectorcall_offset */ // NOLINT: modernize-use-nullptr
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    "Python tracer TLS", /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    nullptr, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    PyType_GenericNew, /* tp_new */
    nullptr /* tp_free */
};

class gil_and_restore_thread {
 public:
  gil_and_restore_thread()
      : gil_(), initial_thread_state_{PyThreadState_Get()} {}
  ~gil_and_restore_thread() {
    PyThreadState_Swap(initial_thread_state_);

    // `gil_scoped_acquire` is a bit fragile in on-demand mode:
    // https://github.com/pytorch/pytorch/pull/91684#issuecomment-1413154458
    if (!Py_IsInitialized()) {
      gil_.disarm();
    }
  }

  PyThreadState* initial_thread_state() const {
    return initial_thread_state_;
  }

 private:
  pybind11::gil_scoped_acquire gil_;
  PyThreadState* initial_thread_state_;
};

// ============================================================================
// == Thread local cache ======================================================
// ============================================================================
class PythonTracer;
struct ThreadLocalResults {
  ThreadLocalResults(
      PyThreadState* thread_state,
      ValueCache* value_cache,
      PythonTracer* active_tracer)
      : thread_state_{thread_state},
        ctx_{(TraceContext*)TraceContextType.tp_alloc(&TraceContextType, 0)},
        value_cache_{value_cache},
        active_tracer_{active_tracer} {
    ctx_->thread_local_results_ = this;
  }

  ThreadLocalResults() = delete;
  ThreadLocalResults(const ThreadLocalResults&) = delete;
  ThreadLocalResults(ThreadLocalResults&&) = delete;
  ThreadLocalResults& operator=(const ThreadLocalResults&) = delete;
  ThreadLocalResults& operator=(const ThreadLocalResults&&) = delete;

  ~ThreadLocalResults() {
    Py_DECREF((PyObject*)ctx_);
  }

  template <CallType C, EventType E, typename Ephemeral, typename... Args>
  TraceKey intern(Ephemeral ephemeral, Args... args) {
    static_assert(
        Config<C>::event_type == E,
        "ThreadLocalResults.intern called from the wrong typed context.");
    auto callsite = Callsite<C>(std::forward<Args>(args)...);
    return std::get<C>(trace_keys_).intern(callsite, ephemeral, *value_cache_);
  }

  static constexpr size_t BLOCK_SIZE = 1024;

  PyThreadState* thread_state_;
  TraceContext* ctx_;
  ValueCache* value_cache_;
  PythonTracer* active_tracer_;
  CallTypeHelper<TraceKeyCacheState>::tuple_type trace_keys_;
  AppendOnlyList<approx_time_t, BLOCK_SIZE> exit_times_;
  AppendOnlyList<approx_time_t, BLOCK_SIZE> c_exit_times_;
};

// ============================================================================
// == Tracing implementation ==================================================
// ============================================================================
class PythonTracer final : public python_tracer::PythonTracerBase {
 public:
  PythonTracer(torch::profiler::impl::RecordQueue* queue);
  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~PythonTracer() override;

  static int pyProfileFn(
      PyObject* obj,
      PyFrameObject* frame,
      int what,
      PyObject* arg);

  void stop() override;
  std::vector<std::shared_ptr<Result>> getEvents(
      std::function<time_t(approx_time_t)> time_converter,
      std::vector<python_tracer::CompressedEvent>& enters,
      time_t end_time_ns) override;

  struct StartFrame {
    TraceKey trace_key_;
    approx_time_t start_time{};
  };

 private:
  void recordPyCall(
      ThreadLocalResults& tls,
      PyFrameObject* frame,
      bool is_startup_frame);

  void recordCCall(
      ThreadLocalResults& tls,
      PyFrameObject* frame,
      PyObject* arg);

  const std::vector<PyThreadState*> interpreterThreads() const;

  std::atomic<bool> active_lock_{false};
  bool active_{false};

  torch::profiler::impl::RecordQueue* queue_;
  PyInterpreterState* interpreter_;
  PyCodeObject* module_call_code_;
  PyCodeObject* optimizer_hook_;

  std::vector<StartFrame> start_frames_;
  std::deque<ThreadLocalResults> thread_local_results_;
  ValueCache value_cache_;
};

const std::vector<PyThreadState*> PythonTracer::interpreterThreads() const {
  pybind11::gil_scoped_acquire gil;
  std::vector<PyThreadState*> out;
  if (SOFT_ASSERT(interpreter_)) {
    auto* thread_state = PyInterpreterState_ThreadHead(interpreter_);
    while (thread_state != nullptr) {
      out.push_back(thread_state);
      thread_state = PyThreadState_Next(thread_state);
    }
  }
  return out;
}

PythonTracer::PythonTracer(torch::profiler::impl::RecordQueue* queue)
    : queue_(queue),
      interpreter_(nullptr),
      module_call_code_(getCode<CallType::PyModuleCall>()),
      optimizer_hook_(getCode<CallType::PyOptimizerCall>()) {
  TORCH_CHECK(queue_ != nullptr);

  bool expected{false};
  active_ = active_lock_.compare_exchange_strong(expected, true);
  if (!active_) {
    TORCH_WARN(
        "There is already an active Python tracer. "
        "Refusing to register profile functions.");
    return;
  }

  gil_and_restore_thread gil;
  interpreter_ = PyInterpreterState_Get();

  if (!gil.initial_thread_state()) {
    TORCH_WARN("PyThreadState_Get returned NULL");
    return;
  }

  // Register the tracer in each thread.
  for (const auto thread_state : interpreterThreads()) {
    PyThreadState_Swap(thread_state);

    thread_local_results_.emplace_back(thread_state, &value_cache_, this);
    auto* ctx = thread_local_results_.back().ctx_;

    // When we begin profiling there are already frames on the Python
    // interpreter stack. To ensure a complete trace, we must push calls
    // to all the prior frames onto our event stack. (We stop at depth=128)

    std::vector<THPFrameObjectPtr> current_stack;
    auto frame = PyEval_GetFrame();
    Py_XINCREF(frame);

    size_t depth = 0; // Make sure we can't infinite loop.
    while (frame != nullptr) {
      current_stack.emplace_back(frame);
      if (++depth == 128) {
        break;
      }

      // NB: `PyFrame_GetBack` returns a strong reference.
      frame = PyFrame_GetBack(frame);
    }

    for (auto it = current_stack.rbegin(); it != current_stack.rend(); it++) {
      recordPyCall(thread_local_results_.back(), it->get(), true);
      auto frame_refcount = Py_REFCNT(it->get());

      // We hold one reference in `current_stack`, and the interpreter holds
      // another.
      TORCH_INTERNAL_ASSERT(frame_refcount >= 2, frame_refcount);
    }

    // Note:
    //   This profile will not compose with other CPython profilers, and
    //   cannot be round tripped via `sys.settrace(sys.gettrace())`
    PyEval_SetProfile(PythonTracer::pyProfileFn, (PyObject*)ctx);
  }
};

void PythonTracer::stop() {
  gil_and_restore_thread gil;
  if (active_) {
    for (const auto thread_state : interpreterThreads()) {
      if (thread_state->c_profilefunc == &PythonTracer::pyProfileFn) {
        PyThreadState_Swap(thread_state);
        PyEval_SetProfile(nullptr, nullptr);
      }
    }

    auto lock_returned = active_lock_.compare_exchange_strong(active_, false);
    active_ = false;
    SOFT_ASSERT(lock_returned, "Failed to return python tracer lock.");
  }
}

// NOLINTNEXTLINE(bugprone-exception-escape)
PythonTracer::~PythonTracer() {
  if (active_) {
    TORCH_WARN("`PythonTracer::stop()` was not called.");
    stop();
  }
}

void PythonTracer::recordPyCall(
    ThreadLocalResults& tls,
    PyFrameObject* frame,
    bool is_startup_frame) {
  static constexpr auto E = EventType::PyCall;
  const auto key = [&]() -> TraceKey {
    auto code = THPCodeObjectPtr(PyFrame_GetCode(frame));
    if (code.get() == module_call_code_) {
      // By default, CPython stores locals in a "fast" format, with an array
      // of names and an array of values. Consequently, frame->f_locals is
      // NULL since the interpreter has no need to populate it.
      //
      // If these arrays were part of the public API then we could very
      // quickly access `self`. Unfortunately they are not, and moreover are
      // not stable across versions. As a result, we are forced to call
      // `PyFrame_FastToLocals` which forces the interpreter to materialize
      // the full dict of locals.
      auto locals = THPObjectPtr(PyFrame_GetLocals(frame));
      auto self = THPObjectPtr(PyDict_GetItemString(locals, "self"));
      Py_INCREF(self.get());
      auto back = THPFrameObjectPtr(PyFrame_GetBack(frame));
      TORCH_INTERNAL_ASSERT(back != nullptr);
      return tls.intern<CallType::PyModuleCall, E>(
          frame, self.get(), back.get());
    } else if (code.get() == optimizer_hook_) {
      auto locals = THPObjectPtr(PyFrame_GetLocals(frame));
      auto self = THPObjectPtr(PyDict_GetItemString(locals, "self"));
      Py_INCREF(self.get());
      auto back = THPFrameObjectPtr(PyFrame_GetBack(frame));
      TORCH_INTERNAL_ASSERT(back != nullptr);
      return tls.intern<CallType::PyOptimizerCall, E>(
          frame, self.get(), back.get());
    } else {
      auto back = THPFrameObjectPtr(PyFrame_GetBack(frame));
      auto f_back = (back.get() != nullptr) ? back.get() : frame;
      return tls.intern<CallType::PyCall, E>(no_ephemeral_t(), frame, f_back);
    }
  }();
  const auto time = getApproximateTime();
  is_startup_frame ? start_frames_.push_back({key, time})
                   : queue_->getSubqueue()->emplace_py_call(key, time);
}

void PythonTracer::recordCCall(
    ThreadLocalResults& tls,
    PyFrameObject* frame,
    PyObject* arg) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(PyCFunction_Check(arg));
  auto fn = reinterpret_cast<PyCFunctionObject*>(arg);

  // NB: For C calls a new frame is not created, so we use `frame` rather than
  //     `frame->f_back`.
  auto key = tls.intern<CallType::PyCCall, EventType::PyCCall>(
      arg, (void*)(fn->m_ml), frame);
  queue_->getSubqueue()->emplace_py_call(key, getApproximateTime());
}

// ============================================================================
// == Post processing =========================================================
// ============================================================================
struct Exit {
  bool operator>(const Exit& other) const {
    return t_ > other.t_;
  }

  time_t t_;
  size_t python_tid_;
};

class PostProcess {
 public:
  PostProcess(
      std::function<time_t(approx_time_t)> time_converter,
      std::deque<ThreadLocalResults>& tls,
      const ValueCache& value_cache,
      time_t end_time_ns)
      : end_time_{end_time_ns}, time_converter_{std::move(time_converter)} {
    for (size_t python_tid : c10::irange(tls.size())) {
      CallTypeHelper<TraceKeyCacheState>::map(
          tls[python_tid].trace_keys_, *this, value_cache, python_tid);

      addExits<EventType::PyCall>(tls[python_tid].exit_times_, python_tid);
      addExits<EventType::PyCCall>(tls[python_tid].c_exit_times_, python_tid);
    }
  }

  void set_start_frames(
      const std::vector<PythonTracer::StartFrame>& start_frames,
      std::vector<python_tracer::CompressedEvent>& enters) {
    for (const auto& frame : start_frames) {
      enters.push_back(
          {frame.trace_key_,
           NoTID, // Allows us to detect unhandled start frames
           {},
           time_converter_(frame.start_time)});
    }
  }

  template <CallType C>
  void operator()(
      const TraceKeyCacheState<C>& trace_cache,
      const ValueCache& value_cache,
      size_t python_tid) {
    for (const auto& it : trace_cache.state_) {
      const auto inserted = get_state<Config<C>::event_type>().fields_.insert(
          {it.second, value_cache.load(it.first, python_tid)});
      TORCH_INTERNAL_ASSERT(inserted.second, "Duplicate key: ", it.second);
    }
  }

  template <EventType E, size_t N>
  void addExits(AppendOnlyList<approx_time_t, N>& exits, size_t python_tid) {
    for (const auto i : exits) {
      get_state<E>().exits_.push({time_converter_(i), python_tid});
    }
  }

  std::vector<std::shared_ptr<Result>> run(
      std::vector<python_tracer::CompressedEvent>& enters) {
    std::stable_sort(
        enters.begin(), enters.end(), [](const auto a, const auto b) {
          return a.enter_t_ < b.enter_t_;
        });
    std::vector<std::shared_ptr<Result>> out;
    populate<EventType::PyCall>(enters, out);
    populate<EventType::PyCCall>(enters, out);
    return out;
  }

 private:
  template <EventType E>
  void populate(
      std::vector<python_tracer::CompressedEvent>& enters,
      std::vector<std::shared_ptr<Result>>& out) {
    using stack_t = std::vector<std::shared_ptr<Result>>;
    const auto initial_size = out.size();
    auto pop = [](stack_t& stack, time_t t) {
      TORCH_INTERNAL_ASSERT(!stack.empty(), "Python replay stack is empty.");
      std::get<ExtraFields<E>>(stack.back()->extra_fields_).end_time_ns_ = t;
      stack.pop_back();
    };

    ska::flat_hash_map<size_t, stack_t> stacks;
    auto& state = get_state<E>();
    for (const auto& enter : enters) {
      auto fields_it = state.fields_.find(enter.key_);
      if (fields_it != state.fields_.end()) {
        while (!state.exits_.empty() &&
               state.exits_.top().t_ < enter.enter_t_) {
          auto& exit = state.exits_.top();
          pop(stacks[exit.python_tid_], exit.t_);
          state.exits_.pop();
        }
        out.push_back(Result::create(
            enter.enter_t_,
            enter.system_tid_,
            enter.kineto_info_,
            fields_it->second));

        stacks[fields_it->second.python_tid_].push_back(out.back());
      }
    }

    // Handle events which were still running when profiling ended.
    for (auto& i : stacks) {
      while (!i.second.empty()) {
        pop(i.second, end_time_);
      }
    }

    // Assign system TIDs to start events based on the system TID of the next
    // observed event with the same Python TID.
    ska::flat_hash_map<size_t, std::pair<size_t, kineto::DeviceAndResource>>
        tid_map;
    auto it = out.rbegin();
    for (C10_UNUSED auto _ : c10::irange(initial_size, out.size())) {
      const auto python_tid =
          std::get<ExtraFields<E>>((*it)->extra_fields_).python_tid_;
      if ((*it)->start_tid_ == NoTID && SOFT_ASSERT(E == EventType::PyCall)) {
        const auto& tid_info =
            tid_map.insert({python_tid, {NoTID, kineto::DeviceAndResource()}})
                .first->second;
        (*it)->start_tid_ = tid_info.first;
        (*it)->kineto_info_ = tid_info.second;
      }
      tid_map[python_tid] = {(*it)->start_tid_, (*it)->kineto_info_};
      ++it;
    }
  }

  template <EventType E>
  struct State {
    ska::flat_hash_map<TraceKey, ExtraFields<E>> fields_;
    std::priority_queue<Exit, std::vector<Exit>, std::greater<>> exits_;
  };

  template <EventType E>
  auto& get_state() {
    return std::get < E == EventType::PyCall ? 0 : 1 > (state_);
  }

  time_t end_time_;
  std::function<time_t(approx_time_t)> time_converter_;
  std::tuple<State<EventType::PyCall>, State<EventType::PyCCall>> state_;
};

struct PythonIDVisitor {
  void operator()(ExtraFields<EventType::PyCall>& py_call) {
    py_call.id_ = ++current_python_id_;
    if (py_call.module_.has_value()) {
      auto& m = py_call.module_;
      auto& module_ids = module_ids_[m->cls_];
      m->id_ = module_ids.insert({m->self_, module_ids.size()}).first->second;
    }
  }

  void operator()(ExtraFields<EventType::PyCCall>& py_call) {
    py_call.id_ = ++current_python_id_;
  }

  template <typename T>
  void operator()(T&) {}

  size_t current_python_id_{0};
  ska::flat_hash_map<PyModuleCls, ska::flat_hash_map<PyModuleSelf, size_t>>
      module_ids_;
};

std::vector<std::shared_ptr<Result>> PythonTracer::getEvents(
    std::function<time_t(approx_time_t)> time_converter,
    std::vector<python_tracer::CompressedEvent>& enters,
    time_t end_time_ns) {
  value_cache_.trimPrefixes();
  PostProcess post_process(
      std::move(time_converter),
      thread_local_results_,
      value_cache_,
      end_time_ns);
  post_process.set_start_frames(start_frames_, enters);
  auto out = post_process.run(enters);

  std::stable_sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
    return a->start_time_ns_ < b->start_time_ns_;
  });

  PythonIDVisitor id_visitor;
  for (auto& i : out) {
    std::visit(id_visitor, i->extra_fields_);
  }

  return out;
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
      local_results.active_tracer_->recordPyCall(local_results, frame, false);
      break;

    case PyTrace_C_CALL:
      local_results.active_tracer_->recordCCall(local_results, frame, arg);
      break;

    case PyTrace_EXCEPTION:
    case PyTrace_RETURN:
      local_results.exit_times_.emplace_back(getApproximateTime());
      break;

    case PyTrace_C_EXCEPTION:
    case PyTrace_C_RETURN:
      local_results.c_exit_times_.emplace_back(getApproximateTime());
      break;
  }
  return 0;
}

std::unique_ptr<python_tracer::PythonTracerBase> getTracer(
    torch::profiler::impl::RecordQueue* queue) {
  return std::make_unique<PythonTracer>(queue);
}
} // namespace
} // namespace impl
} // namespace profiler
} // namespace torch

namespace torch {
namespace autograd {
namespace profiler {
namespace python_tracer {

void init() {
  pybind11::gil_scoped_acquire gil;
  TORCH_CHECK(PyType_Ready(&torch::profiler::impl::TraceContextType) == 0);
  torch::profiler::impl::python_tracer::registerTracer(
      &torch::profiler::impl::getTracer);
}
} // namespace python_tracer
} // namespace profiler
} // namespace autograd
} // namespace torch
