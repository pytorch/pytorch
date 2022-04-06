#pragma once

#include <string>
#include <vector>

#include <torch/csrc/profiler/api.h>
#include <torch/csrc/profiler/kineto_shim.h>
#include <torch/csrc/profiler/util.h>

namespace torch {
namespace autograd {
namespace profiler {

struct TORCH_API KinetoEvent {
  uint64_t startThreadId() const {
    return start_thread_id_;
  }

  KinetoEvent& startThreadId(uint64_t start_thread_id) {
    start_thread_id_ = start_thread_id;
    return *this;
  }

  uint64_t endThreadId() const {
    return end_thread_id_;
  }

  KinetoEvent& endThreadId(uint64_t end_thread_id) {
    end_thread_id_ = end_thread_id;
    return *this;
  }

  uint8_t activityType() const {
    return activity_type_;
  }

  KinetoEvent& activityType(uint8_t activity_type) {
    activity_type_ = activity_type;
    return *this;
  }

  uint64_t fwdThreadId() const {
    return fwd_thread_id_;
  }

  KinetoEvent& fwdThreadId(uint64_t fwd_thread_id) {
    fwd_thread_id_ = fwd_thread_id;
    return *this;
  }

  bool hasShapes() const {
    return shapes_ != c10::nullopt;
  }

  const std::vector<std::vector<int64_t>>& shapes() const {
    return *shapes_;
  }

  KinetoEvent& shapes(const std::vector<std::vector<int64_t>>& shapes) {
    shapes_ = shapes;
    return *this;
  }

  bool hasTypes() const {
    return dtypes_ != c10::nullopt;
  }

  const std::vector<std::string>& dtypes() const {
    return *dtypes_;
  }

  KinetoEvent& dtypes(const std::vector<std::string>& dtypes) {
    dtypes_ = dtypes;
    return *this;
  }

  uint64_t flops() const {
    return flops_;
  }

  KinetoEvent& flops(uint64_t flops) {
    flops_ = flops;
    return *this;
  }

  int64_t sequenceNr() const {
    return sequence_nr_;
  }

  KinetoEvent& sequenceNr(int64_t sequence_nr) {
    sequence_nr_ = sequence_nr;
    return *this;
  }

  bool hasStack() const {
    return stack_ != c10::nullopt;
  }

  const std::vector<std::string>& stack() const {
    return *stack_;
  }

  KinetoEvent& stack(const std::vector<std::string>& st) {
    stack_ = st;
    return *this;
  }

  uint8_t scope() const {
    return scope_;
  }

  KinetoEvent& scope(uint8_t scope) {
    scope_ = scope;
    return *this;
  }

  bool hasModuleHierarchy() const {
    return module_hierarchy_ != c10::nullopt;
  }

  const std::vector<std::string>& moduleHierarchy() const {
    return *module_hierarchy_;
  }

  KinetoEvent& moduleHierarchy(const std::vector<std::string>& module_hierarchy) {
    module_hierarchy_ = module_hierarchy;
    return *this;
  }

  KinetoEvent& debugHandle(int64_t debug_handle) {
    debug_handle_ = debug_handle;
    return *this;
  }

  int64_t debugHandle() const {
    return debug_handle_;
  }

  std::string name() const {
    return name_;
  }

  KinetoEvent& name(const std::string& evt_name) {
    name_ = evt_name;
    return *this;
  }

  KinetoEvent& setAsync(bool is_async) {
    is_async_ = is_async;
    return *this;
  }

  c10::DeviceType deviceType() const {
    return (c10::DeviceType)device_type_;
  }

  KinetoEvent& deviceType(c10::DeviceType device_type) {
    device_type_ = (int8_t)device_type;
    return *this;
  }

  uint8_t deviceIndex() const {
    return device_index_;
  }

  KinetoEvent& deviceIndex(uint8_t device_index) {
    device_index_ = device_index;
    return *this;
  }

  int64_t nBytes() const {
    return nbytes_;
  }

  KinetoEvent& nBytes(int64_t nbytes) {
    nbytes_ = nbytes;
    return *this;
  }

  uint64_t startUs() const {
    return start_us_;
  }

  KinetoEvent& startUs(uint64_t start_us) {
    start_us_ = start_us;
    return *this;
  }

  uint64_t durationUs() const {
    return duration_us_;
  }

  KinetoEvent& durationUs(uint64_t duration_us) {
    duration_us_ = duration_us;
    return *this;
  }

  bool isAsync() const {
    return is_async_;
  }

  uint64_t correlationId() const {
    return correlation_id_;
  }

  KinetoEvent& correlationId(uint64_t correlation_id)  {
    correlation_id_ = correlation_id;
    return *this;
  }

  uint64_t linkedCorrelationId() const {
    return linked_correlation_id_;
  }

  KinetoEvent& linkedCorrelationId(uint64_t linked_correlation_id) {
    linked_correlation_id_ = linked_correlation_id;
    return *this;
  }

  int64_t deviceResourceId() const {
    return device_resource_id_;
  }

  KinetoEvent& deviceResourceId(int64_t device_resource_id) {
    device_resource_id_ = device_resource_id;
    return *this;
  }

  std::string backend() const {
    return backend_;
  }

  KinetoEvent& backend(const std::string& backend) {
    backend_ = backend;
    return *this;
  }

  int64_t cudaElapsedUs() const;

  uint64_t start_thread_id_ = 0;
  uint64_t end_thread_id_ = 0;
  uint64_t fwd_thread_id_ = 0;
  int64_t sequence_nr_ = -1;
  uint8_t scope_ = 0;

  uint8_t activity_type_ = 0;
  c10::optional<std::vector<std::vector<int64_t>>> shapes_;
  c10::optional<std::vector<std::string>> stack_;
  c10::optional<std::vector<std::string>> module_hierarchy_;
  c10::optional<std::vector<std::string>> dtypes_;
  uint64_t flops_ = 0;

  std::string name_;
  uint8_t device_index_ = 0;
  int8_t device_type_ = 0;
  uint64_t start_us_ = 0;
  uint64_t duration_us_ = 0;
  uint64_t correlation_id_ = 0;
  uint64_t linked_correlation_id_ = 0;
  int64_t device_resource_id_ = 0;
  int64_t nbytes_ = 0;
  bool is_async_{false};
  int64_t debug_handle_{-1};
  std::string backend_;

  torch::profiler::impl::CUDAEventStub cuda_event_start_ = nullptr;
  torch::profiler::impl::CUDAEventStub cuda_event_end_ = nullptr;
};

// Consolidating events returned directly from Kineto
// with events manually created by us (e.g. start/stop marks,
// memory allocation events)
struct TORCH_API ProfilerResult {
  ProfilerResult();
  ProfilerResult(
      uint64_t start_time,
      std::vector<KinetoEvent> events,
      torch::profiler::impl::kineto::ActivityTraceWrapper trace);
  ~ProfilerResult();

  uint64_t trace_start_us() const {
    return trace_start_us_;
  }

  const std::vector<KinetoEvent>& events() const {
    return events_;
  }

  void save(const std::string& path);

 private:
  uint64_t trace_start_us_ = 0;
  std::vector<KinetoEvent> events_;
  torch::profiler::impl::kineto::ActivityTraceWrapper trace_;
};

/*
 * This API is used by backends to record latency of events that
 * happened in the backend but were not visible to pytorch runtime.
 * For example, if part of the model is lowered to a dsp backend, then
 * the execution of that part of the model is delegated to the backend.
 * When backend finishes execution it has an option to provide profiling
 * information (latency only at th emoment) corresponding to different operators
 * that were executed in the backend.
 * When such events are recorded by backend using this API, the event
 * records will be collected by active kineto profiler. If no kineto profiler
 * is active then the event is ignored.
 * This provides us with a way to generate all the profiling information
 * for a model regardless of where model (or part of it) executed.
 * @param start_time_us: start time in us of the event
 * @param end_time_us: end time in us of the event
 * @param debug_handle: debug handle to correlate this event/op with
 * model level module/source information
 * @param scope: scope of the event, e.g. LITE_INTERPRETER, RECORD_FN etc.
 * @param event_name: name of the event, e.g. op name
 * @param backend_name: name of the backend where the event took place.
 */
TORCH_API void reportBackendEventToActiveKinetoProfiler(
    const int64_t start_time_us,
    const int64_t end_time_us,
    const int64_t debug_handle,
    const at::RecordScope scope,
    const std::string& event_name,
    const std::string& backend_name);

TORCH_API void enableProfiler(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities,
    const std::unordered_set<at::RecordScope>& scopes = {});

/*
 * Same as enableProfiler but with callback to do post-processing of
 * KinetoEvents.
 * enableProfilerWithEventPostProcess enables profiler to capture
 * specified activities, with specified RecordFunction scope, if any.
 * Additionally, it takes a functor that does in-place post processing of
 * events, e.g. populate stack trace or module hierarchy information lazily
 * using debug_handle.
 * Example usage is with lite interpreter that has recording scope of LITE_INTERPRETER.
 * In this case lite interpreter runtime, records debug handles in RecordFunction, along
 * with other information. Debug handles are eventually passed down to KinetoEvent and
 * recorded as part of the event. KinetoEdgeCPUProfiler,
 * in torch/csrc/jit/mobile/profiler_edge.cpp, enables profiler using post-processing
 * callback, via enableProfilerWithEventPostProcess, that takes these debug handles
 * and generates stack trace and module hierarchy information, once profiling is done.
 */
TORCH_API void enableProfilerWithEventPostProcess(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities,
    std::function<void(std::vector<KinetoEvent>&)>&& cb,
    const std::unordered_set<at::RecordScope>& scopes = {});

TORCH_API std::unique_ptr<ProfilerResult> disableProfiler();

TORCH_API void prepareProfiler(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities);

namespace python_tracer {

/*
Libtorch does not depend on Python (e.g. cannot #include <Python.h>); however
when we call the profiler from libtorch_python we need the profiler to be able
to ingest the data that we collect from the Python tracer. (`PyEval_SetProfile`)

In order to solve this dependency issue we define a set of methods which do not
contain any Python symbols, but can contain the information that Kineto needs
such as times and names. The python tracer then implements these functions and
wraps their registration in an init function which is called from
`torch/csrc/autograd/init.cpp`. This pattern of registration for faux python
dependencies in libtorch is common in the PyTorch codebase.
*/
enum CallType { kPyCall = 0, kPyModuleCall, kCCall };

struct TORCH_API PyTraceEvent {
  int64_t startTime_;
  int64_t endTime_;
  std::string name_;

  uint64_t thread_id_;
  PyTraceEvent* parent_;
  CallType call_type_;
  size_t module_id_;  // Only set call_type_ == kPyModuleCall

  // Index in the list of raw call and return events. This allows one to
  // convert a vector of PyTraceEvents back into the constituent call and
  // return events, even when events share the same timestamp.
  size_t call_idx_;
  size_t return_idx_;
};

enum Command { kStartOne = 0, kStartAll, kStop, kClear };
using CallFn = void (*)(Command);
using TraceEventsFn = std::vector<std::unique_ptr<PyTraceEvent>> (*)();

TORCH_API void registerFunctions(
  CallFn call,
  TraceEventsFn get_events
);

// Because we are interleaving events, the Python tracer should use the same
// timer as the profiler.
TORCH_API int64_t now();
}  // namespace python_tracer

} // namespace profiler
}} // namespace torch::autograd
