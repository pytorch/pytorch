#include <torch/csrc/autograd/profiler_legacy.h>

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <ATen/code_template.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>

#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include <ATen/record_function.h>
#include <c10/core/Allocator.h>
#include <c10/util/ApproximateClock.h>
#include <c10/util/ThreadLocalDebugInfo.h>
#include <c10/util/irange.h>

#include <iostream>

namespace torch::autograd::profiler {

// We decompose the profiler logic into the following components:
//
// ThreadLocalDebugInfo:
//
// ThreadLocalDebugInfo is a thread local mapping from slots into
// the debug information structs.
// ThreadLocalDebugInfo is automatically propagated across thread
// boundaries, including the cases of:
//  - launching async jobs with at::launch
//  - executing JIT continuations
//  - moving from the forward threads into autograd (backward) threads
//
// Entries in ThreadLocalDebugInfo are managed by DebugInfoGuard
// which can be used to add or overwrite an entry in the thread local
// mapping. A corresponding entry is removed when the guard is destroyed,
// potentially revealing the previously set value for the same slot.
//
// For the async tasks, slots previously set in the main thread before
// launching of an async task are shared and visible in the async task.
//
// On the other hand, any adding or overwriting of the mapping by the
// async task is not visible to the main thread and any modification
// (including removal of the entries) in the main thread is not visible
// to the async task if it happens after launching the task.
//
// We use ThreadLocalDebugInfo (slot PROFILER_STATE) to store profiler config,
// as well as a list of events that happen during profiling.
// An instance of ThreadLocalDebugInfo is created each time we enter
// profiler (i.e. enter profiling context manager/call enableConfig) and
// uniquely identifies a profiling run.
//
// We automatically propagate ThreadLocalDebugInfo into async tasks,
// as well as across JIT continuations and autograd thread, so all
// the operations that happen between profiling start and end
// (not necessarily within the same thread) are recorded.
// Unless the profiling slot is overwritten as in the case of nested
// profiling ranges (in this case events for the subrange are handled
// by the nested profiler)
//
// When we exit a profiling range (either by exiting profiling context
// manager or by calling disableProfiler), we remove the previously set
// profiling entry for the given thread local mapping, and consolidate
// events in the profiling result
//
//
// ThreadLocalState:
//
// ThreadLocalState takes a 'snapshot' of thread local variables
// using provided getters. It is used together with ThreadLocalStateGuard
// to transfer the snapshot across thread boundary and set the thread local
// values as in the parent task.
//
// Profiler uses ThreadLocalState to propagate profiler's thread local state.
// ThreadLocalState also automatically propagates profiler callbacks.
//
//
// at::RecordFunction and observers
//
// Profiler uses observers mechanism to add a pair of thread local callbacks
// that are executed on a number of predetermined ranges, including:
//  - c10/ATen ops
//  - TorchScript functions/methods
//  - user defined named ranges (see `record_function` python context manager)
//
// Profiler setups a pair of callbacks that record profiling events and save
// them into the thread local profiler struct (ThreadLocalDebugInfo,
// PROFILER_STATE slot)
//
//
// Thus, the overall logic is:
//
// enableProfiler:
//  - checks that profiler is not enabled (otherwise throws)
//  - pushes new ThreadLocalDebugInfo (slot PROFILER_STATE) as the profiler
//    config for the current thread
//  - pushes profiling callbacks for the current thread
//
// disableProfiler:
//  - pops PROFILER_STATE slot from the current ThreadLocalDebugInfo and
//    consolidates events
//  - removes profiling callbacks
//
// ThreadLocalState:
//  - propagates ThreadLocalDebugInfo across threads
//  - propagates profiler callbacks across threads
//
// Profiler callbacks:
//  - get the current profiling state (PROFILER slot in ThreadLocalDebugInfo)
//  - save profiling events into the profiling state
//

namespace {
using torch::profiler::impl::ActiveProfilerType;
using torch::profiler::impl::ProfilerStateBase;

struct ProfilerLegacyThreadLocalState : public ProfilerStateBase {
  explicit ProfilerLegacyThreadLocalState(
      const torch::profiler::impl::ProfilerConfig& config)
      : ProfilerStateBase(config), remoteProfiledEvents_{c10::nullopt} {}
  ~ProfilerLegacyThreadLocalState() override = default;

  static ProfilerLegacyThreadLocalState* getTLS() {
    auto tls = ProfilerStateBase::get(/*global=*/false);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        tls == nullptr || tls->profilerType() == ActiveProfilerType::LEGACY);
    return static_cast<ProfilerLegacyThreadLocalState*>(tls);
  }

  thread_event_lists consolidate();

  void mark(std::string name, bool include_cuda = true);

  void setOrAddRemoteProfiledEvents(
      std::vector<LegacyEvent>&& remoteProfiledEvents);

  void pushRange(
      const at::RecordFunction& fn,
      const bool record_cuda,
      std::vector<std::vector<int64_t>>&& shapes = {});

  void popRange(const at::RecordFunction& fn, const bool record_cuda);

  void reportMemoryUsage(
      void* /* unused */,
      int64_t alloc_size,
      size_t /* total_allocated, unused for legacy */,
      size_t /* total_reserved, unused for legacy */,
      c10::Device device) override;

  ActiveProfilerType profilerType() override {
    return ActiveProfilerType::LEGACY;
  }

  void leakHandle() {
    handle_ = 0;
  }

 protected:
  RangeEventList& getEventList(
      std::optional<uint64_t> thread_id = std::nullopt);

  std::mutex state_mutex_;
  std::unordered_map<uint64_t, std::shared_ptr<RangeEventList>>
      event_lists_map_;

  c10::optional<std::vector<std::vector<LegacyEvent>>> remoteProfiledEvents_;
};

thread_event_lists ProfilerLegacyThreadLocalState::consolidate() {
  std::lock_guard<std::mutex> g(state_mutex_);
  thread_event_lists result;
  for (auto& kv : event_lists_map_) {
    auto& list = kv.second;
    result.emplace_back(list->consolidate());
  }
  // Consolidate remote events if applicable as well.
  if (remoteProfiledEvents_) {
    result.insert(
        result.end(),
        std::make_move_iterator(remoteProfiledEvents_->begin()),
        std::make_move_iterator(remoteProfiledEvents_->end()));
  }
  return result;
}

void ProfilerLegacyThreadLocalState::mark(std::string name, bool include_cuda) {
  if (config_.disabled()) {
    return;
  }
  if (config_.state == torch::profiler::impl::ProfilerState::NVTX) {
    torch::profiler::impl::cudaStubs()->mark(name.c_str());
  } else {
    LegacyEvent evt(
        EventKind::Mark,
        at::StringView(std::move(name)),
        at::RecordFunction::currentThreadId(),
        include_cuda &&
            config_.state == torch::profiler::impl::ProfilerState::CUDA);
    evt.setNodeId(at::RecordFunction::getDefaultNodeId());
    getEventList().record(std::move(evt));
  }
}

void ProfilerLegacyThreadLocalState::setOrAddRemoteProfiledEvents(
    std::vector<LegacyEvent>&& remoteProfiledEvents) {
  // Lock to serialize access from multiple callback threads.
  std::lock_guard<std::mutex> guard(state_mutex_);
  if (remoteProfiledEvents_) {
    (*remoteProfiledEvents_).emplace_back(remoteProfiledEvents);
  } else {
    remoteProfiledEvents_ = {std::move(remoteProfiledEvents)};
  }
}

void ProfilerLegacyThreadLocalState::pushRange(
    const at::RecordFunction& fn,
    const bool record_cuda,
    std::vector<std::vector<int64_t>>&& shapes) {
  if (config_.disabled()) {
    return;
  }
  if (config_.state == torch::profiler::impl::ProfilerState::NVTX) {
    torch::profiler::impl::cudaStubs()->rangePush(
        torch::profiler::impl::getNvtxStr(fn.name(), fn.seqNr(), shapes)
            .c_str());
  } else {
    LegacyEvent evt(
        EventKind::PushRange,
        at::StringView(std::string(fn.name())),
        at::RecordFunction::currentThreadId(),
        record_cuda,
        fn.handle(),
        std::move(shapes),
        at::RecordFunction::getDefaultNodeId(),
        fn.isAsync());
    evt.setSequenceNr(fn.seqNr());
    evt.setFwdThreadId(fn.forwardThreadId());
    evt.setScope((uint8_t)fn.scope());
    if (config_.with_flops) {
      evt.setExtraArgs(torch::profiler::impl::saveExtraArgs(fn));
      evt.setFlops(torch::profiler::impl::computeFlops(
          std::string(fn.name()), evt.extraArgs()));
    }

// TODO: will unify the two macros BUILD_LITE_INTERPRETER and C10_MOBILE soon.
#if !defined BUILD_LITE_INTERPRETER && !defined C10_MOBILE
    // backward nodes source range corresponds to the forward node
    // TODO: consider using C++ stack trace
    if (config_.with_stack &&
        fn.scope() != at::RecordScope::BACKWARD_FUNCTION) {
      auto cs =
          torch::profiler::impl::prepareCallstack(jit::currentCallstack());
      if (cs.empty()) {
        cs = torch::profiler::impl::prepareCallstack(
            jit::tracer::pythonCallstack());
      }
      evt.setStack(callstackStr(cs));
    }
#endif
    getEventList().record(std::move(evt));
  }
}

void ProfilerLegacyThreadLocalState::popRange(
    const at::RecordFunction& fn,
    const bool record_cuda) {
  if (config_.disabled()) {
    return;
  }
  if (config_.state == torch::profiler::impl::ProfilerState::NVTX) {
    torch::profiler::impl::cudaStubs()->rangePop();
  } else {
    // In some cases RecordFunction (and popRange) may be
    // called on a different thread than pushRange
    // As a convention, we put the async pop on the original
    // thread and save current thread id in pop event
    LegacyEvent evt(
        EventKind::PopRange,
        at::StringView(""),
        at::RecordFunction::currentThreadId(),
        record_cuda,
        fn.handle());
    evt.setNodeId(at::RecordFunction::getDefaultNodeId());
    getEventList(fn.threadId()).record(std::move(evt));
  }
}

void ProfilerLegacyThreadLocalState::reportMemoryUsage(
    void* /* unused */,
    int64_t alloc_size,
    size_t /* total_allocated, unused for legacy */,
    size_t /* total_reserved, unused for legacy */,
    c10::Device device) {
  if (config_.profile_memory && !config_.disabled()) {
    uint64_t thread_id = at::RecordFunction::currentThreadId();
    LegacyEvent evt(
        EventKind::MemoryAlloc,
        at::StringView(""),
        thread_id,
        config_.state == torch::profiler::impl::ProfilerState::CUDA);
    evt.updateMemoryStats(alloc_size, device);
    getEventList(thread_id).record(std::move(evt));
  }
}

RangeEventList& ProfilerLegacyThreadLocalState::getEventList(
    std::optional<uint64_t> thread_id) {
  if (!thread_id.has_value()) {
    thread_id = at::RecordFunction::currentThreadId();
  }
  RangeEventList* list_ptr = nullptr;
  std::lock_guard<std::mutex> guard(state_mutex_);
  auto it = event_lists_map_.find(thread_id.value());
  if (it != event_lists_map_.end()) {
    list_ptr = it->second.get();
  } else {
    auto event_list = std::make_shared<RangeEventList>();
    event_lists_map_[thread_id.value()] = event_list;
    list_ptr = event_list.get();
  }
  return *list_ptr;
}

enum EventIValueIdx {
  KIND = 0,
  NAME,
  THREAD_ID,
  HANDLE,
  NODE_ID,
  CPU_MEM_USAGE,
  CPU_NS,
  CUDA_RECORDED,
  CUDA_MEM_USAGE,
  CUDA_DEVICE,
  CUDA_US,
  SHAPES,
  NUM_EVENT_IVALUE_IDX // must be last in list
};

const std::unordered_set<std::string> disable_cuda_profiling = {
    "aten::view",
    "aten::t",
    "aten::transpose",
    "aten::stride",
    "aten::empty",
    "aten::empty_like",
    "aten::empty_strided",
    "aten::as_strided",
    "aten::expand",
    "aten::resize_",
    "aten::squeeze",
    "aten::unsqueeze",
    "aten::slice",
    "aten::_unsafe_view",
    "aten::size"};

void pushProfilingCallbacksLegacy() {
  auto registration_state_ptr = ProfilerLegacyThreadLocalState::getTLS();
  TORCH_INTERNAL_ASSERT(registration_state_ptr, "Expected profiler state set");
  auto handle = at::addThreadLocalCallback(
      at::RecordFunctionCallback(
          [](const at::RecordFunction& fn)
              -> std::unique_ptr<at::ObserverContext> {
            auto state_ptr = ProfilerLegacyThreadLocalState::getTLS();
            if (!state_ptr || state_ptr->config().disabled()) {
              return nullptr;
            }
            bool record_cuda = state_ptr->config().state ==
                torch::profiler::impl::ProfilerState::CUDA;
            if (record_cuda &&
                disable_cuda_profiling.find(fn.name()) !=
                    disable_cuda_profiling.end()) {
              record_cuda = false;
            }

            if (state_ptr->config().report_input_shapes) {
              auto sizes = torch::profiler::impl::inputSizes(fn);
              state_ptr->pushRange(fn, record_cuda, std::move(sizes));
            } else {
              state_ptr->pushRange(fn, record_cuda);
            }

            return nullptr;
          },
          [](const at::RecordFunction& fn, at::ObserverContext*) {
            auto state_ptr = ProfilerLegacyThreadLocalState::getTLS();
            if (!state_ptr || state_ptr->config().disabled()) {
              return;
            }
            bool record_cuda = state_ptr->config().state ==
                torch::profiler::impl::ProfilerState::CUDA;
            if (record_cuda &&
                disable_cuda_profiling.find(fn.name()) !=
                    disable_cuda_profiling.end()) {
              record_cuda = false;
            }
            state_ptr->popRange(fn, record_cuda);
          })
          .needsInputs(registration_state_ptr->config().report_input_shapes)
          .needsIds(true));
  registration_state_ptr->setCallbackHandle(handle);
}

} // namespace

void enableProfilerLegacy(
    const torch::profiler::impl::ProfilerConfig& new_config) {
  TORCH_CHECK(
      new_config.state != torch::profiler::impl::ProfilerState::NVTX ||
          torch::profiler::impl::cudaStubs()->enabled(),
      "Can't use NVTX profiler - PyTorch was compiled without CUDA");

  TORCH_CHECK(new_config.state != torch::profiler::impl::ProfilerState::KINETO);

  auto state_ptr = ProfilerLegacyThreadLocalState::getTLS();
  TORCH_CHECK(!state_ptr, "Profiler is already enabled on this thread");
  auto state = std::make_shared<ProfilerLegacyThreadLocalState>(new_config);
  c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, state);

  pushProfilingCallbacksLegacy();

  state->mark("__start_profile", false);
}

thread_event_lists disableProfilerLegacy(
    c10::optional<ProfilerDisableOptions> profilerDisableOptions) {
  auto cleanupTLSState =
      profilerDisableOptions ? profilerDisableOptions->cleanupTLSState : true;
  auto consolidate =
      profilerDisableOptions ? profilerDisableOptions->consolidate : true;
  // all the DebugInfoBase objects are scope based and supposed to use
  // DebugInfoGuard
  std::shared_ptr<c10::DebugInfoBase> state;
  if (cleanupTLSState) {
    state = c10::ThreadLocalDebugInfo::_pop(c10::DebugInfoKind::PROFILER_STATE);
  } else {
    state =
        c10::ThreadLocalDebugInfo::_peek(c10::DebugInfoKind::PROFILER_STATE);
  }

  auto state_ptr = static_cast<ProfilerLegacyThreadLocalState*>(state.get());
  TORCH_CHECK(
      state_ptr && !state_ptr->config().disabled(),
      "Can't disable profiler when it's not running");

  cleanupTLSState ? state_ptr->removeCallback() : state_ptr->leakHandle();
  if (!consolidate ||
      state_ptr->config().state == torch::profiler::impl::ProfilerState::NVTX) {
    return thread_event_lists();
  }

  state_ptr->mark("__stop_profile", false);
  // Note that this will erase the underlying events.
  return state_ptr->consolidate();
}

void addEventList(std::vector<LegacyEvent>&& profiledEvents) {
  auto state_ptr = ProfilerLegacyThreadLocalState::getTLS();
  TORCH_CHECK(state_ptr, "Profiler must be enabled.");
  state_ptr->setOrAddRemoteProfiledEvents(std::move(profiledEvents));
}

void LegacyEvent::record(bool record_cuda) {
  if (record_cuda) {
    torch::profiler::impl::cudaStubs()->record(&device_, &cuda_event, &cpu_ns_);
    return;
  }
  cpu_ns_ = c10::getTime();
}

/* static */ LegacyEvent LegacyEvent::fromIValue(
    const at::IValue& eventIValue) {
  TORCH_INTERNAL_ASSERT(
      eventIValue.isList(),
      "Expected IValue to contain type c10::impl::GenericList");
  auto ivalues = eventIValue.toList();
  TORCH_INTERNAL_ASSERT(
      ivalues.size() >= NUM_EVENT_IVALUE_IDX,
      "Expected at least ",
      NUM_EVENT_IVALUE_IDX,
      " elements to reconstruct LegacyEvent.");

  // Reconstruct input shapes from ivalues.
  const auto& shapeListIValue = ivalues.get(EventIValueIdx::SHAPES);
  TORCH_INTERNAL_ASSERT(
      shapeListIValue.isList(),
      "Expected profiler shapes IValue to contain type c10::impl::GenericList.");

  auto shapeList = shapeListIValue.toList();
  std::vector<std::vector<int64_t>> shapes;
  shapes.reserve(shapeList.size());
  for (const auto i : c10::irange(shapeList.size())) {
    std::vector<int64_t> s;
    const auto& shapeIValue = shapeList.get(i);
    TORCH_INTERNAL_ASSERT(
        shapeIValue.isList(),
        "Expected each profiler shape element to contain shapes of type c10::impl::GenericList.")
    auto curShapesList = shapeIValue.toList();
    s.reserve(curShapesList.size());
    for (const auto j : c10::irange(curShapesList.size())) {
      s.emplace_back(curShapesList.get(j).toInt());
    }
    shapes.emplace_back(s);
  }

  LegacyEvent evt(
      static_cast<EventKind>(
          ivalues.get(EventIValueIdx::KIND).toInt()), // EventKind
      at::StringView(ivalues.get(EventIValueIdx::NAME).toStringRef()), // name
      ivalues.get(EventIValueIdx::THREAD_ID).toInt(), // thread_id
      static_cast<at::RecordFunctionHandle>(
          ivalues.get(EventIValueIdx::HANDLE).toDouble()), // handle
      std::move(shapes), // input shapes
      ivalues.get(EventIValueIdx::NODE_ID).toInt(), // node id
      true, // is remote
      ivalues.get(EventIValueIdx::CPU_MEM_USAGE).toInt(), // cpu_mem_usage
      ivalues.get(EventIValueIdx::CPU_NS).toInt(), // cpu_ns
      ivalues.get(EventIValueIdx::CUDA_RECORDED).toBool(), // was cuda recorded
      ivalues.get(EventIValueIdx::CUDA_MEM_USAGE).toInt(), // cuda memory usage
      c10::DeviceIndex(
          ivalues.get(EventIValueIdx::CUDA_DEVICE).toInt()), // device
      static_cast<double>(
          ivalues.get(EventIValueIdx::CUDA_US).toInt()) // cuda_us
  );
  return evt;
}

at::IValue LegacyEvent::toIValue() const {
  c10::impl::GenericList eventIValueList(at::AnyType::get());
  eventIValueList.reserve(NUM_EVENT_IVALUE_IDX);
  eventIValueList.emplace_back(static_cast<int64_t>(kind_));
  eventIValueList.emplace_back(std::string(name_.str()));
  eventIValueList.emplace_back(static_cast<int64_t>(thread_id_));
  eventIValueList.emplace_back(static_cast<double>(handle_));
  eventIValueList.emplace_back(node_id_);
  eventIValueList.emplace_back(cpu_memory_usage_);
  eventIValueList.emplace_back(cpu_ns_);
  // CUDA event information
  bool cuda_profiling_enabled = hasCuda();
  eventIValueList.emplace_back(cuda_profiling_enabled);
  eventIValueList.emplace_back(static_cast<int64_t>(cuda_memory_usage_));
  eventIValueList.emplace_back(device_);
  eventIValueList.emplace_back(cuda_us_);
  // Shapes
  c10::impl::GenericList shapesList =
      c10::impl::GenericList(at::ListType::create(at::IntType::get()));
  shapesList.reserve(shapes_.size());
  for (const auto& shape : shapes_) {
    c10::impl::GenericList s = c10::impl::GenericList(at::IntType::get());
    s.reserve(shape.size());
    for (const auto& k : shape) {
      s.emplace_back(k);
    }
    shapesList.emplace_back(s);
  }
  eventIValueList.emplace_back(shapesList);
  return at::IValue(eventIValueList);
}

double LegacyEvent::cudaElapsedUs(const LegacyEvent& e) const {
  TORCH_CHECK(e.hasCuda() && hasCuda(), "Events were not recorded for CUDA");
  TORCH_CHECK(
      e.device() == device(),
      c10::str(
          "Events are not on the same device: ", e.device(), " vs ", device()));
  if (isRemote() && e.isRemote()) {
    // validate that cuda_us_ has been set properly.
    TORCH_INTERNAL_ASSERT(cuda_us_ >= 0 && e.cuda_us_ >= 0);
    return static_cast<double>(e.cuda_us_ - cuda_us_);
  }
  return torch::profiler::impl::cudaStubs()->elapsed(
      &cuda_event, &e.cuda_event);
}

static const at::jit::CodeTemplate event_template(R"(
{
  "name": "${name}",
  "ph": "X",
  "ts": ${ts},
  "dur": ${dur},
  "tid": ${tid},
  "pid": "CPU Functions",
  "args": {}
})");

void writeProfilerEventsToStream(
    std::ostream& out,
    const std::vector<LegacyEvent*>& events) {
  TORCH_CHECK(out, "Could not open file");
  LegacyEvent* profiler_start = nullptr;
  for (LegacyEvent* e : events) {
    if (0 == strcmp(e->name(), "__start_profile")) {
      profiler_start = e;
      break;
    }
  }
  TORCH_CHECK(profiler_start, "Could not find __start_profile mark");

  struct PairHash {
    size_t operator()(
        std::pair<at::RecordFunctionHandle, int> p) const noexcept {
      return std::hash<at::RecordFunctionHandle>()(p.first) ^
          std::hash<int64_t>()(p.second);
    }
  };
  std::unordered_map<
      std::pair<at::RecordFunctionHandle, int64_t>,
      LegacyEvent*,
      PairHash>
      events_map;
  out << "[\n";
  bool first = true;
  for (LegacyEvent* evt : events) {
    if (evt->kindStr() == "push") {
      events_map[std::make_pair(evt->handle(), evt->nodeId())] = evt;
    } else if (evt->kindStr() == "pop") {
      if (!first) {
        out << ",\n";
      }
      first = false;
      auto it = events_map.find(std::make_pair(evt->handle(), evt->nodeId()));
      TORCH_CHECK(it != events_map.end(), "Unmatched pop event");
      LegacyEvent* evt_start = it->second;
      events_map.erase(it);

      at::jit::TemplateEnv env;
      env.s("name", evt_start->name());
      env.d("ts", profiler_start->cpuElapsedUs(*evt_start));
      env.d("dur", evt_start->cpuElapsedUs(*evt));
      env.d("tid", evt_start->threadId());
      out << event_template.format(env);
    }
  }
  out << "]\n";
}

RecordProfile::RecordProfile(std::ostream& out) : out_(out) {
  init();
}

RecordProfile::RecordProfile(const std::string& filename)
    : file_(new std::ofstream(filename)), out_(*file_) {
  init();
}

void RecordProfile::init() {
  enableProfilerLegacy(torch::profiler::impl::ProfilerConfig(
      torch::profiler::impl::ProfilerState::CPU));
}

RecordProfile::~RecordProfile() {
  try {
    thread_event_lists event_lists = disableProfilerLegacy();
    std::vector<LegacyEvent*> events;
    for (auto& l : event_lists) {
      for (auto& e : l) {
        events.push_back(&e);
      }
    }
    processEvents(events);
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what() << '\n';
  } catch (...) {
    LOG(ERROR) << "Unknown error" << '\n';
  }
}

void RecordProfile::processEvents(const std::vector<LegacyEvent*>& events) {
  writeProfilerEventsToStream(out_, events);
}

} // namespace torch::autograd::profiler
