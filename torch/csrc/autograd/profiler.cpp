#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/jit/frontend/code_template.h>

#include <torch/csrc/jit/runtime/operator.h>

#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>

#include <fstream>
#include <list>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <ATen/record_function.h>
#include <c10/core/Allocator.h>
#include <c10/util/ThreadLocalDebugInfo.h>

#include <iostream>

namespace torch { namespace autograd { namespace profiler {

namespace {

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
    NUM_EVENT_IVALUE_IDX // must be last in list
  };

  enum ProfilerIValueIdx {
    STATE = 0,
    REPORT_INPUT_SHAPES,
    PROFILE_MEMORY,
    NUM_PROFILER_CFG_IVALUE_IDX // must be last in list
  };

CUDAStubs default_stubs;
constexpr CUDAStubs* default_stubs_addr = &default_stubs;
// Constant initialization, so it is guaranteed to be initialized before
// static initialization calls which may invoke registerCUDAMethods
static CUDAStubs* cuda_stubs = default_stubs_addr;

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
// For the async tasks, slots previuosly set in the main thread before
// launching of an async task are shared and visible in the async task.
//
// On the other hand, any adding or overwriting of the mapping by the
// async task is not visible to the main thread and any modification
// (including removal of the entries) in the main thread is not visible
// to the async task if it happends after launching the task.
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
// Profiler setups a pair of callbacks that record profiling events and save them
// into the thread local profiler struct (ThreadLocalDebugInfo, PROFILER_STATE slot)
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

// Profiler state
struct ProfilerThreadLocalState
    : public c10::MemoryReportingInfoBase {
  explicit ProfilerThreadLocalState(
      const ProfilerConfig& config)
    : config_(config), remoteProfiledEvents_{c10::nullopt} {}
  ~ProfilerThreadLocalState() override = default;

  inline const ProfilerConfig& config() const {
    return config_;
  }

  thread_event_lists consolidate() {
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

  void mark(
      std::string name,
      bool include_cuda = true) {
    if (config_.state == ProfilerState::Disabled) {
      return;
    }
    if (config_.state == ProfilerState::NVTX) {
      cuda_stubs->nvtxMarkA(name.c_str());
    } else {
      Event evt(
        EventKind::Mark,
        at::StringView(std::move(name)),
        at::RecordFunction::currentThreadId(),
        include_cuda && config_.state == ProfilerState::CUDA
      );
      evt.setNodeId(at::RecordFunction::getDefaultNodeId());
      getEventList().record(std::move(evt));
    }
  }

  void setOrAddRemoteProfiledEvents(std::vector<Event>&& remoteProfiledEvents) {
    // Lock to serialize access from multiple callback threads.
    std::lock_guard<std::mutex> guard(state_mutex_);
    if (remoteProfiledEvents_) {
      (*remoteProfiledEvents_).emplace_back(remoteProfiledEvents);
    } else {
      remoteProfiledEvents_ = {std::move(remoteProfiledEvents)};
    }
  }

  void pushRange(
      const at::StringView& name,
      const char* msg = "",
      int64_t sequence_nr = -1,
      std::vector<std::vector<int64_t>>&& shapes = {},
      at::RecordFunctionHandle handle = 0) {
    if (config_.state == ProfilerState::Disabled) {
      return;
    }
    if (config_.state == ProfilerState::NVTX) {
      cuda_stubs->nvtxRangePushA(getNvtxStr(
          name, msg, sequence_nr, shapes).c_str());
    } else {
      Event evt(EventKind::PushRange,
          name,
          at::RecordFunction::currentThreadId(),
          config_.state == ProfilerState::CUDA,
          handle,
          std::move(shapes),
          at::RecordFunction::getDefaultNodeId());
      evt.setSequenceNr(sequence_nr);
      getEventList().record(std::move(evt));
    }
  }

  void popRange(uint64_t thread_id, at::RecordFunctionHandle handle) {
    if (config_.state == ProfilerState::Disabled) {
      return;
    }
    if (config_.state == ProfilerState::NVTX) {
      cuda_stubs->nvtxRangePop();
    } else {
      // In some cases RecordFunction (and popRange) may be
      // called on a different thread than pushRange
      // As a convention, we put the async pop on the original
      // thread and save current thread id in pop event
      Event evt(EventKind::PopRange,
          at::StringView(""),
          at::RecordFunction::currentThreadId(),
          config_.state == ProfilerState::CUDA,
          handle);
      evt.setNodeId(at::RecordFunction::getDefaultNodeId());
      getEventList(thread_id).record(std::move(evt));
    }
  }

  void setCallbackHandle(at::CallbackHandle handle) {
    handle_ = handle;
  }

  at::CallbackHandle callbackHandle() const {
    return handle_;
  }

  void reportMemoryUsage(
      void* /* unused */, int64_t alloc_size, c10::Device device) override {
    if (config_.profile_memory && config_.state != ProfilerState::Disabled) {
      uint64_t thread_id = at::RecordFunction::currentThreadId();
      Event evt(
          EventKind::MemoryAlloc,
          at::StringView(""),
          thread_id,
          config_.state == ProfilerState::CUDA);
      evt.updateMemoryStats(alloc_size, device);
      getEventList(thread_id).record(std::move(evt));
    }
  }

  bool memoryProfilingEnabled() const override {
    return config_.profile_memory;
  }

 private:
  std::string getNvtxStr(
      const at::StringView& name,
      const char* msg,
      int64_t sequence_nr,
      const std::vector<std::vector<int64_t>>& shapes) const {
    if (sequence_nr >= 0 || shapes.size() > 0) {
      std::stringstream s;
      if (sequence_nr >= 0) {
        s << name.str() << msg << sequence_nr;
      }
      if (shapes.size() > 0) {
        s << ", sizes = [";
        for (size_t idx = 0; idx < shapes.size(); ++idx) {
          if (shapes[idx].size() > 0) {
            s << "[";
            for (size_t dim = 0; dim < shapes[idx].size(); ++dim) {
              s << shapes[idx][dim];
              if (dim < shapes[idx].size() - 1) {
                s << ", ";
              }
            }
            s << "]";
          } else {
            s << "[]";
          }
          if (idx < shapes.size() - 1) {
            s << ", ";
          }
        }
        s << "]";
      }
      return s.str();
    } else {
      return name.str();
    }
  }

  RangeEventList& getEventList(int64_t thread_id = -1) {
    if (thread_id < 0) {
      thread_id = at::RecordFunction::currentThreadId();
    }
    RangeEventList* list_ptr = nullptr;
    std::lock_guard<std::mutex> guard(state_mutex_);
    auto it = event_lists_map_.find(thread_id);
    if (it != event_lists_map_.end()) {
      list_ptr = it->second.get();
    } else {
      auto event_list = std::make_shared<RangeEventList>();
      event_lists_map_[thread_id] = event_list;
      list_ptr = event_list.get();
    }
    return *list_ptr;
  }

  std::mutex state_mutex_;
  std::unordered_map<uint64_t, std::shared_ptr<RangeEventList>>
      event_lists_map_;

  ProfilerConfig config_ = ProfilerConfig(ProfilerState::Disabled, false, false);
  at::CallbackHandle handle_ = 0;
  c10::optional<std::vector<std::vector<Event>>> remoteProfiledEvents_;
};

ProfilerThreadLocalState* getProfilerTLSState() {
  const auto& state = c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::PROFILER_STATE);
  return dynamic_cast<ProfilerThreadLocalState*>(state.get());
}

void pushProfilingCallbacks() {
  auto state_ptr = getProfilerTLSState();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");
  auto handle = at::addThreadLocalCallback(at::RecordFunctionCallback(
      [](const at::RecordFunction& fn) {
        auto state_ptr = getProfilerTLSState();
        if (!state_ptr || state_ptr->config().state == ProfilerState::Disabled) {
          return;
        }

        auto* msg = (fn.seqNr() >= 0) ? ", seq = " : "";
        if (state_ptr->config().report_input_shapes) {
          std::vector<std::vector<int64_t>> inputSizes;
          inputSizes.reserve(fn.inputs().size());
          for (const c10::IValue& input : fn.inputs()) {
            if (!input.isTensor()) {
              inputSizes.emplace_back();
              continue;
            }
            const at::Tensor& tensor = input.toTensor();
            if (tensor.defined()) {
              inputSizes.push_back(input.toTensor().sizes().vec());
            } else {
              inputSizes.emplace_back();
            }
          }
          state_ptr->pushRange(
              fn.name(), msg, fn.seqNr(), std::move(inputSizes), fn.handle());
        } else {
          state_ptr->pushRange(fn.name(), msg, fn.seqNr(), {}, fn.handle());
        }
      },
      [](const at::RecordFunction& fn) {
        auto state_ptr = getProfilerTLSState();
        if (!state_ptr || state_ptr->config().state == ProfilerState::Disabled) {
          return;
        }
        state_ptr->popRange(fn.getStartCallbacksThreadId(), fn.handle());
      })
    .needsInputs(state_ptr->config().report_input_shapes)
    .needsIds(true));
  state_ptr->setCallbackHandle(handle);
}

const int kCUDAWarmupStart = 5;

// temp. workaround for dispatcher ::Profiler key
thread_local std::vector<std::shared_ptr<at::RecordFunctionGuard>> g_;

} // namespace

void registerCUDAMethods(CUDAStubs* stubs) {
  cuda_stubs = stubs;
}

ProfilerConfig::~ProfilerConfig() = default;

at::IValue ProfilerConfig::toIValue() const {
  c10::impl::GenericList eventIValueList(at::AnyType::get());
  eventIValueList.reserve(NUM_PROFILER_CFG_IVALUE_IDX);
  eventIValueList.emplace_back(static_cast<int64_t>(state));
  eventIValueList.emplace_back(report_input_shapes);
  eventIValueList.emplace_back(profile_memory);
  return eventIValueList;
}

ProfilerConfig ProfilerConfig::fromIValue(
    const at::IValue& profilerConfigIValue) {
  TORCH_INTERNAL_ASSERT(
      profilerConfigIValue.isList(),
      "Expected IValue to contain type c10::impl::GenericList");
  auto ivalues = profilerConfigIValue.toList();
  TORCH_INTERNAL_ASSERT(
      ivalues.size() == NUM_PROFILER_CFG_IVALUE_IDX,
      c10::str(
          "Expected exactly ",
          NUM_PROFILER_CFG_IVALUE_IDX,
          " ivalues to resconstruct ProfilerConfig."));
  return ProfilerConfig(
      static_cast<ProfilerState>(ivalues.get(ProfilerIValueIdx::STATE).toInt()),
      ivalues.get(ProfilerIValueIdx::REPORT_INPUT_SHAPES).toBool(),
      ivalues.get(ProfilerIValueIdx::PROFILE_MEMORY).toBool());
}

ProfilerConfig getProfilerConfig() {
  auto state_ptr = getProfilerTLSState();
  TORCH_CHECK(
      state_ptr,
      "Tried to access profiler config, but profiler is not enabled!");
  return state_ptr->config();
}

bool profilerEnabled() {
  auto state_ptr = getProfilerTLSState();
  return state_ptr && state_ptr->config().state != ProfilerState::Disabled;
}

void enableProfiler(const ProfilerConfig& new_config) {
  TORCH_CHECK(new_config.state != ProfilerState::NVTX || cuda_stubs->enabled(),
    "Can't use NVTX profiler - PyTorch was compiled without CUDA");

  auto state_ptr = getProfilerTLSState();
  TORCH_CHECK(!state_ptr, "Profiler is already enabled on this thread");

  auto state = std::make_shared<ProfilerThreadLocalState>(new_config);
  c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, state);

  pushProfilingCallbacks();
  g_.emplace_back(std::make_shared<at::RecordFunctionGuard>());

  if (new_config.state == ProfilerState::CUDA) {
    // event recording appears to have some startup overhead, so we need to
    // to generate some dummy events first before recording synchronization events
    for (int idx = 0; idx < kCUDAWarmupStart; ++idx) {
      cuda_stubs->onEachDevice([state](int /* unused */) {
          state->mark("__cuda_startup");
          cuda_stubs->synchronize();
      });
    }

    // cuda events must be on the same device, so we need a start event recorded
    // for each gpu. we then use this event to synchronize time on the GPU
    // with the CPU clock.
    cuda_stubs->onEachDevice([state](int d) {
        state->mark("__cuda_start_event");
    });
  }
  state->mark("__start_profile", false);
}

thread_event_lists disableProfiler() {
  // all the DebugInfoBase objects are scope based and supposed to use DebugInfoGuard
  auto state = c10::ThreadLocalDebugInfo::_pop(c10::DebugInfoKind::PROFILER_STATE);
  auto state_ptr = static_cast<ProfilerThreadLocalState*>(state.get());
  TORCH_CHECK(state_ptr && state_ptr->config().state != ProfilerState::Disabled,
      "Can't disable profiler when it's not running");

  g_.pop_back();
  at::removeCallback(state_ptr->callbackHandle());

  if (state_ptr->config().state == ProfilerState::NVTX) {
    return thread_event_lists();
  }

  state_ptr->mark("__stop_profile");

  return state_ptr->consolidate();
}

void addEventList(std::vector<Event>&& profiledEvents) {
  auto state_ptr = getProfilerTLSState();
  TORCH_CHECK(state_ptr, "Profiler must be enabled.");
  state_ptr->setOrAddRemoteProfiledEvents(std::move(profiledEvents));
}

void Event::record(bool record_cuda) {
  if (record_cuda) {
    cuda_stubs->record(&device_, &cuda_event, &cpu_ns_);
    return;
  }
  cpu_ns_ = getTime();
}

/* static */ Event Event::fromIValue(const at::IValue& eventIValue) {
  TORCH_INTERNAL_ASSERT(
      eventIValue.isList(),
      "Expected IValue to contain type c10::impl::GenericList");
  auto ivalues = eventIValue.toList();
  TORCH_INTERNAL_ASSERT(
      ivalues.size() >= NUM_EVENT_IVALUE_IDX,
      "Expected at least ",
      NUM_EVENT_IVALUE_IDX,
      " elements to reconstruct Event.");

  Event evt(
      static_cast<EventKind>(
          ivalues.get(EventIValueIdx::KIND).toInt()), // EventKind
      at::StringView(ivalues.get(EventIValueIdx::NAME).toStringRef()), // name
      ivalues.get(EventIValueIdx::THREAD_ID).toInt(), // thread_id
      static_cast<at::RecordFunctionHandle>(
          ivalues.get(EventIValueIdx::HANDLE).toDouble()), // handle
      {}, // TODO: record shapes
      ivalues.get(EventIValueIdx::NODE_ID).toInt(), // node id
      true, // is remote
      ivalues.get(EventIValueIdx::CPU_MEM_USAGE).toInt(), // cpu_mem_usage
      ivalues.get(EventIValueIdx::CPU_NS).toInt(), // cpu_ns
      ivalues.get(EventIValueIdx::CUDA_RECORDED).toBool(), // was cuda recorded
      ivalues.get(EventIValueIdx::CUDA_MEM_USAGE).toInt(), // cuda memory usage
      ivalues.get(EventIValueIdx::CUDA_DEVICE).toInt(), // device
      ivalues.get(EventIValueIdx::CUDA_US).toInt() // cuda_us
  );
  return evt;
}

at::IValue Event::toIValue() const {
  c10::impl::GenericList eventIValueList(at::AnyType::get());
  eventIValueList.reserve(NUM_EVENT_IVALUE_IDX);
  eventIValueList.emplace_back(static_cast<int64_t>(kind_));
  eventIValueList.emplace_back(std::string(name_.str()));
  eventIValueList.emplace_back(thread_id_);
  eventIValueList.emplace_back(static_cast<double>(handle_));
  eventIValueList.emplace_back(node_id_);
  eventIValueList.emplace_back(cpu_memory_usage_);
  eventIValueList.emplace_back(cpu_ns_);
  // CUDA event information
  bool cuda_profiling_enabled = has_cuda();
  eventIValueList.emplace_back(cuda_profiling_enabled);
  eventIValueList.emplace_back(static_cast<int64_t>(cuda_memory_usage_));
  eventIValueList.emplace_back(device_);
  eventIValueList.emplace_back(cuda_us_);
  return at::IValue(eventIValueList);
}

double Event::cuda_elapsed_us(const Event& e) const {
  TORCH_CHECK(e.has_cuda() && has_cuda(), "Events were not recorded for CUDA");
  TORCH_CHECK(
      e.device() == device(),
      c10::str(
          "Events are not on the same device: ", e.device(), " vs ", device()));
  if (isRemote() && e.isRemote()) {
    // validate that cuda_us_ has been set properly.
    TORCH_INTERNAL_ASSERT(cuda_us_ >= 0 && e.cuda_us_ >= 0);
    return static_cast<double>(e.cuda_us_ - cuda_us_);
  }
  return cuda_stubs->elapsed(&cuda_event, &e.cuda_event);
}

CUDAStubs::~CUDAStubs() = default;


static jit::CodeTemplate event_template(R"(
{
  "name": "${name}",
  "ph": "X",
  "ts": ${ts},
  "dur": ${dur},
  "tid": ${tid},
  "pid": "CPU Functions",
  "args": {}
})");

void writeProfilerEventsToStream(std::ostream& out, const std::vector<Event*>& events) {
  TORCH_CHECK(out, "Could not open file");
  Event* profiler_start = nullptr;
  for (Event* e : events) {
    if (0 == strcmp(e->name(), "__start_profile")) {
      profiler_start = e;
      break;
    }
  }
  TORCH_CHECK(profiler_start, "Could not find __start_profile mark");

  struct PairHash {
    size_t operator()(std::pair<at::RecordFunctionHandle, int> p) const
        noexcept {
      return std::hash<at::RecordFunctionHandle>()(p.first) ^ std::hash<int64_t>()(p.second);
    }
  };
  std::unordered_map<std::pair<at::RecordFunctionHandle, int64_t>, Event*, PairHash> events_map;
  out << "[\n";
  bool first = true;
  for (Event* evt : events) {
    if (evt->kind() == "push") {
      events_map[std::make_pair(evt->handle(), evt->node_id())] = evt;
    } else if (evt->kind() == "pop") {
      if (!first) {
        out << ",\n";
      }
      first = false;
      auto it = events_map.find(std::make_pair(evt->handle(), evt->node_id()));
      TORCH_CHECK(it != events_map.end(), "Unmatched pop event");
      Event* evt_start = it->second;
      events_map.erase(it);

      jit::TemplateEnv env;
      env.s("name", evt_start->name());
      env.d("ts", profiler_start->cpu_elapsed_us(*evt_start));
      env.d("dur", evt_start->cpu_elapsed_us(*evt));
      env.d("tid", evt_start->thread_id());
      out << event_template.format(env);
    }
  }
  out << "]\n";
}


RecordProfile::RecordProfile(std::ostream& out)
: out_(out) {
  init();
}

RecordProfile::RecordProfile(const std::string& filename)
: file_(new std::ofstream(filename)), out_(*file_) {
  init();
}

void RecordProfile::init() {
  enableProfiler(ProfilerConfig(
      ProfilerState::CPU,
      /* report_input_shapes */ false,
      /* profile_memory */ false));
}

RecordProfile::~RecordProfile() {
  thread_event_lists event_lists = disableProfiler();
  std::vector<Event*> events;
  for (auto& l : event_lists) {
    for (auto& e : l) {
        events.push_back(&e);
    }
  }
  processEvents(events);
  if (file_){
    file_->close();
  }
}

void RecordProfile::processEvents(const std::vector<Event*>& events) {
  writeProfilerEventsToStream(out_, events);
}

}}}
