#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/jit/frontend/code_template.h>

#include <fstream>
#include <list>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <ATen/ThreadLocalState.h>

namespace torch { namespace autograd { namespace profiler {

namespace {

CUDAStubs default_stubs;
constexpr CUDAStubs* default_stubs_addr = &default_stubs;
// Constant initialization, so it is guaranteed to be initialized before
// static initialization calls which may invoke registerCUDAMethods
static CUDAStubs* cuda_stubs = default_stubs_addr;

thread_local ProfilerConfig config(ProfilerState::Disabled, false);
// Protects access all_event_lists_map.
std::mutex all_event_lists_map_mutex;
std::unordered_map<uint16_t, std::shared_ptr<RangeEventList>>
    all_event_lists_map;
thread_local std::shared_ptr<RangeEventList> event_list;
thread_local uint16_t thread_id;

bool unused_ = []() {
  at::ThreadLocalState::registerThreadLocalSetting(
    at::ThreadLocalSetting::PROFILER,
    []() {
      return at::SettingValue{.pair =
          {(int32_t)config.state, (int32_t)config.report_input_shapes}};
    },
    [](at::SettingValue v) {
      // ThreadLocalState propagates profiler state across threads and
      // automatically enables/disables profiling in async tasks.
      // Note, that in case of a subtask profiling, we don't
      // want to consolidate profiling results but let the original profiler
      // keep running
      if (v.pair.first != (int32_t)ProfilerState::Disabled) {
        enableProfiler(
            ProfilerConfig{(ProfilerState)v.pair.first, (bool)v.pair.second},
            /* emit_start */ false);
      } else if (config.state != ProfilerState::Disabled) {
        disableProfiler(/* consolidate_results */ false);
      }
    }
  );
  return true;
}();

// Nested profiler depth;
// note: only the outermost profiler is active;
// used to keep a track of enableProfiler/disableProfiler
// within a single thread
thread_local int recursion_depth_ = 0;

} // namespace

void registerCUDAMethods(CUDAStubs* stubs) {
  cuda_stubs = stubs;
}

ProfilerConfig::~ProfilerConfig() = default;

RangeEventList& getEventList() {
  if (!event_list) {
    std::lock_guard<std::mutex> guard(all_event_lists_map_mutex);
    event_list = std::make_shared<RangeEventList>();
    thread_id = RecordFunction::currentThreadId();
    all_event_lists_map.emplace(thread_id, event_list);
  }
  return *event_list;
}

void mark(std::string name, bool include_cuda /* = true */) {
  if (config.state == ProfilerState::Disabled) {
    return;
  }
  if (config.state == ProfilerState::NVTX) {
    cuda_stubs->nvtxMarkA(name.c_str());
  } else {
    auto& list = getEventList();
    std::lock_guard<std::mutex> guard(all_event_lists_map_mutex);
    list.record(
        EventKind::Mark,
        StringView(std::move(name)),
        thread_id,
        include_cuda && config.state == ProfilerState::CUDA);
  }
}

bool profilerEnabled() {
  return config.state != ProfilerState::Disabled;
}

void pushRange(
    const StringView& name,
    const char* msg = "",
    int64_t sequence_nr = -1,
    std::vector<std::vector<int64_t>>&& shapes = {}) {
  if (config.state == ProfilerState::Disabled) {
    return;
  }
  if (config.state == ProfilerState::NVTX) {
    if(sequence_nr >= 0 || shapes.size() > 0) {
      std::stringstream s;
      if(sequence_nr >= 0)
        s << name.str() << msg << sequence_nr;
      if(shapes.size() > 0) {
        s << ", sizes = [";
        for(int i = 0; i < shapes.size(); i++) {
          if(shapes[i].size() > 0) {
            s << "[";
            for(int dim = 0; dim < shapes[i].size(); dim++) {
              s << shapes[i][dim];
              if(dim < shapes[i].size() - 1)
                s << ", ";
            }
            s << "]";
          }
          else
            s << "[]";
          if(i < shapes.size() - 1)
            s << ", ";
        }
        s << "]";
      }
      cuda_stubs->nvtxRangePushA(s.str().c_str());
    } else {
      cuda_stubs->nvtxRangePushA(name.str());
    }
  } else {
    auto& list = getEventList();
    std::lock_guard<std::mutex> guard(all_event_lists_map_mutex);
    list.record(
        EventKind::PushRange,
        name,
        thread_id,
        config.state == ProfilerState::CUDA,
        std::move(shapes));
  }
}

void popRange() {
  if (config.state == ProfilerState::Disabled) {
    return;
  }
  if (config.state == ProfilerState::NVTX) {
    cuda_stubs->nvtxRangePop();
  } else {
    auto& list = getEventList();
    std::lock_guard<std::mutex> guard(all_event_lists_map_mutex);
    list.record(
        EventKind::PopRange,
        StringView(""),
        thread_id,
        config.state == ProfilerState::CUDA);
  }
}

void enableProfiler(ProfilerConfig new_config, bool emit_start) {
  TORCH_CHECK(new_config.state != ProfilerState::Disabled);
  if (config.state != ProfilerState::Disabled) {
    TORCH_WARN("Trying to enable profiler when profiler is already active,"
        " new profiler is inactive");
    ++recursion_depth_;
    return;
  }

  TORCH_CHECK(new_config.state != ProfilerState::NVTX || cuda_stubs->enabled(),
      "Can't use NVTX profiler - PyTorch was compiled without CUDA");

  pushCallback(
      [new_config](const RecordFunction& fn) {
        auto* msg = (fn.seqNr() >= 0) ? ", seq = " : "";
        if (new_config.report_input_shapes) {
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
          pushRange(fn.name(), msg, fn.seqNr(), std::move(inputSizes));
        } else {
          pushRange(fn.name(), msg, fn.seqNr(), {});
        }
        return true;
      },
      [](const RecordFunction& fn) {
        if (fn.getStartCallbacksThreadId() !=
                RecordFunction::currentThreadId()) {
          // If we're not in a thread that ran start callbacks, then find
          // the eventList that was created for the original thread_id. Then,
          // record the end event on this list so that the block is added to
          // the correct list, instead of to a new list. This should only run
          // when calling RecordFunction::end() in a different thread.
          if (config.state == ProfilerState::Disabled) {
            return;
          } else {
            std::lock_guard<std::mutex> guard(all_event_lists_map_mutex);
            const auto& eventListIter =
                all_event_lists_map.find(fn.getStartCallbacksThreadId());
            TORCH_INTERNAL_ASSERT(
                eventListIter != all_event_lists_map.end(),
                "Did not find thread_id matching ",
                fn.getStartCallbacksThreadId());

            auto& eventList = eventListIter->second;
            eventList->record(
                      EventKind::PopRange,
                      StringView(""),
                      fn.getStartCallbacksThreadId(),
                      config.state == ProfilerState::CUDA);
          }
        } else {
          popRange();
        }
      },
      /* needs_inputs */ new_config.report_input_shapes,
      /* sampling_prob */ 1.0,
      /* scopes */ {RecordScope::FUNCTION, RecordScope::USER_SCOPE});
  config = new_config;

  if (emit_start) {
    if (config.state == ProfilerState::CUDA) {
      // event recording appears to have some startup overhead, so we need to
      // to generate some dummy events first before recording synchronization events
      for (int i = 0; i < 5; i++) {
        cuda_stubs->onEachDevice([](int d) {
            mark("__cuda_startup");
            cuda_stubs->synchronize();
        });
      }

      // cuda events must be on the same device, so we need a start event recorded
      // for each gpu. we then use this event to synchronize time on the GPU
      // with the CPU clock.
      cuda_stubs->onEachDevice([](int d) {
          mark("__cuda_start_event");
      });
    }
    mark("__start_profile", false);
  }
}

thread_event_lists disableProfiler(bool consolidate_results) {
  TORCH_CHECK(config.state != ProfilerState::Disabled,
      "Can't disable profiler when it's not running");

  if (recursion_depth_ > 0) {
    --recursion_depth_;
    return thread_event_lists();
  }

  ProfilerState old_state = config.state;
  popCallback();
  config = ProfilerConfig{ProfilerState::Disabled, false};
  if (!consolidate_results || old_state == ProfilerState::NVTX) {
    return thread_event_lists();
  }

  mark("__stop_profile");

  thread_event_lists result;
  std::lock_guard<std::mutex> guard(all_event_lists_map_mutex);
  for (auto it = all_event_lists_map.begin(); it != all_event_lists_map.end();) {
    auto & list = it->second;
    result.emplace_back(list->consolidate());
    // GC lists that are not held by any threads
    if (list.use_count() == 1) {
      auto current_it = it;
      ++it;
      all_event_lists_map.erase(current_it);
    } else {
      ++it;
    }
  }
  return result;
}

void Event::record(bool record_cuda) {
  if (record_cuda) {
    cuda_stubs->record(&device_, &event, &cpu_ns_);
    return;
  }
  cpu_ns_ = getTime();
}

double Event::cuda_elapsed_us(const Event & e) {
  if(!e.has_cuda() || !has_cuda()) {
    throw std::logic_error("Events were not recorded for CUDA");
  }
  if(e.device() != device()) {
    throw std::logic_error("Events are not on the same device");
  }
  return cuda_stubs->elapsed(event, e.event);
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


RecordProfile::RecordProfile(std::ostream& out)
: out_(out) {
  init();
}

RecordProfile::RecordProfile(const std::string& filename)
: file_(new std::ofstream(filename)), out_(*file_) {
  init();
}

void RecordProfile::init() {
  enableProfiler(ProfilerConfig(ProfilerState::CPU, false /* report shapes */));
}

RecordProfile::~RecordProfile() {
  thread_event_lists event_lists = disableProfiler();
  std::vector<Event*> events;
  for(auto& l : event_lists) {
    for(auto& e : l) {
        events.push_back(&e);
    }
  }
  processEvents(events);
  if (file_){
    file_->close();
  }
}

void RecordProfile::processEvents(const std::vector<Event*>& events) {
  TORCH_CHECK(out_, "could not open file");
  Event* start = nullptr;
  for (Event* e : events) {
    if(0 == strcmp(e->name(), "__start_profile")) {
      start = e;
      break;
    }
  }
  TORCH_CHECK(start, "could not find start?");
  std::vector<Event*> stack;
  out_ << "[\n";
  bool first = true;
  for(Event* e : events) {
    if(e->kind() == "push") {
      stack.push_back(e);
    } else if(e->kind() == "pop") {
      if(!first) {
        out_ << ",\n";
      }
      first = false;
      Event* e_start = stack.back();
      stack.pop_back();
      jit::TemplateEnv env;
      env.s("name", e_start->name());
      env.d("ts", start->cpu_elapsed_us(*e_start));
      env.d("dur", e_start->cpu_elapsed_us(*e));
      env.d("tid", e_start->thread_id());
      out_ << event_template.format(env);
    }
  }
  out_ << "]\n";
}

}}}
