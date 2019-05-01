#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/function.h>

#include <sstream>
#include <fstream>

namespace torch { namespace autograd { namespace profiler {

CUDAStubs default_stubs;
constexpr CUDAStubs* default_stubs_addr = &default_stubs;
// constant initialization, so it is guarenteed to be initialized before
// static initialization calls which may invoke registerCUDAMethods
static CUDAStubs* cuda_stubs = default_stubs_addr;

void registerCUDAMethods(CUDAStubs* stubs) {
  cuda_stubs = stubs;
}

ProfilerState state = ProfilerState::Disabled;
uint16_t next_thread_id = 0;
std::mutex all_event_lists_mutex;
std::list<std::shared_ptr<RangeEventList>> all_event_lists;
thread_local std::shared_ptr<RangeEventList> event_list;
thread_local uint16_t thread_id;

RangeEventList& getEventList() {
  if (!event_list) {
    std::lock_guard<std::mutex> guard(all_event_lists_mutex);
    event_list = std::make_shared<RangeEventList>();
    thread_id = next_thread_id++;
    all_event_lists.emplace_front(event_list);
  }
  return *event_list;
}

void mark(std::string name, bool include_cuda /* = true */) {
  if (state == ProfilerState::Disabled) {
    return;
  }
  if (state == ProfilerState::NVTX) {
    cuda_stubs->nvtxMarkA(name.c_str());
  } else {
    getEventList().record(
        EventKind::Mark,
        StringView(std::move(name)),
        thread_id,
        include_cuda && state == ProfilerState::CUDA);
  }
}

void pushRangeImpl(const StringView& name, const char* msg="", int64_t sequence_nr=-1) {
  if (state == ProfilerState::Disabled) {
    return;
  }
  if (state == ProfilerState::NVTX) {
    if(sequence_nr >= 0) {
      std::stringstream s;
      s << name.str() << msg << sequence_nr;
      cuda_stubs->nvtxRangePushA(s.str().c_str());
    } else {
      cuda_stubs->nvtxRangePushA(name.str());
    }
  } else {
    getEventList().record(
        EventKind::PushRange,
        name,
        thread_id,
        state == ProfilerState::CUDA);
  }
}

void pushRange(std::string name) {
  pushRangeImpl(StringView(std::move(name)));
}

void popRange() {
  if (state == ProfilerState::Disabled) {
    return;
  }
  if (state == ProfilerState::NVTX) {
    cuda_stubs->nvtxRangePop();
  } else {
    getEventList().record(
        EventKind::PopRange,
        StringView(""),
        thread_id,
        state == ProfilerState::CUDA);
  }
}

void enableProfiler(ProfilerState new_state) {
  AT_ASSERT(new_state != ProfilerState::Disabled);
  if (new_state == ProfilerState::NVTX && !cuda_stubs->enabled())
    throw std::runtime_error("Can't use NVTX profiler - PyTorch was compiled without CUDA");
  if (state != ProfilerState::Disabled && new_state != state) {
      throw std::runtime_error("can't change kind of profiling (e.g. NVTX to CPU) while profiler is running");
  }

  pushCallback([](const RecordFunction& fn) {
    auto* msg = (fn.seqNr() >= 0) ? ", seq = " : "";
    pushRangeImpl(fn.name(), msg, fn.seqNr());
  },
  [](const RecordFunction& /* unused */) {
    popRange();
  });
  state = new_state;

  if(state == ProfilerState::CUDA) {
    // event recording appears to have some startup overhead, so we need to
    // to generate some dummy events first before recording syncrhonization events
    for(int i = 0; i < 5; i++) {
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

thread_event_lists disableProfiler() {
  if (state == ProfilerState::Disabled) {
    throw std::runtime_error("can't disable profiler when it's not running");
  }
  ProfilerState old_state = state;
  mark("__stop_profile");

  popCallback();
  state = ProfilerState::Disabled;

  if (old_state == ProfilerState::NVTX) {
    return thread_event_lists();
  } else {
    thread_event_lists result;
    std::lock_guard<std::mutex> guard(all_event_lists_mutex);
    for (auto it = all_event_lists.begin(); it != all_event_lists.end();) {
      auto & list = *it;
      result.emplace_back(list->consolidate());
      // GC lists that are not held by any threads
      if (list.use_count() == 1) {
        auto current_it = it;
        ++it;
        all_event_lists.erase(current_it);
      } else {
        ++it;
      }
    }
    return result;
  }
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
  enableProfiler(ProfilerState::CPU);
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
  AT_CHECK(out_, "could not open file");
  Event* start = nullptr;
  for (Event* e : events) {
    if(0 == strcmp(e->name(), "__start_profile")) {
      start = e;
      break;
    }
  }
  AT_CHECK(start, "could not find start?");
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
