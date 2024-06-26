#include <torch/csrc/jit/runtime/script_profile.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <unordered_set>

#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/jit/api/function_impl.h>

namespace torch::jit {

namespace {

class ProfilesRegistry {
 public:
  bool empty() {
    return empty_.load(std::memory_order_relaxed);
  }

  void addProfile(ScriptProfile& p) {
    std::lock_guard<std::mutex> g(mutex_);
    enabledProfiles_.emplace(&p);
    empty_.store(false, std::memory_order_relaxed);
  }

  void removeProfile(ScriptProfile& p) {
    std::lock_guard<std::mutex> g(mutex_);
    enabledProfiles_.erase(&p);
    if (enabledProfiles_.empty()) {
      empty_.store(true, std::memory_order_relaxed);
    }
  }

  void send(std::unique_ptr<profiling::Datapoint> datapoint) {
    auto shared = std::shared_ptr<profiling::Datapoint>(std::move(datapoint));
    std::lock_guard<std::mutex> g(mutex_);
    for (auto* p : enabledProfiles_) {
      p->addDatapoint(shared);
    }
  }

 private:
  std::atomic<bool> empty_{true};
  std::mutex mutex_;
  std::unordered_set<ScriptProfile*> enabledProfiles_;
};

ProfilesRegistry& getProfilesRegistry() {
  static auto registry = std::ref(*new ProfilesRegistry{});
  return registry;
}

auto initBindings() {
  torch::class_<SourceRef>("profiling", "SourceRef")
      .def(
          "starting_lineno",
          [](const c10::intrusive_ptr<SourceRef>& self) {
            return static_cast<int64_t>((*self)->starting_line_no());
          })
      .def("text", [](const c10::intrusive_ptr<SourceRef>& self) {
        return (*self)->text_str().str();
      });

  torch::class_<InstructionStats>("profiling", "InstructionStats")
      .def(
          "count",
          [](const c10::intrusive_ptr<InstructionStats>& self) {
            return self->count;
          })
      .def("duration_ns", [](const c10::intrusive_ptr<InstructionStats>& self) {
        return static_cast<int64_t>(self->duration.count());
      });

  torch::class_<SourceStats>("profiling", "SourceStats")
      .def(
          "source",
          [](const c10::intrusive_ptr<SourceStats>& self) {
            return c10::make_intrusive<SourceRef>(self->getSourceRef());
          })
      .def("line_map", &SourceStats::getLineMap);

  torch::class_<ScriptProfile>("profiling", "_ScriptProfile")
      .def(torch::init<>())
      .def("enable", &ScriptProfile::enable)
      .def("disable", &ScriptProfile::disable)
      .def("_dump_stats", [](const c10::intrusive_ptr<ScriptProfile>& self) {
        const auto& stats = self->dumpStats();
        c10::List<c10::intrusive_ptr<SourceStats>> ret;
        for (const auto& source : stats) {
          SourceStats::LineMap lineMap;
          for (const auto& line : source.second) {
            lineMap.insert(
                line.first, c10::make_intrusive<InstructionStats>(line.second));
          }
          ret.push_back(c10::make_intrusive<SourceStats>(
              source.first, std::move(lineMap)));
        }
        return ret;
      });
  return nullptr;
}

const auto C10_UNUSED torchBindInitializer = initBindings();

} // namespace

namespace profiling {

InstructionSpan::InstructionSpan(Node& node) {
  datapoint_ = std::make_unique<Datapoint>(node.sourceRange());
}

InstructionSpan::~InstructionSpan() {
  datapoint_->end = std::chrono::steady_clock::now();
  getProfilesRegistry().send(std::move(datapoint_));
}

bool isProfilingOngoing() {
  return !getProfilesRegistry().empty();
}

} // namespace profiling

void ScriptProfile::enable() {
  if (!std::exchange(enabled_, true)) {
    getProfilesRegistry().addProfile(*this);
  }
}

void ScriptProfile::disable() {
  if (std::exchange(enabled_, false)) {
    getProfilesRegistry().removeProfile(*this);
  }
}

void ScriptProfile::addDatapoint(
    std::shared_ptr<profiling::Datapoint> datapoint) {
  TORCH_CHECK(enabled_, "Cannot only add datapoint to disabled profilers.");
  datapoints_.push_back(std::move(datapoint));
}

const ScriptProfile::SourceMap& ScriptProfile::dumpStats() {
  TORCH_CHECK(!enabled_, "Only disabled profilers are allowed to dump stats.");

  for (const auto& datapoint : datapoints_) {
    if (const auto& source = datapoint->sourceRange.source()) {
      if (auto fileLineCol = datapoint->sourceRange.file_line_col()) {
        auto it = sourceMap_.find(*source.get());
        if (it == sourceMap_.end()) {
          it = sourceMap_.emplace(SourceRef{source}, LineMap{}).first;
        }
        auto& stats = it->second[std::get<1>(*fileLineCol)];
        stats.count++;
        stats.duration += datapoint->end - datapoint->start;
      }
    }
  }
  datapoints_.clear();

  return sourceMap_;
}

ScriptProfile::~ScriptProfile() {
  if (enabled_) {
    getProfilesRegistry().removeProfile(*this);
  }
}

} // namespace torch::jit
