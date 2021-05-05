#include <torch/csrc/jit/runtime/script_profile.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <unordered_set>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/api/function_impl.h>

namespace torch {
namespace jit {

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

} // namespace

namespace profiling {

InstructionSpan::InstructionSpan(Node& node) {
  if (getProfilesRegistry().empty()) {
    return;
  }

  datapoint_ = std::make_unique<Datapoint>(node.sourceRange());
}

InstructionSpan::~InstructionSpan() {
  if (!datapoint_) {
    return;
  }

  datapoint_->end = std::chrono::steady_clock::now();
  getProfilesRegistry().send(std::move(datapoint_));
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

const ScriptProfile::Stats& ScriptProfile::dumpStats() {
  TORCH_CHECK(!enabled_, "Only disabled profilers are allowed to dump stats.");

  for (const auto& datapoint : datapoints_) {
    if (const auto& source = datapoint->sourceRange.source()) {
      if (auto fileLineCol = datapoint->sourceRange.file_line_col()) {
        auto it = stats_.find(*source.get());
        if (it == stats_.end()) {
          it = stats_.emplace(SourceRef{source}, LineMap{}).first;
        }
        auto& stats = it->second[std::get<1>(*fileLineCol)];
        stats.count++;
        stats.duration += datapoint->end - datapoint->start;
      }
    }
  }
  datapoints_.clear();

  return stats_;
}

ScriptProfile::~ScriptProfile() {
  if (enabled_) {
    getProfilesRegistry().removeProfile(*this);
  }
}

} // namespace jit
} // namespace torch
