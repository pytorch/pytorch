#pragma once
#include <torch/csrc/profiler/standalone/privateuse1_observer.h>

namespace torch::profiler::impl {

void pushOpenRegCallbacks(const ProfilerConfig& config, const std::unordered_set<at::RecordScope>& scopes);

} // namespace torch::profiler::impl