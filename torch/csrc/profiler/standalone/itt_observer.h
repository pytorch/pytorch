#include <torch/csrc/profiler/api.h>

namespace torch::profiler::impl {

void pushITTCallbacks(
    const ProfilerConfig& config,
    const std::unordered_set<at::RecordScope>& scopes);

} // namespace torch::profiler::impl
