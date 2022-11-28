#include <torch/csrc/profiler/api.h>

namespace torch {
namespace profiler {
namespace impl {

void pushNVTXCallbacks(
    const ProfilerConfig& config,
    const std::unordered_set<at::RecordScope>& scopes);

} // namespace impl
} // namespace profiler
} // namespace torch
