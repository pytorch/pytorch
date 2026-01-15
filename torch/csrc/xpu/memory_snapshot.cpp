#include <ATen/Context.h>
#include <c10/util/Exception.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <torch/csrc/profiler/combined_traceback.h>
#include <torch/csrc/xpu/memory_snapshot.h>

namespace torch::xpu {

namespace {

std::shared_ptr<c10::GatheredContext> gather() {
  return CapturedTraceback::gather(true, true, false);
}

std::shared_ptr<c10::GatheredContext> gather_with_cpp() {
  return CapturedTraceback::gather(true, true, true);
}

inline void checkOptionIn(
    const std::string& option,
    std::initializer_list<std::string> valid,
    const char* error) {
  TORCH_CHECK(
      valid.end() != std::find(valid.begin(), valid.end(), option), error);
}

} // anonymous namespace

void _record_memory_history(
    std::optional<std::string> enabled,
    std::optional<std::string> context,
    const std::string& stacks,
    size_t max_entries,
    bool clear_history,
    const std::vector<std::string>& skip_actions) {
  if (enabled) {
    checkOptionIn(
        *enabled,
        {"state", "all"},
        "expected state to be 'state', 'all', or None");
  }
  if (context) {
    checkOptionIn(
        *context,
        {"state", "alloc", "all"},
        "expected context to be 'state', 'alloc', 'all', or None");
  }
  checkOptionIn(
      stacks, {"python", "all"}, "expected stacks to be 'python', or 'all'");

  c10::xpu::XPUCachingAllocator::CreateContextFn recorder = gather;
  if (enabled && context && stacks == "all") {
    recorder = gather_with_cpp;
    // warm up C++ stack unwinding
    unwind::unwind();
  }
  max_entries = (enabled && *enabled == "all") ? max_entries : 1;
  auto when = c10::xpu::XPUCachingAllocator::RecordContext::NEVER;
  if (context) {
    if (context == "all") {
      when = c10::xpu::XPUCachingAllocator::RecordContext::ALL;
    } else if (context == "alloc") {
      when = c10::xpu::XPUCachingAllocator::RecordContext::ALLOC;
    } else if (context == "state") {
      when = c10::xpu::XPUCachingAllocator::RecordContext::STATE;
    }
  }
  at::globalContext().lazyInitDevice(c10::DeviceType::XPU);

  c10::xpu::XPUCachingAllocator::recordHistory(
      enabled.has_value(),
      recorder,
      max_entries,
      when,
      clear_history,
      skip_actions);
}

} // namespace torch::xpu
