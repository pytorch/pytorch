#include <torch/csrc/profiler/standalone/itt_observer.h>

#include <torch/csrc/profiler/stubs/base.h>

namespace torch::profiler::impl {

struct ITTThreadLocalState : ProfilerStateBase {
  explicit ITTThreadLocalState(const ProfilerConfig& config)
      : ProfilerStateBase(config) {
    // Only `report_input_shapes` makes sense in this context.
    TORCH_CHECK(!config.profile_memory);
    TORCH_CHECK(!config.with_stack);
    TORCH_CHECK(!config.with_flops);
    TORCH_CHECK(!config.with_modules);
  }
  ~ITTThreadLocalState() override = default;

  ActiveProfilerType profilerType() override {
    return ActiveProfilerType::ITT;
  }

  void reportMemoryUsage(
      void* /*ptr*/,
      int64_t /*alloc_size*/,
      size_t /*total_allocated*/,
      size_t /*total_reserved*/,
      c10::Device /*device*/) override {}

  static ITTThreadLocalState* getTLS() {
    auto tls = ProfilerStateBase::get(/*global=*/false);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        tls == nullptr || tls->profilerType() == ActiveProfilerType::ITT);
    return static_cast<ITTThreadLocalState*>(tls);
  }
};

template <bool report_input_shapes>
static std::unique_ptr<at::ObserverContext> enterITT(
    const at::RecordFunction& fn) {
  if (ITTThreadLocalState::getTLS() != nullptr) {
    torch::profiler::impl::ittStubs()->rangePush(fn.name());
  }
  return nullptr;
}

void pushITTCallbacks(
    const ProfilerConfig& config,
    const std::unordered_set<at::RecordScope>& scopes) {
  TORCH_CHECK(
      torch::profiler::impl::ittStubs()->enabled(),
      "Can't use ITT profiler - PyTorch was compiled without ITT");

  c10::ThreadLocalDebugInfo::_push(
      c10::DebugInfoKind::PROFILER_STATE,
      std::make_shared<ITTThreadLocalState>(config));

  auto state_ptr = ITTThreadLocalState::getTLS();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");

  auto handle = at::addThreadLocalCallback(
      at::RecordFunctionCallback(
          state_ptr->config().report_input_shapes
              ? &enterITT</*report_input_shapes=*/true>
              : &enterITT</*report_input_shapes=*/false>,
          [](const at::RecordFunction&, at::ObserverContext*) {
            torch::profiler::impl::ittStubs()->rangePop();
          })
          .needsInputs(config.report_input_shapes)
          .scopes(scopes));
  state_ptr->setCallbackHandle(handle);
}

} // namespace torch::profiler::impl
