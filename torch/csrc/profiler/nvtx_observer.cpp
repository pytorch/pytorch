#include <torch/csrc/profiler/nvtx_observer.h>

#include <torch/csrc/profiler/util.h>

namespace torch {
namespace profiler {
namespace impl {

struct NVTXThreadLocalState : ProfilerThreadLocalStateBase {
  explicit NVTXThreadLocalState(const ProfilerConfig& config)
      : ProfilerThreadLocalStateBase(config) {
    // Only `report_input_shapes` makes sense in this context.
    TORCH_CHECK(!config.profile_memory);
    TORCH_CHECK(!config.with_stack);
    TORCH_CHECK(!config.with_flops);
    TORCH_CHECK(!config.with_modules);
  }
  ~NVTXThreadLocalState() override = default;

  ActiveProfilerType profilerType() override {
    return ActiveProfilerType::NVTX;
  }

  void reportMemoryUsage(void*, int64_t, int64_t, int64_t, c10::Device)
      override {}

  static NVTXThreadLocalState* getTLS() {
    auto tls = ProfilerThreadLocalStateBase::getTLS();
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        tls == nullptr || tls->profilerType() == ActiveProfilerType::NVTX);
    return static_cast<NVTXThreadLocalState*>(tls);
  }
};

template <bool report_input_shapes>
std::unique_ptr<at::ObserverContext> enterNVTX(const at::RecordFunction& fn) {
  if (NVTXThreadLocalState::getTLS() != nullptr) {
    torch::profiler::impl::cudaStubs()->nvtxRangePushA(
        torch::profiler::impl::getNvtxStr(
            fn.name(),
            fn.seqNr(),
            report_input_shapes ? torch::profiler::impl::inputSizes(fn)
                                : std::vector<std::vector<int64_t>>())
            .c_str());
  }
  return nullptr;
}

void pushNVTXCallbacks(
    const ProfilerConfig& config,
    const std::unordered_set<at::RecordScope>& scopes) {
  TORCH_CHECK(
      torch::profiler::impl::cudaStubs()->enabled(),
      "Can't use NVTX profiler - PyTorch was compiled without CUDA");

  c10::ThreadLocalDebugInfo::_push(
      c10::DebugInfoKind::PROFILER_STATE,
      std::make_shared<NVTXThreadLocalState>(config));

  auto state_ptr = NVTXThreadLocalState::getTLS();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");

  auto handle = at::addThreadLocalCallback(
      at::RecordFunctionCallback(
          state_ptr->config().report_input_shapes
              ? &enterNVTX</*report_input_shapes=*/true>
              : &enterNVTX</*report_input_shapes=*/false>,
          [](const at::RecordFunction&, at::ObserverContext*) {
            torch::profiler::impl::cudaStubs()->nvtxRangePop();
          })
          .needsInputs(config.report_input_shapes)
          .scopes(scopes));
  state_ptr->setCallbackHandle(handle);
}

} // namespace impl
} // namespace profiler
} // namespace torch
