#include <ATen/record_function.h>
#include <c10/util/ApproximateClock.h>
#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/events.h>
#include <torch/csrc/profiler/stubs/base.h>
#include <torch/csrc/profiler/util.h>
#include <iostream>
#include <limits>
#include <memory>

#include "openreg_observer.h"

namespace torch::profiler::impl {
struct OpenRegThreadLocalState : ProfilerStateBase {
  explicit OpenRegThreadLocalState(const ProfilerConfig& config)
      : ProfilerStateBase(config) {
    // Only `report_input_shapes` makes sense in this context.
    TORCH_CHECK(!config.profile_memory);
    TORCH_CHECK(!config.with_stack);
    TORCH_CHECK(!config.with_flops);
    TORCH_CHECK(!config.with_modules);
  }
  ~OpenRegThreadLocalState() override = default;

  ActiveProfilerType profilerType() override {
    return ActiveProfilerType::PRIVATEUSE1;
  }

  void reportMemoryUsage(
      void* /*ptr*/,
      int64_t /*alloc_size*/,
      size_t /*total_allocated*/,
      size_t /*total_reserved*/,
      c10::Device /*device*/) override {}

  static OpenRegThreadLocalState* getTLS() {
    auto tls = ProfilerStateBase::get(/*global=*/false);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        tls == nullptr ||
        tls->profilerType() == ActiveProfilerType::PRIVATEUSE1);
    return static_cast<OpenRegThreadLocalState*>(tls);
  }
};

struct OpenRegObserverContext : public at::ObserverContext {
  struct Event {
    TorchOpBasicFields basic_fields_;
    c10::approx_time_t start_time_;
    std::vector<std::vector<int64_t>> shapes_;

    c10::approx_time_t end_time_{
        std::numeric_limits<c10::approx_time_t>::min()};
  };

  explicit OpenRegObserverContext(std::unique_ptr<Event> event) : event_(std::move(event)) {}

  std::unique_ptr<Event> event_;
  std::unique_ptr<FallbackPair> fallback_{nullptr};
};

std::unique_ptr<at::ObserverContext> enter(const at::RecordFunction& fn) {
  auto state_ptr = torch::profiler::impl::OpenRegThreadLocalState::getTLS();
  if (state_ptr == nullptr) {
    return nullptr;
  }

  auto overload_name =
      state_ptr->config().experimental_config.capture_overload_names
      ? fn.overload_name()
      : "";

  auto event = std::make_unique<OpenRegObserverContext::Event>();
  auto out = std::make_unique<OpenRegObserverContext>(std::move(event));
  out->fallback_ = std::make_unique<FallbackPair>();
  out->event_->basic_fields_ = TorchOpBasicFields{
      fn.seqNr(),
      fn.forwardThreadId(),
      fn.scope(),
      fn.isAsync(),
      fn.handle(),
      fn.debugHandle(),
      fn.name(),
      overload_name};
  out->event_->start_time_ = c10::getApproximateTime();

  if (state_ptr->config().report_input_shapes) {
    out->event_->shapes_ = torch::profiler::impl::inputSizes(fn);
  }

  privateuse1Stubs()->record(nullptr, &out->fallback_->device_event_start_, nullptr);
  return out;
}

void exit(const at::RecordFunction& fn, at::ObserverContext* ctx_ptr) {
  auto state_ptr = torch::profiler::impl::OpenRegThreadLocalState::getTLS();
  if (!state_ptr) {
    return;
  }
  auto* openreg_ctx_ptr =
  static_cast<torch::profiler::impl::OpenRegObserverContext*>(ctx_ptr);
  TORCH_INTERNAL_ASSERT(openreg_ctx_ptr != nullptr);
  openreg_ctx_ptr->event_->end_time_ = c10::getApproximateTime();

  openreg_ctx_ptr->event_->basic_fields_.end_tid_ =
  at::RecordFunction::currentThreadId();

  TORCH_INTERNAL_ASSERT(openreg_ctx_ptr->fallback_ != nullptr);
  privateuse1Stubs()->record(
    nullptr, &openreg_ctx_ptr->fallback_->device_event_end_, nullptr);
    std::cout << "6" << std::endl;
}

void pushOpenRegCallbacks(
    const ProfilerConfig& config,
    const std::unordered_set<at::RecordScope>& scopes) {
  TORCH_CHECK(
      torch::profiler::impl::privateuse1Stubs()->enabled(),
      "Can't use OpenReg profiler - PyTorch was compiled without OpenReg");

  c10::ThreadLocalDebugInfo::_push(
      c10::DebugInfoKind::PROFILER_STATE,
      std::make_shared<torch::profiler::impl::OpenRegThreadLocalState>(config));

  auto state_ptr = torch::profiler::impl::OpenRegThreadLocalState::getTLS();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");

  auto handle = at::addThreadLocalCallback(
      at::RecordFunctionCallback(enter, exit)
          .needsInputs(config.report_input_shapes)
          .scopes(scopes));

  state_ptr->setCallbackHandle(handle);
}

REGISTER_PRIVATEUSE1_OBSERVER(
    pushPRIVATEUSE1CallbacksStub,
    &pushOpenRegCallbacks);

} // namespace torch::profiler::impl
