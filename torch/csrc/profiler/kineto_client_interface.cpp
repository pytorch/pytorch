#ifdef USE_KINETO
#include <ATen/Context.h>
#include <libkineto.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/mtia/profiler/MTIAMemoryProfiler.h>
#include <torch/csrc/profiler/kineto_client_interface.h>
#include <chrono>
#include <thread>

// Ondemand tracing is not supported on Apple or edge platform
#if defined(__APPLE__) || defined(EDGE_PROFILER_USE_KINETO)
constexpr bool kEnableGlobalObserver = false;
#else
constexpr bool kEnableGlobalObserver = true;
#endif

namespace torch {

namespace profiler::impl {

namespace {

using namespace torch::autograd::profiler;

class LibKinetoClient : public libkineto::ClientInterface {
 public:
  void init() override {
    ::torch::mtia::initMemoryProfiler();
  }

  void prepare(
      bool report_input_shapes = false,
      bool profile_memory = false,
      bool with_stack = false,
      bool with_flops = false,
      bool with_modules = false) override {
    reportInputShapes_ = report_input_shapes;
    profileMemory_ = profile_memory;
    withStack_ = with_stack;
    withFlops_ = with_flops;
    withModules_ = with_modules;
  }

  void start() override {
    ProfilerConfig cfg{
        ProfilerState::KINETO_ONDEMAND,
        /*report_input_shapes=*/reportInputShapes_,
        /*profile_memory=*/profileMemory_,
        /*with_stack=*/withStack_,
        /*with_flops=*/withFlops_,
        /*with_modules=*/withModules_};
    std::set<ActivityType> activities{ActivityType::CPU};
    std::unordered_set<at::RecordScope> scopes;
    scopes.insert(at::RecordScope::FUNCTION);
    scopes.insert(at::RecordScope::USER_SCOPE);
    scopes.insert(at::RecordScope::BACKWARD_FUNCTION);
    enableProfiler(cfg, activities, scopes);
  }

  void stop() override {
    (void)disableProfiler();
  }

  void start_memory_profile() override {
    LOG(INFO) << "Starting on-demand memory profile";
    startMemoryProfile();
  }

  void stop_memory_profile() override {
    LOG(INFO) << "Stopping on-demand memory profile";
    stopMemoryProfile();
  }

  void export_memory_profile(const std::string& path) override {
    exportMemoryProfile(path);
  }

 private:
  // Temporarily disable shape collection until
  // we re-roll out the feature for on-demand cases
  bool reportInputShapes_{false};
  bool profileMemory_{false};
  bool withStack_{false};
  bool withFlops_{false};
  bool withModules_{false};
};

} // namespace

} // namespace profiler::impl

void global_kineto_init() {
  if constexpr (kEnableGlobalObserver) {
    if (c10::utils::get_env("KINETO_USE_DAEMON").has_value()) {
      libkineto_init(
          /*cpuOnly=*/!(at::hasCUDA() || at::hasXPU() || at::hasMTIA()),
          /*logOnError=*/true);
      libkineto::api().suppressLogMessages();
    }
  }
}

namespace {

struct RegisterLibKinetoClient {
  RegisterLibKinetoClient() {
    if constexpr (kEnableGlobalObserver) {
      static profiler::impl::LibKinetoClient client;
      libkineto::api().registerClient(&client);
    }
  }
} register_libkineto_client;

} // namespace

} // namespace torch
#endif // USE_KINETO
