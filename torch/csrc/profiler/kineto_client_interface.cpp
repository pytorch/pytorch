#ifdef USE_KINETO
#include <ATen/Context.h>
#include <libkineto.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/profiler/kineto_client_interface.h>
#include <chrono>
#include <thread>

// Ondemand tracing is not supported on Apple or edge platform
#if defined(__APPLE__) || defined(EDGE_PROFILER_USE_KINETO)
#define ENABLE_GLOBAL_OBSERVER (0)
#else
#define ENABLE_GLOBAL_OBSERVER (1)
#endif

namespace torch {

namespace profiler::impl {

namespace {

using namespace torch::autograd::profiler;

class LibKinetoClient : public libkineto::ClientInterface {
 public:
  void init() override {}

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
#if ENABLE_GLOBAL_OBSERVER
  if (c10::utils::get_env("KINETO_USE_DAEMON").has_value()) {
    libkineto_init(
        /*cpuOnly=*/!(at::hasCUDA() || at::hasXPU() || at::hasMTIA()),
        /*logOnError=*/true);
    libkineto::api().suppressLogMessages();
  }
#endif
}

#if ENABLE_GLOBAL_OBSERVER
namespace {

struct RegisterLibKinetoClient {
  RegisterLibKinetoClient() {
    static profiler::impl::LibKinetoClient client;
    libkineto::api().registerClient(&client);
  }
} register_libkineto_client;

} // namespace
#endif

} // namespace torch
#endif // USE_KINETO
