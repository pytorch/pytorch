#ifdef USE_KINETO
#include <libkineto.h>
#include <torch/csrc/autograd/profiler_kineto.h>

namespace torch {
namespace profiler {
namespace impl {

namespace {

using namespace torch::autograd::profiler;

class LibKinetoClient : public libkineto::ClientInterface {
 public:
  void init() override {}

  void warmup(bool setupOpInputsCollection) override {
    reportInputShapes_ = setupOpInputsCollection;
  }

  void start() override {
    ProfilerConfig cfg{
        ProfilerState::KINETO_ONDEMAND,
        /*report_input_shapes=*/reportInputShapes_,
        /*profile_memory=*/false,
        /*with_stack=*/withStack_,
        /*with_flops=*/false,
        /*with_modules=*/false};
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

  // @lint-ignore CLANGTIDY cppcoreguidelines-explicit-virtual-functions
  void set_withstack(bool withStack) {
    withStack_ = withStack;
  }

 private:
  bool reportInputShapes_{true};
  bool withStack_{false};
};

} // namespace

} // namespace impl
} // namespace profiler

#ifdef ENABLE_LIBKINETO_CLIENT
struct RegisterLibKinetoClient {
  RegisterLibKinetoClient() {
    static profiler::impl::LibKinetoClient client;
    libkineto::api().registerClient(&client);
  }
} register_libkineto_client;
#endif // ENABLE_LIBKINETO_CLIENT

} // namespace torch
#endif // USE_KINETO
