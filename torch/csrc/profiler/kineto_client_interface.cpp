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
        reportInputShapes_,
        false,
        false,
        false,
        false};
    std::set<ActivityType> activities{ActivityType::CPU};
    auto scopes = {at::RecordScope::FUNCTION, at::RecordScope::USER_SCOPE};
    enableProfiler(cfg, activities, scopes);
  }

  void stop() override {
    (void)disableProfiler();
  }
 private:
  bool reportInputShapes_{true};
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
