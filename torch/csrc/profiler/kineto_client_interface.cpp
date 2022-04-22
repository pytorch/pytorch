#ifdef USE_KINETO
#include <libkineto.h>

namespace torch {
namespace profiler {
namespace impl {

namespace {

class LibKinetoClient : public libkineto::ClientInterface {
 public:
  void init() override {
    // TODO: implement
  }

  void warmup(bool setupOpInputsCollection) override {
    // TODO: implement
  }

  void start() override {
    // TODO: implement
  }

  void stop() override {
    // TODO: implement
  }
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
