#ifdef USE_KINETO

#include "profiler/OpenRegActivityProfiler.h"
#include "profiler/OpenRegActivityProfilerSession.h"
#include <torch/csrc/profiler/standalone/privateuse1_profiler.h>

namespace openreg::profiler {

const std::string& OpenRegActivityProfiler::name() const {
  return name_;
}

const std::set<libkineto::ActivityType>&
OpenRegActivityProfiler::availableActivities() const {
  // Real vendor: update activities_ in the header to match your SDK's
  // supported types. PRIVATEUSE1_RUNTIME / PRIVATEUSE1_DRIVER are the
  // correct choices for a PrivateUse1 backend.
  return activities_;
}

std::unique_ptr<libkineto::IActivityProfilerSession>
OpenRegActivityProfiler::configure(
    const std::set<libkineto::ActivityType>& /*activity_types*/,
    const libkineto::Config& /*config*/) {
  // Real vendor: pass activity_types and config into your session constructor
  // to arm the device tracing API.
  return std::make_unique<OpenRegActivityProfilerSession>(/*deviceIndex=*/0);
}

std::unique_ptr<libkineto::IActivityProfilerSession>
OpenRegActivityProfiler::configure(
    int64_t /*ts_ms*/,
    int64_t /*duration_ms*/,
    const std::set<libkineto::ActivityType>& activity_types,
    const libkineto::Config& config) {
  // Real vendor: use ts_ms / duration_ms to schedule a deferred start.
  return configure(activity_types, config);
}

REGISTER_PRIVATEUSE1_PROFILER(OpenRegActivityProfiler);

} // namespace openreg::profiler

#endif // USE_KINETO