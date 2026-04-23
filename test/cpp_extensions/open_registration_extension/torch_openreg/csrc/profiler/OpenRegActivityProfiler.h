#pragma once

#ifdef USE_KINETO

#include <memory>
#include <set>
#include <string>
#include <vector>

#include <IActivityProfiler.h>

namespace openreg::profiler {

// Stateless factory registered with Kineto via REGISTER_PRIVATEUSE1_PROFILER.
// Kineto calls configure() once per trace to obtain a session handle.
class OpenRegActivityProfiler : public libkineto::IActivityProfiler {
 public:
  OpenRegActivityProfiler() = default;
  ~OpenRegActivityProfiler() override = default;

  const std::string& name() const override;
  const std::set<libkineto::ActivityType>& availableActivities() const override;

  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      const std::set<libkineto::ActivityType>& activity_types,
      const libkineto::Config& config) override;

  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      int64_t ts_ms,
      int64_t duration_ms,
      const std::set<libkineto::ActivityType>& activity_types,
      const libkineto::Config& config) override;

 private:
  std::string name_{"openreg"};
    // Real vendor: list every ActivityType your SDK can record
    // (e.g. PRIVATEUSE1_RUNTIME, PRIVATEUSE1_DRIVER)
  std::set<libkineto::ActivityType> activities_{
      libkineto::ActivityType::CONCURRENT_KERNEL,
  };
};

} // namespace openreg::profiler

#endif // USE_KINETO