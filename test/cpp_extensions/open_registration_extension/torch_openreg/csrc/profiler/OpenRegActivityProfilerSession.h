#pragma once

#ifdef USE_KINETO

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <IActivityProfiler.h>
#include <libkineto.h>
#include <csrc/tracer.h>

namespace openreg::profiler {

class OPENREG_EXPORT OpenRegActivityProfilerSession
    : public libkineto::IActivityProfilerSession {
 public:
  explicit OpenRegActivityProfilerSession(int32_t deviceIndex = 0)
      : deviceIndex_(deviceIndex) {}
  ~OpenRegActivityProfilerSession() override = default;

  void start() override;
  void stop() override;
  std::vector<std::string> errors() override;

  void processTrace(libkineto::ActivityLogger& logger) override;
  void processTrace(
      libkineto::ActivityLogger& logger,
      libkineto::getLinkedActivityCallback getLinkedActivity,
      int64_t captureWindowStartTime,
      int64_t captureWindowEndTime) override;

  std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override;
  std::vector<libkineto::ResourceInfo> getResourceInfos() override;
  std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override;

  void pushCorrelationId(uint64_t id) override;
  void popCorrelationId() override;

 private:
  int64_t startTs_{0};   // µs, set in start()
  int64_t endTs_{0};     // µs, set in stop()

  libkineto::CpuTraceBuffer traceBuffer_;
  std::vector<std::string> errors_;
  int32_t deviceIndex_{0};
};

} // namespace openreg::profiler

#endif // USE_KINETO