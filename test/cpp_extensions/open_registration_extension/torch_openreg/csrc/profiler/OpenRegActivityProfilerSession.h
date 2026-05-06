#pragma once

#ifdef USE_KINETO

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <IActivityProfiler.h>
#include <libkineto.h>

#include "profiler/OpenRegTracer.h"

namespace openreg::profiler {

class OpenRegActivityProfilerSession
    : public libkineto::IActivityProfilerSession {
 public:
  OpenRegActivityProfilerSession() = default;
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
  static libkineto::ActivityType toActivityType(ActivityKind kind);

  void convertRecords(
      const std::vector<TraceRecord>& records,
      libkineto::ActivityLogger& logger);

  int64_t startTs_{0};   // µs, set in start()
  int64_t endTs_{0};     // µs, set in stop()

  // Populated by the windowed processTrace overload before delegating.
  int64_t captureWindowStartTime_{0};  // µs
  int64_t captureWindowEndTime_{0};    // µs
  libkineto::getLinkedActivityCallback cpuActivity_;

  libkineto::CpuTraceBuffer traceBuffer_;
  std::vector<std::pair<int32_t, int32_t>> resources_;
  std::vector<std::string> errors_;
};

} // namespace openreg::profiler

#endif // USE_KINETO