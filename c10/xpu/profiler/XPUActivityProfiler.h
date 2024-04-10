#pragma once

#include <unordered_map>
#include <mutex>

#include "XPUProfilerMacros.h"

namespace c10::xpu {

class XPUActivityProfilerSession : public libkineto::IActivityProfilerSession {
  public:
    XPUActivityProfilerSession() = delete;
    XPUActivityProfilerSession(
        XPUActivityApi& xpti,
        const libkineto::Config& config,
        const std::set<act_t>& activity_types);
    XPUActivityProfilerSession(const XPUActivityProfilerSession&) = delete;
    XPUActivityProfilerSession& operator=(const XPUActivityProfilerSession&) = delete;

    ~XPUActivityProfilerSession();

    void start() override;
    void stop() override;
    std::vector<std::string> errors() override {
      return errors_;
    };
    void processTrace(logger_t& logger) override;
    void processTrace(
            logger_t& logger,
            libkineto::getLinkedActivityCallback get_linked_activity,
            int64_t captureWindowStartTime,
            int64_t captureWindowEndTime) override;
    std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override;
    std::vector<libkineto::ResourceInfo> getResourceInfos() override;
    std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override;

    void pushCorrelationId(uint64_t id) override;
    void popCorrelationId() override;
    void pushUserCorrelationId(uint64_t id) override;
    void popUserCorrelationId() override;

  private:
    void checkTimestampOrder(const itrace_t* act1);
    bool outOfRange(const itrace_t& act);
    const itrace_t* linkedActivity(
        int32_t correlationId,
        const std::unordered_map<int64_t, int64_t>& correlationMap);
    void handleCorrelationActivity(const pti_view_record_external_correlation* correlation);
    void handleRuntimeActivity(
        const pti_view_record_sycl_runtime* activity,
        logger_t* logger);
    void handleGpuActivity(const itrace_t& act, logger_t* logger);
    template <class T>
    void handleGpuActivity(const T* act, logger_t* logger);
    void handleOverheadActivity(
        const pti_view_record_overhead* activity,
        logger_t* logger);
    void handlePtiActivity(
        const pti_view_record_base* record,
        logger_t* logger);
 
  private:
    int64_t captureWindowStartTime_{0};
    int64_t captureWindowEndTime_{0};
    std::unordered_map<int64_t, int64_t> cpuCorrelationMap_;
    std::unordered_map<int64_t, int64_t> userCorrelationMap_;
    std::unordered_map<int64_t, const itrace_t*> correlatedPtiActivities_;
    std::vector<std::string> errors_;

    libkineto::getLinkedActivityCallback cpuActivity_;

    XPUActivityApi& xpti_;
    std::unique_ptr<const libkineto::Config> config_{nullptr};
    const std::set<act_t>& activity_types_;
};

class C10_XPU_API XPUActivityProfiler : public libkineto::IActivityProfiler {
  public:
    XPUActivityProfiler() = default;
    XPUActivityProfiler(const XPUActivityProfiler&) = delete;
    XPUActivityProfiler& operator=(const XPUActivityProfiler&) = delete;

    const std::string& name() const override;
    const std::set<act_t>& availableActivities() const override;
    std::unique_ptr<libkineto::IActivityProfilerSession> configure(
            const std::set<act_t>& activity_types,
            const libkineto::Config& config) override;
    std::unique_ptr<libkineto::IActivityProfilerSession> configure(
            int64_t ts_ms,
            int64_t duration_ms,
            const std::set<act_t>& activity_types,
            const libkineto::Config& config) override;

  private:
    std::string name_{"PyTorch XPU Profiler"};
    int64_t profileStartTime_{0};
    int64_t profileEndTime_{0};
};

}
