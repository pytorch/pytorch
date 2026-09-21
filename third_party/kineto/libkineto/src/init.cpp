/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ActivityProfilerProxy.h"
#include "ConfigLoader.h"
#include "DaemonConfigLoader.h"
#ifdef HAS_CUPTI
#include "CuptiActivityApi.h"
#endif
#ifdef HAS_CUPTI_PM_SAMPLING
#include "CuptiPMSamplingProfiler.h"
#endif
#ifdef HAS_ROCTRACER
#include "RocprofLogger.h"
#endif
#ifdef HAS_XPUPTI
#include "plugin/xpupti/XpuptiActivityApi.h"
#include "plugin/xpupti/XpuptiActivityProfiler.h"
#include "plugin/xpupti/XpuptiProfilerMacros.h"
#include "plugin/xpupti/XpuptiScopeProfilerConfig.h"
#endif
#include "libkineto.h"

#include "Logger.h"

namespace KINETO_NAMESPACE {

#ifdef HAS_CUPTI
// Some models suffer from excessive instrumentation code gen
// on dynamic attach which can hang for more than 5+ seconds.
// If the workload was meant to be traced, preload the CUPTI
// to take the performance hit early on.
// https://docs.nvidia.com/cupti/r_main.html#r_overhead
static bool shouldPreloadCuptiInstrumentation() {
#if defined(CUDA_VERSION) && CUDA_VERSION < 11020
  return true;
#else
  return false;
#endif
}

#endif // HAS_CUPTI

} // namespace KINETO_NAMESPACE

// Library initialization entry points
using namespace KINETO_NAMESPACE;
extern "C" {

void libkineto_init(bool cpuOnly, [[maybe_unused]] bool logOnError) {
  // Start with initializing the log level
  const char* logLevelEnv = getenv("KINETO_LOG_LEVEL");
  if (logLevelEnv) {
    // atoi returns 0 on error, so that's what we want - default to VERBOSE
    static_assert(static_cast<int>(VERBOSE) == 0);
    SET_LOG_SEVERITY_LEVEL(atoi(logLevelEnv));
  }

#if __linux__
  // KINETO_USE_DAEMON selects the OSS IPC loader only when the platform has
  // not already registered its own loader.
  const bool registerOssDaemon = libkineto::isDaemonEnvVarSet() &&
      !ConfigLoader::hasDaemonConfigLoaderFactory();
  if (registerOssDaemon) {
    LOG(INFO) << "Registering daemon config loader, cpuOnly = " << cpuOnly;
    DaemonConfigLoader::registerFactory();
  }
#endif

#ifdef HAS_CUPTI
  if (!cpuOnly && shouldPreloadCuptiInstrumentation()) {
    CuptiActivityApi::forceLoadCupti();
  }
#endif // HAS_CUPTI

#ifdef HAS_ROCTRACER
  if (!cpuOnly) {
    RocprofLogger::ensureRegistered();
  }
#endif

  auto& api = libkineto::api();
  ConfigLoader& config_loader = api.configLoader();
  api.registerProfiler(
      std::make_unique<ActivityProfilerProxy>(cpuOnly, config_loader));

#ifdef HAS_CUPTI_PM_SAMPLING
  if (!cpuOnly) {
    libkineto::api().registerProfilerFactory(
        []() -> std::unique_ptr<IActivityProfiler> {
          return std::make_unique<CuptiPMSamplingProfiler>();
        });
  }
#endif

#ifdef HAS_XPUPTI
  if (!cpuOnly) {
    api.registerProfilerFactory([]() -> std::unique_ptr<IActivityProfiler> {
      XPUPTI_CALL(
          ptiViewGPULocalAvailable(),
          /*error_message=*/"Failed to enable Kineto Profiler on XPU.");
      XpuptiScopeProfilerConfig::registerFactory();
      return std::make_unique<XPUActivityProfiler>();
    });
  }
#endif // HAS_XPUPTI

#if __linux__
  if (registerOssDaemon) {
    api.initProfilerIfRegistered();
    VLOG(0) << "libkineto profilers activated";
  }
#endif
}

// The cuda driver calls this function if the CUDA_INJECTION64_PATH environment
// variable is set. Should be skipped if unset or CUDA_INJECTION64_PATH=none.
int InitializeInjection(void) {
  LOG(INFO) << "Injection mode: Initializing libkineto";
  libkineto_init(false /*cpuOnly*/, true /*logOnError*/);
  libkineto::api().initProfilerIfRegistered();
  return 1;
}

bool hasTestEnvVar() {
  return getenv("GTEST_OUTPUT") != nullptr || getenv("FB_TEST") != nullptr ||
      getenv("PYTORCH_TEST") != nullptr || getenv("TEST_PILOT") != nullptr;
}

void suppressLibkinetoLogMessages() {
  // Only suppress messages if explicit override wasn't provided
  const char* logLevelEnv = getenv("KINETO_LOG_LEVEL");
  // For unit tests, don't suppress log verbosity.
  if (!hasTestEnvVar() && (!logLevelEnv || !*logLevelEnv)) {
    SET_LOG_SEVERITY_LEVEL(ERROR);
  }
}

} // extern C
