/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "XpuptiTestUtilities.h"

#include "src/ActivityBuffers.h"
#include "src/plugin/xpupti/XpuptiActivityProfiler.h"
#include "test/xpupti/compute/XpuptiScopeProfilerCompute.h"

#include <libkineto.h>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <gtest/gtest.h>

namespace KN = KINETO_NAMESPACE;

bool IsEnvVerbose() {
  auto verboseEnv = getenv("VERBOSE");
  return verboseEnv && (strcmp(verboseEnv, "1") == 0);
}

std::ostream& operator<<(std::ostream& os, const KN::TraceSpan& span) {
#define PRINT(V) os << std::endl << "      " #V " = " << span.V;
  PRINT(startTime)
  PRINT(endTime)
  PRINT(opCount)
  PRINT(iteration)
  PRINT(name)
  PRINT(prefix)
#undef PRINT
  return os;
}

std::ostream& operator<<(std::ostream& os, KN::ActivityType actType) {
  os << toString(actType);
  return os;
}

static std::pair<unsigned, unsigned> CountMetricsInVector(
    const std::vector<std::string_view>& metrics,
    const decltype(std::declval<KN::GenericTraceActivity>()
                       .counterValues())& vec) {
  unsigned metricsCount = 0;
  unsigned metricsMask = 0;
  for (unsigned i = 0; i < metrics.size(); ++i) {
    const auto metricSv = metrics[i];
    if (std::find_if(vec.begin(), vec.end(), [metricSv](const auto& pair) {
          return pair.first == metricSv;
        }) != vec.end()) {
      ++metricsCount;
      metricsMask |= (1 << i);
    }
  }
  return std::pair{metricsCount, metricsMask};
}

static std::pair<unsigned, unsigned> CountMetricsInString(
    const std::vector<std::string_view>& metrics,
    const std::string_view sv) {
  unsigned metricsCount = 0;
  unsigned metricsMask = 0;
  for (unsigned i = 0; i < metrics.size(); ++i) {
    const auto metricSv = metrics[i];
    if (sv.find(metricSv) != std::string_view::npos) {
      ++metricsCount;
      metricsMask |= (1 << i);
    }
  }
  return std::pair{metricsCount, metricsMask};
}

template <class MAP, class EXPMAP>
void CheckCountsInMap(
    const MAP& map,
    unsigned expSize,
    unsigned repeatCount,
    const EXPMAP& expMap) {
  EXPECT_EQ(map.size(), expSize);

  std::map<unsigned, unsigned> countsMap;
  for (const auto& [key, val] : map) {
    countsMap[val]++;
  }

  EXPECT_EQ(countsMap.size(), expMap.size());

  // Declared separately: countsMap is non-const (iterator) while expMap is
  // const (const_iterator), so a single `auto` cannot deduce both.
  auto itCountsMap = countsMap.begin();
  auto itExpArray = expMap.begin();
  for (; (itCountsMap != countsMap.end()) && (itExpArray != expMap.end());
       ++itCountsMap, ++itExpArray) {
    EXPECT_EQ(itCountsMap->first, itExpArray->first * repeatCount);
    EXPECT_EQ(itCountsMap->second, itExpArray->second);
  }
}

static std::pair<std::map<unsigned, unsigned>, size_t> CountsMap(
    const std::vector<std::string_view>& names) {
  std::map<std::string_view, unsigned> counts;
  for (const auto& name : names) {
    counts[name]++;
  }

  std::map<unsigned, unsigned> countsMap;
  for (const auto& [name, count] : counts) {
    countsMap[count]++;
  }

  return std::pair{countsMap, counts.size()};
};

constexpr const std::string_view driverActivity = "xpu_driver";

static unsigned acceptDriverActivities(
    const std::deque<std::unique_ptr<libkineto::GenericTraceActivity>>&
        pBufferActivities,
    std::vector<std::string_view>& expectedActivities,
    std::vector<std::string_view>& expectedTypes,
    std::set<std::string>& stringStorage) {
  unsigned count = 0;
  for (auto&& pActivity : pBufferActivities) {
    if (pActivity->type() == KN::ActivityType::XPU_DRIVER) {
      auto insertResult = stringStorage.insert(pActivity->name());
      expectedActivities.push_back(*insertResult.first);
      expectedTypes.push_back(driverActivity);
      ++count;
    }
  }
  return count;
}

constexpr const std::string_view overheadActivity = "overhead";

static unsigned acceptOverheadActivities(
    const std::deque<std::unique_ptr<libkineto::GenericTraceActivity>>&
        pBufferActivities,
    std::vector<std::string_view>& expectedActivities,
    std::vector<std::string_view>& expectedTypes,
    std::set<std::string>& stringStorage) {
  unsigned count = 0;
  for (auto&& pActivity : pBufferActivities) {
    if (pActivity->type() == KN::ActivityType::OVERHEAD) {
      auto insertResult = stringStorage.insert(pActivity->name());
      expectedActivities.push_back(*insertResult.first);
      expectedTypes.push_back(overheadActivity);
      ++count;
    }
  }
  return count;
}

class TestActivityLogger : public KN::ActivityLogger {
  void handleDeviceInfo(
      [[maybe_unused]] const KN::DeviceInfo& info,
      [[maybe_unused]] int64_t time) override {}
  void handleResourceInfo(
      [[maybe_unused]] const KN::ResourceInfo& info,
      [[maybe_unused]] int64_t time) override {}
  void handleOverheadInfo(
      [[maybe_unused]] const KN::ActivityLogger::OverheadInfo& info,
      [[maybe_unused]] int64_t time) override {}
  void handleTraceSpan([[maybe_unused]] const KN::TraceSpan& span) override {}
  void handleActivity(
      [[maybe_unused]] const KN::ITraceActivity& activity) override {}
  void handleGenericActivity(
      [[maybe_unused]] const KN::GenericTraceActivity& activity) override {}
  void handleTraceStart(
      [[maybe_unused]] const std::unordered_map<std::string, std::string>&
          metadata,
      [[maybe_unused]] const std::string& device_properties) override {}
  void finalizeMemoryTrace(
      [[maybe_unused]] const std::string&,
      [[maybe_unused]] const KN::Config&) override {}
  void finalizeTrace(
      [[maybe_unused]] const KN::Config& config,
      [[maybe_unused]] std::unique_ptr<KN::ActivityBuffers> buffers,
      [[maybe_unused]] int64_t endTime) override {}
};

std::pair<
    std::unique_ptr<KN::IActivityProfilerSession>,
    std::unique_ptr<KN::CpuTraceBuffer>>
RunProfilerTest(
    const std::vector<std::string_view>& metrics,
    const std::set<KN::ActivityType>& activities,
    const KN::Config& cfg,
    unsigned repeatCount,
    std::vector<std::string_view>&& expectedActivities,
    std::vector<std::string_view>&& expectedTypes,
    int64_t userCorrelationId,
    const KN::ITraceActivity* linkedCpuActivity,
    std::function<const KN::ITraceActivity*(int32_t)> linkedActivityCallback) {
  KN::XPUActivityProfiler profiler;
  EXPECT_TRUE(profiler.name() == "__xpu_profiler__");

  auto pSession = profiler.configure(activities, cfg);

  int64_t startTime = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
  pSession->start();

  if (userCorrelationId != 0) {
    pSession->pushUserCorrelationId(static_cast<uint64_t>(userCorrelationId));
  }

  // A 16x16 GEMM is enough to exercise every activity type while keeping the
  // simulator-based CI runtime small.
  ComputeOnXpu(16, repeatCount);

  if (userCorrelationId != 0) {
    pSession->popUserCorrelationId();
  }

  pSession->stop();
  int64_t endTime = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();

  auto getLinkedActivity = std::move(linkedActivityCallback);
  if (!getLinkedActivity) {
    getLinkedActivity = [userCorrelationId, linkedCpuActivity](
                            int32_t corr) -> const KN::ITraceActivity* {
      if (userCorrelationId != 0 && corr == userCorrelationId) {
        return linkedCpuActivity;
      }
      return nullptr;
    };
  }

  TestActivityLogger logger;
  pSession->processTrace(logger, getLinkedActivity, startTime, endTime);

  EXPECT_TRUE(pSession->errors().empty())
      << fmt::format("{}", fmt::join(pSession->errors(), ","));

  auto pBuffer = pSession->getTraceBuffer();

  std::set<std::string> stringStorage;
  if (activities.find(KN::ActivityType::XPU_DRIVER) != activities.end()) {
    EXPECT_EQ(repeatCount, 1);
    auto count = acceptDriverActivities(
        pBuffer->activities, expectedActivities, expectedTypes, stringStorage);
    EXPECT_GT(count, 0);
  }
  if (activities.find(KN::ActivityType::OVERHEAD) != activities.end()) {
    auto count = acceptOverheadActivities(
        pBuffer->activities, expectedActivities, expectedTypes, stringStorage);
    EXPECT_GT(count, 0);
  }

  static bool isVerbose = IsEnvVerbose();

  if (isVerbose) {
    std::cout << "span = " << pBuffer->span << std::endl;
    std::cout << "gpuOpCount = " << pBuffer->gpuOpCount << std::endl;
    std::cout << "activities.size = " << pBuffer->activities.size()
              << std::endl;
  }

  auto [expectedActivitiesCountsMap, numUniqueActivities] =
      CountsMap(expectedActivities);

  const unsigned numMetrics = std::count_if(
      expectedActivities.begin(),
      expectedActivities.end(),
      [](std::string_view name) { return name.find("metrics") == 0; });

  auto [expectedTypesCountsMap, numUniqueTypes] = CountsMap(expectedTypes);

  const unsigned numActivities = expectedActivities.size();
  EXPECT_EQ(numActivities, expectedTypes.size());

  EXPECT_EQ(pBuffer->activities.size(), numActivities * repeatCount);

  std::map<std::string_view, unsigned> activitiesCount;
  std::map<KN::ActivityType, unsigned> typesCount;
  unsigned scopeProfilerActCount = 0;

  for (auto&& pActivity : pBuffer->activities) {
    auto insertResult = stringStorage.insert(pActivity->name());
    activitiesCount[*insertResult.first]++;
    typesCount[pActivity->type()]++;

    bool isNameXpu = pActivity->name() == "xpu";
    bool nameStartsWithMetrics = pActivity->name().find("metrics:") == 0;

    unsigned metricsCount = 0;
    unsigned metricsMask = 0;
    if (isNameXpu) {
      std::tie(metricsCount, metricsMask) =
          CountMetricsInVector(metrics, pActivity->counterValues());
    } else if (nameStartsWithMetrics) {
      std::tie(metricsCount, metricsMask) =
          CountMetricsInString(metrics, pActivity->metadataJson());
    }

    enum class TestScenario {
      defaultScenario,
      scopeProfiler,
      nameStartsWithMetrics
    };
    TestScenario testScenario = TestScenario::defaultScenario;

    switch (pActivity->type()) {
      case KN::ActivityType::CONCURRENT_KERNEL:
        if (nameStartsWithMetrics) {
          testScenario = TestScenario::nameStartsWithMetrics;
        } else {
          testScenario = TestScenario::defaultScenario;
        }
        break;

      case KN::ActivityType::XPU_SCOPE_PROFILER:
        testScenario = TestScenario::scopeProfiler;
        break;

      default:
        testScenario = TestScenario::defaultScenario;
    }

    switch (testScenario) {
      case TestScenario::scopeProfiler:
        EXPECT_TRUE(isNameXpu);
        [[fallthrough]];
      case TestScenario::nameStartsWithMetrics:
        EXPECT_EQ(metricsCount, metrics.size());
        EXPECT_EQ(metricsMask, (1u << metrics.size()) - 1);
        ++scopeProfilerActCount;
        break;

      case TestScenario::defaultScenario:
        EXPECT_FALSE(isNameXpu);
        EXPECT_EQ(metricsCount, 0);
        EXPECT_EQ(metricsMask, 0);
    }

    if (isVerbose) {
#define PRINT(A) std::cout << #A " = " << pActivity->A() << std::endl;
      PRINT(deviceId)
      PRINT(resourceId)
      PRINT(getThreadId)
      PRINT(timestamp)
      PRINT(duration)
      PRINT(correlationId)
      PRINT(linkedActivity)
      PRINT(flowType)
      PRINT(flowId)
      PRINT(flowStart)
      PRINT(name)
      PRINT(type)
      PRINT(metadataJson)
#undef PRINT

      if (auto pSpan = pActivity->traceSpan()) {
        std::cout << "traceSpan = " << *pSpan << std::endl;
      }

      std::cout << "-----" << std::endl;
    }
  }

  EXPECT_EQ(scopeProfilerActCount, numMetrics * repeatCount);

  CheckCountsInMap(
      activitiesCount,
      numUniqueActivities,
      repeatCount,
      expectedActivitiesCountsMap);

  CheckCountsInMap(
      typesCount, numUniqueTypes, repeatCount, expectedTypesCountsMap);

  return {std::move(pSession), std::move(pBuffer)};
}
