/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "RocprofActivityApi.h"

#include <time.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <unordered_map>
#include <vector>
#include "ApproximateClock.h"
#include "Demangle.h"
#include "Logger.h"
#include "RocmStreamQueue.h"
#include "ThreadUtil.h"
#include "output_base.h"

using namespace std::chrono;

namespace KINETO_NAMESPACE {

namespace {
bool isAsyncCopy(const rocprofAsyncRow& async) {
  return async.domain == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY;
}

bool isAsyncKernel(const rocprofAsyncRow& async) {
  return async.domain == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH;
}
} // namespace

RocprofActivityApi& RocprofActivityApi::singleton() {
  static RocprofActivityApi instance;
  return instance;
}

RocprofActivityApi::RocprofActivityApi() : d(&RocprofLogger::singleton()) {}

RocprofActivityApi::~RocprofActivityApi() {
  disableActivities(std::set<ActivityType>());
}

void RocprofActivityApi::pushCorrelationID(int id, CorrelationFlowType type) {
#ifdef HAS_ROCTRACER
  if (!singleton().d->externalCorrelationEnabled_) {
    return;
  }
  singleton().d->pushCorrelationID(
      id, static_cast<RocLogger::CorrelationDomain>(type));
#endif
}

void RocprofActivityApi::popCorrelationID(CorrelationFlowType type) {
#ifdef HAS_ROCTRACER
  if (!singleton().d->externalCorrelationEnabled_) {
    return;
  }
  singleton().d->popCorrelationID(
      static_cast<RocLogger::CorrelationDomain>(type));
#endif
}

void RocprofActivityApi::setMaxEvents(uint32_t maxEvents) {
  d->setMaxEvents(maxEvents);
}

void RocprofActivityApi::setMaxBufferSize([[maybe_unused]] int64_t size) {
  // FIXME: implement?
  // maxGpuBufferCount_ = 1 + size / kBufSize;
}

inline bool inRange(int64_t start, int64_t end, int64_t stamp) {
  return ((stamp > start) && (stamp < end));
}

inline bool RocprofActivityApi::isLogged(libkineto::ActivityType atype) const {
  return activityMaskSnapshot_ & (1 << static_cast<uint32_t>(atype));
}

timestamp_t getTimeOffset() {
  int64_t t0, t00;
  timespec t1;
  t0 = libkineto::getApproximateTime();
  clock_gettime(CLOCK_MONOTONIC, &t1);
  t00 = libkineto::getApproximateTime();

  // Confvert to ns (if necessary)
  t0 = libkineto::get_time_converter()(t0);
  t00 = libkineto::get_time_converter()(t00);

  // Our stored timestamps (from roctracer and generated) are in CLOCK_MONOTONIC
  // domain (in ns).
  return (t0 >> 1) + (t00 >> 1) - timespec_to_ns(t1);
}

int RocprofActivityApi::processActivities(
    std::function<void(const rocprofBase*)> handler,
    std::function<void(uint64_t, uint64_t, RocLogger::CorrelationDomain)>
        correlationHandler) {
  // Find offset to map from monotonic clock to system clock.
  // This will break time-ordering of events but is status quo.

  int count = 0;

  // Process all external correlations pairs
  for (int it = RocLogger::CorrelationDomain::begin;
       it < RocLogger::CorrelationDomain::end;
       ++it) {
    auto& externalCorrelations = d->externalCorrelations_[it];
    for (auto& item : externalCorrelations) {
      correlationHandler(
          item.first,
          item.second,
          static_cast<RocLogger::CorrelationDomain>(it));
    }
    std::lock_guard<std::mutex> lock(d->externalCorrelationsMutex_);
    externalCorrelations.clear();
  }

  // Async ops are in CLOCK_MONOTONIC rather than junk clock.
  // Convert these timestamps, poorly.
  // These accurate timestamps will skew when converted to approximate time
  // The time_converter is not available at collection time.  Or we could do a
  // much better job.
  auto toffset = getTimeOffset();

  const bool logCopies = isLogged(ActivityType::GPU_MEMCPY);
  const bool logKernels = isLogged(ActivityType::CONCURRENT_KERNEL);
  if (logCopies || logKernels) {
    detail::backfillAsyncStreams(
        d->rows_, [logCopies, logKernels](const rocprofAsyncRow& async) {
          return (logCopies && isAsyncCopy(async)) ||
              (logKernels && isAsyncKernel(async));
        });
  }

  // All Runtime API Calls
  for (auto& item : d->rows_) {
    bool filtered = false;
    if (item->type != ROCTRACER_ACTIVITY_ASYNC &&
        !isLogged(ActivityType::CUDA_RUNTIME)) {
      filtered = true;
    } else {
      switch (reinterpret_cast<rocprofAsyncRow*>(item)->domain) {
        case ROCPROFILER_BUFFER_TRACING_MEMORY_COPY:
          if (!isLogged(ActivityType::GPU_MEMCPY))
            filtered = true;
          break;
        case ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH:
        default:
          if (!isLogged(ActivityType::CONCURRENT_KERNEL))
            filtered = true;
          break;
      }
    }
    if (!filtered) {
      // Convert the begin and end timestamps from monotonic clock to system
      // clock.
      if (item->type == ROCTRACER_ACTIVITY_ASYNC) {
        // Async ops are in CLOCK_MONOTONIC, apply offset to converted
        // approximate
        item->begin += toffset;
        item->end += toffset;
      } else {
        // Runtime ranges are in approximate clock, just apply conversion
        item->begin = libkineto::get_time_converter()(item->begin);
        item->end = libkineto::get_time_converter()(item->end);
      }
      handler(item);
      ++count;
    }
  }
  return count;
}

// TODO: implement the actual flush with roctracer_flush_activity
void RocprofActivityApi::flushActivities() {}

void RocprofActivityApi::clearActivities() {
  d->clearLogs();
}

void RocprofActivityApi::enableActivities(
    const std::set<ActivityType>& selected_activities) {
#ifdef HAS_ROCTRACER
  d->startLogging();

  for (const auto& activity : selected_activities) {
    activityMask_ |= (1 << static_cast<uint32_t>(activity));
    if (activity == ActivityType::EXTERNAL_CORRELATION) {
      d->externalCorrelationEnabled_ = true;
    }
  }
#endif
}

void RocprofActivityApi::disableActivities(
    const std::set<ActivityType>& selected_activities) {
#ifdef HAS_ROCTRACER
  d->stopLogging();

  activityMaskSnapshot_ = activityMask_;

  for (const auto& activity : selected_activities) {
    activityMask_ &= ~(1 << static_cast<uint32_t>(activity));
    if (activity == ActivityType::EXTERNAL_CORRELATION) {
      d->externalCorrelationEnabled_ = false;
    }
  }
#endif
}

} // namespace KINETO_NAMESPACE
