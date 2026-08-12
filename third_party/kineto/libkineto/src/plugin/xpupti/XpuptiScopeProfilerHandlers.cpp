/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "XpuptiProfilerMacros.h"
#include "XpuptiScopeProfilerSession.h"

namespace KINETO_NAMESPACE {

enum class MetadataOrCounterValue {
  Metadata = 0,
  CounterValue = 1,
};

static void AddPtiValueToMetadataOrCounterValue(
    GenericTraceActivity* scopeActivity,
    MetadataOrCounterValue metadataOrCounterValue,
    const std::string& metricName,
    pti_metric_value_type valueType,
    const pti_value_t& value) {
  switch (valueType) {
#define CASE(T, FIELD)                                                \
  case PTI_METRIC_VALUE_TYPE_##T:                                     \
    if (metadataOrCounterValue == MetadataOrCounterValue::Metadata) { \
      scopeActivity->addMetadata(metricName, value.FIELD);            \
    } else {                                                          \
      scopeActivity->addCounterValue(metricName, value.FIELD);        \
    }                                                                 \
    return;

    CASE(UINT32, ui32);
    CASE(UINT64, ui64);
    CASE(FLOAT32, fp32);
    CASE(FLOAT64, fp64);

#undef CASE

    case PTI_METRIC_VALUE_TYPE_BOOL8:
      if (metadataOrCounterValue == MetadataOrCounterValue::Metadata) {
        scopeActivity->addMetadata(metricName, value.b8 ? "true" : "false ");
      } else {
        scopeActivity->addCounterValue(metricName, value.b8);
      }
      return;

    default:
      break;
  }
}

void XpuptiScopeProfilerSession::handleScopeRecord(
    const pti_metrics_scope_record_t* record,
    const pti_metrics_scope_record_metadata_t& metadata,
    ActivityLogger& logger) {
  std::array<GenericTraceActivity*, 3> scopeActivities{};

  traceBuffer_.emplace_activity(
      traceBuffer_.span,
      ActivityType::CONCURRENT_KERNEL,
      record->_kernel_name
          ? fmt::format("metrics: {}", record->_kernel_name)
          : fmt::format("metrics: kernel_{}", record->_kernel_id));

  scopeActivities[0] = traceBuffer_.activities.back().get();

  for (auto itSa = scopeActivities.begin() + 1; itSa != scopeActivities.end();
       ++itSa) {
    traceBuffer_.emplace_activity(
        traceBuffer_.span, ActivityType::XPU_SCOPE_PROFILER, "xpu");

    *itSa = traceBuffer_.activities.back().get();
  }

  std::function<void(GenericTraceActivity*)> FillActivityRecord{};
  auto it = kernelActivities_.find(record->_kernel_id);
  if (it != kernelActivities_.end()) {
    FillActivityRecord = [it](GenericTraceActivity* act) {
      act->startTime = it->second.startTime_ - 1;
      act->endTime = it->second.endTime_ + 1;
      act->device = it->second.device_;
      act->resource = it->second.resource_;
    };
  } else {
    FillActivityRecord = [this](GenericTraceActivity* act) {
      act->startTime = lastKernelActivityEndTime_ + 1;
      act->endTime = act->startTime + 1;
      act->device = 0;
      act->resource = 0;
    };
  }
  for (auto sa : scopeActivities) {
    FillActivityRecord(sa);
  }
  scopeActivities[2]->startTime = scopeActivities[2]->endTime;

  if (it != kernelActivities_.end()) {
    kernelActivities_.erase(it);
  }
  lastKernelActivityEndTime_ = scopeActivities[0]->endTime;

  scopeActivities[0]->addMetadata("kernel_id", record->_kernel_id);
  scopeActivities[0]->addMetadataQuoted(
      "queue", fmt::format("{}", record->_queue));

  for (uint32_t m = 0; m < metadata._metrics_count; ++m) {
    const auto& unit = metadata._metric_units[m];
    std::string unitSuffix = unit ? fmt::format("::{}", unit) : "";
    std::string metricNameWithUnit =
        fmt::format("{}{}", metadata._metric_names[m], unitSuffix);

    AddPtiValueToMetadataOrCounterValue(
        scopeActivities[0],
        MetadataOrCounterValue::Metadata,
        metricNameWithUnit,
        metadata._value_types[m],
        record->_metrics_values[m]);

    AddPtiValueToMetadataOrCounterValue(
        scopeActivities[1],
        MetadataOrCounterValue::CounterValue,
        metadata._metric_names[m],
        metadata._value_types[m],
        record->_metrics_values[m]);

    scopeActivities[2]->addCounterValue(metadata._metric_names[m], 0);
  }

  for (auto sa : scopeActivities) {
    sa->log(logger);
  }
}

} // namespace KINETO_NAMESPACE
