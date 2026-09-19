/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct CUpti_PmSampling_Object;
struct CUpti_Profiler_Host_Object;

namespace KINETO_NAMESPACE {

struct CuptiPMSample {
  // Raw timestamps produced through the same registered timestamp source as
  // CUPTI Activity API records.
  uint64_t rawStartTimestamp;
  uint64_t rawEndTimestamp;
  std::vector<double> values; // One value per metric
};

struct CuptiPMSamplingConfig {
  int32_t deviceId{-1};
  std::vector<std::string> metricNames;
  std::chrono::nanoseconds samplingInterval{0};
};

class CuptiPMSamplingApi {
 public:
  CuptiPMSamplingApi() = default;
  CuptiPMSamplingApi(const CuptiPMSamplingApi&) = delete;
  CuptiPMSamplingApi& operator=(const CuptiPMSamplingApi&) = delete;
  CuptiPMSamplingApi(CuptiPMSamplingApi&&) = delete;
  CuptiPMSamplingApi& operator=(CuptiPMSamplingApi&&) = delete;

  virtual ~CuptiPMSamplingApi();

  virtual void configure(const CuptiPMSamplingConfig& config);
  virtual void start();
  // Appends one batch and returns true if all current records were decoded.
  virtual bool decode(std::vector<CuptiPMSample>& samples);
  virtual void stop();
  virtual void disable();

 private:
  void ensureConfigured() const;
  void configureCupti();
  void resetImage();
  CuptiPMSample decodeSample(size_t sampleIndex);

  CuptiPMSamplingConfig config_;
  // Points into config_.metricNames and must be rebuilt whenever config_ is
  // replaced. config_ remains unchanged while these pointers are in use.
  std::vector<const char*> metricNamePtrs_;
  // Non-owning pointers to CUPTI-managed objects. disable() ends their
  // lifetimes.
  CUpti_Profiler_Host_Object* hostObject_{nullptr};
  CUpti_PmSampling_Object* samplingObject_{nullptr};
  bool profilerInitialized_{false};
  // CUPTI retains this buffer until PM sampling is disabled.
  std::vector<uint8_t> configImage_;
  std::vector<uint8_t> counterDataImage_;
};

} // namespace KINETO_NAMESPACE
