/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "CuptiPMSamplingApi.h"

namespace KINETO_NAMESPACE {

// CuptiPMSamplingSession depends on this interface so unit tests can provide a
// mock controller instead of constructing the real CUPTI controller.
class ICuptiPMSamplingController {
 public:
  virtual ~ICuptiPMSamplingController() = default;

  [[nodiscard]] virtual bool prepare() = 0;
  virtual void start() = 0;
  [[nodiscard]] virtual bool stop() = 0;

  [[nodiscard]] virtual int32_t deviceId() const = 0;
  [[nodiscard]] virtual const std::vector<std::string>& metricNames() const = 0;
  [[nodiscard]] virtual std::vector<CuptiPMSample> takeSamples() = 0;
};

class CuptiPMSamplingController final : public ICuptiPMSamplingController {
 public:
  explicit CuptiPMSamplingController(CuptiPMSamplingConfig config);
  CuptiPMSamplingController(const CuptiPMSamplingController&) = delete;
  CuptiPMSamplingController& operator=(const CuptiPMSamplingController&) =
      delete;
  CuptiPMSamplingController(CuptiPMSamplingController&&) = delete;
  CuptiPMSamplingController& operator=(CuptiPMSamplingController&&) = delete;

  ~CuptiPMSamplingController() override;

  [[nodiscard]] bool prepare() override;
  void start() override;
  // Returns whether collection was active when stop() was called.
  [[nodiscard]] bool stop() override;

  [[nodiscard]] int32_t deviceId() const override;
  [[nodiscard]] const std::vector<std::string>& metricNames() const override;
  [[nodiscard]] std::vector<CuptiPMSample> takeSamples() override;

 private:
  void decodeLoop();
  bool decodeBatch(std::vector<CuptiPMSample>& decodedSamples);
  void drain(std::vector<CuptiPMSample>& decodedSamples);
  bool validateConfig() const;
  bool validateSample(const CuptiPMSample& sample) const;
  void logCurrentException(const char* fallback);

  CuptiPMSamplingConfig config_;
  CuptiPMSamplingApi api_;
  std::thread decodeThread_;
  std::atomic_bool stopRequested_{false};
  std::atomic_bool decodeFailed_{false};
  std::mutex samplesMutex_;
  std::mutex waitMutex_;
  std::condition_variable waitCondition_;
  std::vector<CuptiPMSample> samples_;
  bool discardFirstSample_{true};
  bool prepared_{false};
  // True while a collection started by api_.start() is active.
  bool active_{false};
};

} // namespace KINETO_NAMESPACE
