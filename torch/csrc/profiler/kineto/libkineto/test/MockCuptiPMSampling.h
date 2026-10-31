/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <cupti_pmsampling.h>

namespace libkineto::test {

struct MockSample {
  uint64_t startTimestamp;
  uint64_t endTimestamp;
  std::vector<double> values;
};

struct MockCuptiState {
  int major{9};
  int minor{0};
  int clockRate{1'500'000};
  int clockRateDevice{-1};
  cudaError_t clockRateResult{cudaSuccess};
  size_t numPasses{1};
  CUpti_PmSampling_DecodeStopReason decodeStopReason{
      CUPTI_PM_SAMPLING_DECODE_STOP_REASON_END_OF_RECORDS};
  std::vector<MockSample> decodedSamples;

  std::vector<std::string> calls;

  std::vector<std::string> configMetricNames;
  size_t enabledDeviceIndex{0};
  CUptiResult setConfigResult{CUPTI_SUCCESS};
  size_t hardwareBufferSize{0};
  uint64_t samplingInterval{0};
  CUpti_PmSampling_TriggerMode triggerMode{};
  CUpti_PmSampling_HardwareBuffer_AppendMode appendMode{};
  std::vector<std::string> counterDataMetricNames;
  uint32_t maxSamples{0};
  size_t counterDataSize{1};

  std::byte hostObject{};
  std::byte samplingObject{};
};

MockCuptiState& mockCupti();

} // namespace libkineto::test
