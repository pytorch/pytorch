/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "CuptiPMSamplingApi.h"

#include <stdexcept>

#include <cuda_runtime_api.h>
#include <cupti_pmsampling.h>
#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>
#include <cupti_target.h>

#include "DeviceUtil.h"
#include "Logger.h"
#include "ThrowUtil.h"

// The CUPTI APIs used in this file follow a versioned struct pattern: each
// function takes a params struct whose first two fields are structSize (so
// CUPTI knows which version of the struct the caller compiled against) and
// pPriv (an internal pointer, always nullptr from the caller's side). The
// remaining fields are the actual parameters.
//
// NVIDIA's recommended initialization pattern explicitly initializes only
// structSize and relies on C++ aggregate zero-initialization for pPriv and the
// remaining fields, then sets specific fields afterward:
//
//   CUpti_PmSampling_Start_Params params{
//       CUpti_PmSampling_Start_Params_STRUCT_SIZE};
//   params.pPmSamplingObject = samplingObject;
//
// -Wmissing-field-initializers (enabled by -Wextra) warns when an aggregate
// initializer does not provide values for every field, which fires at all of
// the CUPTI params call sites in this file. We suppress it rather than
// enumerating every field, since the struct definitions change between CUPTI
// SDK versions and explicit listing would break whenever NVIDIA adds or
// removes fields.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

namespace KINETO_NAMESPACE {
namespace {

constexpr size_t kHardwareBufferSizeBytes = 64 * 1024 * 1024;
constexpr uint32_t kMaxSamplesPerDecode = 1024;

// GA100 only supports variable-frequency SYSCLK sampling. One option is to
// measure hardware clock frequency and estimate the number of cycles needed to
// fill a certain sampling duration, but clock frequency is not necessarily
// fixed. This is almost certainly something we'll need to revisit/tune.
constexpr uint64_t kGa100SamplingIntervalCycles = 1'000'000;

/*
 * ========================== CUPTI PM SAMPLING API ==========================
 *
 * CUPTI PM sampling periodically collects device metrics and exports completed
 * sample intervals as CuptiPMSample values.
 *
 * The lifecycle here is:
 *
 * 1. [Configure]
 *    Builds the metric configuration, calls cuptiPmSamplingEnable(),
 *    initializes the data image (counterDataImage_). The data image is a
 *    caller-owned byte buffer initialized in CUPTI's format.
 *
 * 2. [Start]
 *    Starts hardware counter collection. CUPTI starts writing raw counter
 *    records into its hardware buffer.
 *
 * 3. [Decode] (PERIODICALLY, UNTIL STOP)
 *    Transfers records from the CUPTI's hardware buffer into the data image.
 *    Each completed sample in the data image is translated from CUPTI's format
 *    into a CuptiPMSample and pushed to the output vector. If records remain in
 *    the hardware buffer (COUNTER_DATA_FULL), we reset the image and return
 * false.
 *
 * 4. [Stop]
 *    Instruct CUPTI to stop producing records.
 *
 * 5. [Decode]
 *    Run one more decode to empty the hardware buffer.
 *
 * 6. [Disable]
 *    Disable the sampler, clear all objects.
 *
 * ===========================================================================
 */

} // namespace

CuptiPMSamplingApi::~CuptiPMSamplingApi() {
  disable();
  if (samplingObject_ != nullptr || hostObject_ != nullptr ||
      profilerInitialized_) {
    LOG(WARNING) << "CUPTI PM sampling cleanup failed during destruction; "
                    "CUPTI resources may leak";
  }
}

void CuptiPMSamplingApi::configure(const CuptiPMSamplingConfig& config) {
  if (samplingObject_ != nullptr || hostObject_ != nullptr ||
      profilerInitialized_) {
    KINETO_THROW(
        std::runtime_error,
        "Cannot configure CUPTI PM sampling before the previous "
        "configuration is fully disabled");
  }
  config_ = config;

  // CUPTI expects metric names as std::vector<const char*>
  metricNamePtrs_.clear();
  metricNamePtrs_.reserve(config_.metricNames.size());
  for (const auto& metricName : config_.metricNames) {
    metricNamePtrs_.push_back(metricName.c_str());
  }
  configureCupti();
}

void CuptiPMSamplingApi::ensureConfigured() const {
  if (samplingObject_ == nullptr) {
    KINETO_THROW(
        std::runtime_error, "CUPTI PM sampling: samplingObject_ is null");
  }
  if (hostObject_ == nullptr) {
    KINETO_THROW(std::runtime_error, "CUPTI PM sampling: hostObject_ is null");
  }
  if (counterDataImage_.empty()) {
    KINETO_THROW(
        std::runtime_error, "CUPTI PM sampling: counterDataImage_ is empty");
  }
}

void CuptiPMSamplingApi::configureCupti() {
  CUpti_Profiler_Initialize_Params params{
      CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
  CUPTI_CALL_THROW(cuptiProfilerInitialize(&params));
  profilerInitialized_ = true;

  // CUpti_Device_GetChipName_Params stores device info used in other CUPTI
  // calls.
  CUpti_Device_GetChipName_Params chip{
      CUpti_Device_GetChipName_Params_STRUCT_SIZE};
  chip.deviceIndex = static_cast<size_t>(config_.deviceId);
  CUPTI_CALL_THROW(cuptiDeviceGetChipName(&chip));
  if (chip.pChipName == nullptr) {
    KINETO_THROW(
        std::runtime_error, "cuptiDeviceGetChipName returned no chip name");
  }

  cudaDeviceProp deviceProperties{};
  const auto cudaStatus =
      cudaGetDeviceProperties(&deviceProperties, config_.deviceId);
  if (cudaStatus != cudaSuccess) {
    KINETO_THROW(
        std::runtime_error,
        std::string{"cudaGetDeviceProperties failed: "} +
            cudaGetErrorString(cudaStatus));
  }
  CUpti_PmSampling_TriggerMode triggerMode;
  uint64_t samplingInterval;
  // GPU_TIME_INTERVAL is unavailable on GA100. The wall-time duration of this
  // fixed SYSCLK interval varies with the GPU's clock frequency.
  if (deviceProperties.major == 8 && deviceProperties.minor == 0) {
    triggerMode = CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_SYSCLK_INTERVAL;
    samplingInterval = kGa100SamplingIntervalCycles;
  } else if (
      deviceProperties.major > 8 ||
      (deviceProperties.major == 8 && deviceProperties.minor >= 6)) {
    if (config_.samplingInterval.count() <= 0) {
      KINETO_THROW(
          std::runtime_error, "CUPTI PM sampling interval must be positive");
    }
    triggerMode = CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL;
    samplingInterval = static_cast<uint64_t>(config_.samplingInterval.count());
  } else {
    KINETO_THROW(
        std::runtime_error,
        "Minimum supported GPU for CUPTI PM sampling is GA100 (compute "
        "capability 8.0); other GPUs require compute capability 8.6 or newer");
  }

  // Building the availability image. This is a CUPTI byte buffer describing
  // which raw hardware counters are available on this machine. The first call
  // to cuptiPmSamplingGetCounterAvailability is to obtain
  // counterAvailabilityImageSize, the number of bytes we need to allocate for
  // counterAvailabilityImage. Then the second call populates
  // counterAvailabilityImage.
  CUpti_PmSampling_GetCounterAvailability_Params availability{
      CUpti_PmSampling_GetCounterAvailability_Params_STRUCT_SIZE};
  availability.deviceIndex = static_cast<size_t>(config_.deviceId);
  CUPTI_CALL_THROW(cuptiPmSamplingGetCounterAvailability(&availability));

  std::vector<uint8_t> counterAvailabilityImage(
      availability.counterAvailabilityImageSize);
  availability.pCounterAvailabilityImage = counterAvailabilityImage.data();
  CUPTI_CALL_THROW(cuptiPmSamplingGetCounterAvailability(&availability));

  // CUpti_Profiler_Host_Initialize_Params is the evaluator which translates
  // from requested metric names into the raw counters the GPU should collect.
  // Later on, cuptiProfilerHostEvaluateToGpuValues() uses the initialized
  // hostObject_ to convert raw hardware counters into the requested metric
  // values using chip-specific formulas.
  CUpti_Profiler_Host_Initialize_Params host{
      CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
  host.profilerType = CUPTI_PROFILER_TYPE_PM_SAMPLING;
  host.pChipName = chip.pChipName;
  host.pCounterAvailabilityImage = counterAvailabilityImage.data();
  CUPTI_CALL_THROW(cuptiProfilerHostInitialize(&host));
  hostObject_ = host.pHostObject;

  // Translating the requested metric strings and passing them to CUPTI.
  // This updates hostObject_ with the high-level metric names inplace.
  CUpti_Profiler_Host_ConfigAddMetrics_Params addMetrics{
      CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
  addMetrics.pHostObject = hostObject_;
  addMetrics.ppMetricNames = metricNamePtrs_.data();
  addMetrics.numMetrics = metricNamePtrs_.size();
  CUPTI_CALL_THROW(cuptiProfilerHostConfigAddMetrics(&addMetrics));

  // Allocation/configuration for configImage. This describes
  // the raw (low-level) counters the sampler needs to collect for the requested
  // metrics and how to schedule them.
  CUpti_Profiler_Host_GetConfigImageSize_Params size{
      CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE};
  size.pHostObject = hostObject_;
  CUPTI_CALL_THROW(cuptiProfilerHostGetConfigImageSize(&size));

  configImage_.resize(size.configImageSize);
  CUpti_Profiler_Host_GetConfigImage_Params image{
      CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE};
  image.pHostObject = hostObject_;
  image.configImageSize = configImage_.size();
  image.pConfigImage = configImage_.data();
  CUPTI_CALL_THROW(cuptiProfilerHostGetConfigImage(&image));

  // Asking CUPTI how many passes it will take to fetch the requested metrics.
  // We do not want to make multiple passes, so reject the config if this is the
  // case.
  CUpti_Profiler_Host_GetNumOfPasses_Params passes{
      CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE};
  passes.configImageSize = configImage_.size();
  passes.pConfigImage = configImage_.data();
  CUPTI_CALL_THROW(cuptiProfilerHostGetNumOfPasses(&passes));
  if (passes.numOfPasses != 1) {
    KINETO_THROW(
        std::runtime_error,
        "CUPTI PM sampling requires one pass; configuration requires " +
            std::to_string(passes.numOfPasses));
  }

  // Enabling PM sampling and setting the config
  CUpti_PmSampling_Enable_Params enable{
      CUpti_PmSampling_Enable_Params_STRUCT_SIZE};
  enable.deviceIndex = static_cast<size_t>(config_.deviceId);
  CUPTI_CALL_THROW(cuptiPmSamplingEnable(&enable));
  samplingObject_ = enable.pPmSamplingObject;

  CUpti_PmSampling_SetConfig_Params setConfig{
      CUpti_PmSampling_SetConfig_Params_STRUCT_SIZE};
  setConfig.pPmSamplingObject = samplingObject_;
  setConfig.configSize = configImage_.size();
  setConfig.pConfig = configImage_.data();
  setConfig.hardwareBufferSize = kHardwareBufferSizeBytes;
  setConfig.samplingInterval = samplingInterval;
  setConfig.triggerMode = triggerMode;
  setConfig.hwBufferAppendMode =
      CUPTI_PM_SAMPLING_HARDWARE_BUFFER_APPEND_MODE_KEEP_LATEST;
  CUPTI_CALL_THROW(cuptiPmSamplingSetConfig(&setConfig));

  // Asking CUPTI how large the (counter) data image should be.
  // Since the data image has an opage CUPTI-defined layout, the size
  // depends on sampling config, metrics, etc.
  CUpti_PmSampling_GetCounterDataSize_Params counterDataSize{
      CUpti_PmSampling_GetCounterDataSize_Params_STRUCT_SIZE};
  counterDataSize.pPmSamplingObject = samplingObject_;
  counterDataSize.pMetricNames = metricNamePtrs_.data();
  counterDataSize.numMetrics = metricNamePtrs_.size();
  counterDataSize.maxSamples = kMaxSamplesPerDecode;
  CUPTI_CALL_THROW(cuptiPmSamplingGetCounterDataSize(&counterDataSize));

  counterDataImage_.resize(counterDataSize.counterDataSize);
  resetImage();
}

void CuptiPMSamplingApi::resetImage() {
  CUpti_PmSampling_CounterDataImage_Initialize_Params params{
      CUpti_PmSampling_CounterDataImage_Initialize_Params_STRUCT_SIZE};
  params.pPmSamplingObject = samplingObject_;
  params.counterDataSize = counterDataImage_.size();
  params.pCounterData = counterDataImage_.data();
  CUPTI_CALL_THROW(cuptiPmSamplingCounterDataImageInitialize(&params));
}

void CuptiPMSamplingApi::start() {
  ensureConfigured();
  CUpti_PmSampling_Start_Params params{
      CUpti_PmSampling_Start_Params_STRUCT_SIZE};
  params.pPmSamplingObject = samplingObject_;
  CUPTI_CALL_THROW(cuptiPmSamplingStart(&params));
}

CuptiPMSample CuptiPMSamplingApi::decodeSample(size_t sampleIndex) {
  // Decode a single CUPTI PM sample into CuptiPMSample via
  // cuptiProfilerHostEvaluateToGpuValues.
  CUpti_PmSampling_CounterData_GetSampleInfo_Params info{
      CUpti_PmSampling_CounterData_GetSampleInfo_Params_STRUCT_SIZE};
  info.pPmSamplingObject = samplingObject_;
  info.pCounterDataImage = counterDataImage_.data();
  info.counterDataImageSize = counterDataImage_.size();
  info.sampleIndex = sampleIndex;
  CUPTI_CALL_THROW(cuptiPmSamplingCounterDataGetSampleInfo(&info));

  CuptiPMSample sample{
      info.startTimestamp,
      info.endTimestamp,
      std::vector<double>(metricNamePtrs_.size())};
  CUpti_Profiler_Host_EvaluateToGpuValues_Params evaluate{
      CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
  evaluate.pHostObject = hostObject_;
  evaluate.pCounterDataImage = counterDataImage_.data();
  evaluate.counterDataImageSize = counterDataImage_.size();
  evaluate.rangeIndex = sampleIndex;
  evaluate.ppMetricNames = metricNamePtrs_.data();
  evaluate.numMetrics = metricNamePtrs_.size();
  evaluate.pMetricValues = sample.values.data();
  CUPTI_CALL_THROW(cuptiProfilerHostEvaluateToGpuValues(&evaluate));
  return sample;
}

bool CuptiPMSamplingApi::decode(std::vector<CuptiPMSample>& samples) {
  ensureConfigured();
  // Fetch the next result from CUPTI. Decodes the collected counters
  // inline into the counterDataImage_.
  CUpti_PmSampling_DecodeData_Params decode{
      CUpti_PmSampling_DecodeData_Params_STRUCT_SIZE};
  decode.pPmSamplingObject = samplingObject_;
  decode.pCounterDataImage = counterDataImage_.data();
  decode.counterDataImageSize = counterDataImage_.size();
  CUPTI_CALL_THROW(cuptiPmSamplingDecodeData(&decode));
  if (decode.overflow != 0) {
    LOG_FIRST_N(WARNING, 3)
        << "CUPTI PM sampling hardware buffer overflowed; older samples were "
           "dropped.";
  }

  const auto reason = decode.decodeStopReason;
  bool isBufferDrained;
  switch (reason) {
    case CUPTI_PM_SAMPLING_DECODE_STOP_REASON_OTHER:
    case CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNTER_DATA_FULL:
      isBufferDrained = false;
      break;
    case CUPTI_PM_SAMPLING_DECODE_STOP_REASON_END_OF_RECORDS:
      isBufferDrained = true;
      break;
    default:
      KINETO_THROW(
          std::runtime_error,
          "cuptiPmSamplingDecodeData returned unexpected stop reason " +
              std::to_string(static_cast<int>(reason)));
  }

  CUpti_PmSampling_GetCounterDataInfo_Params info{
      CUpti_PmSampling_GetCounterDataInfo_Params_STRUCT_SIZE};
  info.pCounterDataImage = counterDataImage_.data();
  info.counterDataImageSize = counterDataImage_.size();
  CUPTI_CALL_THROW(cuptiPmSamplingGetCounterDataInfo(&info));

  samples.reserve(samples.size() + info.numCompletedSamples);
  for (size_t sampleIndex = 0; sampleIndex < info.numCompletedSamples;
       ++sampleIndex) {
    samples.push_back(decodeSample(sampleIndex));
  }

  resetImage();
  return isBufferDrained;
}

void CuptiPMSamplingApi::stop() {
  ensureConfigured();
  CUpti_PmSampling_Stop_Params params{CUpti_PmSampling_Stop_Params_STRUCT_SIZE};
  params.pPmSamplingObject = samplingObject_;
  CUPTI_CALL_THROW(cuptiPmSamplingStop(&params));
}

void CuptiPMSamplingApi::disable() {
  // When releasing resources, stop at the first failure to avoid
  // a partially torn-down configuration and preserve the remaining
  // state for a later disable() retry.
  if (samplingObject_ != nullptr) {
    CUpti_PmSampling_Disable_Params params{
        CUpti_PmSampling_Disable_Params_STRUCT_SIZE};
    params.pPmSamplingObject = samplingObject_;
    if (CUPTI_CALL(cuptiPmSamplingDisable(&params)) != CUPTI_SUCCESS) {
      return;
    }
    samplingObject_ = nullptr;
  }
  if (hostObject_ != nullptr) {
    CUpti_Profiler_Host_Deinitialize_Params params{
        CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
    params.pHostObject = hostObject_;
    if (CUPTI_CALL(cuptiProfilerHostDeinitialize(&params)) != CUPTI_SUCCESS) {
      return;
    }
    hostObject_ = nullptr;
  }
  if (profilerInitialized_) {
    CUpti_Profiler_DeInitialize_Params params{
        CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
    if (CUPTI_CALL(cuptiProfilerDeInitialize(&params)) != CUPTI_SUCCESS) {
      return;
    }
    profilerInitialized_ = false;
  }

  configImage_.clear();
  counterDataImage_.clear();
  config_.deviceId = -1;
  metricNamePtrs_.clear();
  config_.metricNames.clear();
  config_.samplingInterval = std::chrono::nanoseconds::zero();
}

} // namespace KINETO_NAMESPACE

#pragma GCC diagnostic pop
