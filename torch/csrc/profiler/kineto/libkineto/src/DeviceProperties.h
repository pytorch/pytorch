/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <string>

#ifdef HAS_CUPTI
#include <cuda_occupancy.h>
#include <cupti.h>
#endif

namespace KINETO_NAMESPACE {

// Return compute properties for each device as a json string
const std::string& devicePropertiesJson();

int smCount(uint32_t deviceId);

// TODO: Implement the below for HAS_ROCTRACER
#ifdef HAS_CUPTI

// Use newer CUPTI activity structs for extended fields
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13010
using CUpti_ActivityKernelType = CUpti_ActivityKernel11;
#elif defined(CUDA_VERSION) && CUDA_VERSION >= 12000
using CUpti_ActivityKernelType = CUpti_ActivityKernel9;
#else
using CUpti_ActivityKernelType = CUpti_ActivityKernel4;
#endif

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
using CUpti_ActivityMemcpyType = CUpti_ActivityMemcpy5;
using CUpti_ActivityMemcpyPtoPType = CUpti_ActivityMemcpyPtoP4;
using CUpti_ActivityMemsetType = CUpti_ActivityMemset4;
#else
using CUpti_ActivityMemcpyType = CUpti_ActivityMemcpy;
using CUpti_ActivityMemcpyPtoPType = CUpti_ActivityMemcpy2;
using CUpti_ActivityMemsetType = CUpti_ActivityMemset;
#endif

float blocksPerSm(const CUpti_ActivityKernelType& kernel);
float warpsPerSm(const CUpti_ActivityKernelType& kernel);

// Occupancy results from CUDA occupancy calculator
// Returns cudaOccResult from cuda_occupancy.h plus a computed occupancy metric
struct OccupancyMetrics {
  float occupancy = -1.0f; // Computed effective occupancy in number of threads
  cudaOccResult result =
      {}; // Raw results from cudaOccMaxActiveBlocksPerMultiprocessor
};

// Return detailed occupancy metrics including limiting factors
OccupancyMetrics computeOccupancyMetrics(
    const CUpti_ActivityKernelType& kernel);
#endif

} // namespace KINETO_NAMESPACE
