/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "src/cupti_strings.h"

using namespace KINETO_NAMESPACE;

TEST(CuptiStringsTest, RuntimeStripsFourDigitVersionSuffix) {
  EXPECT_EQ(
      runtimeCbidName(CUPTI_RUNTIME_TRACE_CBID_cudaDriverGetVersion_v3020),
      "cudaDriverGetVersion");
  EXPECT_EQ(
      runtimeCbidName(CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020),
      "cudaDeviceSynchronize");
  EXPECT_EQ(
      runtimeCbidName(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000),
      "cudaLaunchKernel");
}

TEST(CuptiStringsTest, RuntimeStripsFiveDigitVersionSuffix) {
#if defined(CUPTI_API_VERSION) && CUPTI_API_VERSION >= 18
  EXPECT_EQ(
      runtimeCbidName(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060),
      "cudaLaunchKernelExC");
#endif
  EXPECT_EQ(
      runtimeCbidName(
          CUPTI_RUNTIME_TRACE_CBID_cudaStreamSetAttribute_ptsz_v11000),
      "cudaStreamSetAttribute_ptsz");
}

// Non-greedy strip: only the trailing CUDA-version suffix `_v12030` is removed.
// The `_v3` API-generation suffix (single digit) must survive. This case also
// exercises a cbid >= 446 — the range that returned "INVALID" before this
// refactor (motivation for D104900166).
#if defined(CUPTI_API_VERSION) && CUPTI_API_VERSION >= 21
TEST(CuptiStringsTest, RuntimeStripsOnlyTrailingVersionSuffix) {
  EXPECT_EQ(
      runtimeCbidName(
          CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetCaptureInfo_v3_v12030),
      "cudaStreamGetCaptureInfo_v3");
}
#endif

TEST(CuptiStringsTest, RuntimePreservesNamesWithoutVersionSuffix) {
  EXPECT_EQ(runtimeCbidName(CUPTI_RUNTIME_TRACE_CBID_INVALID), "INVALID");
}

TEST(CuptiStringsTest, RuntimeReturnsInvalidForUnknownCbids) {
  EXPECT_EQ(runtimeCbidName(static_cast<CUpti_CallbackId>(-1)), "INVALID");
  EXPECT_EQ(
      runtimeCbidName(static_cast<CUpti_CallbackId>(0xFFFFFFFF)), "INVALID");
  EXPECT_EQ(runtimeCbidName(100000), "INVALID");
}

TEST(CuptiStringsTest, DriverPreservesNamesWithoutVersionSuffix) {
  EXPECT_EQ(driverCbidName(CUPTI_DRIVER_TRACE_CBID_INVALID), "INVALID");
  EXPECT_EQ(
      driverCbidName(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel), "cuLaunchKernel");
  EXPECT_EQ(driverCbidName(CUPTI_DRIVER_TRACE_CBID_cuMemCreate), "cuMemCreate");
}

TEST(CuptiStringsTest, DriverReturnsInvalidForUnknownCbids) {
  EXPECT_EQ(driverCbidName(static_cast<CUpti_CallbackId>(-1)), "INVALID");
  EXPECT_EQ(
      driverCbidName(static_cast<CUpti_CallbackId>(0xFFFFFFFF)), "INVALID");
}
