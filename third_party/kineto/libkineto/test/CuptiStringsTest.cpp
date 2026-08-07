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

// Non-greedy strip: only the trailing CUDA-version suffix is removed. The
// single-digit API-generation suffix must survive.
TEST(CuptiStringsTest, RuntimeStripsOnlyTrailingVersionSuffix) {
  EXPECT_EQ(
      runtimeCbidName(
          CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetCaptureInfo_v2_v11030),
      "cudaStreamGetCaptureInfo_v2");

  // CUDA 12.3 added a v3 generation of the same call and CUDA 13 dropped it,
  // so the identifier only exists in between. Where it does, it also covers a
  // cbid above 446 — the range that came back as "INVALID" from the
  // table-driven lookup this replaced (motivation for D104900166).
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030 && CUDART_VERSION < 13000
  EXPECT_EQ(
      runtimeCbidName(
          CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetCaptureInfo_v3_v12030),
      "cudaStreamGetCaptureInfo_v3");
#endif
}

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
