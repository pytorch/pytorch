/*
 * Copyright (C) Intel Corporation
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define XPUPTI_COMPUTE_BUILDING
#include "XpuptiPeerCopyCompute.h"

#include <sycl/sycl.hpp>

XPUPTI_COMPUTE_API bool CopyPeerToPeerOnXpu(std::size_t bytes) {
  auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  if (devices.size() < 2) {
    return false;
  }
  auto& src = devices[0];
  auto& dst = devices[1];
  if (!src.ext_oneapi_can_access_peer(dst)) {
    return false;
  }

  const sycl::context context({src, dst});
  sycl::queue queue(context, src, sycl::property::queue::in_order());
  auto* srcPtr = sycl::malloc_device<char>(bytes, src, context);
  auto* dstPtr = sycl::malloc_device<char>(bytes, dst, context);
  queue.memcpy(dstPtr, srcPtr, bytes).wait_and_throw();
  sycl::free(srcPtr, context);
  sycl::free(dstPtr, context);
  return true;
}
