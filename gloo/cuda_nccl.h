/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <nccl.h>
#include <vector>

#include "gloo/common/logging.h"
#include "gloo/cuda.h"

namespace gloo {
namespace nccl {

#define NCCL_CHECK(condition)          \
  do {                                 \
    ncclResult_t status = (condition); \
    GLOO_ENFORCE_EQ(                   \
        status,                        \
        ncclSuccess,                   \
        " ",                           \
        "Error at: ",                  \
        __FILE__,                      \
        __LINE__,                      \
        ": ",                          \
        ncclGetErrorString(status));   \
  } while (0)

template <typename T>
struct NCCLElement {
  NCCLElement(CudaDevicePointer<T> src, CudaDevicePointer<T> dst)
      : src(std::move(src)),
        dst(std::move(dst)),
        count(src.getCount()),
        device(src.getDeviceID()) {
    GLOO_ENFORCE_EQ(
        src.getCount(),
        dst.getCount(),
        "NCCL source and destination must be the same size");
    GLOO_ENFORCE_EQ(
        src.getDeviceID(),
        dst.getDeviceID(),
        "NCCL source and destination must be on same device");
  }

  CudaDevicePointer<T> src;
  CudaDevicePointer<T> dst;
  const size_t count;
  const int device;
};

template <typename T>
class NCCLContext {
 public:
  NCCLContext(std::vector<NCCLElement<T>>&& elements, int root);
  NCCLContext(NCCLContext&&) = default;
  ~NCCLContext();

  // Instances cannot be copied or copy-assigned
  NCCLContext(const NCCLContext&) = delete;
  NCCLContext& operator=(const NCCLContext&) = delete;

  const int root;
  std::vector<NCCLElement<T>> elements;
  std::vector<ncclComm_t> comms;
  std::vector<cudaEvent_t> ncclEvents;
};

template <typename T>
class NCCLOp {
 public:
  explicit NCCLOp(NCCLContext<T>&& context) : context_(std::move(context)) {}
  NCCLOp(NCCLOp&&) = default;
  virtual ~NCCLOp() = default;

  // Kick off the operation
  virtual void runAsync() = 0;

  // Wait for the operation to complete
  virtual void wait();

 protected:
  // Run the NCCL operation
  template <typename F>
  void runNCCL(F&& f);

  NCCLContext<T> context_;
};

template <typename T>
class ReduceOp : public NCCLOp<T> {
 public:
  explicit ReduceOp(NCCLContext<T>&& context) : NCCLOp<T>(std::move(context)) {}
  void runAsync() override;
};

template <typename T>
class BroadcastOp : public NCCLOp<T> {
 public:
  explicit BroadcastOp(NCCLContext<T>&& context)
      : NCCLOp<T>(std::move(context)) {}
  void runAsync() override;
};

} // namespace nccl
} // namespace gloo
