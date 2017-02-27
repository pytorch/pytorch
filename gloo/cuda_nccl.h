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

struct NCCLElement {
  NCCLElement(
      void* src,
      void* dst,
      const size_t length,
      const int device,
      const cudaStream_t stream)
      : src(src), dst(dst), length(length), device(device), stream(stream) {}

  void* src;
  void* dst;
  const size_t length;
  const int device;
  const cudaStream_t stream;
};

class NCCLContext {
 public:
  explicit NCCLContext(
      int device,
      cudaStream_t stream,
      std::vector<NCCLElement>&& elements,
      int root);

  NCCLContext(NCCLContext&& other) noexcept;

  ~NCCLContext();

  // Instances cannot be copied or copy-assigned
  NCCLContext(const NCCLContext&) = delete;
  NCCLContext& operator=(const NCCLContext&) = delete;

  const int masterDevice_;
  cudaEvent_t masterEvent_;
  const cudaStream_t masterStream_;
  const int root_;
  std::vector<NCCLElement> elements_;
  std::vector<ncclComm_t> comms_;
  std::vector<cudaEvent_t> events_;
};

template <typename T>
class NCCLOp {
 public:
  explicit NCCLOp(NCCLContext&& context) : context_(std::move(context)) {}
  NCCLOp(NCCLOp&& other) = default;
  virtual ~NCCLOp() = default;

  // Kick off the operation
  virtual void runAsync() = 0;

  // Wait for the operation to complete
  virtual void wait();

 protected:
  // Run the NCCL operation
  template <typename F>
  void runNCCL(F&& f);

  NCCLContext context_;
};

template <typename T>
class ReduceOp : public NCCLOp<T> {
 public:
  explicit ReduceOp(NCCLContext&& context) : NCCLOp<T>(std::move(context)) {}
  void runAsync() override;
};

template <typename T>
class BroadcastOp : public NCCLOp<T> {
 public:
  explicit BroadcastOp(NCCLContext&& context) : NCCLOp<T>(std::move(context)) {}
  void runAsync() override;
};

} // namespace nccl
} // namespace gloo
