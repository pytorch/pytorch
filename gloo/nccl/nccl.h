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
#include <memory>
#include <string>
#include <vector>

#include "gloo/common/logging.h"
#include "gloo/cuda.h"
#include "gloo/types.h"

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

#define NCCL_VERSION_MIN(major, minor, patch) \
  ((NCCL_MAJOR > major) || \
    ((NCCL_MAJOR == major) && ((NCCL_MINOR > minor) || \
      ((NCCL_MINOR == minor) && (NCCL_PATCH >= patch)) )))

template <typename T>
struct NCCLElement {
  NCCLElement(
      CudaDevicePointer<T> srcParam,
      CudaStream& srcStreamParam,
      CudaDevicePointer<T> dstParam,
      CudaStream& dstStreamParam)
      : src(std::move(srcParam)),
        srcStream(srcStreamParam),
        dst(std::move(dstParam)),
        dstStream(dstStreamParam),
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
  CudaStream& srcStream;
  CudaDevicePointer<T> dst;
  CudaStream& dstStream;
  const int device;
};

template <typename T>
class NCCLExecution {
 public:
  /* implicit */ NCCLExecution(std::vector<NCCLElement<T>>&& elements);
  NCCLExecution(NCCLExecution&&) = default;
  ~NCCLExecution();

  std::vector<int> getDevices() const;
  std::string getKey() const;

  std::vector<NCCLElement<T>> elements;
  std::vector<cudaEvent_t> ncclEvents;
};

template <typename T>
class NCCLContext;

template <typename T>
class NCCLOp {
 public:
  explicit NCCLOp(NCCLExecution<T>&& execution);
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

  NCCLExecution<T> execution_;
  std::shared_ptr<NCCLContext<T>> context_;
};

template <typename T>
ncclRedOp_t toReductionOp(const CudaReductionFunction<T>* fn) {
  switch (fn->type()) {
    case SUM:
      return ncclSum;
    case PRODUCT:
      return ncclProd;
    case MAX:
      return ncclMax;
    case MIN:
      return ncclMin;
    default:
      GLOO_ENFORCE(false, "NCCL does not support reduction type ", fn->type());
  }
}

template <typename T>
class ReduceOp : public NCCLOp<T> {
 public:
  ReduceOp(
    NCCLExecution<T>&& execution,
    const CudaReductionFunction<T>* fn,
    const int root)
      : NCCLOp<T>(std::move(execution)), op_(toReductionOp(fn)), root_(root) {
    for (const auto& element : execution.elements) {
      GLOO_ENFORCE_EQ(
        element.src.getCount(),
        element.dst.getCount(),
        "NCCL source and destination must be the same size");
    }
  }

  void runAsync() override;

 protected:
  const ncclRedOp_t op_;
  const int root_;
};

template <typename T>
class AllreduceOp : public NCCLOp<T> {
 public:
  AllreduceOp(
    NCCLExecution<T>&& execution,
    const CudaReductionFunction<T>* fn)
      : NCCLOp<T>(std::move(execution)), op_(toReductionOp(fn)) {
    for (const auto& element : execution.elements) {
      GLOO_ENFORCE_EQ(
        element.src.getCount(),
        element.dst.getCount(),
        "NCCL source and destination must be the same size");
    }
  }

  void runAsync() override;

 protected:
  const ncclRedOp_t op_;
};

template <typename T>
class ReduceScatterOp : public NCCLOp<T> {
 public:
  ReduceScatterOp(
    NCCLExecution<T>&& execution,
    const CudaReductionFunction<T>* fn)
      : NCCLOp<T>(std::move(execution)), op_(toReductionOp(fn)) {
    for (const auto& element : execution.elements) {
      GLOO_ENFORCE_EQ(
        element.src.getCount() / execution.elements.size(),
        element.dst.getCount(),
        "NCCL source must be ",
        execution.elements.size(),
        " times as big as the destination");
    }
  }

  void runAsync() override;

 protected:
  const ncclRedOp_t op_;
};

template <typename T>
class BroadcastOp : public NCCLOp<T> {
 public:
  explicit BroadcastOp(NCCLExecution<T>&& execution, int root)
      : NCCLOp<T>(std::move(execution)), root_(root) {
    for (const auto& element : execution.elements) {
      GLOO_ENFORCE_EQ(
        element.src.getCount(),
        element.dst.getCount(),
        "NCCL source and destination must be the same size");
    }
  }

  void runAsync() override;

 protected:
  const int root_;
};

template <typename T>
class AllgatherOp : public NCCLOp<T> {
 public:
  explicit AllgatherOp(NCCLExecution<T>&& execution)
      : NCCLOp<T>(std::move(execution)) {
    for (const auto& element : execution.elements) {
      GLOO_ENFORCE_EQ(
        element.src.getCount(),
        element.dst.getCount() / execution.elements.size(),
        "NCCL destination must be ",
        execution.elements.size(),
        " times as big as the source");
    }
  }

  void runAsync() override;
};

} // namespace nccl
} // namespace gloo
