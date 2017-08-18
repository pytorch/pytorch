/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "gloo/common/common.h"
#include "gloo/common/logging.h"
#include "gloo/nccl/nccl.h"

namespace gloo {

template <typename T>
std::vector<nccl::NCCLElement<T> > toDeviceElements(
    std::vector<CudaStream>& streams,
    const std::vector<CudaDevicePointer<T> >& ptrs,
    size_t offset,
    size_t count) {
  std::vector<nccl::NCCLElement<T> > elements;
  elements.reserve(ptrs.size());
  for (auto i = 0; i < ptrs.size(); i++) {
    elements.push_back(
        nccl::NCCLElement<T>(
            ptrs[i].range(offset, count),
            streams[i],
            ptrs[i].range(offset, count),
            streams[i]));
  }
  return elements;
}

// Forward declaration
template <typename T, typename Dst>
class CudaLocalNCCLReduce;

// Partial specialization for device pointer target
template <typename T>
class CudaLocalNCCLReduce<T, CudaDevicePointer<T> > : public LocalOp<T> {
 public:
  CudaLocalNCCLReduce(
      std::vector<CudaStream>& streams,
      std::vector<CudaDevicePointer<T> >& devicePtrs,
      CudaDevicePointer<T>& targetPtr,
      const CudaReductionFunction<T>* fn,
      size_t offset,
      size_t count) {
    // The targetPtr must be one of devicePtrs.
    auto root = -1;
    for (auto i = 0; i < devicePtrs.size(); i++) {
      if (devicePtrs[i] == targetPtr) {
        root = i;
        break;
      }
    }
    GLOO_ENFORCE_GE(root, 0, "targetPtr must be one of devicePtrs");

    // Only if we have multiple device pointers does this
    // operation need to execute.
    if (devicePtrs.size() > 1) {
      reduceOp_ = make_unique<nccl::ReduceOp<T> >(
          toDeviceElements(streams, devicePtrs, offset, count),
          fn,
          devicePtrs[root].getDeviceID());
    }
  }

  virtual ~CudaLocalNCCLReduce() {}

  virtual void runAsync() {
    if (reduceOp_) {
      reduceOp_->runAsync();
    }
  }

  virtual void wait() {
    if (reduceOp_) {
      reduceOp_->wait();
    }
  }

 protected:
  std::unique_ptr<nccl::ReduceOp<T> > reduceOp_;
};

// Partial specialization for host pointer target
template <typename T>
class CudaLocalNCCLReduce<T, CudaHostPointer<T> > : public LocalOp<T> {
 public:
  CudaLocalNCCLReduce(
      std::vector<CudaStream>& streams,
      std::vector<CudaDevicePointer<T> >& devicePtrs,
      CudaHostPointer<T>& targetPtr,
      const CudaReductionFunction<T>* fn,
      size_t offset,
      size_t count)
      : root_(0),
        stream_(streams[root_]),
        devicePtr_(devicePtrs[root_].range(offset, count)),
        hostPtr_(targetPtr.range(offset, count)) {
    if (devicePtrs.size() > 1) {
      reduceOp_ = make_unique<nccl::ReduceOp<T> >(
          toDeviceElements(streams, devicePtrs, offset, count),
          fn,
          devicePtrs[root_].getDeviceID());
    }
  }

  virtual ~CudaLocalNCCLReduce() {}

  virtual void runAsync() {
    if (reduceOp_) {
      reduceOp_->runAsync();
    }

    // The stream for operations on devicePtrs_[0] now includes an
    // asynchronous wait for completion of the reduce operation, if it
    // was executed. This means we can sequence an asynchronous memory
    // copy and wait on completion of that to signal completion of
    // both operations.
    stream_.copyAsync(hostPtr_, devicePtr_);
  }

  virtual void wait() {
    stream_.wait();
  }

 protected:
  const int root_;
  CudaStream& stream_;
  CudaDevicePointer<T> devicePtr_;
  CudaHostPointer<T> hostPtr_;
  std::unique_ptr<nccl::ReduceOp<T> > reduceOp_;
};

// Forward declaration
template <typename T, typename Src>
class CudaLocalNCCLBroadcast;

// Specialization for device pointer source
template <typename T>
class CudaLocalNCCLBroadcast<T, CudaDevicePointer<T> > : public LocalOp<T> {
 public:
  CudaLocalNCCLBroadcast(
      std::vector<CudaStream>& streams,
      std::vector<CudaDevicePointer<T> >& devicePtrs,
      CudaDevicePointer<T>& sourcePtr,
      size_t offset,
      size_t count) {
    // The sourcePtr must be one of devicePtrs.
    auto root = -1;
    for (auto i = 0; i < devicePtrs.size(); i++) {
      if (devicePtrs[i] == sourcePtr) {
        root = i;
        break;
      }
    }
    GLOO_ENFORCE_GE(root, 0, "sourcePtr must be one of devicePtrs");

    // Only if we have multiple device pointers does this
    // operation need to execute.
    if (devicePtrs.size() > 1) {
      broadcastOp_ = make_unique<nccl::BroadcastOp<T> >(
          toDeviceElements(streams, devicePtrs, offset, count),
          devicePtrs[root].getDeviceID());
    }
  }

  virtual ~CudaLocalNCCLBroadcast() {}

  virtual void runAsync() {
    if (broadcastOp_) {
      broadcastOp_->runAsync();
    }
  }

  virtual void wait() {
    if (broadcastOp_) {
      broadcastOp_->wait();
    }
  }

 protected:
  std::unique_ptr<nccl::BroadcastOp<T> > broadcastOp_;
};

// Specialization for host pointer source
template <typename T>
class CudaLocalNCCLBroadcast<T, CudaHostPointer<T> > : public LocalOp<T> {
 public:
  CudaLocalNCCLBroadcast(
      std::vector<CudaStream>& streams,
      std::vector<CudaDevicePointer<T> >& devicePtrs,
      CudaHostPointer<T>& sourcePtr,
      size_t offset,
      size_t count)
      : root_(0),
        stream_(streams[root_]),
        devicePtr_(devicePtrs[root_].range(offset, count)),
        sourcePtr_(sourcePtr.range(offset, count)) {
    if (devicePtrs.size() > 1) {
      broadcastOp_ = make_unique<nccl::BroadcastOp<T> >(
          toDeviceElements(streams, devicePtrs, offset, count),
          devicePtrs[root_].getDeviceID());
    }
  }

  virtual ~CudaLocalNCCLBroadcast() {}

  virtual void runAsync() {
    // Since we run an asynchronous memcpy to devicePtr_ which is
    // executed on the stream associated with that device pointer, the
    // broadcast operation will only start after the memcpy completes.
    stream_.copyAsync(devicePtr_, sourcePtr_);
    if (broadcastOp_) {
      broadcastOp_->runAsync();
    }
  }

  virtual void wait() {
    stream_.wait();
    if (broadcastOp_) {
      broadcastOp_->wait();
    }
  }

 protected:
  const int root_;
  CudaStream& stream_;
  CudaDevicePointer<T> devicePtr_;
  CudaHostPointer<T> sourcePtr_;
  std::unique_ptr<nccl::BroadcastOp<T> > broadcastOp_;
};

template <typename T, typename Dst>
std::unique_ptr<LocalOp<T> > cudaNCCLReduce(
    std::vector<CudaStream>& streams,
    std::vector<CudaDevicePointer<T> >& devicePtrs,
    Dst& targetPtr,
    const CudaReductionFunction<T>* fn,
    size_t offset,
    size_t count) {
  GLOO_ENFORCE_EQ(streams.size(), devicePtrs.size());
  return make_unique<CudaLocalNCCLReduce<T, Dst> >(
      streams, devicePtrs, targetPtr, fn, offset, count);
}

template <typename T, typename Src>
std::unique_ptr<LocalOp<T> > cudaNCCLBroadcast(
    std::vector<CudaStream>& streams,
    std::vector<CudaDevicePointer<T> >& devicePtrs,
    Src& sourcePtr,
    size_t offset,
    size_t count) {
  GLOO_ENFORCE_EQ(streams.size(), devicePtrs.size());
  return make_unique<CudaLocalNCCLBroadcast<T, Src> >(
      streams, devicePtrs, sourcePtr, offset, count);
}

} // namespace gloo
