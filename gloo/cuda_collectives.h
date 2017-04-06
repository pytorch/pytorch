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
#include "gloo/cuda.h"
#include "gloo/nccl/nccl.h"

namespace gloo {

template <typename T, typename Src, typename Dst>
class CudaLocalMemcpy : public LocalOp<T> {
 public:
  CudaLocalMemcpy(CudaStream& stream, Src& src, Dst& dst)
      : stream_(stream),
        src_(src),
        dst_(dst) {}

  virtual void runAsync() {
    stream_.copyAsync(dst_, src_);
  }

  virtual void wait() {
    stream_.wait();
  }

 protected:
  CudaStream& stream_;
  Src& src_;
  Dst& dst_;
};

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
class CudaLocalDeviceReduce;

// Partial specialization for device pointer target
template <typename T>
class CudaLocalDeviceReduce<T, CudaDevicePointer<T> > : public LocalOp<T> {
 public:
  CudaLocalDeviceReduce(
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

  virtual ~CudaLocalDeviceReduce() {}

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
class CudaLocalDeviceReduce<T, CudaHostPointer<T> > : public LocalOp<T> {
 public:
  CudaLocalDeviceReduce(
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

  virtual ~CudaLocalDeviceReduce() {}

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
class CudaLocalDeviceBroadcast;

// Specialization for device pointer target
template <typename T>
class CudaLocalDeviceBroadcast<T, CudaDevicePointer<T> > : public LocalOp<T> {
 public:
  CudaLocalDeviceBroadcast(
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

  virtual ~CudaLocalDeviceBroadcast() {}

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

// Specialization for host pointer target
template <typename T>
class CudaLocalDeviceBroadcast<T, CudaHostPointer<T> > : public LocalOp<T> {
 public:
  CudaLocalDeviceBroadcast(
      std::vector<CudaStream>& streams,
      std::vector<CudaDevicePointer<T> >& devicePtrs,
      CudaHostPointer<T>& sourcePtr,
      size_t offset,
      size_t count)
      : root_(0),
        stream_(streams[root_]),
        devicePtr_(devicePtrs[root_].range(offset, count)),
        hostPtr_(sourcePtr.range(offset, count)) {
    if (devicePtrs.size() > 1) {
      broadcastOp_ = make_unique<nccl::BroadcastOp<T> >(
          toDeviceElements(streams, devicePtrs, offset, count),
          devicePtrs[root_].getDeviceID());
    }
  }

  virtual ~CudaLocalDeviceBroadcast() {}

  virtual void runAsync() {
    // Since we run an asynchronous memcpy to devicePtr_ which is
    // executed on the stream associated with that device pointer, the
    // broadcast operation will only start after the memcpy completes.
    stream_.copyAsync(devicePtr_, hostPtr_);
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
  CudaHostPointer<T> hostPtr_;
  std::unique_ptr<nccl::BroadcastOp<T> > broadcastOp_;
};

template <typename T, typename Dst>
std::unique_ptr<LocalOp<T> > cudaDeviceReduce(
    std::vector<CudaStream>& streams,
    std::vector<CudaDevicePointer<T> >& devicePtrs,
    Dst& targetPtr,
    const CudaReductionFunction<T>* fn,
    size_t offset,
    size_t count) {
  GLOO_ENFORCE_EQ(streams.size(), devicePtrs.size());
  return make_unique<CudaLocalDeviceReduce<T, Dst> >(
      streams, devicePtrs, targetPtr, fn, offset, count);
}

template <typename T, typename Src>
std::unique_ptr<LocalOp<T> > cudaDeviceBroadcast(
    std::vector<CudaStream>& streams,
    std::vector<CudaDevicePointer<T> >& devicePtrs,
    Src& sourcePtr,
    size_t offset,
    size_t count) {
  GLOO_ENFORCE_EQ(streams.size(), devicePtrs.size());
  return make_unique<CudaLocalDeviceBroadcast<T, Src> >(
      streams, devicePtrs, sourcePtr, offset, count);
}

template <typename T>
class CudaLocalHostReduce : public LocalOp<T> {
 public:
  CudaLocalHostReduce(
      std::vector<CudaStream>& streams,
      std::vector<CudaDevicePointer<T> >& devicePtrs,
      CudaHostPointer<T>& hostPtr,
      const CudaReductionFunction<T>* fn)
      : streams_(streams),
        devicePtrs_(devicePtrs),
        hostPtr_(hostPtr),
        count_(hostPtr_.getCount()),
        fn_(fn) {
    // Allocate N-1 temporary buffers to asynchronously
    // copy device memory into.
    for (auto i = 1; i < devicePtrs_.size(); i++) {
      tmpPtrs_.push_back(CudaHostPointer<T>::alloc(count_));
    }
  }

  virtual void runAsync() {
    // Asynchronously copy device memory to host
    streams_[0].copyAsync(hostPtr_, devicePtrs_[0]);
    for (auto i = 1; i < devicePtrs_.size(); i++) {
      streams_[i].copyAsync(tmpPtrs_[i-1], devicePtrs_[i]);
    }
    // Reduce specified pointers into hostPtr_
    streams_[0].wait();
    for (auto i = 1; i < devicePtrs_.size(); i++) {
      streams_[i].wait();
      fn_->call(hostPtr_, tmpPtrs_[i-1], count_, streams_[i]);
    }
  }

  virtual void wait() {
    // Because reduction happens on CPU, this op is synchronous.
  }

 protected:
  std::vector<CudaStream>& streams_;
  std::vector<CudaDevicePointer<T> >& devicePtrs_;
  CudaHostPointer<T>& hostPtr_;
  const size_t count_;
  const CudaReductionFunction<T>* fn_;

  // Temporary buffers used for async memory copies
  std::vector<CudaHostPointer<T> > tmpPtrs_;
};

template <typename T>
std::unique_ptr<LocalOp<T> > cudaHostReduce(
    std::vector<CudaStream>& streams,
    std::vector<CudaDevicePointer<T> >& devicePtrs,
    CudaHostPointer<T>& hostPtr,
    const CudaReductionFunction<T>* fn) {
  GLOO_ENFORCE_EQ(streams.size(), devicePtrs.size());
  // Simple copy operation if there is only a single device pointer.
  if (devicePtrs.size() == 1) {
    return make_unique<
      CudaLocalMemcpy<T, CudaDevicePointer<T>, CudaHostPointer<T> > >(
          streams[0],
          devicePtrs[0],
          hostPtr);
  }

  return make_unique<CudaLocalHostReduce<T> >(
      streams,
      devicePtrs,
      hostPtr,
      fn);
}

template <typename T>
class CudaLocalHostBroadcast : public LocalOp<T> {
 public:
  CudaLocalHostBroadcast(
      std::vector<CudaStream>& streams,
      std::vector<CudaDevicePointer<T> >& devicePtrs,
      CudaHostPointer<T>& hostPtr)
      : streams_(streams),
        devicePtrs_(devicePtrs),
        hostPtr_(hostPtr) {}

  virtual void runAsync() {
    // Asynchronously copy host memory to device
    for (auto i = 0; i < devicePtrs_.size(); i++) {
      streams_[i].copyAsync(devicePtrs_[i], hostPtr_);
    }
  }

  virtual void wait() {
    for (auto i = 0; i < devicePtrs_.size(); i++) {
      streams_[i].wait();
    }
  }

 protected:
  std::vector<CudaStream>& streams_;
  std::vector<CudaDevicePointer<T> >& devicePtrs_;
  CudaHostPointer<T>& hostPtr_;
};

template <typename T>
std::unique_ptr<LocalOp<T> > cudaHostBroadcast(
    std::vector<CudaStream>& streams,
    std::vector<CudaDevicePointer<T> >& devicePtrs,
    CudaHostPointer<T>& hostPtr) {
  GLOO_ENFORCE_EQ(streams.size(), devicePtrs.size());
  // Simple copy operation if there is only a single device pointer.
  if (devicePtrs.size() == 1) {
    return make_unique<
      CudaLocalMemcpy<T, CudaHostPointer<T>, CudaDevicePointer<T> > >(
          streams[0],
          hostPtr,
          devicePtrs[0]);
  }
  return make_unique<CudaLocalHostBroadcast<T> >(
      streams,
      devicePtrs,
      hostPtr);
}

} // namespace gloo
