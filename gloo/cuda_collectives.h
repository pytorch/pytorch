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
  CudaLocalMemcpy(Src& src, Dst& dst) : src_(src), dst_(dst) {}

  virtual void runAsync() {
    src_.copyToAsync(dst_);
  }

  virtual void wait() {
    src_.wait();
  }

 protected:
  Src& src_;
  Dst& dst_;
};

template <typename T>
std::vector<nccl::NCCLElement<T> > toDeviceElements(
    const std::vector<CudaDevicePointer<T> >& ptrs,
    size_t offset,
    size_t count) {
  std::vector<nccl::NCCLElement<T> > elements;
  elements.reserve(ptrs.size());
  for (const auto& ptr : ptrs) {
    const auto stream = ptr.getStream();
    elements.push_back(
        nccl::NCCLElement<T>(
            CudaDevicePointer<T>::create(*ptr + offset, count, stream),
            CudaDevicePointer<T>::create(*ptr + offset, count, stream)));
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
          toDeviceElements(devicePtrs, offset, count),
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
      std::vector<CudaDevicePointer<T> >& devicePtrs,
      CudaHostPointer<T>& targetPtr,
      const CudaReductionFunction<T>* fn,
      size_t offset,
      size_t count)
      : root_(0),
        devicePtr_(devicePtrs[root_].range(offset, count)),
        hostPtr_(targetPtr.range(offset, count)) {
    if (devicePtrs.size() > 1) {
      reduceOp_ = make_unique<nccl::ReduceOp<T> >(
          toDeviceElements(devicePtrs, offset, count),
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
    devicePtr_.copyToAsync(hostPtr_);
  }

  virtual void wait() {
    devicePtr_.wait();
  }

 protected:
  const int root_;
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
          toDeviceElements(devicePtrs, offset, count),
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
      std::vector<CudaDevicePointer<T> >& devicePtrs,
      CudaHostPointer<T>& sourcePtr,
      size_t offset,
      size_t count)
      : root_(0),
        devicePtr_(devicePtrs[root_].range(offset, count)),
        hostPtr_(sourcePtr.range(offset, count)) {
    if (devicePtrs.size() > 1) {
      broadcastOp_ = make_unique<nccl::BroadcastOp<T> >(
          toDeviceElements(devicePtrs, offset, count),
          devicePtrs[root_].getDeviceID());
    }
  }

  virtual ~CudaLocalDeviceBroadcast() {}

  virtual void runAsync() {
    // Since we run an asynchronous memcpy to devicePtr_ which is
    // executed on the stream associated with that device pointer, the
    // broadcast operation will only start after the memcpy completes.
    devicePtr_.copyFromAsync(hostPtr_);
    if (broadcastOp_) {
      broadcastOp_->runAsync();
    }
  }

  virtual void wait() {
    devicePtr_.wait();
    if (broadcastOp_) {
      broadcastOp_->wait();
    }
  }

 protected:
  const int root_;
  CudaDevicePointer<T> devicePtr_;
  CudaHostPointer<T> hostPtr_;
  std::unique_ptr<nccl::BroadcastOp<T> > broadcastOp_;
};

template <typename T, typename Dst>
std::unique_ptr<LocalOp<T> > cudaDeviceReduce(
    std::vector<CudaDevicePointer<T> >& devicePtrs,
    Dst& targetPtr,
    const CudaReductionFunction<T>* fn,
    size_t offset,
    size_t count) {
  return make_unique<CudaLocalDeviceReduce<T, Dst> >(
      devicePtrs, targetPtr, fn, offset, count);
}

template <typename T, typename Src>
std::unique_ptr<LocalOp<T> > cudaDeviceBroadcast(
    std::vector<CudaDevicePointer<T> >& devicePtrs,
    Src& sourcePtr,
    size_t offset,
    size_t count) {
  return make_unique<CudaLocalDeviceBroadcast<T, Src> >(
      devicePtrs, sourcePtr, offset, count);
}

template <typename T>
class CudaLocalHostReduce : public LocalOp<T> {
 public:
  CudaLocalHostReduce(
    std::vector<CudaDevicePointer<T> >& devicePtrs,
    CudaHostPointer<T>& hostPtr,
    const CudaReductionFunction<T>* fn)
      : devicePtrs_(devicePtrs),
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
    hostPtr_.copyFromAsync(devicePtrs_[0]);
    for (auto i = 0; i < tmpPtrs_.size(); i++) {
      tmpPtrs_[i].copyFromAsync(devicePtrs_[i+1]);
    }
    // Reduce specified pointers into hostPtr_
    hostPtr_.wait();
    for (auto i = 0; i < tmpPtrs_.size(); i++) {
      tmpPtrs_[i].wait();
      fn_->call(hostPtr_, tmpPtrs_[i], count_);
    }
  }

  virtual void wait() {
    // Because reduction happens on CPU, this op is synchronous.
  }

 protected:
  std::vector<CudaDevicePointer<T> >& devicePtrs_;
  CudaHostPointer<T>& hostPtr_;
  const size_t count_;
  const CudaReductionFunction<T>* fn_;

  // Temporary buffers used for async memory copies
  std::vector<CudaHostPointer<T> > tmpPtrs_;
};

template <typename T>
std::unique_ptr<LocalOp<T> > cudaHostReduce(
    std::vector<CudaDevicePointer<T> >& devicePtrs,
    CudaHostPointer<T>& hostPtr,
    const CudaReductionFunction<T>* fn) {
  // Simple copy operation if there is only a single device pointer.
  if (devicePtrs.size() == 1) {
    return make_unique<
      CudaLocalMemcpy<T, CudaDevicePointer<T>, CudaHostPointer<T> > >(
        devicePtrs[0],
        hostPtr);
  }

  return make_unique<CudaLocalHostReduce<T> >(
    devicePtrs,
    hostPtr,
    fn);
}

template <typename T>
class CudaLocalHostBroadcast : public LocalOp<T> {
 public:
  CudaLocalHostBroadcast(
    std::vector<CudaDevicePointer<T> >& devicePtrs,
    CudaHostPointer<T>& hostPtr)
      : devicePtrs_(devicePtrs),
        hostPtr_(hostPtr) {}

  virtual void runAsync() {
    // Asynchronously copy host memory to device
    for (auto i = 0; i < devicePtrs_.size(); i++) {
      devicePtrs_[i].copyFromAsync(hostPtr_);
    }
  }

  virtual void wait() {
    for (auto i = 0; i < devicePtrs_.size(); i++) {
      devicePtrs_[i].wait();
    }
  }

 protected:
  std::vector<CudaDevicePointer<T> >& devicePtrs_;
  CudaHostPointer<T>& hostPtr_;
};

template <typename T>
std::unique_ptr<LocalOp<T> > cudaHostBroadcast(
    std::vector<CudaDevicePointer<T> >& devicePtrs,
    CudaHostPointer<T>& hostPtr) {
  // Simple copy operation if there is only a single device pointer.
  if (devicePtrs.size() == 1) {
    return make_unique<
      CudaLocalMemcpy<T, CudaHostPointer<T>, CudaDevicePointer<T> > >(
        hostPtr,
        devicePtrs[0]);
  }
  return make_unique<CudaLocalHostBroadcast<T> >(
    devicePtrs,
    hostPtr);
}

} // namespace gloo
