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

// CudaLocalDeviceOp wraps NCCL collectives as a LocalOp.
template <typename T, typename DeviceOp>
class CudaLocalDeviceOp : public LocalOp<T> {
 public:
  explicit CudaLocalDeviceOp(DeviceOp&& op) : op_(std::move(op)) {}

  virtual ~CudaLocalDeviceOp() {}

  virtual void runAsync() {
    op_.runAsync();
  }

  virtual void wait() {
    op_.wait();
  }

 protected:
  DeviceOp op_;
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
    elements.push_back(nccl::NCCLElement<T>(
        CudaDevicePointer<T>::create(*ptr + offset, count, stream),
        CudaDevicePointer<T>::create(*ptr + offset, count, stream)));
  }
  return elements;
}

template <typename T>
std::unique_ptr<LocalOp<T> > cudaDeviceReduce(
    const std::vector<CudaDevicePointer<T> >& devicePtrs,
    const CudaReductionFunction<T>* fn,
    const int root,
    size_t offset = 0,
    size_t count = 0) {
  if (count == 0) {
    count = devicePtrs[root].getCount();
  }
  auto op = make_unique<CudaLocalDeviceOp<T, nccl::ReduceOp<T> > >(
    nccl::ReduceOp<T>(
      toDeviceElements(devicePtrs, offset, count),
      fn,
      devicePtrs[root].getDeviceID()));
  return std::move(op);
}

template <typename T>
std::unique_ptr<LocalOp<T> > cudaDeviceBroadcast(
    const std::vector<CudaDevicePointer<T> >& devicePtrs,
    const int root,
    size_t offset = 0,
    size_t count = 0) {
  if (count == 0) {
    count = devicePtrs[root].getCount();
  }
  auto op = make_unique<CudaLocalDeviceOp<T, nccl::BroadcastOp<T> > >(
    nccl::BroadcastOp<T>(
      toDeviceElements(devicePtrs, offset, count),
      devicePtrs[root].getDeviceID()));
  return std::move(op);
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
