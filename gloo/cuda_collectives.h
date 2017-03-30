/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "gloo/cuda.h"
#include "gloo/nccl/nccl.h"

namespace gloo {

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
  auto op = std::make_unique<CudaLocalDeviceOp<T, nccl::ReduceOp<T> > >(
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
  auto op = std::make_unique<CudaLocalDeviceOp<T, nccl::BroadcastOp<T> > >(
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
    std::vector<T*>& hostPtrs,
    const ReductionFunction<T>* fn,
    const int root)
      : devicePtrs_(devicePtrs),
        hostPtrs_(hostPtrs),
        fn_(CudaReductionFunction<T>::toHostFunction(fn)),
        root_(root) {}

  virtual ~CudaLocalHostReduce() {}

  virtual void runAsync() {
    // Asynchronously copy all device buffers to host
    for (int i = 0; i < devicePtrs_.size(); i++) {
      devicePtrs_[i].copyToHostAsync(hostPtrs_[i]);
    }
    // Reduce specified pointers into hostPtrs_[root_]
    devicePtrs_[root_].wait();
    for (int i = 0; i < devicePtrs_.size(); i++) {
      if (i == root_) {
        continue;
      }
      devicePtrs_[i].wait();
      fn_->call(hostPtrs_[root_], hostPtrs_[i], devicePtrs_[0].getCount());
    }
  }

  virtual void wait() {
    // Because reduction happens on CPU, this op is synchronous.
  }

 protected:
  std::vector<CudaDevicePointer<T> >& devicePtrs_;
  std::vector<T*>& hostPtrs_;
  const ReductionFunction<T>* fn_;
  const int root_;
};

template <typename T>
std::unique_ptr<LocalOp<T> > cudaHostReduce(
  std::vector<CudaDevicePointer<T> >& devicePtrs,
  std::vector<T*>& hostPtrs,
  const ReductionFunction<T>* fn,
  int root) {
  return std::make_unique<CudaLocalHostReduce<T> >(
    devicePtrs,
    hostPtrs,
    fn,
    root);
}

template <typename T>
class CudaLocalHostBroadcast : public LocalOp<T> {
 public:
  CudaLocalHostBroadcast(
    std::vector<CudaDevicePointer<T> >& devicePtrs,
    std::vector<T*>& hostPtrs,
    int root)
      : devicePtrs_(devicePtrs), hostPtrs_(hostPtrs), root_(root) {}

  virtual ~CudaLocalHostBroadcast() {}

  virtual void runAsync() {
    // Asynchronously copy host buffer to all device buffers
    for (int i = 0; i < devicePtrs_.size(); i++) {
      devicePtrs_[i].copyFromHostAsync(hostPtrs_[root_]);
    }
  }

  virtual void wait() {
    for (int i = 0; i < devicePtrs_.size(); i++) {
      devicePtrs_[i].wait();
    }
  }

 protected:
  std::vector<CudaDevicePointer<T> >& devicePtrs_;
  std::vector<T*>& hostPtrs_;
  const int root_;
};

template <typename T>
std::unique_ptr<LocalOp<T> > cudaHostBroadcast(
  std::vector<CudaDevicePointer<T> >& devicePtrs,
  std::vector<T*>& hostPtrs,
  int root) {
  return std::make_unique<CudaLocalHostBroadcast<T> >(
    devicePtrs,
    hostPtrs,
    root);
}

} // namespace gloo
