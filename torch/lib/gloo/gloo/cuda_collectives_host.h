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

namespace gloo {

// Forward declaration
template <typename T, typename Dst>
class CudaLocalHostReduce;

// Partial specialization for device pointer target
template <typename T>
class CudaLocalHostReduce<T, CudaDevicePointer<T> > : public LocalOp<T> {
 public:
  CudaLocalHostReduce(
      std::vector<CudaStream>& streams,
      std::vector<CudaDevicePointer<T> >& devicePtrs,
      CudaDevicePointer<T>& targetPtr,
      const CudaReductionFunction<T>* fn,
      size_t offset,
      size_t count)
      : streams_(streams),
        targetPtr_(targetPtr.range(offset, count)),
        offset_(offset),
        count_(count),
        fn_(fn) {
    // Incorporate offset/count into devicePtrs
    devicePtrs_.reserve(devicePtrs.size());
    for (const auto& ptr : devicePtrs) {
      devicePtrs_.push_back(ptr.range(offset, count));
    }
    // Allocate N temporary buffers to async copy device ptrs into.
    for (auto i = 0; i < devicePtrs_.size(); i++) {
      tmpPtrs_.push_back(CudaHostPointer<T>::alloc(count));
    }
  }

  virtual void runAsync() {
    // Asynchronously copy device memory to host
    for (auto i = 0; i < devicePtrs_.size(); i++) {
      streams_[i].copyAsync(tmpPtrs_[i], devicePtrs_[i]);
    }
    // Reduce specified pointers into tmpPtrs_[0]
    streams_[0].wait();
    for (auto i = 1; i < devicePtrs_.size(); i++) {
      streams_[i].wait();
      fn_->call(tmpPtrs_[0], tmpPtrs_[i], count_, streams_[i]);
    }
    // Copy final reduction back to device
    streams_[0].copyAsync(targetPtr_, tmpPtrs_[0]);
  }

  virtual void wait() {
    // Reduction happens on CPU but we still have to wait for the
    // memory copy of the result back to device.
    streams_[0].wait();
  }

 protected:
  std::vector<CudaStream>& streams_;
  std::vector<CudaDevicePointer<T> > devicePtrs_;
  CudaDevicePointer<T> targetPtr_;
  const size_t offset_;
  const size_t count_;
  const CudaReductionFunction<T>* fn_;

  // Temporary buffers used for async memory copies
  std::vector<CudaHostPointer<T> > tmpPtrs_;
};

// Partial specialization for host pointer target
template <typename T>
class CudaLocalHostReduce<T, CudaHostPointer<T> > : public LocalOp<T> {
 public:
  CudaLocalHostReduce(
      std::vector<CudaStream>& streams,
      std::vector<CudaDevicePointer<T> >& devicePtrs,
      CudaHostPointer<T>& targetPtr,
      const CudaReductionFunction<T>* fn,
      size_t offset,
      size_t count)
      : streams_(streams),
        targetPtr_(targetPtr.range(offset, count)),
        offset_(offset),
        count_(count),
        fn_(fn) {
    // Incorporate offset/count into devicePtrs
    devicePtrs_.reserve(devicePtrs.size());
    for (const auto& ptr : devicePtrs) {
      devicePtrs_.push_back(ptr.range(offset, count));
    }
    // Allocate N-1 temporary buffers to async copy device ptrs into.
    for (auto i = 1; i < devicePtrs_.size(); i++) {
      tmpPtrs_.push_back(CudaHostPointer<T>::alloc(count));
    }
  }

  virtual void runAsync() {
    // Asynchronously copy device memory to host
    streams_[0].copyAsync(targetPtr_, devicePtrs_[0]);
    for (auto i = 1; i < devicePtrs_.size(); i++) {
      streams_[i].copyAsync(tmpPtrs_[i-1], devicePtrs_[i]);
    }
    // Reduce specified pointers into targetPtr_
    streams_[0].wait();
    for (auto i = 1; i < devicePtrs_.size(); i++) {
      streams_[i].wait();
      fn_->call(targetPtr_, tmpPtrs_[i-1], count_, streams_[i]);
    }
  }

  virtual void wait() {
    // Because reduction happens on CPU, this op is synchronous.
  }

 protected:
  std::vector<CudaStream>& streams_;
  std::vector<CudaDevicePointer<T> > devicePtrs_;
  CudaHostPointer<T> targetPtr_;
  const size_t offset_;
  const size_t count_;
  const CudaReductionFunction<T>* fn_;

  // Temporary buffers used for async memory copies
  std::vector<CudaHostPointer<T> > tmpPtrs_;
};

// Forward declaration
template <typename T, typename Src>
class CudaLocalHostBroadcast;

// Specialization for device pointer source
template <typename T>
class CudaLocalHostBroadcast<T, CudaDevicePointer<T> > : public LocalOp<T> {
 public:
  CudaLocalHostBroadcast(
      std::vector<CudaStream>& streams,
      std::vector<CudaDevicePointer<T> >& devicePtrs,
      CudaDevicePointer<T>& sourcePtr,
      size_t offset,
      size_t count)
      : streams_(streams),
        sourcePtr_(sourcePtr.range(offset, count)) {
    // Incorporate offset/count into devicePtrs
    devicePtrs_.reserve(devicePtrs.size());
    for (const auto& ptr : devicePtrs) {
      devicePtrs_.push_back(ptr.range(offset, count));
    }
  }

  virtual void runAsync() {
    // Asynchronously copy source to device ptrs
    for (auto i = 0; i < devicePtrs_.size(); i++) {
      streams_[i].copyAsync(devicePtrs_[i], sourcePtr_);
    }
  }

  virtual void wait() {
    for (auto i = 0; i < devicePtrs_.size(); i++) {
      streams_[i].wait();
    }
  }

 protected:
  std::vector<CudaStream>& streams_;
  std::vector<CudaDevicePointer<T> > devicePtrs_;
  CudaDevicePointer<T> sourcePtr_;
};

// Specialization for host pointer source
template <typename T>
class CudaLocalHostBroadcast<T, CudaHostPointer<T> > : public LocalOp<T> {
 public:
  CudaLocalHostBroadcast(
      std::vector<CudaStream>& streams,
      std::vector<CudaDevicePointer<T> >& devicePtrs,
      CudaHostPointer<T>& sourcePtr,
      size_t offset,
      size_t count)
      : streams_(streams),
        sourcePtr_(sourcePtr.range(offset, count)) {
    // Incorporate offset/count into devicePtrs
    devicePtrs_.reserve(devicePtrs.size());
    for (const auto& ptr : devicePtrs) {
      devicePtrs_.push_back(ptr.range(offset, count));
    }
  }

  virtual void runAsync() {
    // Asynchronously copy host memory to device
    for (auto i = 0; i < devicePtrs_.size(); i++) {
      streams_[i].copyAsync(devicePtrs_[i], sourcePtr_);
    }
  }

  virtual void wait() {
    for (auto i = 0; i < devicePtrs_.size(); i++) {
      streams_[i].wait();
    }
  }

 protected:
  std::vector<CudaStream>& streams_;
  std::vector<CudaDevicePointer<T> > devicePtrs_;
  CudaHostPointer<T> sourcePtr_;
};

template <typename T, typename Dst>
std::unique_ptr<LocalOp<T> > cudaHostReduce(
    std::vector<CudaStream>& streams,
    std::vector<CudaDevicePointer<T> >& devicePtrs,
    Dst& targetPtr,
    const CudaReductionFunction<T>* fn,
    size_t offset,
    size_t count) {
  GLOO_ENFORCE_EQ(streams.size(), devicePtrs.size());
  // Simple copy operation if there is only a single device pointer.
  if (devicePtrs.size() == 1) {
    return make_unique<
      CudaLocalMemcpy<T, CudaDevicePointer<T>, Dst> >(
          streams[0],
          devicePtrs[0],
          targetPtr,
          offset,
          count);
  }
  return make_unique<CudaLocalHostReduce<T, Dst> >(
      streams,
      devicePtrs,
      targetPtr,
      fn,
      offset,
      count);
}

template <typename T, typename Src>
std::unique_ptr<LocalOp<T> > cudaHostBroadcast(
    std::vector<CudaStream>& streams,
    std::vector<CudaDevicePointer<T> >& devicePtrs,
    Src& sourcePtr,
    size_t offset,
    size_t count) {
  GLOO_ENFORCE_EQ(streams.size(), devicePtrs.size());
  // Simple copy operation if there is only a single device pointer.
  if (devicePtrs.size() == 1) {
    return make_unique<
      CudaLocalMemcpy<T, Src, CudaDevicePointer<T> > >(
          streams[0],
          sourcePtr,
          devicePtrs[0],
          offset,
          count);
  }
  return make_unique<CudaLocalHostBroadcast<T, Src> >(
      streams,
      devicePtrs,
      sourcePtr,
      offset,
      count);
}

} // namespace gloo
