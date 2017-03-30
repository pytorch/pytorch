/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <atomic>
#include <mutex>

#include <cuda.h>
#include <cuda_runtime.h>

#include "gloo/common.h"

namespace gloo {

extern const cudaStream_t kStreamNotSet;

class CudaShared {
 public:
  // Get the mutex used to synchronize CUDA and NCCL operations
  static std::mutex& getMutex() {
    return *mutex_;
  }

  // Set the mutex used to synchronize CUDA and NCCL operations
  static void setMutex(std::mutex* m) {
    mutex_ = m;
  }

 private:
  static std::atomic<std::mutex*> mutex_;
};

template<typename T>
class CudaDevicePointer {
 public:
  static CudaDevicePointer create(
    T* ptr,
    size_t count,
    cudaStream_t stream = kStreamNotSet);

  CudaDevicePointer(CudaDevicePointer&&) noexcept;
  ~CudaDevicePointer();

  // Default constructor creates invalid instance
  CudaDevicePointer() : count_(0), deviceId_(-1) {}

  // Move assignment operator
  CudaDevicePointer& operator=(CudaDevicePointer&&);

  T* operator*() const {
    return device_;
  }

  int getCount() const {
    return count_;
  }

  int getDeviceID() const {
    return deviceId_;
  }

  cudaStream_t getStream() const {
    return stream_;
  }

  cudaEvent_t getEvent() const {
    return event_;
  }

  // Copy contents of device pointer to host.
  void copyToHostAsync(T* dst);

  // Copy contents of host pointer to device.
  void copyFromHostAsync(T* src);

  // Copy contents of device pointer to other device pointer.
  void copyToDeviceAsync(T* dst);

  // Copy contents of device pointer to other device pointer.
  void copyFromDeviceAsync(T* src);

  // Wait for copy to complete.
  void wait();

 protected:
  // Instances must be created through static functions
  CudaDevicePointer(T* ptr, size_t count);

  // Instances cannot be copied or copy-assigned
  CudaDevicePointer(const CudaDevicePointer&) = delete;
  CudaDevicePointer& operator=(const CudaDevicePointer&) = delete;

  // Device pointer
  T* device_;

  // Number of T elements in device pointer
  size_t count_;

  // GPU that the device pointer lives on
  int deviceId_;

  // Operations on this pointer are run always run on a stream such
  // that they don't block other operations being executed on the GPU
  // this pointer lives on. The stream can be specified at
  // construction time if one has already been created outside this
  // library. If it is not specified, a new stream is created.
  cudaStream_t stream_ = kStreamNotSet;
  cudaEvent_t event_ = 0;

  // If no stream is specified at construction time, this class
  // allocates a new stream for operations against this pointer.
  // Record whether or not this instance is a stream's owner so that
  // it is destroyed when this instance is destructed.
  bool streamOwner_ = false;
};

template <typename T>
void cudaSum(T* x, const T* y, size_t n, const cudaStream_t stream);

template <typename T>
void cudaProduct(T* x, const T* y, size_t n, const cudaStream_t stream);

template <typename T>
void cudaMax(T* x, const T* y, size_t n, const cudaStream_t stream);

template <typename T>
void cudaMin(T* x, const T* y, size_t n, const cudaStream_t stream);

template <typename T>
class CudaReductionFunction : public ReductionFunction<T> {
  using Fn = void(T*, const T*, size_t n, const cudaStream_t stream);

 public:
  static const CudaReductionFunction<T>* sum;
  static const CudaReductionFunction<T>* product;
  static const CudaReductionFunction<T>* min;
  static const CudaReductionFunction<T>* max;

  static const CudaReductionFunction<T>* toCudaReductionFunction(
      const ReductionFunction<T>* fn) {
    switch (fn->type()) {
      case SUM:
        return sum;
      case PRODUCT:
        return product;
      case MIN:
        return min;
      case MAX:
        return max;
      default:
        return nullptr;
    }
  }

  CudaReductionFunction(ReductionType type, Fn* fn)
      : type_(type), fn_(fn) {}

  virtual ReductionType type() const override {
    return type_;
  }

  virtual void call(T* x, const T* y, size_t n) const override {
    fn_(x, y, n, 0);
  }

  virtual void callAsync(
      T* x,
      const T* y,
      size_t n,
      const cudaStream_t stream) const {
    fn_(x, y, n, stream);
  }

 protected:
  ReductionType type_;
  Fn* fn_;
};

template <typename T>
const CudaReductionFunction<T>* CudaReductionFunction<T>::sum =
  new CudaReductionFunction<T>(SUM, &::gloo::cudaSum<T>);
template <typename T>
const CudaReductionFunction<T>* CudaReductionFunction<T>::product =
  new CudaReductionFunction<T>(PRODUCT, &::gloo::cudaProduct<T>);
template <typename T>
const CudaReductionFunction<T>* CudaReductionFunction<T>::min =
  new CudaReductionFunction<T>(MIN, &::gloo::cudaMin<T>);
template <typename T>
const CudaReductionFunction<T>* CudaReductionFunction<T>::max =
  new CudaReductionFunction<T>(MAX, &::gloo::cudaMax<T>);

} // namespace gloo
