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

#include "gloo/algorithm.h"
#include "gloo/common/logging.h"

namespace gloo {

extern const cudaStream_t kStreamNotSet;

// Forward declarations
template <typename T>
class CudaHostPointer;
template<typename T>
class CudaReductionFunction;

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
  static CudaDevicePointer<T> alloc(
    size_t count,
    cudaStream_t stream);

  static CudaDevicePointer<T> create(
    T* ptr,
    size_t count,
    cudaStream_t stream = kStreamNotSet);

  CudaDevicePointer(CudaDevicePointer&&) noexcept;
  ~CudaDevicePointer();

  // Default constructor creates invalid instance
  CudaDevicePointer()
      : device_(nullptr),
        count_(0),
        owner_(false),
        deviceId_(-1) {}

  // Move assignment operator
  CudaDevicePointer& operator=(CudaDevicePointer&&);

  bool operator ==(const CudaDevicePointer<T>& other) const {
    return device_ == other.device_ && count_ == other.count_;
  }

  T* operator*() const {
    return device_;
  }

  T& operator[](size_t index) const {
    return device_[index];
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

  // Copy between pointers.
  void copyToAsync(CudaHostPointer<T>& dst);
  void copyFromAsync(CudaHostPointer<T>& src);
  void copyToAsync(CudaDevicePointer<T>& dst);
  void copyFromAsync(CudaDevicePointer<T>& src);

  // Wait for copy to complete.
  void wait();

  // Call reduction function against this pointer
  void reduceAsync(
      const CudaReductionFunction<T>* fn,
      CudaDevicePointer<T>& src);

  // Create range into this pointer
  CudaDevicePointer<T> range(size_t offset, size_t count) {
    GLOO_ENFORCE_LE(offset + count, count_);
    CudaDevicePointer<T> p(device_ + offset, count, false);
    p.stream_ = stream_;
    p.streamOwner_ = false;
    return p;
  }

 protected:
  // Instances must be created through static functions
  CudaDevicePointer(T* ptr, size_t count, bool owner);

  // Instances cannot be copied or copy-assigned
  CudaDevicePointer(const CudaDevicePointer&) = delete;
  CudaDevicePointer& operator=(const CudaDevicePointer&) = delete;

  // Device pointer
  T* device_;

  // Number of T elements in device pointer
  size_t count_;

  // Record whether or not this instance is this pointer's owner so
  // that it is freed when this instance is destructed.
  bool owner_ = false;

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
class CudaHostPointer {
 public:
  static CudaHostPointer<T> alloc(size_t count);

  CudaHostPointer(CudaHostPointer&&) noexcept;
  ~CudaHostPointer();

  // Default constructor creates invalid instance
  CudaHostPointer() : CudaHostPointer(nullptr, 0, false) {}

  // Move assignment operator
  CudaHostPointer& operator=(CudaHostPointer&&);

  bool operator ==(const CudaHostPointer<T>& other) const {
    return host_ == other.host_ && count_ == other.count_;
  }

  T* operator*() const {
    return host_;
  }

  T& operator[](size_t index) const {
    return host_[index];
  }

  int getCount() const {
    return count_;
  }

  // Copy between pointers.
  void copyToAsync(CudaHostPointer<T>& dst);
  void copyFromAsync(CudaHostPointer<T>& src);
  void copyToAsync(CudaDevicePointer<T>& dst);
  void copyFromAsync(CudaDevicePointer<T>& src);

  // Wait for copy to complete.
  void wait();

  // Call reduction function against this pointer
  void reduceAsync(
      const CudaReductionFunction<T>* fn,
      CudaHostPointer<T>& src);

  // Create range into this pointer
  CudaHostPointer<T> range(size_t offset, size_t count) {
    GLOO_ENFORCE_LE(offset + count, count_);
    CudaHostPointer<T> p(host_ + offset, count, false);
    return p;
  }

 protected:
  // Instances must be created through static functions
  CudaHostPointer(T* ptr, size_t count, bool owner);

  // Instances cannot be copied or copy-assigned
  CudaHostPointer(const CudaHostPointer&) = delete;
  CudaHostPointer& operator=(const CudaHostPointer&) = delete;

  // Host pointer
  T* host_;

  // Number of T elements in host pointer
  size_t count_;

  // Record whether or not this instance is this pointer's owner so
  // that it is freed when this instance is destructed.
  bool owner_ = false;

  // References to stream/event of most recent async copy;
  int deviceId_;
  cudaStream_t stream_;
  cudaEvent_t event_;
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
class CudaReductionFunction {
  using DeviceFunction =
    void(T*, const T*, size_t n, const cudaStream_t stream);
  using HostFunction =
    void(T*, const T*, size_t n);

 public:
  static const CudaReductionFunction<T>* sum;
  static const CudaReductionFunction<T>* product;
  static const CudaReductionFunction<T>* min;
  static const CudaReductionFunction<T>* max;

  CudaReductionFunction(
    ReductionType type,
    DeviceFunction* deviceFn,
    HostFunction* hostFn)
      : type_(type),
        deviceFn_(deviceFn),
        hostFn_(hostFn) {}

  ReductionType type() const {
    return type_;
  }

  // Backwards compatibility.
  // Can be removed when all CUDA algorithms use CudaHostPointer.
  void call(T* x, const T* y, size_t n) const {
    hostFn_(x, y, n);
  }

  void call(
      CudaHostPointer<T>& dst,
      const CudaHostPointer<T>& src,
      size_t n) const {
    hostFn_(*dst, *src, n);
  }

  void call(
      CudaDevicePointer<T>& dst,
      const CudaDevicePointer<T>& src,
      size_t n) const {
    deviceFn_(*dst, *src, n, dst.getStream());
  }

 protected:
  const ReductionType type_;
  DeviceFunction* deviceFn_;
  HostFunction* hostFn_;

  friend class CudaDevicePointer<T>;
  friend class CudaHostPointer<T>;
};

template <typename T>
const CudaReductionFunction<T>* CudaReductionFunction<T>::sum =
  new CudaReductionFunction<T>(
    SUM, &::gloo::cudaSum<T>, &::gloo::sum<T>);
template <typename T>
const CudaReductionFunction<T>* CudaReductionFunction<T>::product =
  new CudaReductionFunction<T>(
    PRODUCT, &::gloo::cudaProduct<T>, &::gloo::product<T>);
template <typename T>
const CudaReductionFunction<T>* CudaReductionFunction<T>::min =
  new CudaReductionFunction<T>(
    MIN, &::gloo::cudaMin<T>, &::gloo::min<T>);
template <typename T>
const CudaReductionFunction<T>* CudaReductionFunction<T>::max =
  new CudaReductionFunction<T>(
    MAX, &::gloo::cudaMax<T>, &::gloo::max<T>);

} // namespace gloo
