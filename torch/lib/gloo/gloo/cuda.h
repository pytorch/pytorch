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
#include "gloo/config.h"
#include "gloo/common/logging.h"

// Check that configuration header was properly generated
#if !GLOO_USE_CUDA
#error "Expected GLOO_USE_CUDA to be defined"
#endif

namespace gloo {

extern const cudaStream_t kStreamNotSet;
extern const int kInvalidDeviceId;

// Forward declarations
template<typename T>
class CudaDevicePointer;
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

class CudaStream {
 public:
  explicit CudaStream(int deviceId, cudaStream_t stream = kStreamNotSet);

  // Move constructor
  CudaStream(CudaStream&& other) noexcept;

  ~CudaStream();

  cudaStream_t operator*() const {
    return stream_;
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

  template <typename T>
  void copyAsync(CudaHostPointer<T>& dst, CudaDevicePointer<T>& src);
  template <typename T>
  void copyAsync(CudaHostPointer<T>& dst, CudaHostPointer<T>& src);
  template <typename T>
  void copyAsync(CudaDevicePointer<T>& dst, CudaDevicePointer<T>& src);
  template <typename T>
  void copyAsync(CudaDevicePointer<T>& dst, CudaHostPointer<T>& src);

  void record();

  void wait();

 protected:
  // Instances cannot be copied or copy-assigned
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  // GPU that the stream belongs to.
  int deviceId_;

  // Operations are always run on a stream such that they can run
  // concurrently with other operations. The stream can be specified
  // at construction time if one has already been created outside this
  // library. If it is not specified, a new stream is created.
  cudaStream_t stream_;
  cudaEvent_t event_;

  // If no stream is specified at construction time, this class
  // allocates a new stream for operations against CUDA pointers.
  // Record whether or not this instance is a stream's owner so that
  // it is destroyed when this instance is destructed.
  bool streamOwner_;
};

template<typename T>
class CudaDevicePointer {
 public:
  static CudaDevicePointer<T> alloc(size_t count);

  static CudaDevicePointer<T> create(T* ptr, size_t count);

  CudaDevicePointer(CudaDevicePointer&&) noexcept;
  ~CudaDevicePointer();

  // Default constructor creates invalid instance
  CudaDevicePointer()
      : device_(nullptr),
        count_(0),
        owner_(false),
        deviceId_(kInvalidDeviceId) {}

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

  // Create range into this pointer
  CudaDevicePointer<T> range(size_t offset, size_t count) const {
    GLOO_ENFORCE_LE(offset + count, count_);
    return CudaDevicePointer<T>(device_ + offset, count, false);
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

  // Create range into this pointer
  CudaHostPointer<T> range(size_t offset, size_t count) const {
    GLOO_ENFORCE_LE(offset + count, count_);
    return CudaHostPointer<T>(host_ + offset, count, false);
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
};

template <typename T, typename Src, typename Dst>
class CudaLocalMemcpy : public LocalOp<T> {
 public:
  CudaLocalMemcpy(
    CudaStream& stream,
    Src& src,
    Dst& dst,
    size_t offset,
    size_t count)
      : stream_(stream),
        src_(src.range(offset, count)),
        dst_(dst.range(offset, count)) {}

  virtual void runAsync() {
    stream_.copyAsync(dst_, src_);
  }

  virtual void wait() {
    stream_.wait();
  }

 protected:
  CudaStream& stream_;
  Src src_;
  Dst dst_;
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
      size_t n,
      CudaStream& stream) const {
    // The specified stream may still have a memcpy in flight to
    // either of the CudaHostPointers. Wait on the stream to make sure
    // they have finished before executing the reduction function.
    stream.wait();
    hostFn_(*dst, *src, n);
  }

  void call(
      CudaDevicePointer<T>& dst,
      const CudaDevicePointer<T>& src,
      size_t n,
      CudaStream& stream) const {
    deviceFn_(*dst, *src, n, *stream);
    stream.record();
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
