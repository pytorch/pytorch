/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <mutex>

#include "hip/hip_runtime.h"

#include "gloo/algorithm.h"
#include "gloo/config.h"
#include "gloo/common/logging.h"

// Check that configuration header was properly generated
#if !GLOO_USE_HIP
//#error "Expected GLOO_USE_HIP to be defined"
#endif

namespace gloo {

extern const hipStream_t kStreamNotSet;
extern const int kInvalidDeviceId;

// Forward declarations
template<typename T>
class HipDevicePointer;
template <typename T>
class HipHostPointer;
template<typename T>
class HipReductionFunction;

class HipShared {
 public:
  // Get the mutex used to synchronize HIP and NCCL operations
  static std::mutex& getMutex() {
    return *mutex_;
  }

  // Set the mutex used to synchronize HIP and NCCL operations
  static void setMutex(std::mutex* m) {
    mutex_ = m;
  }

 private:
  static std::atomic<std::mutex*> mutex_;
};

class HipStream {
 public:
  explicit HipStream(int deviceId, hipStream_t stream = kStreamNotSet);

  // Move constructor
  HipStream(HipStream&& other) noexcept;

  ~HipStream();

  hipStream_t operator*() const {
    return stream_;
  }

  int getDeviceID() const {
    return deviceId_;
  }

  hipStream_t getStream() const {
    return stream_;
  }

  hipEvent_t getEvent() const {
    return event_;
  }

  template <typename T>
  void copyAsync(HipHostPointer<T>& dst, HipDevicePointer<T>& src);
  template <typename T>
  void copyAsync(HipHostPointer<T>& dst, HipHostPointer<T>& src);
  template <typename T>
  void copyAsync(HipDevicePointer<T>& dst, HipDevicePointer<T>& src);
  template <typename T>
  void copyAsync(HipDevicePointer<T>& dst, HipHostPointer<T>& src);

  void record();

  void wait();

 protected:
  // Instances cannot be copied or copy-assigned
  HipStream(const HipStream&) = delete;
  HipStream& operator=(const HipStream&) = delete;

  // GPU that the stream belongs to.
  int deviceId_;

  // Operations are always run on a stream such that they can run
  // concurrently with other operations. The stream can be specified
  // at construction time if one has already been created outside this
  // library. If it is not specified, a new stream is created.
  hipStream_t stream_;
  hipEvent_t event_;

  // If no stream is specified at construction time, this class
  // allocates a new stream for operations against HIP pointers.
  // Record whether or not this instance is a stream's owner so that
  // it is destroyed when this instance is destructed.
  bool streamOwner_;
};

template<typename T>
class BuilderHelpers {
  public:
    // Checks if all the pointers are GPU pointers.
    static bool checkAllPointersGPU(std::vector<T*> inputs){
      return std::all_of(inputs.begin(), inputs.end(), [](const T* ptr) {
        hipPointerAttribute_t attr;
        auto rv = hipPointerGetAttributes(&attr, ptr);
        return rv == hipSuccess && attr.memoryType == hipMemoryTypeDevice;
      });
    }
};

template<typename T>
class HipDevicePointer {
 public:
  static HipDevicePointer<T> alloc(size_t count);

  static HipDevicePointer<T> create(T* ptr, size_t count);

  static HipDevicePointer<T> create(const HipDevicePointer<T>& ptr) {
    return HipDevicePointer<T>::create(*ptr, ptr.getCount());
  }

  HipDevicePointer(HipDevicePointer&&) noexcept;
  ~HipDevicePointer();

  // Default constructor creates invalid instance
  HipDevicePointer()
      : device_(nullptr),
        count_(0),
        owner_(false),
        deviceId_(kInvalidDeviceId) {}

  // Move assignment operator
  HipDevicePointer& operator=(HipDevicePointer&&);

  bool operator ==(const HipDevicePointer<T>& other) const {
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
  HipDevicePointer<T> range(size_t offset, size_t count) const {
    GLOO_ENFORCE_LE(offset + count, count_);
    return HipDevicePointer<T>(device_ + offset, count, false);
  }

 protected:
  // Instances must be created through static functions
  HipDevicePointer(T* ptr, size_t count, bool owner);

  // Instances cannot be copied or copy-assigned
  HipDevicePointer(const HipDevicePointer&) = delete;
  HipDevicePointer& operator=(const HipDevicePointer&) = delete;

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
class HipHostPointer {
 public:
  static HipHostPointer<T> alloc(size_t count);

  HipHostPointer(HipHostPointer&&) noexcept;
  ~HipHostPointer();

  // Default constructor creates invalid instance
  HipHostPointer() : HipHostPointer(nullptr, 0, false) {}

  // Move assignment operator
  HipHostPointer& operator=(HipHostPointer&&);

  bool operator ==(const HipHostPointer<T>& other) const {
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
  HipHostPointer<T> range(size_t offset, size_t count) const {
    GLOO_ENFORCE_LE(offset + count, count_);
    return HipHostPointer<T>(host_ + offset, count, false);
  }

 protected:
  // Instances must be created through static functions
  HipHostPointer(T* ptr, size_t count, bool owner);

  // Instances cannot be copied or copy-assigned
  HipHostPointer(const HipHostPointer&) = delete;
  HipHostPointer& operator=(const HipHostPointer&) = delete;

  // Host pointer
  T* host_;

  // Number of T elements in host pointer
  size_t count_;

  // Record whether or not this instance is this pointer's owner so
  // that it is freed when this instance is destructed.
  bool owner_ = false;
};

template <typename T, typename Src, typename Dst>
class HipLocalMemcpy : public LocalOp<T> {
 public:
  HipLocalMemcpy(
    HipStream& stream,
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
  HipStream& stream_;
  Src src_;
  Dst dst_;
};

template <typename T>
void hipSum(T* x, const T* y, size_t n, const hipStream_t stream);

template <typename T>
void hipProduct(T* x, const T* y, size_t n, const hipStream_t stream);

template <typename T>
void hipMax(T* x, const T* y, size_t n, const hipStream_t stream);

template <typename T>
void hipMin(T* x, const T* y, size_t n, const hipStream_t stream);

template <typename T>
class HipReductionFunction {
  using DeviceFunction =
    void(T*, const T*, size_t n, const hipStream_t stream);
  using HostFunction =
    void(T*, const T*, size_t n);

 public:
  static const HipReductionFunction<T>* sum;
  static const HipReductionFunction<T>* product;
  static const HipReductionFunction<T>* min;
  static const HipReductionFunction<T>* max;

  HipReductionFunction(
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
  // Can be removed when all HIP algorithms use HipHostPointer.
  void call(T* x, const T* y, size_t n) const {
    hostFn_(x, y, n);
  }

  void call(
      HipHostPointer<T>& dst,
      const HipHostPointer<T>& src,
      size_t n,
      HipStream& stream) const {
    // The specified stream may still have a memcpy in flight to
    // either of the HipHostPointers. Wait on the stream to make sure
    // they have finished before executing the reduction function.
    stream.wait();
    hostFn_(*dst, *src, n);
  }

  void call(
      HipDevicePointer<T>& dst,
      const HipDevicePointer<T>& src,
      size_t n,
      HipStream& stream) const {
    deviceFn_(*dst, *src, n, *stream);
    stream.record();
  }

 protected:
  const ReductionType type_;
  DeviceFunction* deviceFn_;
  HostFunction* hostFn_;

  friend class HipDevicePointer<T>;
  friend class HipHostPointer<T>;
};

template <typename T>
const HipReductionFunction<T>* HipReductionFunction<T>::sum =
  new HipReductionFunction<T>(
    SUM, &::gloo::hipSum<T>, &::gloo::sum<T>);
template <typename T>
const HipReductionFunction<T>* HipReductionFunction<T>::product =
  new HipReductionFunction<T>(
    PRODUCT, &::gloo::hipProduct<T>, &::gloo::product<T>);
template <typename T>
const HipReductionFunction<T>* HipReductionFunction<T>::min =
  new HipReductionFunction<T>(
    MIN, &::gloo::hipMin<T>, &::gloo::min<T>);
template <typename T>
const HipReductionFunction<T>* HipReductionFunction<T>::max =
  new HipReductionFunction<T>(
    MAX, &::gloo::hipMax<T>, &::gloo::max<T>);

} // namespace gloo
