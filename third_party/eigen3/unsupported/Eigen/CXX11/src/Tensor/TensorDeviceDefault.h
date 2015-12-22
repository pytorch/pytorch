// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_DEVICE_DEFAULT_H
#define EIGEN_CXX11_TENSOR_TENSOR_DEVICE_DEFAULT_H


namespace Eigen {

// Default device for the machine (typically a single cpu core)
struct DefaultDevice {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void* allocate(size_t num_bytes) const {
    return internal::aligned_malloc(num_bytes);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void deallocate(void* buffer) const {
    internal::aligned_free(buffer);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpy(void* dst, const void* src, size_t n) const {
    ::memcpy(dst, src, n);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpyHostToDevice(void* dst, const void* src, size_t n) const {
    memcpy(dst, src, n);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpyDeviceToHost(void* dst, const void* src, size_t n) const {
    memcpy(dst, src, n);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memset(void* buffer, int c, size_t n) const {
    ::memset(buffer, c, n);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t numThreads() const {
#ifndef __CUDA_ARCH__
    // Running on the host CPU
    return 1;
#else
    // Running on a CUDA device
    return 32;
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int majorDeviceVersion() const {
#ifndef __CUDA_ARCH__
    // Running single threaded on the host CPU
    // Should return an enum that encodes the ISA supported by the CPU
    return 1;
#else
    // Running on a CUDA device
    return __CUDA_ARCH__ / 100;
#endif
  }
};

}  // namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_DEFAULT_H
