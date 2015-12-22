// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#if defined(EIGEN_USE_GPU) && !defined(EIGEN_CXX11_TENSOR_TENSOR_DEVICE_CUDA_H)
#define EIGEN_CXX11_TENSOR_TENSOR_DEVICE_CUDA_H


namespace Eigen {

// This defines an interface that GPUDevice can take to use
// CUDA streams underneath.
class StreamInterface {
 public:
  virtual ~StreamInterface() {}

  virtual const cudaStream_t& stream() const = 0;
  virtual const cudaDeviceProp& deviceProperties() const = 0;

  // Allocate memory on the actual device where the computation will run
  virtual void* allocate(size_t num_bytes) const = 0;
  virtual void deallocate(void* buffer) const = 0;
};

static cudaDeviceProp* m_deviceProperties;
static bool m_devicePropInitialized = false;

static void initializeDeviceProp() {
  if (!m_devicePropInitialized) {
    if (!m_devicePropInitialized) {
      int num_devices;
      cudaError_t status = cudaGetDeviceCount(&num_devices);
      EIGEN_UNUSED_VARIABLE(status)
      assert(status == cudaSuccess);
      m_deviceProperties = new cudaDeviceProp[num_devices];
      for (int i = 0; i < num_devices; ++i) {
        status = cudaGetDeviceProperties(&m_deviceProperties[i], i);
        assert(status == cudaSuccess);
      }
      m_devicePropInitialized = true;
    }
  }
}

static const cudaStream_t default_stream = cudaStreamDefault;

class CudaStreamDevice : public StreamInterface {
 public:
  // Use the default stream on the current device
  CudaStreamDevice() : stream_(&default_stream) {
    cudaGetDevice(&device_);
    initializeDeviceProp();
  }
  // Use the default stream on the specified device
  CudaStreamDevice(int device) : stream_(&default_stream), device_(device) {
    initializeDeviceProp();
  }
  // Use the specified stream. Note that it's the
  // caller responsibility to ensure that the stream can run on
  // the specified device. If no device is specified the code
  // assumes that the stream is associated to the current gpu device.
  CudaStreamDevice(const cudaStream_t* stream, int device = -1)
      : stream_(stream), device_(device) {
    if (device < 0) {
      cudaGetDevice(&device_);
    } else {
      int num_devices;
      cudaError_t err = cudaGetDeviceCount(&num_devices);
      EIGEN_UNUSED_VARIABLE(err)
      assert(err == cudaSuccess);
      assert(device < num_devices);
      device_ = device;
    }
    initializeDeviceProp();
  }

  const cudaStream_t& stream() const { return *stream_; }
  const cudaDeviceProp& deviceProperties() const {
    return m_deviceProperties[device_];
  }
  virtual void* allocate(size_t num_bytes) const {
    cudaError_t err = cudaSetDevice(device_);
    EIGEN_UNUSED_VARIABLE(err)
    assert(err == cudaSuccess);
    void* result;
    err = cudaMalloc(&result, num_bytes);
    assert(err == cudaSuccess);
    assert(result != NULL);
    return result;
  }
  virtual void deallocate(void* buffer) const {
    cudaError_t err = cudaSetDevice(device_);
    EIGEN_UNUSED_VARIABLE(err)
    assert(err == cudaSuccess);
    assert(buffer != NULL);
    err = cudaFree(buffer);
    assert(err == cudaSuccess);
  }

 private:
  const cudaStream_t* stream_;
  int device_;
};

struct GpuDevice {
  // The StreamInterface is not owned: the caller is
  // responsible for its initialization and eventual destruction.
  explicit GpuDevice(const StreamInterface* stream) : stream_(stream) {
    eigen_assert(stream);
  }

  // TODO(bsteiner): This is an internal API, we should not expose it.
  EIGEN_STRONG_INLINE const cudaStream_t& stream() const {
    return stream_->stream();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void* allocate(size_t num_bytes) const {
#ifndef __CUDA_ARCH__
    return stream_->allocate(num_bytes);
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
    return NULL;
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void deallocate(void* buffer) const {
#ifndef __CUDA_ARCH__
    stream_->deallocate(buffer);

#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpy(void* dst, const void* src, size_t n) const {
#ifndef __CUDA_ARCH__
    cudaError_t err = cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice,
                                      stream_->stream());
    EIGEN_UNUSED_VARIABLE(err)
    assert(err == cudaSuccess);
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpyHostToDevice(void* dst, const void* src, size_t n) const {
#ifndef __CUDA_ARCH__
    cudaError_t err =
        cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, stream_->stream());
    EIGEN_UNUSED_VARIABLE(err)
    assert(err == cudaSuccess);
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpyDeviceToHost(void* dst, const void* src, size_t n) const {
#ifndef __CUDA_ARCH__
    cudaError_t err =
        cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, stream_->stream());
    EIGEN_UNUSED_VARIABLE(err)
    assert(err == cudaSuccess);
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memset(void* buffer, int c, size_t n) const {
#ifndef __CUDA_ARCH__
    cudaError_t err = cudaMemsetAsync(buffer, c, n, stream_->stream());
    EIGEN_UNUSED_VARIABLE(err)
    assert(err == cudaSuccess);
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t numThreads() const {
    // FIXME
    return 32;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t firstLevelCacheSize() const {
    // FIXME
    return 48*1024;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t lastLevelCacheSize() const {
    // We won't try to take advantage of the l2 cache for the time being, and
    // there is no l3 cache on cuda devices.
    return firstLevelCacheSize();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void synchronize() const {
#if defined(__CUDACC__) && !defined(__CUDA_ARCH__)
    cudaError_t err = cudaStreamSynchronize(stream_->stream());
    EIGEN_UNUSED_VARIABLE(err)
    assert(err == cudaSuccess);
#else
    assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  inline int getNumCudaMultiProcessors() const {
    return stream_->deviceProperties().multiProcessorCount;
  }
  inline int maxCudaThreadsPerBlock() const {
    return stream_->deviceProperties().maxThreadsPerBlock;
  }
  inline int maxCudaThreadsPerMultiProcessor() const {
    return stream_->deviceProperties().maxThreadsPerMultiProcessor;
  }
  inline int sharedMemPerBlock() const {
    return stream_->deviceProperties().sharedMemPerBlock;
  }
  inline int majorDeviceVersion() const {
    return stream_->deviceProperties().major;
  }

  // This function checks if the CUDA runtime recorded an error for the
  // underlying stream device.
  inline bool ok() const {
#ifdef __CUDACC__
    cudaError_t error = cudaStreamQuery(stream_->stream());
    return (error == cudaSuccess) || (error == cudaErrorNotReady);
#else
    return false;
#endif
  }

 private:
  const StreamInterface* stream_;

};


#define LAUNCH_CUDA_KERNEL(kernel, gridsize, blocksize, sharedmem, device, ...)            \
  (kernel) <<< (gridsize), (blocksize), (sharedmem), (device).stream() >>> (__VA_ARGS__);  \
  assert(cudaGetLastError() == cudaSuccess);


// FIXME: Should be device and kernel specific.
#ifdef __CUDACC__
static inline void setCudaSharedMemConfig(cudaSharedMemConfig config) {
  cudaError_t status = cudaDeviceSetSharedMemConfig(config);
  EIGEN_UNUSED_VARIABLE(status)
  assert(status == cudaSuccess);
}
#endif

}  // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_TYPE_H
