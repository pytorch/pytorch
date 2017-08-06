// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

#import <Metal/Metal.h>

namespace caffe2 {

struct MetalAllocator final : CPUAllocator {
  id<MTLDevice> device;

  MetalAllocator(id<MTLDevice> _device);

  ~MetalAllocator();

  void *New(size_t nbytes) override;

  void Delete(void *data) override;

  id<MTLBuffer> Buffer(void *data);
};

MetalAllocator *GetMetalAllocator();

class MetalCaffeContext final {
 public:
  MetalCaffeContext() : random_seed_(math::randomNumberSeed()) {}
  explicit MetalCaffeContext(const DeviceOption &option)
      : random_seed_(option.has_random_seed() ? option.random_seed() : math::randomNumberSeed()) {
    CHECK_EQ(option.device_type(), CPU);
  }

  ~MetalCaffeContext() {}

  inline void SwitchToDevice(int stream_id) {}
  inline void SwitchToDevice() {
    SwitchToDevice(0);
  }

  inline bool FinishDeviceComputation() { return true; }

  inline std::mt19937 &RandGenerator() {
    if (!random_generator_.get()) {
      random_generator_.reset(new std::mt19937(random_seed_));
    }
    return *random_generator_.get();
  }

  static void *New(size_t nbytes);

  static void Delete(void *data);

  // Two copy functions that deals with cross-device copies.
  template <class SrcContext, class DstContext>
  inline void CopyBytes(size_t nbytes, const void *src, void *dst);

  template <typename T, class SrcContext, class DstContext>
  inline void Copy(size_t n, const T *src, T *dst) {
    if (std::is_fundamental<T>::value) {
      CopyBytes<SrcContext, DstContext>(n * sizeof(T), static_cast<const void *>(src), static_cast<void *>(dst));
    } else {
      for (int i = 0; i < n; ++i) {
        dst[i] = src[i];
      }
    }
  }

  template <class SrcContext, class DstContext>
  inline void CopyItems(const TypeMeta &meta, size_t n, const void *src, void *dst) {
    if (meta.copy()) {
      meta.copy()(src, dst, n);
    } else {
      CopyBytes<SrcContext, DstContext>(n * meta.itemsize(), src, dst);
    }
  }

 protected:
  int random_seed_{1701};
  std::unique_ptr<std::mt19937> random_generator_;
};

typedef Tensor<MetalCaffeContext> TensorMetal;
}
