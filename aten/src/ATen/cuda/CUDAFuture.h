#pragma once

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include <ATen/core/ivalue.h>
#include <ATen/core/ivalue_inl.h>
#include <ATen/core/jit_type.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAMultiStreamGuard.h>
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Export.h>
#include <c10/util/intrusive_ptr.h>

namespace at { namespace cuda {

struct TORCH_CUDA_API CUDAFuture : at::ivalue::Future {
 public:
  using at::ivalue::Future::Future;

  void setDataPtrExtractor(DataPtrExtractor data_ptr_extractor) override {
    dataPtrExtractor_ = std::move(data_ptr_extractor);
  }

 protected:
  c10::intrusive_ptr<Future> createInstance(at::TypePtr type) override {
    return c10::make_intrusive<CUDAFuture>(std::move(type));
  }

  void postMarkCompletedHook(const at::IValue& value) override {
    std::vector<bool> isCudaDeviceUsed(c10::cuda::device_count(), false);
    for (const at::DataPtr& data_ptr : extractDataPtrs(value)) {
      if (data_ptr.device().is_cuda()) {
        isCudaDeviceUsed[data_ptr.device().index()] = true;
      }
    }

    cudaEvents_ = std::make_shared<std::vector<at::cuda::CUDAEvent>>();
    for (c10::DeviceIndex idx = 0; idx < isCudaDeviceUsed.size(); idx++) {
      if (isCudaDeviceUsed[idx]) {
        at::cuda::CUDAEvent cudaEvent;
        cudaEvent.record(at::cuda::getDefaultCUDAStream(idx));
        (*cudaEvents_).push_back(std::move(cudaEvent));
      }
    }
  }

  std::function<void(void)> wrapCallback(
      std::function<void(void)> callback) override {
    return [this, callback{std::move(callback)}]() {
      // Get a stream for all devices, even those that are not used by the
      // value, because the user's callback could use those other devices.
      std::vector<at::cuda::CUDAStream> streams;
      for (c10::DeviceIndex idx = 0; idx < c10::cuda::device_count(); idx++) {
        // FIXME Should we find a way to allow to change the priority of
        // streams?
        streams.push_back(
            at::cuda::getStreamFromPool(/*isHighPriority=*/false, idx));
      }

      // Do not free the underlying data storage of value_ before its
      // usage on the stream finishes.
      for (const at::DataPtr& data_ptr : extractDataPtrs(constValue())) {
        if (data_ptr.device().is_cuda()) {
          c10::cuda::CUDACachingAllocator::recordStream(
              data_ptr, streams[data_ptr.device().index()]);
        }
      }

      for (at::cuda::CUDAEvent& cudaEvent : *cudaEvents_) {
        cudaEvent.block(streams[cudaEvent.device_index()]);
      }

      // Use the dedicated callback stream to run callback.
      at::cuda::CUDAMultiStreamGuard streamGuard(streams);

      callback();
    };
  }

  void postWaitHook() override {
    for (at::cuda::CUDAEvent& cudaEvent : *cudaEvents_) {
      cudaEvent.block(at::cuda::getCurrentCUDAStream(cudaEvent.device_index()));
    }
  }

  // FIXME This field is protected (rather than private) and wrapped in a
  // shared_ptr in order to support the FutureNCCL subclass, which wants to set
  // the events on its own in order to use the same ones as its WorkNCCL class.
  // Once WorkNCCL is gone (as part of the Future and Work merge) this should be
  // fixed.
 protected:
  std::shared_ptr<std::vector<at::cuda::CUDAEvent>> cudaEvents_;

 private:
  DataPtrExtractor dataPtrExtractor_;

  std::vector<std::reference_wrapper<const at::DataPtr>> extractDataPtrs(
      const at::IValue& value) {
    std::vector<std::reference_wrapper<const at::DataPtr>> data_ptrs;
    if (dataPtrExtractor_ != nullptr) {
      // If a Python communication hook is used, dataPtrExtractor_ will be
      // set in torch/csrc/jit/python/pybind_utils.h, which allows Python
      // dependency to be imported.
      data_ptrs = dataPtrExtractor_(value);
    } else {
      // If a C++ communication hook is used, use the default extractor.
      data_ptrs = at::ivalue::Future::defaultDataPtrExtractor(value);
    }
    return data_ptrs;
  }
};

} // namespace cuda
} // namespace at
