#pragma once

#include <functional>
#include <memory>
#include <mutex>
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

namespace at {
namespace cuda {

struct TORCH_CUDA_CPP_API CUDAFuture : at::ivalue::Future {
 public:
  CUDAFuture(at::TypePtr type) : at::ivalue::Future(std::move(type)) {
    // Use current device to initialize currentDevice_. This is necessary
    // because preMarkCompletedHook won't be called when the Future contains
    // an error. Uninitialized currentDevice_ could lead to crash when used
    // in CUDAGuard.
    currentDevice_ = c10::cuda::current_device();
  }

  c10::intrusive_ptr<Future> createInstance(at::TypePtr type) override {
    return c10::make_intrusive<CUDAFuture>(std::move(type));
  }

 protected:
  /**
   * The dataPtrs field contains storage pointers of all tensors in the IValue.
   * This method records CUDAEvents on participating devices and uses those
   * CUDAEvents to synchronize streams when calling postWaitHook().
   * If dataPtrs does not have a value, this method will try to inspect the
   * given IValue by walking through all subvalues and extracting data pointers
   * from CUDA tensors.
   */
  void preMarkCompletedHook(
      const at::IValue& value,
      c10::optional<std::vector<std::reference_wrapper<const at::DataPtr>>>
          dataPtrs) override {
    currentDevice_ = c10::cuda::current_device();

    // Extract them once and cache them for later uses.
    dataPtrs_ =
        dataPtrs.has_value() ? std::move(*dataPtrs) : extractDataPtrs(value);

    std::vector<bool> isCudaDeviceUsed(c10::cuda::device_count(), false);
    for (const at::DataPtr& data_ptr : dataPtrs_) {
      if (data_ptr.device().is_cuda()) {
        isCudaDeviceUsed[data_ptr.device().index()] = true;
      }
    }

    for (c10::DeviceIndex idx = 0; idx < isCudaDeviceUsed.size(); idx++) {
      if (isCudaDeviceUsed[idx]) {
        at::cuda::CUDAEvent cudaEvent;
        cudaEvent.record(at::cuda::getCurrentCUDAStream(idx));
        cudaEvents_.push_back(std::move(cudaEvent));
      }
    }
  }

  std::function<void(void)> wrapCallback(
      std::function<void(void)> callback) override {
    return [this, callback{std::move(callback)}]() {
      // We'd love to get a stream for all devices, even those that are not used
      // by the value, because the callback could use those other devices, but
      // unfortunately this could cause a deadlock with NCCL. See
      // https://github.com/pytorch/pytorch/pull/48500#issuecomment-735395414
      // In general, if some devices haven't been used yet, by getting a stream
      // for them we'd initialize them, and in addition to causing NCCL to
      // misbehaving this also ends up using memory on those devices, which the
      // user might not want.
      std::vector<at::cuda::CUDAStream> streams;
      for (at::cuda::CUDAEvent& cudaEvent : cudaEvents_) {
        c10::DeviceIndex idx = cudaEvent.device_index();
        // FIXME Should we find a way to allow to change the priority of
        // streams?
        at::cuda::CUDAStream stream =
            at::cuda::getStreamFromPool(/*isHighPriority=*/false, idx);
        cudaEvent.block(stream);
        streams.push_back(stream);
      }

      // Use the dedicated callback stream to run callback.
      at::cuda::CUDAMultiStreamGuard streamGuard(streams);

      // Do not free the underlying data storage of value_ before its
      // usage on the stream finishes.
      for (const at::DataPtr& data_ptr : dataPtrs_) {
        if (data_ptr.device().is_cuda()) {
          c10::cuda::CUDACachingAllocator::recordStream(
              data_ptr,
              at::cuda::getCurrentCUDAStream(data_ptr.device().index()));
        }
      }

      c10::cuda::CUDAGuard deviceGuard(currentDevice_);

      callback();
    };
  }

  void postWaitHook(const at::IValue& value) override {
    for (at::cuda::CUDAEvent& cudaEvent : cudaEvents_) {
      cudaEvent.block(at::cuda::getCurrentCUDAStream(cudaEvent.device_index()));
    }

    for (const at::DataPtr& data_ptr : dataPtrs_) {
      if (data_ptr.device().is_cuda()) {
        c10::cuda::CUDACachingAllocator::recordStream(
            data_ptr,
            at::cuda::getCurrentCUDAStream(data_ptr.device().index()));
      }
    }
  }

  virtual std::vector<std::reference_wrapper<const at::DataPtr>> extractDataPtrs(
      const at::IValue& value) {
    at::IValue::HashAliasedIValues sub_values;
    // Prefer getSubValues() over visit() as the latter is a silent no-op for
    // some unsupported types, whereas the former at least fails loudly.
    value.getSubValues(sub_values);

    std::vector<std::reference_wrapper<const at::DataPtr>> data_ptrs;
    for (const at::IValue& sub_value : sub_values) {
      if (sub_value.isTensor()) {
        data_ptrs.emplace_back(sub_value.toTensor().storage().data_ptr());
      }
    }
    return data_ptrs;
  }

 private:
  // The device that was current when markCompleted was called, which we'll
  // restore when invoking callbacks.
  c10::DeviceIndex currentDevice_;

  // The events that correspond to the completion of the async I/O kernels. They
  // are recorded on the appropriate streams when the future is marked completed
  // and can then be queried/waited/blocked on. There is one event for each
  // distinct device on which the value's tensors reside.
  std::vector<at::cuda::CUDAEvent> cudaEvents_;

  // A cached version of the data ptrs extracted from the value when the future
  // is first marked completed.
  std::vector<std::reference_wrapper<const at::DataPtr>> dataPtrs_;
};

} // namespace cuda
} // namespace at
