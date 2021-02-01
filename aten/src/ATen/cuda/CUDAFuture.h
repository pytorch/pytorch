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
    // because postMarkCompletedHook won't be called when the Future contains
    // an error. Uninitialized currentDevice_ could lead to crash when used
    // in CUDAGuard.
    currentDevice_ = c10::cuda::current_device();
    dataPtrs_ = std::make_shared<DataPtrs>();
    cudaEvents_ = std::make_shared<std::vector<at::cuda::CUDAEvent>>();
  }

 protected:
  using DataPtrs = std::vector<std::reference_wrapper<const at::DataPtr>>;

  c10::intrusive_ptr<Future> createChild(at::TypePtr type) override {
    auto child = c10::make_intrusive<CUDAFuture>(std::move(type));
    // child future initially holds on to parent DataPtrs and CUDAEvents, and
    // will wait for these CUDAEvents if its own value does not contain tensors
    // or is not supported by JIT.
    child->dataPtrs_ = dataPtrs_;
    child->cudaEvents_ = cudaEvents_;
    return child;
  }

  void postMarkCompletedHook(const at::IValue& value) override {
    currentDevice_ = c10::cuda::current_device();

    // Extract them once and cache them for later uses.
    auto dataPtrs = extractDataPtrs(value);
    // Only replace parent Future's DataPtrs if value of this child Future is
    // no empty.
    if (!dataPtrs->empty()) {
      dataPtrs_ = dataPtrs;
      cudaEvents_->clear();
      std::vector<bool> isCudaDeviceUsed(c10::cuda::device_count(), false);
      for (const at::DataPtr& dataPtr : *dataPtrs_) {
        if (dataPtr.device().is_cuda()) {
          isCudaDeviceUsed[dataPtr.device().index()] = true;
        }
      }

      for (c10::DeviceIndex idx = 0; idx < isCudaDeviceUsed.size(); idx++) {
        if (isCudaDeviceUsed[idx]) {
          at::cuda::CUDAEvent cudaEvent;
          cudaEvent.record(at::cuda::getCurrentCUDAStream(idx));
          cudaEvents_->push_back(std::move(cudaEvent));
        }
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
      for (at::cuda::CUDAEvent& cudaEvent : *cudaEvents_) {
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
      for (const at::DataPtr& dataPtr : *dataPtrs_) {
        if (dataPtr.device().is_cuda()) {
          c10::cuda::CUDACachingAllocator::recordStream(
              dataPtr,
              at::cuda::getCurrentCUDAStream(dataPtr.device().index()));
        }
      }

      c10::cuda::CUDAGuard deviceGuard(currentDevice_);

      callback();
    };
  }

  void postWaitHook(const at::IValue& value) override {
    for (at::cuda::CUDAEvent& cudaEvent : *cudaEvents_) {
      cudaEvent.block(at::cuda::getCurrentCUDAStream(cudaEvent.device_index()));
    }

    for (const at::DataPtr& dataPtr : *dataPtrs_) {
      if (dataPtr.device().is_cuda()) {
        c10::cuda::CUDACachingAllocator::recordStream(
            dataPtr, at::cuda::getCurrentCUDAStream(dataPtr.device().index()));
      }
    }
  }

  virtual std::shared_ptr<DataPtrs> extractDataPtrs(const at::IValue& value) {
    auto dataPtrs = std::make_shared<DataPtrs>();
    // former fails loudly while the latter is a silent no-op for unsupported
    // types. We made the change because it's common for RPC user functions to
    // return unsupported Python object types, and users don't have easy
    // solutions to get around. Besides, the RemoteException in RPC is also an
    // unsupported type. Adding it to JIT might be overkill. If we don't add
    // RemoteException and use getSubValues() here, any errors in RPC functions
    // would result in a crash on the caller.
    // The current solution is that, the createChild() API initializes child
    // Future dataPtrs_ and cudaEvents_ with parent dataPtrs_ and cudaEvents_.
    // Note that, when creating a child Future, the parent Future might not be
    // completed yet. Hence, the parent passes a shared_ptr of the DataPtrs
    // cudaEvents_ instead of using a copy of the vector.
    // When the child Future is marked as completed (the parent Future is
    // guaranteed to complete at this time), it inspects the returned IValue
    // using the `visit()` API and extracts DataPtrs accordingly. If it fails
    // to extract any DataPtrs, it will use parent's DataPtrs, so that when
    // application code only calls wait() on child Future but not parent Future,
    // it should still work.
    // FIXME: this is not perfect, as users functions can return unsupported
    // custom Python objects with Tensors and the current solution won't
    // recognize them. A better solution might be allowing users to decorate
    // the callback function and provide custom logic to extract tensors.
    value.visit([&dataPtrs](const IValue& subValue) {
      if (subValue.isTensor()) {
        dataPtrs->emplace_back(subValue.toTensor().storage().data_ptr());
      }
      return false;
    });

    return dataPtrs;
  }

 private:
  // The device that was current when markCompleted was called, which we'll
  // restore when invoking callbacks.
  c10::DeviceIndex currentDevice_;

  // The events that correspond to the completion of the async I/O kernels. They
  // are recorded on the appropriate streams when the future is marked completed
  // and can then be queried/waited/blocked on. There is one event for each
  // distinct device on which the value's tensors reside.
  std::shared_ptr<std::vector<at::cuda::CUDAEvent>> cudaEvents_;

  // A cached version of the data ptrs extracted from the value when the future
  // is first marked completed.
  std::shared_ptr<DataPtrs> dataPtrs_;
};

} // namespace cuda
} // namespace at
