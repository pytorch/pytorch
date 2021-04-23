#include <ATen/cuda/CUDAFuture.h>

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

namespace {

std::vector<std::reference_wrapper<const at::DataPtr>> extractDataPtrs(
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

std::vector<c10::DeviceIndex> getDevicesOfDataPtrs(
    const std::vector<std::reference_wrapper<const at::DataPtr>>& data_ptrs) {
  std::vector<bool> isCudaDeviceUsed(c10::cuda::device_count(), false);
  for (const at::DataPtr& data_ptr : data_ptrs) {
    if (data_ptr.device().is_cuda()) {
      isCudaDeviceUsed[data_ptr.device().index()] = true;
    }
  }
  std::vector<c10::DeviceIndex> deviceIndices;
  for (c10::DeviceIndex idx = 0; idx < isCudaDeviceUsed.size(); idx++) {
    if (isCudaDeviceUsed[idx]) {
      deviceIndices.push_back(idx);
    }
  }
  return deviceIndices;
}

std::string formatSetOfDevices(const std::vector<c10::DeviceIndex>& devices) {
  if (devices.empty()) {
    return "(none)";
  }
  std::ostringstream oss;
  oss << "cuda:" << static_cast<int64_t>(devices[0]);
  for (size_t idx = 1; idx < devices.size(); idx++) {
    if (idx == devices.size() - 1) {
      oss << " and ";
    } else {
      oss << ", ";
    }
    oss << "cuda:" << static_cast<int64_t>(devices[idx]);
  }
  return oss.str();
}

// We need devices to be sorted in order to use set_difference.
std::vector<c10::DeviceIndex> sortDevices(
    std::vector<c10::DeviceIndex> devices) {
  std::sort(devices.begin(), devices.end());
  return devices;
}

} // namespace

CUDAFuture::CUDAFuture(at::TypePtr type, std::vector<c10::DeviceIndex> devices)
    : at::ivalue::Future(std::move(type)),
      devices_(sortDevices(std::move(devices))) {
  // Use current device to initialize currentDevice_. This is necessary
  // because preMarkCompletedHook won't be called when the Future contains
  // an error. Uninitialized currentDevice_ could lead to crash when used
  // in CUDAGuard.
  currentDevice_ = c10::cuda::current_device();
}

c10::intrusive_ptr<ivalue::Future> CUDAFuture::createInstance(
    at::TypePtr type) {
  return c10::make_intrusive<CUDAFuture>(std::move(type), devices_);
}

/**
 * The dataPtrs field contains storage pointers of all tensors in the IValue.
 * This method records CUDAEvents on participating devices and uses those
 * CUDAEvents to synchronize streams when calling postWaitHook().
 * If dataPtrs does not have a value, this method will try to inspect the
 * given IValue by walking through all subvalues and extracting data pointers
 * from CUDA tensors.
 */
void CUDAFuture::preMarkCompletedHook(
    const at::IValue& value,
    c10::optional<std::vector<std::reference_wrapper<const at::DataPtr>>>
        dataPtrs) {
  // Start by performing all steps that can throw, before setting any field.
  std::vector<std::reference_wrapper<const at::DataPtr>> actualDataPtrs =
      dataPtrs.has_value() ? std::move(*dataPtrs) : extractDataPtrs(value);
  std::vector<c10::DeviceIndex> usedDevices =
      getDevicesOfDataPtrs(actualDataPtrs);
  std::vector<c10::DeviceIndex> excessDevices;
  std::set_difference(
      usedDevices.begin(),
      usedDevices.end(),
      devices_.begin(),
      devices_.end(),
      std::back_inserter(excessDevices));
  TORCH_CHECK_VALUE(
      excessDevices.empty(),
      "The result contained tensors residing on device(s) ",
      formatSetOfDevices(excessDevices),
      " which are not among the expected device(s) ",
      formatSetOfDevices(devices_));

  currentDevice_ = c10::cuda::current_device();

  // Extract them once and cache them for later uses.
  dataPtrs_ = std::move(actualDataPtrs);

  for (const c10::DeviceIndex& idx : usedDevices) {
    at::cuda::CUDAEvent cudaEvent;
    cudaEvent.record(at::cuda::getCurrentCUDAStream(idx));
    cudaEvents_.push_back(std::move(cudaEvent));
  }
}

std::function<void(void)> CUDAFuture::wrapCallback(
    std::function<void(void)> callback) {
  return [this, callback{std::move(callback)}]() {
    std::vector<at::cuda::CUDAStream> streams;
    for (const c10::DeviceIndex& idx : devices_) {
      // FIXME Should we find a way to allow to change the priority of
      // streams?
      streams.push_back(
          at::cuda::getStreamFromPool(/*isHighPriority=*/false, idx));
    }

    // Use the dedicated callback stream to run callback.
    at::cuda::CUDAMultiStreamGuard streamGuard(streams);

    for (at::cuda::CUDAEvent& cudaEvent : cudaEvents_) {
      cudaEvent.block(at::cuda::getCurrentCUDAStream(cudaEvent.device_index()));
    }

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

void CUDAFuture::postWaitHook(const at::IValue& value) {
  for (at::cuda::CUDAEvent& cudaEvent : cudaEvents_) {
    cudaEvent.block(at::cuda::getCurrentCUDAStream(cudaEvent.device_index()));
  }

  for (const at::DataPtr& data_ptr : dataPtrs_) {
    if (data_ptr.device().is_cuda()) {
      c10::cuda::CUDACachingAllocator::recordStream(
          data_ptr, at::cuda::getCurrentCUDAStream(data_ptr.device().index()));
    }
  }
}

} // namespace cuda
} // namespace at
