#include <ATen/cuda/CUDAFuture.h>

#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <ATen/core/ivalue.h>
#include <ATen/core/ivalue_inl.h>
#include <ATen/core/jit_type.h>
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/StreamGuard.h>
#include <c10/macros/Export.h>
#include <c10/util/intrusive_ptr.h>

namespace at {
namespace cuda {

namespace {

std::vector<std::reference_wrapper<const at::DataPtr>> extractDataPtrs(
    const at::IValue& value) {
  std::vector<std::reference_wrapper<const at::DataPtr>> data_ptrs;
  // getSubValues works poorly on Python objects: it only works if they can be
  // converted to a "regular" IValue type hence, for example, it doesn't support
  // custom subclasses. Thus, instead, we extract the tensors through pickling.
  if (value.isPyObject()) {
    std::vector<at::Tensor> tensors =
        value.toPyObjectHolder()->extractTensors();
    data_ptrs.reserve(tensors.size());
    for (const at::Tensor& tensor : tensors) {
      data_ptrs.emplace_back(tensor.storage().data_ptr());
    }
  } else {
    at::IValue::HashAliasedIValues sub_values;
    // Prefer getSubValues() over visit() as the latter is a silent no-op for
    // some unsupported types, whereas the former at least fails loudly.
    value.getSubValues(sub_values);
    for (const at::IValue& sub_value : sub_values) {
      if (sub_value.isTensor()) {
        data_ptrs.emplace_back(sub_value.toTensor().storage().data_ptr());
      }
    }
  }
  return data_ptrs;
}

std::vector<c10::DeviceIndex> getDevicesOfDataPtrs(
    const c10::impl::DeviceGuardImplInterface* impl,
    const std::vector<std::reference_wrapper<const at::DataPtr>>& data_ptrs) {
  std::vector<bool> isDeviceUsed(impl->deviceCount(), false);
  for (const at::DataPtr& data_ptr : data_ptrs) {
    if (!data_ptr.device().is_cpu()) {
      TORCH_CHECK_VALUE(
          data_ptr.device().type() == impl->type(),
          "Expected all data ptrs to be on a device of type ",
          impl->type(),
          ", got one on device ",
          data_ptr.device());
      isDeviceUsed[data_ptr.device().index()] = true;
    }
  }
  std::vector<c10::DeviceIndex> deviceIndices;
  for (c10::DeviceIndex idx = 0; idx < isDeviceUsed.size(); idx++) {
    if (isDeviceUsed[idx]) {
      deviceIndices.push_back(idx);
    }
  }
  return deviceIndices;
}

std::string formatSetOfDevices(
    const c10::impl::DeviceGuardImplInterface* impl,
    const std::vector<c10::DeviceIndex>& devices) {
  if (devices.empty()) {
    return "(none)";
  }
  std::ostringstream oss;
  oss << c10::Device(impl->type(), devices[0]);
  for (size_t idx = 1; idx < devices.size(); idx++) {
    if (idx == devices.size() - 1) {
      oss << " and ";
    } else {
      oss << ", ";
    }
    oss << c10::Device(impl->type(), devices[idx]);
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
      impl_(c10::impl::getDeviceGuardImpl(c10::kCUDA)),
      devices_(sortDevices(std::move(devices))) {
  // Use current device to initialize currentDevice_. This is necessary
  // because preMarkCompletedHook won't be called when the Future contains
  // an error. Uninitialized currentDevice_ could lead to crash when used
  // in CUDAGuard.
  currentDevice_ = impl_->getDevice().index();
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
      getDevicesOfDataPtrs(impl_, actualDataPtrs);
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
      formatSetOfDevices(impl_, excessDevices),
      " which are not among the expected device(s) ",
      formatSetOfDevices(impl_, devices_));

  currentDevice_ = impl_->getDevice().index();

  // Extract them once and cache them for later uses.
  dataPtrs_ = std::move(actualDataPtrs);

  for (const c10::DeviceIndex& idx : usedDevices) {
    c10::Event event(impl_->type());
    event.record(impl_->getStream(c10::Device(impl_->type(), idx)));
    events_.push_back(std::move(event));
  }
}

std::function<void(void)> CUDAFuture::wrapCallback(
    std::function<void(void)> callback) {
  return [this, callback{std::move(callback)}]() {
    std::vector<c10::Stream> streams;
    for (const c10::DeviceIndex& idx : devices_) {
      // FIXME Should we find a way to allow to change the priority of
      // streams?
      streams.push_back(impl_->getStreamFromPool(
          c10::Device(impl_->type(), idx), /*isHighPriority=*/false));
    }

    // Use the dedicated callback stream to run callback.
    c10::MultiStreamGuard streamGuard(streams);

    for (c10::Event& event : events_) {
      event.block(impl_->getStream(
          c10::Device(event.device_type(), event.device_index())));
    }

    // Do not free the underlying data storage of value_ before its
    // usage on the stream finishes.
    for (const at::DataPtr& data_ptr : dataPtrs_) {
      if (!data_ptr.device().is_cpu()) {
        impl_->recordDataPtrOnStream(
            data_ptr, impl_->getStream(data_ptr.device()));
      }
    }

    c10::DeviceGuard deviceGuard(c10::Device(impl_->type(), currentDevice_));

    callback();
  };
}

void CUDAFuture::postWaitHook(const at::IValue& value) {
  for (c10::Event& event : events_) {
    event.block(impl_->getStream(
        c10::Device(event.device_type(), event.device_index())));
  }

  for (const at::DataPtr& data_ptr : dataPtrs_) {
    if (!data_ptr.device().is_cpu()) {
      impl_->recordDataPtrOnStream(
          data_ptr, impl_->getStream(data_ptr.device()));
    }
  }
}

} // namespace cuda
} // namespace at
