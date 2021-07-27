#include "lazy_tensor_core/csrc/tensor_impl.h"

#include <c10/core/ScalarType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"
#include "lazy_tensor_core/csrc/device.h"
#include "lazy_tensor_core/csrc/layout_manager.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensors/computation_client/computation_client.h"
#include "lazy_tensors/computation_client/debug_macros.h"

namespace torch_lazy_tensors {
namespace {

struct LTCGuardImpl : public c10::impl::DeviceGuardImplInterface {
  at::DeviceType type() const override { return at::DeviceType::Lazy; }

  c10::Device exchangeDevice(c10::Device device) const override {
    return bridge::SetCurrentDevice(device);
  }

  c10::Device getDevice() const override {
    return bridge::GetCurrentAtenDevice();
  }

  void setDevice(c10::Device device) const override {
    bridge::SetCurrentDevice(device);
  }

  void uncheckedSetDevice(c10::Device device) const noexcept override {
    bridge::SetCurrentDevice(device);
  }

  c10::Stream getStream(c10::Device device) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, device);
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, bridge::GetCurrentAtenDevice());
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return lazy_tensors::ComputationClient::Get()->GetNumDevices();
  }
};

C10_REGISTER_GUARD_IMPL(Lazy, LTCGuardImpl);

}  // namespace

LTCTensorImpl::LTCTensorImpl(LazyTensor tensor)
    : c10::TensorImpl(c10::DispatchKeySet{c10::DispatchKey::Lazy,
                                          c10::DispatchKey::AutogradLazy},
                      GetTypeMeta(tensor),
                      bridge::LtcDeviceToAtenDevice(tensor.GetDevice())),
      tensor_(std::move(tensor)) {
  is_non_overlapping_and_dense_ = false;
}

void LTCTensorImpl::set_tensor(LazyTensor lazy_tensor) {
  tensor_ = std::move(lazy_tensor);
  generation_ = 0;
}

c10::intrusive_ptr<c10::TensorImpl> LTCTensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  auto impl = c10::make_intrusive<LTCTensorImpl>(tensor_);
  if (is_interop_view_) {
    impl->MarkAsInteropView();
  }
  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/version_counter,
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  return impl;
}

c10::intrusive_ptr<c10::TensorImpl> LTCTensorImpl::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  auto impl = c10::make_intrusive<LTCTensorImpl>(tensor_);
  if (is_interop_view_) {
    impl->MarkAsInteropView();
  }
  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/std::move(version_counter),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  return impl;
}

void LTCTensorImpl::shallow_copy_from(
    const c10::intrusive_ptr<TensorImpl>& impl) {
  LTCTensorImpl* ltc_impl = dynamic_cast<LTCTensorImpl*>(impl.get());
  copy_tensor_metadata(
      /*src_impl=*/ltc_impl,
      /*dest_impl=*/this,
      /*version_counter=*/version_counter(),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());
  ltc_impl->tensor_.ShallowCopyTo(&tensor_);
  generation_ = 0;
}

at::IntArrayRef LTCTensorImpl::sizes() const {
  const_cast<LTCTensorImpl*>(this)->SetupSizeProperties();
  return c10::TensorImpl::sizes();
}

int64_t LTCTensorImpl::dim() const {
  const_cast<LTCTensorImpl*>(this)->SetupSizeProperties();
  return c10::TensorImpl::dim();
}

int64_t LTCTensorImpl::numel() const {
  const_cast<LTCTensorImpl*>(this)->SetupSizeProperties();
  return c10::TensorImpl::numel();
}

bool LTCTensorImpl::is_contiguous(at::MemoryFormat memory_format) const {
  if (tensor_.CurrentTensorData()) {
    return tensor_.CurrentTensorData()->is_contiguous();
  }
  // Only check that the storage is already contiguous.
  LTC_CHECK(is_contiguous_) << "Non-contiguous storage for lazy tensor";
  return true;
}

int64_t LTCTensorImpl::size(int64_t d) const {
  const_cast<LTCTensorImpl*>(this)->SetupSizeProperties();
  return c10::TensorImpl::size(d);
}

void LTCTensorImpl::SetupSizeProperties() {
  size_t generation = tensor_.generation();
  if (generation != generation_) {
    // Fill up the basic dimension data members which the base class
    // implementation uses in its APIs.
    auto shape = tensor_.shape();
    c10::SmallVector<int64_t, 5> updated_sizes;
    numel_ = 1;
    for (auto dim : shape.get().dimensions()) {
      updated_sizes.push_back(dim);
      numel_ *= dim;
    }
    sizes_and_strides_.set_sizes(updated_sizes);
    std::vector<lazy_tensors::int64> updated_strides;
    if (is_interop_view_ && tensor_.CurrentTensorData()) {
      at::IntArrayRef strides = tensor_.CurrentTensorData()->strides();
      updated_strides.assign(strides.begin(), strides.end());
    } else {
      updated_strides = ComputeArrayStrides(shape.get().dimensions());
    }
    for (int i = 0; i < updated_strides.size(); i++) {
      sizes_and_strides_.stride_at_unchecked(i) = updated_strides[i];
    }
    generation_ = generation;
  }
}

caffe2::TypeMeta LTCTensorImpl::GetTypeMeta(const LazyTensor& tensor) {
  return c10::scalarTypeToTypeMeta(tensor.dtype());
}

void LTCTensorImpl::AtenInitialize() {
  // ATEN specific initialization calls placed below.
}

const at::Storage& LTCTensorImpl::storage() const {
  LTC_ERROR() << "Lazy tensors do not have storage";
}

bool LTCTensorImpl::has_storage() const { return false; }

}  // namespace torch_lazy_tensors
