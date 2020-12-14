#import <ATen/native/metal/MetalTensor.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/mpscnn/MPSImageWrapper.h>

namespace at {
namespace native {
namespace metal {

class API_AVAILABLE(ios(10.0), macos(10.13)) MetalTensor::Impl {
 public:
  Impl(const std::vector<int64_t>& sizes)
      : Impl(sizes, std::vector<int64_t>(sizes.size())) {}

  Impl(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides)
      : _sizes(sizes),
        _strides(strides),
        _numel(std::accumulate(
            std::begin(_sizes),
            std::end(_sizes),
            (int64_t)1,
            std::multiplies<int64_t>())),
        _textureImpl(std::make_unique<MPSImageWrapper>(sizes)) {}

  IntArrayRef sizes() const {
    return _sizes;
  }
  IntArrayRef strides() const {
    return _strides;
  }
  int64_t dim() const {
    return _sizes.size();
  }
  int64_t numel() const {
    return _numel;
  }
  void set_data_from_host(const float* inputData) {
    _textureImpl->copyDataFromHost(inputData);
  }
  void copy_data_to_host(float* host) {
    _textureImpl->copyDataToHost(host);
  }
  MPSImageWrapper* texture() const {
    return _textureImpl.get();
  }

 private:
  std::vector<int64_t> _sizes;
  std::vector<int64_t> _strides;
  int64_t _numel;
  std::unique_ptr<MPSImageWrapper> _textureImpl;
};

MetalTensor::MetalTensor(const std::vector<int64_t>& sizes)
    : MetalTensor(sizes, std::vector<int64_t>(sizes.size())) {} // fake strides

MetalTensor::MetalTensor(
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides)
    : _impl(std::make_shared<Impl>(std::move(sizes), std::move(strides))) {}

bool MetalTensor::defined() const {
  return static_cast<bool>(_impl);
}

at::Tensor MetalTensor::toTensor(
    MetalTensor&& mt,
    const TensorOptions& options) {
  using MetalTensorImpl = at::MetalTensorImpl<MetalTensor>;
  auto sizes = mt.sizes(); // sizes is stored in TensorImpl
  auto strides = mt.strides(); // strides is stored in MetalTensorImpl
  return detail::make_tensor<MetalTensorImpl>(
      DispatchKeySet(DispatchKey::Metal),
      options.dtype(),
      at::Device(at::kMetal),
      std::move(mt),
      std::vector<int64_t>(sizes.begin(), sizes.end()),
      std::vector<int64_t>(strides.begin(), strides.end()));
}

MetalTensor& MetalTensor::fromTensor(const at::Tensor& tensor) {
  using MetalTensorImpl = at::MetalTensorImpl<MetalTensor>;
  TORCH_INTERNAL_ASSERT(
      tensor.is_metal(), "unbox expects Metal tensor as inputs");
  MetalTensorImpl* impl =
      static_cast<MetalTensorImpl*>(tensor.unsafeGetTensorImpl());
  return impl->unsafe_opaque_handle();
}

std::shared_ptr<MetalTensor::Impl> MetalTensor::impl() {
  return _impl;
}

std::shared_ptr<const MetalTensor::Impl> MetalTensor::impl() const {
  return _impl;
}

IntArrayRef MetalTensor::sizes() const {
  return impl()->sizes();
}

IntArrayRef MetalTensor::strides() const {
  return impl()->strides();
}

int64_t MetalTensor::dim() const {
  return impl()->dim();
}

int64_t MetalTensor::numel() const {
  return impl()->numel();
}

void MetalTensor::set_data_from_host(const float* inputData) {
  impl()->set_data_from_host(inputData);
}

void MetalTensor::copy_data_to_host(float* hostData) {
  impl()->copy_data_to_host(hostData);
}

API_AVAILABLE(ios(10.0))
MPSImageWrapper* MetalTensor::texture() const {
  return impl()->texture();
}

std::ostream& operator<<(std::ostream& output, const MetalTensor& mt) {
  auto&& sizes = mt.sizes();
  auto&& strides = mt.strides();
  output << "[MetalTensor] | Size:{";
  std::ostringstream oss;
  std::copy(
      sizes.begin(), sizes.end() - 1, std::ostream_iterator<int>(oss, ","));
  oss << sizes.back();
  output << oss.str() << "}, Stride:{";
  std::string sizesStr = oss.str();
  oss.str("");
  oss.clear();
  std::copy(
      strides.begin(), strides.end() - 1, std::ostream_iterator<int>(oss, ","));
  oss << sizes.back();
  output << oss.str() << "}";
  return output;
}

}
}
}
