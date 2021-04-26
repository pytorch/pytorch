#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalTensorImplStorage.h>
#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSImageWrapper.h>

#include <ATen/Utils.h>
#include <c10/util/accumulate.h>

namespace at {
namespace native {
namespace metal {

class API_AVAILABLE(ios(10.0), macos(10.13)) MetalTensorImplStorage::Impl {
 public:
  Impl(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides)
      : _sizes(sizes),
        _strides(strides),
        _numel(c10::multiply_integers(std::begin(_sizes), std::end(_sizes))),
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

MetalTensorImplStorage::MetalTensorImplStorage(
    const std::vector<int64_t>& sizes)
    : MetalTensorImplStorage(sizes, compute_strides(sizes)) {}

MetalTensorImplStorage::MetalTensorImplStorage(
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides)
    : _impl(std::make_shared<Impl>(std::move(sizes), std::move(strides))) {}

bool MetalTensorImplStorage::defined() const {
  return static_cast<bool>(_impl);
}

std::shared_ptr<MetalTensorImplStorage::Impl> MetalTensorImplStorage::impl() {
  return _impl;
}

std::shared_ptr<const MetalTensorImplStorage::Impl> MetalTensorImplStorage::
    impl() const {
  return _impl;
}

IntArrayRef MetalTensorImplStorage::sizes() const {
  return impl()->sizes();
}

IntArrayRef MetalTensorImplStorage::strides() const {
  return impl()->strides();
}

int64_t MetalTensorImplStorage::dim() const {
  return impl()->dim();
}

int64_t MetalTensorImplStorage::numel() const {
  return impl()->numel();
}

void MetalTensorImplStorage::set_data_from_host(const float* inputData) {
  impl()->set_data_from_host(inputData);
}

void MetalTensorImplStorage::copy_data_to_host(float* hostData) {
  impl()->copy_data_to_host(hostData);
}

API_AVAILABLE(ios(10.0))
MPSImageWrapper* MetalTensorImplStorage::texture() const {
  return impl()->texture();
}

std::ostream& operator<<(
    std::ostream& output,
    const MetalTensorImplStorage& mt) {
  auto&& sizes = mt.sizes();
  auto&& strides = mt.strides();
  output << "[MetalTensorImplStorage] | Size:{";
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
