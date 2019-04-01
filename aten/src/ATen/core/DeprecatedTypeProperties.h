#pragma once

#include <c10/core/Backend.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Layout.h>



namespace at {

// This class specifies a Backend and a ScalarType. Currently, it primarily
// serves as a replacement return value for Tensor::type(). Previously,
// Tensor::type() returned Type&, but we are changing Type to not be
// dtype-specific.
class DeprecatedTypeProperties {
 public:
  DeprecatedTypeProperties(Backend backend, ScalarType scalar_type)
    : backend_(backend), scalar_type_(scalar_type) {}

  Backend backend() const {
    return backend_;
  }

  bool is_sparse() const {
    return layout_from_backend(backend()) == kSparse;
  }

  DeviceType device_type() const {
    return backendToDeviceType(backend_);
  }

  bool is_cuda() const {
    return backendToDeviceType(backend_) == kCUDA;
  }

  ScalarType scalarType() const {
    return scalar_type_;
  }

  caffe2::TypeMeta typeMeta() const {
    return scalarTypeToTypeMeta(scalar_type_);
  }

  bool is_defined() const {
    return backend_ != Backend::Undefined && scalar_type_ != ScalarType::Undefined;
  }

  bool operator==(const DeprecatedTypeProperties& other) const {
    return backend_ == other.backend() && scalar_type_ == other.scalarType();
  }

  bool operator!=(const DeprecatedTypeProperties& other) const {
    return !(*this == other);
  }

  std::string toString() const {
    std::stringstream ss;
    ss << at::toString(backend()) << at::toString(scalarType()) << "Type";
    return ss.str();
  }

 private:
  Backend backend_;
  ScalarType scalar_type_;
};

}  // namespace at
