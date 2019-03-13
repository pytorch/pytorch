#pragma once

#include <c10/core/Backend.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Layout.h>

namespace at {

class TypeProperties {
 public:
  TypeProperties(Backend backend=Backend::Undefined, ScalarType scalar_type=ScalarType::Undefined)
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

  TypeProperties toBackend(Backend b) const {
    return TypeProperties(b, scalarType());
  }

  TypeProperties toScalarType(ScalarType t) const {
    return TypeProperties(backend(), t);
  }

  TypeProperties& operator=(const TypeProperties& other) {
    if (this != &other) {
      backend_ = other.backend();
      scalar_type_ == other.scalarType();
    }
    return *this;
  }

  bool operator==(const TypeProperties& other) const {
    return backend_ == other.backend() && scalar_type_ == other.scalarType();
  }

  bool operator!=(const TypeProperties& other) const {
    return !(*this == other);
  }

  std::string toString() const {
    std::stringstream ss;
    ss << at::toString(backend()) << at::toString(scalarType()) << "Type";
    return ss.str();
  }

  /// Constructs the `TensorOptions` from a type and a `device_index`.
  TensorOptions options(int16_t device_index = -1) const {
    return TensorOptions().dtype(scalarType())
                          .device(device_type(), device_index)
                          .layout(layout_from_backend(backend()));
  }

  /// Constructs the `TensorOptions` from a type and a Device.  Asserts that
  /// the device type matches the device type of the type.
  TensorOptions options(c10::optional<Device> device_opt) const {
    if (!device_opt.has_value()) {
      return options(-1);
    } else {
      Device device = device_opt.value();
      AT_ASSERT(device.type() == device_type());
      return options(device.index());
    }
  }

  operator TensorOptions() const {
    return options();
  }

 private:
  Backend backend_;
  ScalarType scalar_type_;
};

}  // namespace at
