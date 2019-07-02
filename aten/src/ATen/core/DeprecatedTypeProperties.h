#pragma once

#include <c10/core/Backend.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Layout.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/Storage.h>
#include <ATen/core/DeprecatedTypePropertiesRegistry.h>
#include <ATen/core/Generator.h>


namespace at {

class Tensor;

// This class specifies a Backend and a ScalarType. Currently, it primarily
// serves as a replacement return value for Tensor::type(). Previously,
// Tensor::type() returned Type&, but we are changing Type to not be
// dtype-specific.
class CAFFE2_API DeprecatedTypeProperties {
 public:
  DeprecatedTypeProperties(Backend backend, ScalarType scalar_type, bool is_variable)
    : backend_(backend), scalar_type_(scalar_type), is_variable_(is_variable) {}

  Backend backend() const {
    return backend_;
  }

  Layout layout() const {
    return layout_from_backend(backend_);
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

  bool is_variable() const {
    return is_variable_;
  }

  bool operator==(const DeprecatedTypeProperties& other) const {
    return backend_ == other.backend() && scalar_type_ == other.scalarType();
  }

  bool operator!=(const DeprecatedTypeProperties& other) const {
    return !(*this == other);
  }

  std::string toString() const {
    std::string base_str;
    if (backend_ == Backend::Undefined || scalar_type_ == ScalarType::Undefined) {
      base_str = "UndefinedType";
    } else {
      base_str = std::string(at::toString(backend_)) + at::toString(scalar_type_) + "Type";
    }
    if (is_variable_) {
      return "Variable[" + base_str + "]";
    }
    return base_str;
  }

  DeprecatedTypeProperties & toBackend(Backend b) const {
    return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
        b, scalar_type_, is_variable_);
  }

  DeprecatedTypeProperties & toScalarType(ScalarType s) const {
    return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
        backend_, s, is_variable_);
  }

  DeprecatedTypeProperties & cpu() const {
    return toBackend(Backend::CPU);
  }

  DeprecatedTypeProperties & cuda() const {
    return toBackend(Backend::CUDA);
  }

  DeprecatedTypeProperties & hip() const {
    return toBackend(Backend::HIP);
  }

  /// Constructs the `TensorOptions` from a type and a `device_index`.
  TensorOptions options(int16_t device_index = -1) const {
    return TensorOptions().dtype(typeMeta())
                          .device(device_type(), device_index)
                          .layout(layout())
                          .is_variable(is_variable());
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

  int64_t id() const {
    return static_cast<int64_t>(backend()) *
        static_cast<int64_t>(ScalarType::NumOptions) +
        static_cast<int64_t>(scalarType());
  }

  Tensor unsafeTensorFromTH(void * th_pointer, bool retain) const;
  Storage unsafeStorageFromTH(void * th_pointer, bool retain) const;
  Tensor copy(const Tensor & src, bool non_blocking=false, c10::optional<Device> to_device={}) const;

 private:
  Backend backend_;
  ScalarType scalar_type_;
  bool is_variable_;
};

}  // namespace at
