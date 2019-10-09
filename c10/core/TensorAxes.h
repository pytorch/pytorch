#pragma once

#include <c10/core/DefaultDtype.h>
#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorTypeSet.h>

namespace c10 {

/// `TensorAxes` holds the effectively used data type, device and layout of a
/// `Tensor`. Additionally it indicates whether a `Tensor` is an autograd
/// variable.
///
/// The class is in general immutable but allows to change the representation
/// as a whole via the standard assignment operator.
///
/// In contrast to `TensorOptions` which allows to specify (optional)
/// construction arguments this class always holds a full axes specification of
/// a `Tensor`, i.e. it has no optional fields.
///
/// An (implicit) conversion from `TensorAxes` to `TensorOptions` via the
/// `TensorOptions(TensorAxes&)` ctor exists but not vice versa.

class C10_API TensorAxes final {
 public:
  /// Constructs a `TensorAxes` object with the given parameters.
  TensorAxes(
      caffe2::TypeMeta dtype,
      Device device,
      Layout layout = at::kStrided,
      bool is_variable = false)
      : dtype_(dtype),
        device_(device),
        layout_(layout),
        is_variable_(is_variable) {}

  /// Default copy constructor
  TensorAxes(const TensorAxes&) = default;

  /// Default assignment operator
  TensorAxes& operator=(const TensorAxes&) = default;

  /// Returns the `device` property of the `TensorAxes`.
  Device device() const noexcept {
    return device_;
  }

  /// Return a copy of `TensorAxes` with `device` set to the given one.
  C10_NODISCARD TensorAxes device(Device device) const noexcept {
    auto r = *this;
    r.device_ = device;
    return r;
  }

  /// Returns a copy of `TensorAxes` with `device` set to the given one.
  /// (This overload ensures that variadic template c10::optional constructor
  /// for Device work correctly.)
  template <typename... Args>
  C10_NODISCARD TensorAxes device(Args&&... args) const noexcept {
    return device(Device(std::forward<Args>(args)...));
  }

  /// Returns the device index of the `TensorAxes`.
  int32_t device_index() const noexcept {
    return device().index();
  }

  /// Returns the `dtype` property of the `TensorAxes`.
  caffe2::TypeMeta dtype() const noexcept {
    return dtype_;
  }

  /// Returns a copy of `TensorAxes` with `dtype` set to the given one.
  C10_NODISCARD TensorAxes dtype(caffe2::TypeMeta dtype) const noexcept {
    auto r = *this;
    r.dtype_ = dtype;
    return r;
  }

  /// Returns a copy of `TensorAxes` with `dtype` set to the given one
  /// (legacy function to support ScalarType).
  C10_NODISCARD TensorAxes dtype(ScalarType dtype) const noexcept {
    return this->dtype(scalarTypeToTypeMeta(dtype));
  }

  /// Returns the layout of the `TensorAxes`.
  Layout layout() const noexcept {
    return layout_;
  }

  /// Returns a copy of `TensorAxes` with `layout` set to the given one.
  C10_NODISCARD TensorAxes layout(Layout layout) const noexcept {
    auto r = *this;
    r.layout_ = layout;
    return r;
  }

  /// Returns the `is_variable` property of the `TensorAxes`.
  bool is_variable() const noexcept {
    return is_variable_;
  }

  // Returns a copy of `TensorAxes` with `is_variable` set to the given value.
  C10_NODISCARD TensorAxes is_variable(bool is_variable) const noexcept {
    auto r = *this;
    r.is_variable_ = is_variable;
    return r;
  }

  /// True if all elements of the `TensorAxes` match that of the other.
  bool operator==(const TensorAxes& other) const noexcept {
    return dtype_ == other.dtype_ && layout_ == other.layout_ &&
        device_ == other.device_ && is_variable_ == other.is_variable_;
  }

  /// True if any of the elements of this `TensorOptions` do not match that of
  /// the other.
  bool operator!=(const TensorAxes& other) const noexcept {
    return !(*this == other);
  }

  // Resolves the tensor type set specified by the current construction axes.
  TensorTypeSet type_set() const noexcept {
    TensorTypeSet r{tensorTypeId()};
    if (is_variable()) {
      r = r.add(TensorTypeId::VariableTensorId);
    }
    return r;
  }

  inline TensorTypeId tensorTypeId() const {
    switch (layout()) {
      case Layout::Strided:
        switch (device().type()) {
          case DeviceType::CPU:
            if (isComplexType(typeMetaToScalarType(dtype()))) {
              return TensorTypeId::ComplexCPUTensorId;
            }
            if (isQIntType(typeMetaToScalarType(dtype()))) {
              return TensorTypeId::QuantizedCPUTensorId;
            }
            return TensorTypeId::CPUTensorId;
          case DeviceType::CUDA:
            if (isComplexType(typeMetaToScalarType(dtype()))) {
              return TensorTypeId::ComplexCUDATensorId;
            }
            return TensorTypeId::CUDATensorId;
          case DeviceType::MKLDNN:
            return TensorTypeId::MKLDNNTensorId;
          case DeviceType::OPENGL:
            return TensorTypeId::OpenGLTensorId;
          case DeviceType::OPENCL:
            return TensorTypeId::OpenCLTensorId;
          case DeviceType::IDEEP:
            return TensorTypeId::IDEEPTensorId;
          case DeviceType::HIP:
            return TensorTypeId::HIPTensorId;
          case DeviceType::MSNPU:
            return TensorTypeId::MSNPUTensorId;
          case DeviceType::XLA:
            return TensorTypeId::XLATensorId;
          default:
            AT_ERROR(
                "Unsupported device type for dense layout: ", device().type());
        }
      case Layout::Sparse:
        switch (device().type()) {
          case DeviceType::CPU:
            return TensorTypeId::SparseCPUTensorId;
          case DeviceType::CUDA:
            return TensorTypeId::SparseCUDATensorId;
          case DeviceType::HIP:
            return TensorTypeId::SparseHIPTensorId;
          default:
            AT_ERROR(
                "Unsupported device type for sparse layout: ", device().type());
        }
      case Layout::Mkldnn:
        switch (device().type()) {
          case DeviceType::CPU:
            return TensorTypeId::MkldnnCPUTensorId;
          default:
            AT_ERROR(
                "Unsupported device type for mkldnn layout: ", device().type());
        }
      default:
        AT_ERROR("Unsupported layout: ", layout());
    }
  }

  std::string toString() const;

 private:
  caffe2::TypeMeta dtype_;
  Device device_;
  Layout layout_;
  bool is_variable_;
};

C10_API std::string toString(const TensorAxes&);
C10_API std::ostream& operator<<(std::ostream&, const TensorAxes&);

} // namespace c10
