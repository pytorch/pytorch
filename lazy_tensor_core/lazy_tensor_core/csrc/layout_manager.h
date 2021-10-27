#pragma once

#include "lazy_tensor_core/csrc/device.h"
#include "lazy_tensors/shape.h"

namespace torch_lazy_tensors {

// Creates a minor-to-major layout from given dimensions. The dynamic_dimensions
// slice should be either empty, or of the same size as dimensions.
lazy_tensors::Shape MakeTorchTensorLayout(
    c10::ArrayRef<int64_t> dimensions, c10::ArrayRef<bool> dynamic_dimensions,
    c10::ScalarType type);

// Create an shape with the given dimensions and type, suitable to be used in
// the specified device type. The type of device can affect the choice of the
// layout. The dynamic_dimensions slice should be either empty, or of the same
// size as dimensions.
lazy_tensors::Shape MakeArrayShapeFromDimensions(
    c10::ArrayRef<int64_t> dimensions, c10::ArrayRef<bool> dynamic_dimensions,
    c10::ScalarType type, DeviceType device_type);

}  // namespace torch_lazy_tensors
