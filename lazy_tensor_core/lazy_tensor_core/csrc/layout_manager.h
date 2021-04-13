#pragma once

#include "lazy_tensor_core/csrc/device.h"
#include "lazy_tensors/shape.h"
#include "lazy_tensors/span.h"
#include "lazy_tensors/types.h"

namespace torch_lazy_tensors {

// Creates a minor-to-major layout from given dimensions. The dynamic_dimensions
// slice should be either empty, or of the same size as dimensions.
lazy_tensors::Shape MakeTorchTensorLayout(
    lazy_tensors::Span<const lazy_tensors::int64> dimensions,
    lazy_tensors::Span<const bool> dynamic_dimensions,
    lazy_tensors::PrimitiveType type);

// Create an shape with the given dimensions and type, suitable to be used in
// the specified device type. The type of device can affect the choice of the
// layout. The dynamic_dimensions slice should be either empty, or of the same
// size as dimensions.
lazy_tensors::Shape MakeArrayShapeFromDimensions(
    lazy_tensors::Span<const lazy_tensors::int64> dimensions,
    lazy_tensors::Span<const bool> dynamic_dimensions,
    lazy_tensors::PrimitiveType type, DeviceType device_type);

}  // namespace torch_lazy_tensors
