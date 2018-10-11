#include <ATen/core/TensorOptions.h>

#include <ATen/core/Device.h>
#include <ATen/core/Layout.h>
#include <ATen/core/OptionsGuard.h>
#include <ATen/core/ScalarType.h>
#include <ATen/core/optional.h>
#include <ATen/core/ScalarType.h>

#include <iostream>

namespace at {

std::ostream& operator<<(
    std::ostream& stream,
    const TensorOptions& options) {
  return stream << "TensorOptions(dtype=" << options.dtype()
                << ", device=" << options.device()
                << ", layout=" << options.layout()
                << ", requires_grad=" << std::boolalpha
                << options.requires_grad() << ")";
}

} // namespace at
