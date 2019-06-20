#include <ATen/core/TensorOptions.h>

#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

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
