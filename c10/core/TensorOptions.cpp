#include <c10/core/TensorOptions.h>

#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

#include <iostream>

namespace c10 {

// Note: TensorOptions properties are all optional, but (almost) all have
// getters that supply a default when the corresponding property is missing.
// Here we print the values returned by the default-supplying getters for
// properties that have them, along with an annotation if the value is
// returned by default. This gives the full picture of both the object's
// internal state and what its getters will return.

std::ostream& operator<<(std::ostream& stream, const TensorOptions& options) {
  auto print = [&](const char* label, auto prop, bool has_prop) {
    stream << label << std::boolalpha << prop << (has_prop ? "" : " (default)");
  };

  print("TensorOptions(dtype=", options.dtype(), options.has_dtype());
  print(", device=", options.device(), options.has_device());
  print(", layout=", options.layout(), options.has_layout());
  print(
      ", requires_grad=", options.requires_grad(), options.has_requires_grad());
  print(
      ", pinned_memory=", options.pinned_memory(), options.has_pinned_memory());

  // note: default-supplying memory_format() getter not provided; no canonical
  // default
  stream << ", memory_format=";
  if (options.has_memory_format()) {
    stream << *options.memory_format_opt();
  } else {
    stream << "(nullopt)";
  }
  stream << ")";

  return stream;
}

} // namespace c10
