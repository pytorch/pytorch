#pragma once

#include <cstddef>

namespace torch {
namespace data {
namespace samplers {
/// A base class for custom index types.
struct CustomIndex {
  virtual ~CustomIndex() = default;

  /// The number of elements accessed by this index.
  virtual size_t size() const = 0;
};
} // namespace samplers
} // namespace data
} // namespace torch
