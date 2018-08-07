#pragma once

#include <iostream>
#include <mutex>
#include <string>
#include <unordered_set>
#include "ATen/core/IdWrapper.h"

namespace at {
class TensorTypeId;
}

std::ostream& operator<<(std::ostream&, at::TensorTypeId);

namespace at {

namespace details {
using _tensorTypeId_underlyingType = uint8_t;
}

/**
 * Dynamic type ID of a Tensor argument.  It represents something like
 * CPUTensor, etc.
 */
class TensorTypeId final
    : public at::
          IdWrapper<TensorTypeId, details::_tensorTypeId_underlyingType> {
 public:
  // Don't use this!
  // Unfortunately, a default constructor needs to be defined because of
  // https://reviews.llvm.org/D41223
  constexpr TensorTypeId() noexcept : IdWrapper(0) {}

 private:
  constexpr explicit TensorTypeId(
      details::_tensorTypeId_underlyingType id) noexcept
      : IdWrapper(id) {}

  friend class TensorTypeIdCreator;
  friend std::ostream& ::operator<<(std::ostream&, TensorTypeId);
};

} // namespace at

AT_DEFINE_HASH_FOR_IDWRAPPER(at::TensorTypeId)
