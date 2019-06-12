#pragma once

#include <iostream>
#include <string>
#include "c10/util/IdWrapper.h"
#include "c10/macros/Macros.h"

namespace at {

namespace details {
using _tensorTypeId_underlyingType = uint8_t;
}

/**
 * Dynamic type ID of a Tensor argument.  It represents something like
 * CPUTensor, etc.
 */
class CAFFE2_API TensorTypeId final
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
  friend CAFFE2_API std::ostream& operator<<(std::ostream&, TensorTypeId);
};

CAFFE2_API std::ostream& operator<<(std::ostream&, at::TensorTypeId);

} // namespace at

C10_DEFINE_HASH_FOR_IDWRAPPER(at::TensorTypeId)
