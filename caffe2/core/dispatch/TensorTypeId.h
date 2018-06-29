#pragma once

#include "caffe2/utils/IdWrapper.h"
#include <string>
#include <iostream>
#include <mutex>
#include <unordered_set>

namespace c10 {
class TensorTypeId;
}

std::ostream& operator<<(std::ostream&, c10::TensorTypeId);

namespace c10 {

namespace details {
  using _tensorTypeId_underlyingType = uint8_t;
}

/**
 * Dynamic type ID of a Tensor argument.  It represents something like CPUTensor, etc.
 */
class TensorTypeId final : public guts::IdWrapper<TensorTypeId, details::_tensorTypeId_underlyingType> {
public:
  // Don't use this!
  // Unfortunately, a default constructor needs to be defined because of https://reviews.llvm.org/D41223
  constexpr TensorTypeId() noexcept: IdWrapper(0) {}
private:
  constexpr explicit TensorTypeId(details::_tensorTypeId_underlyingType id) noexcept: IdWrapper(id) {}

  friend class TensorTypeIdCreator;
  friend std::ostream& ::operator<<(std::ostream&, TensorTypeId);
};

}  // namespace c10

C10_DEFINE_HASH_FOR_IDWRAPPER(c10::TensorTypeId)
