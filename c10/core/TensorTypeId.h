#ifndef TENSOR_TYPE_ID_H_
#define TENSOR_TYPE_ID_H_

#include <iostream>
#include <string>
#include "c10/macros/Macros.h"
#include "c10/util/IdWrapper.h"

namespace c10 {

namespace details {
using _tensorTypeId_underlyingType = uint8_t;
}

/**
 * Dynamic type ID of a Tensor argument.  It represents something like
 * CPUTensor, etc.
 */
class C10_API TensorTypeId final
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
  friend C10_API std::ostream& operator<<(std::ostream&, TensorTypeId);
  friend C10_API std::string toString(TensorTypeId);
};

} // namespace c10

C10_DEFINE_HASH_FOR_IDWRAPPER(c10::TensorTypeId)

#endif // TENSOR_TYPE_ID_H_
