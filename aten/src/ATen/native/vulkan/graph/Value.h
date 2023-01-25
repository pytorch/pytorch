#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/api/Tensor.h>
#include <ATen/native/vulkan/graph/Staging.h>
#include <ATen/native/vulkan/graph/Types.h>

#include <iostream>

namespace at {
namespace native {
namespace vulkan {

/*
 * This class is modelled after c10::IValue; however, it is simplified and does
 * not support as many types. However, the core design is the same; it is a
 * tagged union over the types supported by the Vulkan Graph type.
 */
struct Value final {
 private:
  /*
   * The union type which is used to store the value of the Value.
   */
  union Payload {
    /*
     * Similar to IValue::Payload, trivially copyable types are nested in their
     * own union.
     */
    union TriviallyCopyablePayload {
      TriviallyCopyablePayload() : as_int(0) {}
      int64_t as_int;
      double as_double;
      bool as_bool;
    } u;

    vTensor as_tensor;
    TensorStaging as_staging;

    Payload() : u() {}
    ~Payload() {}
  };

 public:
  /*
   Copy constructor and assignment (disabled)
  */

  Value(const Value& rhs) = delete;
  Value& operator=(const Value&) = delete;

  /*
   Move constructor and assignment; Move assignment is disabled but construction
   is implemented to allow for use in container types.
  */

  Value& operator=(Value&&) = delete;

  Value(Value&& rhs) noexcept : tag(rhs.tag) {
    if (rhs.isTensor()) {
      new (&payload.as_tensor) vTensor(std::move(rhs.payload.as_tensor));
    } else if (rhs.isStaging()) {
      new (&payload.as_staging)
          TensorStaging(std::move(rhs.payload.as_staging));
    } else {
      payload.u = rhs.payload.u;
    }
    tag = rhs.tag;
    rhs.clearToNone();
  }

  /*
   Destructor
  */

  ~Value() {
    if (this->isTensor()) {
      payload.as_tensor.~vTensor();
    } else if (this->isStaging()) {
      payload.as_staging.~TensorStaging();
    }
  }

  /*
   Tensor
  */

  Value(vTensor&& t) : tag(TypeTag::TENSOR) {
    new (&payload.as_tensor) vTensor(std::move(t));
  }

  inline bool isTensor() const {
    return TypeTag::TENSOR == tag;
  }

  inline vTensor& toTensor() {
    if (!isTensor()) {
      throw 1;
    }
    return payload.as_tensor;
  }

  /*
   Staging
  */

  Value(TensorStaging&& t) : tag(TypeTag::STAGING) {
    new (&payload.as_staging) TensorStaging(std::move(t));
  }

  inline bool isStaging() const {
    return TypeTag::STAGING == tag;
  }

  inline TensorStaging& toStaging() {
    if (!isStaging()) {
      throw 1;
    }
    return payload.as_staging;
  }

 private:
  Payload payload;
  TypeTag tag;

  /*
   Utility Functions
  */

  inline void clearToNone() noexcept {
    payload.u.as_int = 0;
    tag = TypeTag::NONE;
  }
};

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
