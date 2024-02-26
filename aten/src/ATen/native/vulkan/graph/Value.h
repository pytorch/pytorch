#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/api/Tensor.h>

#include <ATen/native/vulkan/graph/Constant.h>
#include <ATen/native/vulkan/graph/Types.h>

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
    api::StorageBuffer as_staging;
    TensorRef as_tensorref;

    Payload() : u() {}
    // NOLINTNEXTLINE
    ~Payload(){};
  };

 public:
  //
  // Copy constructor and assignment (disabled)
  //

  Value(const Value& rhs) = delete;
  Value& operator=(const Value&) = delete;

  //
  // Move constructor and assignment; Move assignment is disabled but
  // construction is implemented to allow for use in container types.
  //

  Value& operator=(Value&&) = delete;

  Value(Value&& rhs) noexcept : tag(rhs.tag) {
    if (rhs.isTensor()) {
      new (&payload.as_tensor) vTensor(std::move(rhs.payload.as_tensor));
    } else if (rhs.isStaging()) {
      new (&payload.as_staging)
          api::StorageBuffer(std::move(rhs.payload.as_staging));
    } else if (rhs.isTensorRef()) {
      payload.as_tensorref = std::move(rhs.payload.as_tensorref);
    } else {
      payload.u = rhs.payload.u;
    }
    tag = rhs.tag;
    rhs.clearToNone();
  }

  //
  // Accessors
  //

  inline TypeTag type() const {
    return tag;
  }

  //
  // Destructor
  //

  ~Value() {
    if (this->isTensor()) {
      payload.as_tensor.~vTensor();
    } else if (this->isStaging()) {
      payload.as_staging.~StorageBuffer();
    } else if (this->isTensorRef()) {
      payload.as_tensorref.~TensorRef();
    }
  }

  //
  // Tensor
  //

  explicit Value(vTensor&& t) : tag(TypeTag::TENSOR) {
    new (&payload.as_tensor) vTensor(std::move(t));
  }

  inline bool isTensor() const {
    return TypeTag::TENSOR == tag;
  }

  inline vTensor& toTensor() {
    VK_CHECK_COND(
        isTensor(),
        "Expected value to have type TENSOR, got ",
        tag,
        " instead.");
    return payload.as_tensor;
  }

  //
  // Staging
  //

  explicit Value(api::StorageBuffer&& t) : tag(TypeTag::STAGING) {
    new (&payload.as_staging) api::StorageBuffer(std::move(t));
  }

  inline bool isStaging() const {
    return TypeTag::STAGING == tag;
  }

  inline api::StorageBuffer& toStaging() {
    VK_CHECK_COND(
        isStaging(),
        "Expected value to have type STAGING, got ",
        tag,
        " instead.");
    return payload.as_staging;
  }

  //
  // TensorRef
  //

  explicit Value(TensorRef&& t) : tag(TypeTag::TENSORREF) {
    payload.as_tensorref = std::move(t);
  }

  inline bool isTensorRef() const {
    return TypeTag::TENSORREF == tag;
  }

  inline TensorRef& toTensorRef() {
    VK_CHECK_COND(
        isTensorRef(),
        "Expected value to have type TENSORREF, got ",
        tag,
        " instead.");
    return payload.as_tensorref;
  }

 private:
  Payload payload;
  TypeTag tag;

  //
  // Utility Functions
  //

  inline void clearToNone() noexcept {
    payload.u.as_int = 0;
    tag = TypeTag::NONE;
  }
};

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
