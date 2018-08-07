#pragma once

/**
 * To register your own tensor types, do in a header file:
 *   AT_DECLARE_TENSOR_TYPE(MY_TENSOR)
 * and in one (!) cpp file:
 *   AT_DEFINE_TENSOR_TYPE(MY_TENSOR)
 * Both must be in the same namespace.
 */

#include "ATen/core/Macros.h"
#include "ATen/core/TensorTypeId.h"

#include <atomic>
#include <unordered_set>

namespace at {

class TensorTypeIdCreator final {
 public:
  TensorTypeIdCreator();

  at::TensorTypeId create();

  static constexpr at::TensorTypeId undefined() noexcept {
    return TensorTypeId(0);
  }

 private:
  std::atomic<details::_tensorTypeId_underlyingType> last_id_;

  static constexpr at::TensorTypeId max_id_ = TensorTypeId(
      std::numeric_limits<details::_tensorTypeId_underlyingType>::max());

  AT_DISABLE_COPY_AND_ASSIGN(TensorTypeIdCreator);
};

class TensorTypeIdRegistry final {
 public:
  TensorTypeIdRegistry();

  void registerId(at::TensorTypeId id);
  void deregisterId(at::TensorTypeId id);

 private:
  std::unordered_set<at::TensorTypeId> registeredTypeIds_;
  std::mutex mutex_;

  AT_DISABLE_COPY_AND_ASSIGN(TensorTypeIdRegistry);
};

class TensorTypeIds final {
 public:
  static TensorTypeIds& singleton();

  at::TensorTypeId createAndRegister();
  void deregister(at::TensorTypeId id);

  static constexpr at::TensorTypeId undefined() noexcept;

 private:
  TensorTypeIds();

  TensorTypeIdCreator creator_;
  TensorTypeIdRegistry registry_;

  AT_DISABLE_COPY_AND_ASSIGN(TensorTypeIds);
};

inline constexpr at::TensorTypeId TensorTypeIds::undefined() noexcept {
  return TensorTypeIdCreator::undefined();
}

class TensorTypeIdRegistrar final {
 public:
  TensorTypeIdRegistrar();
  ~TensorTypeIdRegistrar();

  at::TensorTypeId id() const noexcept;

 private:
  at::TensorTypeId id_;

  AT_DISABLE_COPY_AND_ASSIGN(TensorTypeIdRegistrar);
};

inline at::TensorTypeId TensorTypeIdRegistrar::id() const noexcept {
  return id_;
}

} // namespace at

#define AT_DECLARE_TENSOR_TYPE(TensorName) at::TensorTypeId TensorName();

#define AT_DEFINE_TENSOR_TYPE(TensorName)           \
  at::TensorTypeId TensorName() {                   \
    static TensorTypeIdRegistrar registration_raii; \
    return registration_raii.id();                  \
  }
