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

class AT_CORE_API TensorTypeIdCreator final {
 public:
  TensorTypeIdCreator();

  at::TensorTypeId create();

  static constexpr at::TensorTypeId undefined() noexcept {
    return TensorTypeId(0);
  }

 private:
  std::atomic<details::_tensorTypeId_underlyingType> last_id_;

  AT_DISABLE_COPY_AND_ASSIGN(TensorTypeIdCreator);
};

class AT_CORE_API TensorTypeIdRegistry final {
 public:
  TensorTypeIdRegistry();

  void registerId(at::TensorTypeId id);
  void deregisterId(at::TensorTypeId id);

 private:
  std::unordered_set<at::TensorTypeId> registeredTypeIds_;
  std::mutex mutex_;

  AT_DISABLE_COPY_AND_ASSIGN(TensorTypeIdRegistry);
};

class AT_CORE_API TensorTypeIds final {
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

class AT_CORE_API TensorTypeIdRegistrar final {
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

#define AT_DECLARE_TENSOR_TYPE(TensorName) AT_CORE_API at::TensorTypeId TensorName();

#define AT_DEFINE_TENSOR_TYPE(TensorName)           \
  at::TensorTypeId TensorName() {                   \
    static TensorTypeIdRegistrar registration_raii; \
    return registration_raii.id();                  \
  }

AT_DECLARE_TENSOR_TYPE(UndefinedTensorId);
AT_DECLARE_TENSOR_TYPE(CPUTensorId); // Caffe2 supported
AT_DECLARE_TENSOR_TYPE(CUDATensorId); // Caffe2 supported
AT_DECLARE_TENSOR_TYPE(SparseCPUTensorId);
AT_DECLARE_TENSOR_TYPE(SparseCUDATensorId);

} // namespace at
