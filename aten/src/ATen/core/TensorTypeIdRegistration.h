#pragma once

/**
 * To register your own tensor types, do in a header file:
 *   AT_DECLARE_TENSOR_TYPE(MY_TENSOR)
 * and in one (!) cpp file:
 *   AT_DEFINE_TENSOR_TYPE(MY_TENSOR)
 * Both must be in the same namespace.
 */

#include "ATen/core/TensorTypeId.h"
#include "c10/macros/Macros.h"

#include <atomic>
#include <mutex>
#include <unordered_set>

namespace at {

class CAFFE2_API TensorTypeIdCreator final {
 public:
  TensorTypeIdCreator();

  at::TensorTypeId create();

  static constexpr at::TensorTypeId undefined() noexcept {
    return TensorTypeId(0);
  }

 private:
  std::atomic<details::_tensorTypeId_underlyingType> last_id_;

  C10_DISABLE_COPY_AND_ASSIGN(TensorTypeIdCreator);
};

class CAFFE2_API TensorTypeIdRegistry final {
 public:
  TensorTypeIdRegistry();

  void registerId(at::TensorTypeId id);
  void deregisterId(at::TensorTypeId id);

 private:
  std::unordered_set<at::TensorTypeId> registeredTypeIds_;
  std::mutex mutex_;

  C10_DISABLE_COPY_AND_ASSIGN(TensorTypeIdRegistry);
};

class CAFFE2_API TensorTypeIds final {
 public:
  static TensorTypeIds& singleton();

  at::TensorTypeId createAndRegister();
  void deregister(at::TensorTypeId id);

  static constexpr at::TensorTypeId undefined() noexcept;

 private:
  TensorTypeIds();

  TensorTypeIdCreator creator_;
  TensorTypeIdRegistry registry_;

  C10_DISABLE_COPY_AND_ASSIGN(TensorTypeIds);
};

inline constexpr at::TensorTypeId TensorTypeIds::undefined() noexcept {
  return TensorTypeIdCreator::undefined();
}

class CAFFE2_API TensorTypeIdRegistrar final {
 public:
  TensorTypeIdRegistrar();
  ~TensorTypeIdRegistrar();

  at::TensorTypeId id() const noexcept;

 private:
  at::TensorTypeId id_;

  C10_DISABLE_COPY_AND_ASSIGN(TensorTypeIdRegistrar);
};

inline at::TensorTypeId TensorTypeIdRegistrar::id() const noexcept {
  return id_;
}

#define AT_DECLARE_TENSOR_TYPE(TensorName) \
  CAFFE2_API at::TensorTypeId TensorName()

#define AT_DEFINE_TENSOR_TYPE(TensorName)           \
  at::TensorTypeId TensorName() {                   \
    static TensorTypeIdRegistrar registration_raii; \
    return registration_raii.id();                  \
  }

AT_DECLARE_TENSOR_TYPE(UndefinedTensorId);
AT_DECLARE_TENSOR_TYPE(CPUTensorId); // PyTorch/Caffe2 supported
AT_DECLARE_TENSOR_TYPE(CUDATensorId); // PyTorch/Caffe2 supported
AT_DECLARE_TENSOR_TYPE(SparseCPUTensorId); // PyTorch only
AT_DECLARE_TENSOR_TYPE(SparseCUDATensorId); // PyTorch only
AT_DECLARE_TENSOR_TYPE(MKLDNNTensorId); // Caffe2 only
AT_DECLARE_TENSOR_TYPE(OpenGLTensorId); // Caffe2 only
AT_DECLARE_TENSOR_TYPE(OpenCLTensorId); // Caffe2 only
AT_DECLARE_TENSOR_TYPE(IDEEPTensorId); // Caffe2 only
AT_DECLARE_TENSOR_TYPE(HIPTensorId); // Caffe2 only

} // namespace at
