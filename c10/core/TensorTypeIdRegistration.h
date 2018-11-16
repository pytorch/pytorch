#pragma once

/**
 * To register your own tensor types, do in a header file:
 *   C10_DECLARE_TENSOR_TYPE(MY_TENSOR)
 * and in one (!) cpp file:
 *   C10_DEFINE_TENSOR_TYPE(MY_TENSOR)
 * Both must be in the same namespace.
 */

#include <c10/core/TensorTypeId.h>
#include <c10/macros/Macros.h>

#include <atomic>
#include <mutex>
#include <unordered_set>

namespace c10 {

class C10_API TensorTypeIdCreator final {
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

class C10_API TensorTypeIdRegistry final {
 public:
  TensorTypeIdRegistry();

  void registerId(at::TensorTypeId id);
  void deregisterId(at::TensorTypeId id);

 private:
  std::unordered_set<at::TensorTypeId> registeredTypeIds_;
  std::mutex mutex_;

  C10_DISABLE_COPY_AND_ASSIGN(TensorTypeIdRegistry);
};

class C10_API TensorTypeIds final {
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

class C10_API TensorTypeIdRegistrar final {
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

#define C10_DECLARE_TENSOR_TYPE(TensorName) \
  C10_API at::TensorTypeId TensorName()

#define C10_DEFINE_TENSOR_TYPE(TensorName)           \
  at::TensorTypeId TensorName() {                   \
    static TensorTypeIdRegistrar registration_raii; \
    return registration_raii.id();                  \
  }

C10_DECLARE_TENSOR_TYPE(UndefinedTensorId);
C10_DECLARE_TENSOR_TYPE(CPUTensorId); // PyTorch/Caffe2 supported
C10_DECLARE_TENSOR_TYPE(CUDATensorId); // PyTorch/Caffe2 supported
C10_DECLARE_TENSOR_TYPE(SparseCPUTensorId); // PyTorch only
C10_DECLARE_TENSOR_TYPE(SparseCUDATensorId); // PyTorch only
C10_DECLARE_TENSOR_TYPE(MKLDNNTensorId); // Caffe2 only
C10_DECLARE_TENSOR_TYPE(OpenGLTensorId); // Caffe2 only
C10_DECLARE_TENSOR_TYPE(OpenCLTensorId); // Caffe2 only
C10_DECLARE_TENSOR_TYPE(IDEEPTensorId); // Caffe2 only
C10_DECLARE_TENSOR_TYPE(HIPTensorId); // Caffe2 only

} // namespace c10
