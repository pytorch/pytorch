#ifndef TENSOR_TYPE_ID_REGISTRATION_H_
#define TENSOR_TYPE_ID_REGISTRATION_H_

/**
 * To register your own tensor types, do in a header file:
 *   C10_DECLARE_TENSOR_TYPE(MY_TENSOR)
 * and in one (!) cpp file:
 *   C10_DEFINE_TENSOR_TYPE(MY_TENSOR)
 * Both must be in the same namespace.
 */

#include <c10/macros/Macros.h>
#include <c10/core/TensorTypeId.h>

#include <atomic>
#include <mutex>
#include <unordered_set>

namespace c10 {

class C10_API TensorTypeIdCreator final {
 public:
  TensorTypeIdCreator();

  c10::TensorTypeId create();

  static constexpr c10::TensorTypeId undefined() noexcept {
    return c10::TensorTypeId(0);
  }

 private:
  std::atomic<details::_tensorTypeId_underlyingType> last_id_;

  C10_DISABLE_COPY_AND_ASSIGN(TensorTypeIdCreator);
};

class C10_API TensorTypeIdRegistry final {
 public:
  TensorTypeIdRegistry();

  void registerId(c10::TensorTypeId id);
  void deregisterId(c10::TensorTypeId id);

 private:
  std::unordered_set<c10::TensorTypeId> registeredTypeIds_;
  std::mutex mutex_;

  C10_DISABLE_COPY_AND_ASSIGN(TensorTypeIdRegistry);
};

class C10_API TensorTypeIds final {
 public:
  static TensorTypeIds& singleton();

  c10::TensorTypeId createAndRegister();
  void deregister(c10::TensorTypeId id);

  static constexpr c10::TensorTypeId undefined() noexcept;

 private:
  TensorTypeIds();

  TensorTypeIdCreator creator_;
  TensorTypeIdRegistry registry_;

  C10_DISABLE_COPY_AND_ASSIGN(TensorTypeIds);
};

inline constexpr c10::TensorTypeId TensorTypeIds::undefined() noexcept {
  return TensorTypeIdCreator::undefined();
}

class C10_API TensorTypeIdRegistrar final {
 public:
  TensorTypeIdRegistrar();
  ~TensorTypeIdRegistrar();

  c10::TensorTypeId id() const noexcept;

 private:
  c10::TensorTypeId id_;

  C10_DISABLE_COPY_AND_ASSIGN(TensorTypeIdRegistrar);
};

inline c10::TensorTypeId TensorTypeIdRegistrar::id() const noexcept {
  return id_;
}

#define C10_DECLARE_TENSOR_TYPE(TensorName) \
  C10_API c10::TensorTypeId TensorName()

#define C10_DEFINE_TENSOR_TYPE(TensorName)          \
  c10::TensorTypeId TensorName() {                  \
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
C10_DECLARE_TENSOR_TYPE(HIPTensorId); // PyTorch/Caffe2 supported
C10_DECLARE_TENSOR_TYPE(SparseHIPTensorId); // PyTorch only
C10_DECLARE_TENSOR_TYPE(MSNPUTensorId); // PyTorch only
C10_DECLARE_TENSOR_TYPE(XLATensorId); // PyTorch only

} // namespace c10

#endif // TENSOR_TYPE_ID_REGISTRATION_H_
