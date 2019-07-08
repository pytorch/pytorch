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
#include <c10/util/flat_hash_map.h>

#include <atomic>
#include <mutex>
#include <unordered_set>

namespace c10 {

class C10_API TensorTypeIdCreator final {
 public:
  TensorTypeIdCreator();

  TensorTypeId create();

  static constexpr TensorTypeId undefined() noexcept {
    return TensorTypeId(0);
  }

 private:
  std::atomic<details::_tensorTypeId_underlyingType> last_id_;

  C10_DISABLE_COPY_AND_ASSIGN(TensorTypeIdCreator);
};

class C10_API TensorTypeIdRegistry final {
 public:
  TensorTypeIdRegistry();

  void registerId(TensorTypeId id, std::string name);
  void deregisterId(TensorTypeId id);

  const std::string& toString(TensorTypeId id) const;

 private:
  using TypeIdName = std::string;
  ska::flat_hash_map<TensorTypeId, TypeIdName> registeredTypeIds_;
  mutable std::mutex mutex_;

  C10_DISABLE_COPY_AND_ASSIGN(TensorTypeIdRegistry);
};

class C10_API TensorTypeIds final {
 public:
  static TensorTypeIds& singleton();

  TensorTypeId createAndRegister(std::string name);
  void deregister(TensorTypeId id);

  const std::string& toString(TensorTypeId id) const;

  static constexpr TensorTypeId undefined() noexcept;

 private:
  TensorTypeIds();

  TensorTypeIdCreator creator_;
  TensorTypeIdRegistry registry_;

  C10_DISABLE_COPY_AND_ASSIGN(TensorTypeIds);
};

inline constexpr TensorTypeId TensorTypeIds::undefined() noexcept {
  return TensorTypeIdCreator::undefined();
}

class C10_API TensorTypeIdRegistrar final {
 public:
  explicit TensorTypeIdRegistrar(std::string name);
  ~TensorTypeIdRegistrar();

  TensorTypeId id() const noexcept;

 private:
  TensorTypeId id_;

  C10_DISABLE_COPY_AND_ASSIGN(TensorTypeIdRegistrar);
};

inline TensorTypeId TensorTypeIdRegistrar::id() const noexcept {
  return id_;
}

C10_API std::string toString(TensorTypeId id);
C10_API std::ostream& operator<<(std::ostream&, c10::TensorTypeId);

#define C10_DECLARE_TENSOR_TYPE(TensorName)                             \
  C10_API ::c10::TensorTypeId TensorName()

#define C10_DEFINE_TENSOR_TYPE(TensorName)                              \
  C10_EXPORT ::c10::TensorTypeId TensorName() {                         \
    static ::c10::TensorTypeIdRegistrar registration_raii(#TensorName); \
    return registration_raii.id();                                      \
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
C10_DECLARE_TENSOR_TYPE(MkldnnCPUTensorId);
C10_DECLARE_TENSOR_TYPE(QuantizedCPUTensorId); // PyTorch only
C10_DECLARE_TENSOR_TYPE(ComplexCPUTensorId); // PyTorch only
C10_DECLARE_TENSOR_TYPE(ComplexCUDATensorId); // PyTorch only

} // namespace c10

#endif // TENSOR_TYPE_ID_REGISTRATION_H_
