#include <c10/core/Scalar.h>
#include <ATen/core/TensorBody.h>

#include <string_view>

namespace at {

namespace {

// Verifies the requested type is the same as the Tensor's type.
void check_type(const TensorBase& tensor, ScalarType type) {
  TORCH_CHECK(
      tensor.scalar_type() == type
      || (isQIntType(tensor.scalar_type())
          && toUnderlying(tensor.scalar_type()) == type),
      "expected scalar type ", type, " but found ", tensor.scalar_type());
}

} // namespace

template <typename T>
const T* TensorBase::const_data_ptr() const {
  using NonConstT = std::remove_const_t<T>;
  check_type(*this, c10::CppTypeToScalarType<NonConstT>());
  return this->unsafeGetTensorImpl()->data_ptr_impl<NonConstT>();
}

template <typename T>
T* TensorBase::mutable_data_ptr() const {
  check_type(*this, c10::CppTypeToScalarType<T>());
  return this->unsafeGetTensorImpl()->mutable_data_ptr_impl<T>();
}

template <typename T>
T* TensorBase::data_ptr() const {
  return this->mutable_data_ptr<T>();
}

#define DEFINE_CAST(T, name)                                                \
   template TORCH_API const T* TensorBase::const_data_ptr<T>() const;       \
   template TORCH_API const T* TensorBase::const_data_ptr<const T>() const; \
   template TORCH_API T* TensorBase::mutable_data_ptr() const;              \
   template TORCH_API T* TensorBase::data_ptr() const;

 AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CAST)
 AT_FORALL_QINT_TYPES(DEFINE_CAST)
 DEFINE_CAST(uint16_t, UInt16)
 DEFINE_CAST(uint32_t, UInt32)
 DEFINE_CAST(uint64_t, UInt64)
 #undef DEFINE_CAST

 #define DEFINE_ITEM(T, name)      \
   template <>                     \
   TORCH_API T Tensor::item() const { \
     return item().to##name();     \
   }

 AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_ITEM)
 #undef DEFINE_ITEM

 } //namespace at
