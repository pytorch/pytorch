#include <c10/core/Scalar.h>
#include <ATen/core/TensorBody.h>

namespace at {

#define DEFINE_CAST(T, name)                                         \
   template <>                                                       \
   TORCH_API T* TensorBase::data_ptr() const {                       \
     TORCH_CHECK(                                                    \
         scalar_type() == ScalarType::name                           \
         || (isQIntType(scalar_type())                               \
         && toUnderlying(scalar_type()) == ScalarType::name),        \
         "expected scalar type "                                     \
         #name                                                       \
         " but found ",                                              \
         scalar_type());                                             \
     return this->unsafeGetTensorImpl()->data_ptr_impl<T>();         \
   }

 AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CAST)
 AT_FORALL_QINT_TYPES(DEFINE_CAST)
 #undef DEFINE_CAST

 #define DEFINE_ITEM(T, name)      \
   template <>                     \
   TORCH_API T Tensor::item() const { \
     return item().to##name();     \
   }

 AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_ITEM)
 #undef DEFINE_ITEM

 } //namespace at
