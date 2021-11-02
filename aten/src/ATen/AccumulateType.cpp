#include <ATen/AccumulateType.h>

namespace at {

c10::ScalarType toAccumulateType(c10::ScalarType type, bool is_cuda) {
  switch (type) {
#define DEFINE_CASE(scalar_t, TypeNum)                                  \
    case ScalarType::TypeNum:                                           \
      return is_cuda ?                                                  \
          CppTypeToScalarType<at::acc_type<scalar_t, true>>::value :    \
          CppTypeToScalarType<at::acc_type<scalar_t, false>>::value;

    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(DEFINE_CASE)
#undef DEFINE_CASE

    default: TORCH_INTERNAL_ASSERT(false, "Unrecognized ScalarType: ", type);
  }
}

}
