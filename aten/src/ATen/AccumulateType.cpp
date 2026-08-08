#include <ATen/AccumulateType.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/util/Exception.h>

namespace at {

c10::ScalarType toAccumulateType(c10::ScalarType type, c10::DeviceType device) {
  switch (type) {
#define DEFINE_CASE(scalar_t, TypeNum)                                                             \
    case ScalarType::TypeNum:                                                                      \
      switch (device) {                                                                            \
        case DeviceType::CUDA:                                                                     \
          return CppTypeToScalarType<at::acc_type_device<scalar_t, c10::DeviceType::CUDA>>::value; \
        case DeviceType::PrivateUse1: {                                                            \
          if (at::isPrivateUse1HooksRegistered()) {                                               \
            if (auto acc = at::detail::getPrivateUse1Hooks().toAccumulateType(type)) {             \
              return *acc;                                                                         \
            }                                                                                      \
          }                                                                                        \
          TORCH_WARN_ONCE(                                                                         \
              "PrivateUse1 backend has not registered an accumulate-type mapping; ",               \
              "falling back to the CPU accumulation type. Override ",                              \
              "`toAccumulateType` in a `PrivateUse1HooksInterface` subclass to silence this.");   \
          return CppTypeToScalarType<at::acc_type_device<scalar_t, c10::DeviceType::CPU>>::value;  \
        }                                                                                          \
        case DeviceType::XPU:                                                                      \
          return CppTypeToScalarType<at::acc_type_device<scalar_t, c10::DeviceType::XPU>>::value;  \
        case DeviceType::MPS:                                                                      \
          return CppTypeToScalarType<at::acc_type_device<scalar_t, c10::DeviceType::MPS>>::value;  \
        default:                                                                                   \
          return CppTypeToScalarType<at::acc_type_device<scalar_t, c10::DeviceType::CPU>>::value;  \
      }

    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF_F8NZ(DEFINE_CASE)
#undef DEFINE_CASE

    default: TORCH_INTERNAL_ASSERT(false, "Unrecognized ScalarType: ", type);
  }
}

c10::ScalarType toAccumulateType(c10::ScalarType type, bool is_cuda) {
  return is_cuda ? toAccumulateType(type, c10::DeviceType::CUDA) : toAccumulateType(type, c10::DeviceType::CPU);
}

} // namespace at
