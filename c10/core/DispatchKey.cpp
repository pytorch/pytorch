#include <c10/core/DispatchKey.h>

namespace c10 {
#define FOR_EACH_DISPATCH_KEY(_) \
  _(Undefined)                   \
  _(CPU)                         \
  _(CUDA)                        \
  _(HIP)                         \
  _(FPGA)                        \
  _(MSNPU)                       \
  _(XLA)                         \
  _(Vulkan)                      \
  _(MKLDNN)                      \
  _(OpenGL)                      \
  _(OpenCL)                      \
  _(IDEEP)                       \
  _(QuantizedCPU)                \
  _(QuantizedCUDA)               \
  _(ComplexCPU)                  \
  _(ComplexCUDA)                 \
  _(CustomRNGKeyId)              \
  _(MkldnnCPU)                   \
  _(SparseCPU)                   \
  _(SparseCUDA)                  \
  _(SparseHIP)                   \
  _(PrivateUse1)                 \
  _(PrivateUse2)                 \
  _(PrivateUse3)                 \
  _(Meta)                        \
  _(Autograd)                    \
  _(AutogradOther)               \
  _(AutogradCPU)                 \
  _(AutogradCUDA)                \
  _(AutogradXLA)                 \
  _(AutogradPrivateUse1)         \
  _(AutogradPrivateUse2)         \
  _(AutogradPrivateUse3)         \
  _(BackendSelect)               \
  _(Named)                       \
  _(Tracer)                      \
  _(Autocast)                    \
  _(Batched)                     \
  _(VmapMode)                    \
  _(Math)                        \
  _(TESTING_ONLY_GenericWrapper) \
  _(TESTING_ONLY_GenericMode)

const char* toString(DispatchKey t) {
#define DEFINE_CASE(dk) \
  case DispatchKey::dk: \
    return #dk;

  switch (t) {
    FOR_EACH_DISPATCH_KEY(DEFINE_CASE)
    default:
      return "UNKNOWN_TENSOR_TYPE_ID";
  }
#undef DEFINE_CASE
}

std::ostream& operator<<(std::ostream& str, DispatchKey rhs) {
  return str << toString(rhs);
}

DispatchKey getAutogradKeyFromBackend(DispatchKey t) {
  switch (t) {
    case DispatchKey::CPU:
      return DispatchKey::AutogradCPU;
    case DispatchKey::CUDA:
      return DispatchKey::AutogradCUDA;
    case DispatchKey::XLA:
      return DispatchKey::AutogradXLA;
    case DispatchKey::PrivateUse1:
      return DispatchKey::AutogradPrivateUse1;
    case DispatchKey::PrivateUse2:
      return DispatchKey::AutogradPrivateUse2;
    case DispatchKey::PrivateUse3:
      return DispatchKey::AutogradPrivateUse3;
    default:
      return DispatchKey::AutogradOther;
  }
}

} // namespace c10
