#include <c10/core/DispatchKey.h>

namespace c10 {

const char* toString(DispatchKey t) {
  switch (t) {
    case DispatchKey::Undefined:
      return "Undefined";

    case DispatchKey::CPU:
      return "CPU";
    case DispatchKey::CUDA:
      return "CUDA";
    case DispatchKey::HIP:
      return "HIP";
    case DispatchKey::FPGA:
      return "FPGA";
    case DispatchKey::MSNPU:
      return "MSNPU";
    case DispatchKey::XLA:
      return "XLA";
    case DispatchKey::Vulkan:
      return "Vulkan";

    case DispatchKey::MKLDNN:
      return "MKLDNN";
    case DispatchKey::OpenGL:
      return "OpenGL";
    case DispatchKey::OpenCL:
      return "OpenCL";
    case DispatchKey::IDEEP:
      return "IDEEP";

    case DispatchKey::QuantizedCPU:
      return "QuantizedCPU";
    case DispatchKey::QuantizedCUDA:
      return "QuantizedCUDA";

    case DispatchKey::ComplexCPU:
      return "ComplexCPU";
    case DispatchKey::ComplexCUDA:
      return "ComplexCUDA";

    case DispatchKey::CustomRNGKeyId:
      return "CustomRNGKeyId";

    case DispatchKey::MkldnnCPU:
      return "MkldnnCPU";
    case DispatchKey::SparseCPU:
      return "SparseCPU";
    case DispatchKey::SparseCUDA:
      return "SparseCUDA";
    case DispatchKey::SparseHIP:
      return "SparseHIP";

    case DispatchKey::PrivateUse1:
      return "PrivateUse1";
    case DispatchKey::PrivateUse2:
      return "PrivateUse2";
    case DispatchKey::PrivateUse3:
      return "PrivateUse3";

    case DispatchKey::Meta:
      return "Meta";

    case DispatchKey::BackendSelect:
      return "BackendSelect";
    case DispatchKey::Named:
      return "Named";

    case DispatchKey::Conjugate:
      return "Conjugate";

    case DispatchKey::Autograd:
      return "Autograd";

    case DispatchKey::Tracer:
      return "Tracer";

    case DispatchKey::AutogradXLA:
      return "AutogradXLA";

    case DispatchKey::Autocast:
      return "Autocast";

    case DispatchKey::PrivateUse1_PreAutograd:
      return "PrivateUse1_PreAutograd";
    case DispatchKey::PrivateUse2_PreAutograd:
      return "PrivateUse2_PreAutograd";
    case DispatchKey::PrivateUse3_PreAutograd:
      return "PrivateUse3_PreAutograd";

    case DispatchKey::Batched:
      return "Batched";

    case DispatchKey::VmapMode:
      return "VmapMode";

    case DispatchKey::TESTING_ONLY_GenericWrapper:
      return "TESTING_ONLY_GenericWrapper";

    case DispatchKey::TESTING_ONLY_GenericMode:
      return "TESTING_ONLY_GenericMode";

    default:
      return "UNKNOWN_TENSOR_TYPE_ID";
  }
}

std::ostream& operator<<(std::ostream& str, DispatchKey rhs) {
  return str << toString(rhs);
}

} // namespace c10
