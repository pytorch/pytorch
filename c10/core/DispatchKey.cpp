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

    case DispatchKey::Autograd:
      return "Autograd";
    case DispatchKey::AutogradCPU:
      return "AutogradCPU";
    case DispatchKey::AutogradCUDA:
      return "AutogradCUDA";
    case DispatchKey::AutogradXLA:
      return "AutogradXLA";
    case DispatchKey::AutogradPrivateUse1:
      return "AutogradPrivateUse1";
    case DispatchKey::AutogradPrivateUse2:
      return "AutogradPrivateUse2";
    case DispatchKey::AutogradPrivateUse3:
      return "AutogradPrivateUse3";
    case DispatchKey::AutogradOther:
      return "AutogradOther";
    case DispatchKey::BackendSelect:
      return "BackendSelect";
    case DispatchKey::Named:
      return "Named";

    case DispatchKey::Tracer:
      return "Tracer";

    case DispatchKey::Autocast:
      return "Autocast";

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

DispatchKey getBackendKeyFromAutograd(DispatchKey t) {
  switch (t) {
    case DispatchKey::AutogradCPU:
      return DispatchKey::CPU;
    case DispatchKey::AutogradCUDA:
      return DispatchKey::CUDA;
    case DispatchKey::AutogradXLA:
      return DispatchKey::XLA;
    case DispatchKey::AutogradPrivateUse1:
      return DispatchKey::PrivateUse1;
    case DispatchKey::AutogradPrivateUse2:
      return DispatchKey::PrivateUse2;
    case DispatchKey::AutogradPrivateUse3:
      return DispatchKey::PrivateUse3;
    default:
      return DispatchKey::Undefined;
  }
}


} // namespace c10
