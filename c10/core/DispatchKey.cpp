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
    case DispatchKey::XPU:
      return "XPU";
    case DispatchKey::MSNPU:
      return "MSNPU";
    case DispatchKey::XLA:
      return "XLA";
    case DispatchKey::MLC:
      return "MLC";
    case DispatchKey::Vulkan:
      return "Vulkan";
    case DispatchKey::Metal:
      return "Metal";
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
    case DispatchKey::QuantizedXPU:
      return "QuantizedXPU";

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
    case DispatchKey::SparseXPU:
      return "SparseXPU";

    case DispatchKey::NestedTensor:
      return "NestedTensor";

    case DispatchKey::PrivateUse1:
      return "PrivateUse1";
    case DispatchKey::PrivateUse2:
      return "PrivateUse2";
    case DispatchKey::PrivateUse3:
      return "PrivateUse3";

    case DispatchKey::Meta:
      return "Meta";

    case DispatchKey::InplaceOrView:
      return "InplaceOrView";

    case DispatchKey::Autograd:
      return "Autograd";
    case DispatchKey::AutogradCPU:
      return "AutogradCPU";
    case DispatchKey::AutogradCUDA:
      return "AutogradCUDA";
    case DispatchKey::AutogradXLA:
      return "AutogradXLA";
    case DispatchKey::AutogradMLC:
      return "AutogradMLC";
    case DispatchKey::AutogradNestedTensor:
      return "AutogradNestedTensor";
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

    case DispatchKey::CompositeImplicitAutograd:
      return "CompositeImplicitAutograd";

    case DispatchKey::CompositeExplicitAutograd:
      return "CompositeExplicitAutograd";

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

// for a given backend key, return the associated autograd key.
// for non-backend keys, return AutogradOther as a default.
// Note: it's convenient and fast to return a default here rather than (say)
// returning an optional<DispatchKey>, or throwing. But it makes callers
// responsible for either a) enforcing the invariant that only backend keys
// be passed as arguments, or b) interpreting our return value carefully.
//
DispatchKey getAutogradKeyFromBackend(DispatchKey t) {
  switch (t) {
    case DispatchKey::CPU:
      return DispatchKey::AutogradCPU;
    case DispatchKey::CUDA:
      return DispatchKey::AutogradCUDA;
    case DispatchKey::XLA:
      return DispatchKey::AutogradXLA;
    case DispatchKey::MLC:
      return DispatchKey::AutogradMLC;
    case DispatchKey::NestedTensor:
      return DispatchKey::AutogradNestedTensor;
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
