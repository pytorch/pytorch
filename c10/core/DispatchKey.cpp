#include <c10/core/DispatchKey.h>

#include <unordered_map>

namespace c10 {

const char* toString(BackendBit t) {
  switch (t) {
    case BackendBit::CPUBit:
      return "CPUBit";
    case BackendBit::CUDABit:
      return "CUDABit";
    case BackendBit::HIPBit:
      return "HIPBit";
    case BackendBit::XLABit:
      return "XLABit";
    case BackendBit::LazyBit:
      return "LazyBit";
    case BackendBit::XPUBit:
      return "XPUBit";
    case BackendBit::NestedTensorBit:
      return "NestedTensorBit";
    case BackendBit::MLCBit:
      return "MLCBit";
    case BackendBit::HPUBit:
      return "HPUBit";
    case BackendBit::VEBit:
      return "VEBit";
    case BackendBit::PrivateUse1Bit:
      return "PrivateUse1Bit";
    case BackendBit::PrivateUse2Bit:
      return "PrivateUse2Bit";
    case BackendBit::PrivateUse3Bit:
      return "PrivateUse3Bit";
    case BackendBit::InvalidBit:
      return "InvalidBit";
    default:
      return "UNKNOWN_BACKEND_BIT";
  }
}

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
    case DispatchKey::VE:
      return "VE";
    case DispatchKey::FPGA:
      return "FPGA";
    case DispatchKey::XPU:
      return "XPU";
    case DispatchKey::ORT:
      return "ORT";
    case DispatchKey::XLA:
      return "XLA";
    case DispatchKey::Lazy:
      return "Lazy";
    case DispatchKey::MLC:
      return "MLC";
    case DispatchKey::HPU:
      return "HPU";
    case DispatchKey::Vulkan:
      return "Vulkan";
    case DispatchKey::Metal:
      return "Metal";
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
    case DispatchKey::SparseCsrCPU:
      return "SparseCsrCPU";
    case DispatchKey::SparseCsrCUDA:
      return "SparseCsrCUDA";
    case DispatchKey::SparseHIP:
      return "SparseHIP";
    case DispatchKey::SparseVE:
      return "SparseVE";
    case DispatchKey::SparseXPU:
      return "SparseXPU";

    case DispatchKey::NestedTensor:
      return "NestedTensor";

    case DispatchKey::Python:
      return "Python";

    case DispatchKey::PrivateUse1:
      return "PrivateUse1";
    case DispatchKey::PrivateUse2:
      return "PrivateUse2";
    case DispatchKey::PrivateUse3:
      return "PrivateUse3";

    case DispatchKey::Negative:
      return "Negative";
    case DispatchKey::Conjugate:
      return "Conjugate";
    case DispatchKey::Meta:
      return "Meta";

    case DispatchKey::ADInplaceOrView:
      return "ADInplaceOrView";

    case DispatchKey::Autograd:
      return "Autograd";
    case DispatchKey::AutogradCPU:
      return "AutogradCPU";
    case DispatchKey::AutogradXPU:
      return "AutogradXPU";
    case DispatchKey::AutogradCUDA:
      return "AutogradCUDA";
    case DispatchKey::AutogradXLA:
      return "AutogradXLA";
    case DispatchKey::AutogradLazy:
      return "AutogradLazy";
    case DispatchKey::AutogradMLC:
      return "AutogradMLC";
    case DispatchKey::AutogradHPU:
      return "AutogradHPU";
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

    case DispatchKey::ZeroTensor:
      return "ZeroTensor";
    case DispatchKey::BackendSelect:
      return "BackendSelect";
    case DispatchKey::Named:
      return "Named";

    case DispatchKey::Functionalize:
      return "Functionalize";

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

    // Note [Out-of-tree vmap+grad prototype]
    // The following keys are used in the implementation of the out-of-tree
    // composable functions transforms (vmap+grad) prototype that lives at
    // https://github.com/zou3519/functorch
    // We plan on eventually upstreaming the prototype into core, at which
    // point it will have a different design that should use fewer keys.
    case DispatchKey::FuncTorchDynamicLayerBackMode:
      return "FuncTorchDynamicLayerBackMode";
    case DispatchKey::FuncTorchDynamicLayerFrontMode:
      return "FuncTorchDynamicLayerFrontMode";
    case DispatchKey::FuncTorchGradWrapper:
      return "FuncTorchGradWrapper";
    case DispatchKey::FuncTorchVmapMode:
      return "FuncTorchVmapMode";
    case DispatchKey::FuncTorchBatched:
      return "FuncTorchBatched";

    case DispatchKey::Dense:
      return "Dense";
    case DispatchKey::Quantized:
      return "Quantized";
    case DispatchKey::Sparse:
      return "Sparse";
    case DispatchKey::AutogradFunctionality:
      return "AutogradFunctionality";

    default:
      return "UNKNOWN_TENSOR_TYPE_ID";
  }
}

std::ostream& operator<<(std::ostream& str, DispatchKey rhs) {
  return str << toString(rhs);
}
std::ostream& operator<<(std::ostream& str, BackendBit rhs) {
  return str << toString(rhs);
}

// for a given backend key, return the associated autograd key.
// for non-backend keys, return AutogradOther as a default.
// Note: it's convenient and fast to return a default here rather than (say)
// returning an optional<DispatchKey>, or throwing. But it makes callers
// responsible for either a) enforcing the invariant that only backend keys
// be passed as arguments, or b) interpreting our return value carefully.
//
DispatchKey getAutogradKeyFromBackend(BackendBit t) {
  switch (t) {
    case BackendBit::CPUBit:
      return DispatchKey::AutogradCPU;
    case BackendBit::XPUBit:
      return DispatchKey::AutogradXPU;
    case BackendBit::CUDABit:
      return DispatchKey::AutogradCUDA;
    case BackendBit::XLABit:
      return DispatchKey::AutogradXLA;
    case BackendBit::LazyBit:
      return DispatchKey::AutogradLazy;
    case BackendBit::MLCBit:
      return DispatchKey::AutogradMLC;
    case BackendBit::HPUBit:
      return DispatchKey::AutogradHPU;
    case BackendBit::NestedTensorBit:
      return DispatchKey::AutogradNestedTensor;
    case BackendBit::PrivateUse1Bit:
      return DispatchKey::AutogradPrivateUse1;
    case BackendBit::PrivateUse2Bit:
      return DispatchKey::AutogradPrivateUse2;
    case BackendBit::PrivateUse3Bit:
      return DispatchKey::AutogradPrivateUse3;
    default:
      return DispatchKey::AutogradOther;
  }
}

c10::DispatchKey parseDispatchKey(const std::string& k) {
  static std::unordered_map<std::string, c10::DispatchKey> key_map = {
      {"XPU", c10::DispatchKey::XPU},
      {"Undefined", c10::DispatchKey::Undefined},
      {"Dense", c10::DispatchKey::Dense},
      {"FPGA", c10::DispatchKey::FPGA},
      {"ORT", c10::DispatchKey::ORT},
      {"Vulkan", c10::DispatchKey::Vulkan},
      {"Metal", c10::DispatchKey::Metal},
      {"VE", c10::DispatchKey::VE},
      {"Meta", c10::DispatchKey::Meta},
      {"Quantized", c10::DispatchKey::Quantized},
      {"CustomRNGKeyId", c10::DispatchKey::CustomRNGKeyId},
      {"MkldnnCPU", c10::DispatchKey::MkldnnCPU},
      {"Sparse", c10::DispatchKey::Sparse},
      {"SparseCsrCPU", c10::DispatchKey::SparseCsrCPU},
      {"SparseCsrCUDA", c10::DispatchKey::SparseCsrCUDA},
      {"BackendSelect", c10::DispatchKey::BackendSelect},
      {"Python", c10::DispatchKey::Python},
      {"Named", c10::DispatchKey::Named},
      {"Conjugate", c10::DispatchKey::Conjugate},
      {"Negative", c10::DispatchKey::Negative},
      {"ZeroTensor", c10::DispatchKey::ZeroTensor},
      {"FuncTorchDynamicLayerBackMode",
       c10::DispatchKey::FuncTorchDynamicLayerBackMode},
      {"ADInplaceOrView", c10::DispatchKey::ADInplaceOrView},
      {"AutogradOther", c10::DispatchKey::AutogradOther},
      {"AutogradFunctionality", c10::DispatchKey::AutogradFunctionality},
      {"Tracer", c10::DispatchKey::Tracer},
      {"AutocastCPU", c10::DispatchKey::AutocastCPU},
      {"AutocastCUDA", c10::DispatchKey::AutocastCUDA},
      {"FuncTorchBatched", c10::DispatchKey::FuncTorchBatched},
      {"FuncTorchVmapMode", c10::DispatchKey::FuncTorchVmapMode},
      {"Batched", c10::DispatchKey::Batched},
      {"VmapMode", c10::DispatchKey::VmapMode},
      {"FuncTorchGradWrapper", c10::DispatchKey::FuncTorchGradWrapper},
      {"FuncTorchDynamicLayerFrontMode",
       c10::DispatchKey::FuncTorchDynamicLayerFrontMode},
      {"TESTING_ONLY_GenericWrapper",
       c10::DispatchKey::TESTING_ONLY_GenericWrapper},
      {"TESTING_ONLY_GenericMode", c10::DispatchKey::TESTING_ONLY_GenericMode},

      {"CPU", c10::DispatchKey::CPU},
      {"CUDA", c10::DispatchKey::CUDA},
      {"HIP", c10::DispatchKey::HIP},
      {"XLA", c10::DispatchKey::XLA},
      {"MLC", c10::DispatchKey::MLC},
      {"XPU", c10::DispatchKey::XPU},
      {"HPU", c10::DispatchKey::HPU},
      {"Lazy", c10::DispatchKey::Lazy},
      {"NestedTensor", c10::DispatchKey::NestedTensor},
      {"PrivateUse1", c10::DispatchKey::PrivateUse1},
      {"PrivateUse2", c10::DispatchKey::PrivateUse2},
      {"PrivateUse3", c10::DispatchKey::PrivateUse3},

      {"QuantizedCPU", c10::DispatchKey::QuantizedCPU},
      {"QuantizedCUDA", c10::DispatchKey::QuantizedCUDA},
      {"QuantizedXPU", c10::DispatchKey::QuantizedXPU},

      {"SparseCPU", c10::DispatchKey::SparseCPU},
      {"SparseCUDA", c10::DispatchKey::SparseCUDA},
      {"SparseHIP", c10::DispatchKey::SparseHIP},
      {"SparseXPU", c10::DispatchKey::SparseXPU},
      {"SparseVE", c10::DispatchKey::SparseVE},

      {"AutogradCPU", c10::DispatchKey::AutogradCPU},
      {"AutogradCUDA", c10::DispatchKey::AutogradCUDA},
      {"AutogradXLA", c10::DispatchKey::AutogradXLA},
      {"AutogradLazy", c10::DispatchKey::AutogradLazy},
      {"AutogradXPU", c10::DispatchKey::AutogradXPU},
      {"AutogradMLC", c10::DispatchKey::AutogradMLC},
      {"AutogradHPU", c10::DispatchKey::AutogradHPU},
      {"AutogradNestedTensor", c10::DispatchKey::AutogradNestedTensor},
      {"AutogradPrivateUse1", c10::DispatchKey::AutogradPrivateUse1},
      {"AutogradPrivateUse2", c10::DispatchKey::AutogradPrivateUse2},
      {"AutogradPrivateUse3", c10::DispatchKey::AutogradPrivateUse3},

      {"Autograd", c10::DispatchKey::Autograd},
      {"CompositeImplicitAutograd",
       c10::DispatchKey::CompositeImplicitAutograd},
      {"CompositeExplicitAutograd",
       c10::DispatchKey::CompositeExplicitAutograd},
  };
  auto it = key_map.find(k);
  TORCH_CHECK(it != key_map.end(), "could not parse dispatch key: ", k);
  return it->second;
}

} // namespace c10
