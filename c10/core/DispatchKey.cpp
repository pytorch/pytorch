#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>

#include <unordered_map>

namespace c10 {

const char* toString(BackendComponent t) {
  switch (t) {
    case BackendComponent::CPUBit:
      return "CPUBit";
    case BackendComponent::CUDABit:
      return "CUDABit";
    case BackendComponent::HIPBit:
      return "HIPBit";
    case BackendComponent::XLABit:
      return "XLABit";
    case BackendComponent::LazyBit:
      return "LazyBit";
    case BackendComponent::XPUBit:
      return "XPUBit";
    case BackendComponent::IPUBit:
      return "IPUBit";
    case BackendComponent::MLCBit:
      return "MLCBit";
    case BackendComponent::HPUBit:
      return "HPUBit";
    case BackendComponent::VEBit:
      return "VEBit";
    case BackendComponent::PrivateUse1Bit:
      return "PrivateUse1Bit";
    case BackendComponent::PrivateUse2Bit:
      return "PrivateUse2Bit";
    case BackendComponent::PrivateUse3Bit:
      return "PrivateUse3Bit";
    case BackendComponent::InvalidBit:
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
    case DispatchKey::IPU:
      return "IPU";
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
    case DispatchKey::QuantizedHIP:
      return "QuantizedHIP";
    case DispatchKey::QuantizedVE:
      return "QuantizedVE";
    case DispatchKey::QuantizedXPU:
      return "QuantizedXPU";
    case DispatchKey::QuantizedIPU:
      return "QuantizedIPU";
    case DispatchKey::QuantizedXLA:
      return "QuantizedXLA";
    case DispatchKey::QuantizedLazy:
      return "QuantizedLazy";
    case DispatchKey::QuantizedMLC:
      return "QuantizedMLC";
    case DispatchKey::QuantizedHPU:
      return "QuantizedHPU";
    case DispatchKey::QuantizedPrivateUse1:
      return "QuantizedPrivateUse1";
    case DispatchKey::QuantizedPrivateUse2:
      return "QuantizedPrivateUse2";
    case DispatchKey::QuantizedPrivateUse3:
      return "QuantizedPrivateUse3";

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
    case DispatchKey::SparseVE:
      return "SparseVE";
    case DispatchKey::SparseXPU:
      return "SparseXPU";
    case DispatchKey::SparseIPU:
      return "SparseIPU";
    case DispatchKey::SparseXLA:
      return "SparseXLA";
    case DispatchKey::SparseLazy:
      return "SparseLazy";
    case DispatchKey::SparseMLC:
      return "SparseMLC";
    case DispatchKey::SparseHPU:
      return "SparseHPU";
    case DispatchKey::SparsePrivateUse1:
      return "SparsePrivateUse1";
    case DispatchKey::SparsePrivateUse2:
      return "SparsePrivateUse2";
    case DispatchKey::SparsePrivateUse3:
      return "SparsePrivateUse3";

    case DispatchKey::SparseCsrCPU:
      return "SparseCsrCPU";
    case DispatchKey::SparseCsrCUDA:
      return "SparseCsrCUDA";
    case DispatchKey::SparseCsrHIP:
      return "SparseCsrHIP";
    case DispatchKey::SparseCsrVE:
      return "SparseCsrVE";
    case DispatchKey::SparseCsrXPU:
      return "SparseCsrXPU";
    case DispatchKey::SparseCsrIPU:
      return "SparseCsrIPU";
    case DispatchKey::SparseCsrXLA:
      return "SparseCsrXLA";
    case DispatchKey::SparseCsrLazy:
      return "SparseCsrLazy";
    case DispatchKey::SparseCsrMLC:
      return "SparseCsrMLC";
    case DispatchKey::SparseCsrHPU:
      return "SparseCsrHPU";
    case DispatchKey::SparseCsrPrivateUse1:
      return "SparseCsrPrivateUse1";
    case DispatchKey::SparseCsrPrivateUse2:
      return "SparseCsrPrivateUse2";
    case DispatchKey::SparseCsrPrivateUse3:
      return "SparseCsrPrivateUse3";

    case DispatchKey::NestedTensor:
      return "NestedTensor";
    case DispatchKey::NestedTensorCPU:
      return "NestedTensorCPU";
    case DispatchKey::NestedTensorCUDA:
      return "NestedTensorCUDA";
    case DispatchKey::NestedTensorHIP:
      return "NestedTensorHIP";
    case DispatchKey::NestedTensorVE:
      return "NestedTensorVE";
    case DispatchKey::NestedTensorXPU:
      return "NestedTensorXPU";
    case DispatchKey::NestedTensorIPU:
      return "NestedTensorIPU";
    case DispatchKey::NestedTensorXLA:
      return "NestedTensorXLA";
    case DispatchKey::NestedTensorLazy:
      return "NestedTensorLazy";
    case DispatchKey::NestedTensorMLC:
      return "NestedTensorMLC";
    case DispatchKey::NestedTensorHPU:
      return "NestedTensorHPU";
    case DispatchKey::NestedTensorPrivateUse1:
      return "NestedTensorPrivateUse1";
    case DispatchKey::NestedTensorPrivateUse2:
      return "NestedTensorPrivateUse2";
    case DispatchKey::NestedTensorPrivateUse3:
      return "NestedTensorPrivateUse3";

    case DispatchKey::Python:
      return "Python";
    case DispatchKey::PythonTLSSnapshot:
      return "PythonTLSSnapshot";

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
    case DispatchKey::AutogradIPU:
      return "AutogradIPU";
    case DispatchKey::AutogradXPU:
      return "AutogradXPU";
    case DispatchKey::AutogradCUDA:
      return "AutogradCUDA";
    case DispatchKey::AutogradHIP:
      return "AutogradHIP";
    case DispatchKey::AutogradXLA:
      return "AutogradXLA";
    case DispatchKey::AutogradLazy:
      return "AutogradLazy";
    case DispatchKey::AutogradMLC:
      return "AutogradMLC";
    case DispatchKey::AutogradHPU:
      return "AutogradHPU";
    case DispatchKey::AutogradVE:
      return "AutogradVE";
    case DispatchKey::AutogradPrivateUse1:
      return "AutogradPrivateUse1";
    case DispatchKey::AutogradPrivateUse2:
      return "AutogradPrivateUse2";
    case DispatchKey::AutogradPrivateUse3:
      return "AutogradPrivateUse3";
    case DispatchKey::AutogradOther:
      return "AutogradOther";
    case DispatchKey::AutogradNestedTensor:
      return "AutogradNestedTensor";

    case DispatchKey::AutocastCPU:
      return "AutocastCPU";
    case DispatchKey::AutocastIPU:
      return "AutocastIPU";
    case DispatchKey::AutocastXPU:
      return "AutocastXPU";
    case DispatchKey::AutocastCUDA:
      return "AutocastCUDA";
    case DispatchKey::AutocastHIP:
      return "AutocastHIP";
    case DispatchKey::AutocastXLA:
      return "AutocastXLA";
    case DispatchKey::AutocastLazy:
      return "AutocastLazy";
    case DispatchKey::AutocastMLC:
      return "AutocastMLC";
    case DispatchKey::AutocastHPU:
      return "AutocastHPU";
    case DispatchKey::AutocastVE:
      return "AutocastVE";
    case DispatchKey::AutocastPrivateUse1:
      return "AutocastPrivateUse1";
    case DispatchKey::AutocastPrivateUse2:
      return "AutocastPrivateUse2";
    case DispatchKey::AutocastPrivateUse3:
      return "AutocastPrivateUse3";

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
    case DispatchKey::SparseCsr:
      return "SparseCsr";
    case DispatchKey::AutogradFunctionality:
      return "AutogradFunctionality";
    case DispatchKey::AutocastFunctionality:
      return "AutocastFunctionality";

    default:
      return "UNKNOWN_TENSOR_TYPE_ID";
  }
}

std::ostream& operator<<(std::ostream& str, DispatchKey rhs) {
  return str << toString(rhs);
}
std::ostream& operator<<(std::ostream& str, BackendComponent rhs) {
  return str << toString(rhs);
}

DispatchKey getAutogradKeyFromBackend(BackendComponent k) {
  // We want this to return an autograd key. We're relying on the fact that
  // getAutogradRelatedKeySetFromBackend returns an autograd key +
  // ADInplaceOrView, and autograd has higher precedence. The core mapping from
  // backend -> autograd key lives in `getAutogradRelatedKeySetFromBackend`
  // instead of here for performance. `getAutogradRelatedKeySetFromBackend` is a
  // hotpath function, and we want to make sure that it doesn't have to
  // construct any DispatchKeySets at runtime.
  return getAutogradRelatedKeySetFromBackend(k).highestPriorityTypeId();
}

c10::DispatchKey parseDispatchKey(const std::string& k) {
  static std::unordered_map<std::string, c10::DispatchKey> key_map = {
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
      {"PythonTLSSnapshot", c10::DispatchKey::PythonTLSSnapshot},
      {"Named", c10::DispatchKey::Named},
      {"Conjugate", c10::DispatchKey::Conjugate},
      {"Negative", c10::DispatchKey::Negative},
      {"ZeroTensor", c10::DispatchKey::ZeroTensor},
      {"FuncTorchDynamicLayerBackMode",
       c10::DispatchKey::FuncTorchDynamicLayerBackMode},
      {"ADInplaceOrView", c10::DispatchKey::ADInplaceOrView},
      {"AutogradOther", c10::DispatchKey::AutogradOther},
      {"AutogradFunctionality", c10::DispatchKey::AutogradFunctionality},
      {"AutogradNestedTensor", c10::DispatchKey::AutogradNestedTensor},
      {"Tracer", c10::DispatchKey::Tracer},
      {"AutocastCPU", c10::DispatchKey::AutocastCPU},
      {"AutocastXPU", c10::DispatchKey::AutocastXPU},
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
      {"IPU", c10::DispatchKey::IPU},
      {"HPU", c10::DispatchKey::HPU},
      {"Lazy", c10::DispatchKey::Lazy},
      {"NestedTensor", c10::DispatchKey::NestedTensor},
      {"NestedTensorCPU", c10::DispatchKey::NestedTensorCPU},
      {"NestedTensorCUDA", c10::DispatchKey::NestedTensorCUDA},
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
      {"AutogradIPU", c10::DispatchKey::AutogradIPU},
      {"AutogradXPU", c10::DispatchKey::AutogradXPU},
      {"AutogradMLC", c10::DispatchKey::AutogradMLC},
      {"AutogradHPU", c10::DispatchKey::AutogradHPU},
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
