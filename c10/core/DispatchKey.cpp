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
    case BackendComponent::MetaBit:
      return "MetaBit";
    case BackendComponent::XPUBit:
      return "XPUBit";
    case BackendComponent::IPUBit:
      return "IPUBit";
    case BackendComponent::MPSBit:
      return "MPSBit";
    case BackendComponent::HPUBit:
      return "HPUBit";
    case BackendComponent::VEBit:
      return "VEBit";
    case BackendComponent::MTIABit:
      return "MTIA";
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

BackendComponent toBackendComponent(DeviceType device_type) {
  switch (device_type) {
#define DO_CASE(device, _)                          \
  case DeviceType::device: {                        \
    return toBackendComponent(DispatchKey::device); \
  }
    C10_FORALL_BACKEND_DEVICE_TYPES(DO_CASE, unused)
#undef DO_CASE
    default:
      return BackendComponent::InvalidBit;
  }
}

const char* toString(DispatchKey t) {
  switch (t) {
    case DispatchKey::Undefined:
      return "Undefined";

    case DispatchKey::Dense:
      return "Dense";
    case DispatchKey::FPGA:
      return "FPGA";
    case DispatchKey::ORT:
      return "ORT";
    case DispatchKey::Vulkan:
      return "Vulkan";
    case DispatchKey::Metal:
      return "Metal";

    case DispatchKey::Lazy:
      return "Lazy";
    case DispatchKey::MPS:
      return "MPS";
    case DispatchKey::HPU:
      return "HPU";
    case DispatchKey::MTIA:
      return "MTIA";

    case DispatchKey::Quantized:
      return "Quantized";
    case DispatchKey::CustomRNGKeyId:
      return "CustomRNGKeyId";
    case DispatchKey::MkldnnCPU:
      return "MkldnnCPU";

    case DispatchKey::Sparse:
      return "Sparse";
    case DispatchKey::SparseCsrCPU:
      return "SparseCsrCPU";
    case DispatchKey::SparseCsrCUDA:
      return "SparseCsrCUDA";

    case DispatchKey::NestedTensor:
      return "NestedTensor";

    case DispatchKey::BackendSelect:
      return "BackendSelect";

    case DispatchKey::Python:
      return "Python";

    case DispatchKey::Fake:
      return "Fake";
    case DispatchKey::FuncTorchDynamicLayerBackMode:
      return "FuncTorchDynamicLayerBackMode";

    case DispatchKey::Functionalize:
      return "Functionalize";

    case DispatchKey::Named:
      return "Named";

    case DispatchKey::Conjugate:
      return "Conjugate";
    case DispatchKey::Negative:
      return "Negative";
    case DispatchKey::ZeroTensor:
      return "ZeroTensor";

    case DispatchKey::ADInplaceOrView:
      return "ADInplaceOrView";

    case DispatchKey::AutogradOther:
      return "AutogradOther";
    case DispatchKey::AutogradFunctionality:
      return "AutogradFunctionality";
    case DispatchKey::AutogradNestedTensor:
      return "AutogradNestedTensor";

    case DispatchKey::Tracer:
      return "Tracer";

    case DispatchKey::AutocastCPU:
      return "AutocastCPU";
    case DispatchKey::AutocastXPU:
      return "AutocastXPU";
    case DispatchKey::AutocastIPU:
      return "AutocastIPU";
    case DispatchKey::AutocastHPU:
      return "AutocastHPU";
    case DispatchKey::AutocastCUDA:
      return "AutocastCUDA";
    case DispatchKey::AutocastPrivateUse1:
      return "AutocastPrivateUse1";

    case DispatchKey::FuncTorchBatched:
      return "FuncTorchBatched";
    case DispatchKey::FuncTorchVmapMode:
      return "FuncTorchVmapMode";

    case DispatchKey::Batched:
      return "Batched";
    case DispatchKey::VmapMode:
      return "VmapMode";

    case DispatchKey::FuncTorchGradWrapper:
      return "FuncTorchGradWrapper";

    case DispatchKey::DeferredInit:
      return "DeferredInit";
    case DispatchKey::PythonTLSSnapshot:
      return "PythonTLSSnapshot";

    // Note [Out-of-tree vmap+grad prototype]
    // The following keys are used in the implementation of the out-of-tree
    // composable functions transforms (vmap+grad) prototype that lives at
    // https://github.com/zou3519/functorch
    // We plan on eventually upstreaming the prototype into core, at which
    // point it will have a different design that should use fewer keys.
    case DispatchKey::FuncTorchDynamicLayerFrontMode:
      return "FuncTorchDynamicLayerFrontMode";

    case DispatchKey::TESTING_ONLY_GenericWrapper:
      return "TESTING_ONLY_GenericWrapper";

    case DispatchKey::TESTING_ONLY_GenericMode:
      return "TESTING_ONLY_GenericMode";

    case DispatchKey::PreDispatch:
      return "PreDispatch";

    case DispatchKey::PythonDispatcher:
      return "PythonDispatcher";

      // Aliases

    case DispatchKey::Autograd:
      return "Autograd";
    case DispatchKey::CompositeImplicitAutograd:
      return "CompositeImplicitAutograd";
    case DispatchKey::CompositeImplicitAutogradNestedTensor:
      return "CompositeImplicitAutogradNestedTensor";
    case DispatchKey::CompositeExplicitAutograd:
      return "CompositeExplicitAutograd";
    case DispatchKey::CompositeExplicitAutogradNonFunctional:
      return "CompositeExplicitAutogradNonFunctional";
    case DispatchKey::FuncTorchBatchedDecomposition:
      return "FuncTorchBatchedDecomposition";

      // Per-backend dispatch keys

    default:
      auto bc = toBackendComponent(t);
      auto fk = toFunctionalityKey(t);

      switch (fk) {
#define ENTRY(backend, functionality)  \
  case BackendComponent::backend##Bit: \
    return #functionality #backend;

#define FORALL_BC(dkname, prefix)                  \
  case DispatchKey::dkname:                        \
    switch (bc) {                                  \
      C10_FORALL_BACKEND_COMPONENTS(ENTRY, prefix) \
      default:                                     \
        return #prefix "Undefined";                \
    }

        C10_FORALL_FUNCTIONALITY_KEYS(FORALL_BC)

        default:
          switch (bc) {
            C10_FORALL_BACKEND_COMPONENTS(ENTRY, Unknown)
            default:
              return "UnknownUnknown";
          }

#undef FORALL_BC
#undef ENTRY
      }
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
      {"MPS", c10::DispatchKey::MPS},
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
      {"Fake", c10::DispatchKey::Fake},
      {"Named", c10::DispatchKey::Named},
      {"Conjugate", c10::DispatchKey::Conjugate},
      {"Negative", c10::DispatchKey::Negative},
      {"ZeroTensor", c10::DispatchKey::ZeroTensor},
      {"FuncTorchDynamicLayerBackMode",
       c10::DispatchKey::FuncTorchDynamicLayerBackMode},
      {"Functionalize", c10::DispatchKey::Functionalize},
      {"ADInplaceOrView", c10::DispatchKey::ADInplaceOrView},
      {"AutogradOther", c10::DispatchKey::AutogradOther},
      {"AutogradFunctionality", c10::DispatchKey::AutogradFunctionality},
      {"AutogradNestedTensor", c10::DispatchKey::AutogradNestedTensor},
      {"Tracer", c10::DispatchKey::Tracer},
      {"AutocastCPU", c10::DispatchKey::AutocastCPU},
      {"AutocastXPU", c10::DispatchKey::AutocastXPU},
      {"AutocastIPU", c10::DispatchKey::AutocastIPU},
      {"AutocastHPU", c10::DispatchKey::AutocastHPU},
      {"AutocastCUDA", c10::DispatchKey::AutocastCUDA},
      {"AutocastPrivateUse1", c10::DispatchKey::AutocastPrivateUse1},
      {"FuncTorchBatched", c10::DispatchKey::FuncTorchBatched},
      {"FuncTorchVmapMode", c10::DispatchKey::FuncTorchVmapMode},
      {"Batched", c10::DispatchKey::Batched},
      {"VmapMode", c10::DispatchKey::VmapMode},
      {"DeferredInit", c10::DispatchKey::DeferredInit},
      {"FuncTorchGradWrapper", c10::DispatchKey::FuncTorchGradWrapper},
      {"FuncTorchDynamicLayerFrontMode",
       c10::DispatchKey::FuncTorchDynamicLayerFrontMode},
      {"TESTING_ONLY_GenericWrapper",
       c10::DispatchKey::TESTING_ONLY_GenericWrapper},
      {"TESTING_ONLY_GenericMode", c10::DispatchKey::TESTING_ONLY_GenericMode},
      {"PythonDispatcher", c10::DispatchKey::PythonDispatcher},
      {"PreDispatch", c10::DispatchKey::PreDispatch},

      {"CPU", c10::DispatchKey::CPU},
      {"CUDA", c10::DispatchKey::CUDA},
      {"HIP", c10::DispatchKey::HIP},
      {"XLA", c10::DispatchKey::XLA},
      {"MPS", c10::DispatchKey::MPS},
      {"XPU", c10::DispatchKey::XPU},
      {"IPU", c10::DispatchKey::IPU},
      {"HPU", c10::DispatchKey::HPU},
      {"Lazy", c10::DispatchKey::Lazy},
      {"MTIA", c10::DispatchKey::MTIA},
      {"NestedTensor", c10::DispatchKey::NestedTensor},
      {"NestedTensorCPU", c10::DispatchKey::NestedTensorCPU},
      {"NestedTensorCUDA", c10::DispatchKey::NestedTensorCUDA},
      {"NestedTensorMeta", c10::DispatchKey::NestedTensorMeta},
      {"NestedTensorPrivateUse1", c10::DispatchKey::NestedTensorPrivateUse1},
      {"PrivateUse1", c10::DispatchKey::PrivateUse1},
      {"PrivateUse2", c10::DispatchKey::PrivateUse2},
      {"PrivateUse3", c10::DispatchKey::PrivateUse3},

      {"QuantizedCPU", c10::DispatchKey::QuantizedCPU},
      {"QuantizedCUDA", c10::DispatchKey::QuantizedCUDA},
      {"QuantizedXPU", c10::DispatchKey::QuantizedXPU},
      {"QuantizedPrivateUse1", c10::DispatchKey::QuantizedPrivateUse1},

      {"SparseCPU", c10::DispatchKey::SparseCPU},
      {"SparseCUDA", c10::DispatchKey::SparseCUDA},
      {"SparseHIP", c10::DispatchKey::SparseHIP},
      {"SparseXPU", c10::DispatchKey::SparseXPU},
      {"SparseVE", c10::DispatchKey::SparseVE},
      {"SparseMeta", c10::DispatchKey::SparseMeta},
      {"SparsePrivateUse1", c10::DispatchKey::SparsePrivateUse1},

      {"AutogradCPU", c10::DispatchKey::AutogradCPU},
      {"AutogradCUDA", c10::DispatchKey::AutogradCUDA},
      {"AutogradXLA", c10::DispatchKey::AutogradXLA},
      {"AutogradLazy", c10::DispatchKey::AutogradLazy},
      {"AutogradMeta", c10::DispatchKey::AutogradMeta},
      {"AutogradIPU", c10::DispatchKey::AutogradIPU},
      {"AutogradXPU", c10::DispatchKey::AutogradXPU},
      {"AutogradMPS", c10::DispatchKey::AutogradMPS},
      {"AutogradHPU", c10::DispatchKey::AutogradHPU},
      {"AutogradPrivateUse1", c10::DispatchKey::AutogradPrivateUse1},
      {"AutogradPrivateUse2", c10::DispatchKey::AutogradPrivateUse2},
      {"AutogradPrivateUse3", c10::DispatchKey::AutogradPrivateUse3},

      {"Autograd", c10::DispatchKey::Autograd},
      {"CompositeImplicitAutograd",
       c10::DispatchKey::CompositeImplicitAutograd},
      {"CompositeImplicitAutogradNestedTensor",
       c10::DispatchKey::CompositeImplicitAutogradNestedTensor},
      {"CompositeExplicitAutograd",
       c10::DispatchKey::CompositeExplicitAutograd},
      {"CompositeExplicitAutogradNonFunctional",
       c10::DispatchKey::CompositeExplicitAutogradNonFunctional},
      {"FuncTorchBatchedDecomposition",
       c10::DispatchKey::FuncTorchBatchedDecomposition},
  };
  auto it = key_map.find(k);
  TORCH_CHECK(it != key_map.end(), "could not parse dispatch key: ", k);
  return it->second;
}

} // namespace c10
