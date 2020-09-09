#include <c10/core/DispatchKeySet.h>

namespace c10 {

constexpr DispatchKeySet autograd_dispatch_keyset = DispatchKeySet({
  DispatchKey::AutogradCPU,
  DispatchKey::AutogradCUDA,
  DispatchKey::AutogradXLA,
  DispatchKey::AutogradPrivateUse1,
  DispatchKey::AutogradPrivateUse2,
  DispatchKey::AutogradPrivateUse3,
  DispatchKey::AutogradOther,
});

constexpr DispatchKeySet math_dispatch_keyset = DispatchKeySet({
  DispatchKey::CPU,
  DispatchKey::CUDA,
  DispatchKey::HIP,
  DispatchKey::FPGA,
  DispatchKey::MSNPU,
  DispatchKey::XLA,
  DispatchKey::Vulkan,
  DispatchKey::MKLDNN,
  DispatchKey::OpenGL,
  DispatchKey::OpenCL,
  DispatchKey::IDEEP,
  DispatchKey::QuantizedCPU,
  DispatchKey::QuantizedCUDA,
  DispatchKey::ComplexCPU,
  DispatchKey::ComplexCUDA,
  DispatchKey::CustomRNGKeyId,
  DispatchKey::MkldnnCPU,
  DispatchKey::SparseCPU,
  DispatchKey::SparseCUDA,
  DispatchKey::SparseHIP,
  DispatchKey::PrivateUse1,
  DispatchKey::PrivateUse2,
  DispatchKey::PrivateUse3,
  DispatchKey::AutogradCPU,
  DispatchKey::AutogradCUDA,
  DispatchKey::AutogradXLA,
  DispatchKey::AutogradPrivateUse1,
  DispatchKey::AutogradPrivateUse2,
  DispatchKey::AutogradPrivateUse3,
  DispatchKey::AutogradOther,
});


DispatchKeySet getRuntimeDispatchKeySet(DispatchKey t) {
  switch (t) {
    case DispatchKey::Autograd:
      return autograd_dispatch_keyset;
    case DispatchKey::Math:
      return math_dispatch_keyset;
    case DispatchKey::Undefined:
     return DispatchKeySet();
   default:
     return DispatchKeySet(t);
  }
}

template <std::size_t... Is>
constexpr auto make_array_from_sequence(std::index_sequence<Is...>) {
  return std::array<DispatchKey, sizeof...(Is)>{static_cast<DispatchKey>(Is)...};
}

constexpr auto runtime_dispatch_keys = make_array_from_sequence(
  std::make_index_sequence<static_cast<uint8_t>(DispatchKey::NumDispatchKeys)>{});

// Create singleton for alias keys separately to make sure we don't
// accidentally support DispatchKey::NumDispatchKeys in std::array.
constexpr std::array<DispatchKey, 7> autograd_dispatch_keys {
    DispatchKey::AutogradCPU, DispatchKey::AutogradCUDA, DispatchKey::AutogradXLA,
    DispatchKey::AutogradPrivateUse1, DispatchKey::AutogradPrivateUse2,
    DispatchKey::AutogradPrivateUse3, DispatchKey::AutogradOther};

constexpr std::array<DispatchKey, 30> math_dispatch_keys {
  DispatchKey::CPU,
  DispatchKey::CUDA,
  DispatchKey::HIP,
  DispatchKey::FPGA,
  DispatchKey::MSNPU,
  DispatchKey::XLA,
  DispatchKey::Vulkan,
  DispatchKey::MKLDNN,
  DispatchKey::OpenGL,
  DispatchKey::OpenCL,
  DispatchKey::IDEEP,
  DispatchKey::QuantizedCPU,
  DispatchKey::QuantizedCUDA,
  DispatchKey::ComplexCPU,
  DispatchKey::ComplexCUDA,
  DispatchKey::CustomRNGKeyId,
  DispatchKey::MkldnnCPU,
  DispatchKey::SparseCPU,
  DispatchKey::SparseCUDA,
  DispatchKey::SparseHIP,
  DispatchKey::PrivateUse1,
  DispatchKey::PrivateUse2,
  DispatchKey::PrivateUse3,
  DispatchKey::AutogradCPU,
  DispatchKey::AutogradCUDA,
  DispatchKey::AutogradXLA,
  DispatchKey::AutogradPrivateUse1,
  DispatchKey::AutogradPrivateUse2,
  DispatchKey::AutogradPrivateUse3,
  DispatchKey::AutogradOther};

ArrayRef<DispatchKey> getRuntimeDispatchKeys(DispatchKey k) {
  if (isAliasDispatchKey(k)) {
    switch (k) {
      case DispatchKey::Autograd:
        return autograd_dispatch_keys;
      case DispatchKey::Math:
        return math_dispatch_keys;
      default:
        TORCH_INTERNAL_ASSERT(false, "Unable to resolve alias dispatch key");
    }
  }
  return c10::ArrayRef<DispatchKey>(runtime_dispatch_keys).slice(static_cast<uint8_t>(k), 1);
}

bool isIncludedInAlias(DispatchKey k, DispatchKey alias) {
  return k != DispatchKey::Undefined && getRuntimeDispatchKeySet(alias).has(k);
}

std::string toString(DispatchKeySet ts) {
  std::stringstream ss;
  ss << ts;
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, DispatchKeySet ts) {
  if (ts.empty()) {
    os << "DispatchKeySet()";
    return os;
  }
  os << "DispatchKeySet(";
  DispatchKey tid;
  bool first = true;
  while ((tid = ts.highestPriorityTypeId()) != DispatchKey::Undefined) {
    if (!first) {
      os << ", ";
    }
    os << tid;
    ts = ts.remove(tid);
    first = false;
  }
  os << ")";
  return os;
}

}
