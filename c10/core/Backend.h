#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/util/Exception.h>

#include <stdexcept>

namespace c10 {

/**
 * This legacy enum class defines the set of backends supported by old school,
 * code generated Type-based ATen.  A "backend" in this sense roughly
 * corresponds to the cartesian product of (device type, layout), but restricted
 * only to combinations which we actually have kernels for.  Backend does NOT
 * include dtype.
 *
 * The reason we are sunsetting this enum class is because it doesn't allow for
 * open registration; e.g., if you want to add SparseXLA, you'd have to
 * edit this enum; you wouldn't be able to do it out of tree.  DispatchKey is
 * the replacement for Backend which supports open registration.
 *
 * NB: The concept of 'Backend' here disagrees with the notion of backend
 * exposed to users in torch.backends.  Backend here is something like "CPU"
 * or "SparseCUDA"; backend in torch.backends is something like "MKL" or
 * "CUDNN".
 */
enum class Backend {
  CPU,
  CUDA,
  HIP,
  VE,
  FPGA,
  IPU,
  XPU,
  SparseCPU,
  SparseCUDA,
  SparseCsrCPU,
  SparseCsrCUDA,
  SparseCsrMPS,
  SparseMPS,
  SparseHIP,
  SparseVE,
  SparseXPU,
  SparsePrivateUse1,
  SparseCsrHIP,
  SparseCsrVE,
  SparseCsrXPU,
  SparseCsrPrivateUse1,
  MAIA,
  XLA,
  Vulkan,
  Metal,
  Meta,
  QuantizedCPU,
  QuantizedCUDA,
  QuantizedXPU,
  QuantizedPrivateUse1,
  Undefined,
  MkldnnCPU,
  MPS,
  HPU,
  Lazy,
  MTIA,
  PrivateUse1,
  NumOptions
};

inline Backend dispatchKeyToBackend(DispatchKey t) {
  if (t == DispatchKey::CPU || t == DispatchKey::AutogradCPU) {
    return Backend::CPU;
  } else if (t == DispatchKey::CUDA || t == DispatchKey::AutogradCUDA) {
    return Backend::CUDA;
  } else if (t == DispatchKey::HIP) {
    return Backend::HIP;
  } else if (t == DispatchKey::VE) {
    return Backend::VE;
  } else if (t == DispatchKey::FPGA) {
    return Backend::FPGA;
  } else if (t == DispatchKey::MAIA || t == DispatchKey::AutogradMAIA) {
    return Backend::MAIA;
  } else if (t == DispatchKey::XLA || t == DispatchKey::AutogradXLA) {
    return Backend::XLA;
  } else if (t == DispatchKey::Lazy || t == DispatchKey::AutogradLazy) {
    return Backend::Lazy;
  } else if (t == DispatchKey::MPS || t == DispatchKey::AutogradMPS) {
    return Backend::MPS;
  } else if (t == DispatchKey::Vulkan) {
    return Backend::Vulkan;
  } else if (t == DispatchKey::Metal) {
    return Backend::Metal;
  } else if (t == DispatchKey::Meta) {
    return Backend::Meta;
  } else if (t == DispatchKey::SparseCPU) {
    return Backend::SparseCPU;
  } else if (t == DispatchKey::SparseCUDA) {
    return Backend::SparseCUDA;
  } else if (t == DispatchKey::SparseMPS) {
    return Backend::SparseMPS;
  } else if (t == DispatchKey::SparseCsrMPS) {
    return Backend::SparseCsrMPS;
  } else if (t == DispatchKey::SparseHIP) {
    return Backend::SparseHIP;
  } else if (t == DispatchKey::SparseVE) {
    return Backend::SparseVE;
  } else if (t == DispatchKey::SparsePrivateUse1) {
    return Backend::SparsePrivateUse1;
  } else if (t == DispatchKey::SparseCsrCPU) {
    return Backend::SparseCsrCPU;
  } else if (t == DispatchKey::SparseCsrCUDA) {
    return Backend::SparseCsrCUDA;
  } else if (t == DispatchKey::SparseCsrHIP) {
    return Backend::SparseCsrHIP;
  } else if (t == DispatchKey::SparseCsrVE) {
    return Backend::SparseCsrVE;
  } else if (t == DispatchKey::SparseCsrPrivateUse1) {
    return Backend::SparseCsrPrivateUse1;
  } else if (t == DispatchKey::MkldnnCPU) {
    return Backend::MkldnnCPU;
  } else if (t == DispatchKey::QuantizedCPU) {
    return Backend::QuantizedCPU;
  } else if (t == DispatchKey::QuantizedCUDA) {
    return Backend::QuantizedCUDA;
  } else if (t == DispatchKey::IPU || t == DispatchKey::AutogradIPU) {
    return Backend::IPU;
  } else if (t == DispatchKey::XPU || t == DispatchKey::AutogradXPU) {
    return Backend::XPU;
  } else if (t == DispatchKey::SparseXPU) {
    return Backend::SparseXPU;
  } else if (t == DispatchKey::SparseCsrXPU) {
    return Backend::SparseCsrXPU;
  } else if (t == DispatchKey::QuantizedXPU) {
    return Backend::QuantizedXPU;
  } else if (t == DispatchKey::QuantizedPrivateUse1) {
    return Backend::QuantizedPrivateUse1;
  } else if (t == DispatchKey::HPU || t == DispatchKey::AutogradHPU) {
    return Backend::HPU;
  } else if (t == DispatchKey::MTIA || t == DispatchKey::AutogradMTIA) {
    return Backend::MTIA;
  } else if (
      t == DispatchKey::PrivateUse1 || t == DispatchKey::AutogradPrivateUse1) {
    return Backend::PrivateUse1;
  } else if (t == DispatchKey::Undefined) {
    return Backend::Undefined;
  } else {
    TORCH_CHECK(false, "Unrecognized tensor type ID: ", t);
  }
}

inline DispatchKey backendToDispatchKey(Backend b) {
  switch (b) {
    case Backend::CPU:
      return DispatchKey::CPU;
    case Backend::CUDA:
      return DispatchKey::CUDA;
    case Backend::HIP:
      return DispatchKey::HIP;
    case Backend::VE:
      return DispatchKey::VE;
    case Backend::FPGA:
      return DispatchKey::FPGA;
    case Backend::MAIA:
      return DispatchKey::MAIA;
    case Backend::XLA:
      return DispatchKey::XLA;
    case Backend::Lazy:
      return DispatchKey::Lazy;
    case Backend::IPU:
      return DispatchKey::IPU;
    case Backend::XPU:
      return DispatchKey::XPU;
    case Backend::SparseXPU:
      return DispatchKey::SparseXPU;
    case Backend::SparseCsrXPU:
      return DispatchKey::SparseCsrXPU;
    case Backend::SparseCPU:
      return DispatchKey::SparseCPU;
    case Backend::SparseCUDA:
      return DispatchKey::SparseCUDA;
    case Backend::SparseMPS:
      return DispatchKey::SparseMPS;
    case Backend::SparseCsrMPS:
      return DispatchKey::SparseCsrMPS;
    case Backend::SparseHIP:
      return DispatchKey::SparseHIP;
    case Backend::SparseVE:
      return DispatchKey::SparseVE;
    case Backend::SparsePrivateUse1:
      return DispatchKey::SparsePrivateUse1;
    case Backend::SparseCsrCPU:
      return DispatchKey::SparseCsrCPU;
    case Backend::SparseCsrCUDA:
      return DispatchKey::SparseCsrCUDA;
    case Backend::SparseCsrHIP:
      return DispatchKey::SparseCsrHIP;
    case Backend::SparseCsrVE:
      return DispatchKey::SparseCsrVE;
    case Backend::SparseCsrPrivateUse1:
      return DispatchKey::SparseCsrPrivateUse1;
    case Backend::MkldnnCPU:
      return DispatchKey::MkldnnCPU;
    case Backend::Vulkan:
      return DispatchKey::Vulkan;
    case Backend::Metal:
      return DispatchKey::Metal;
    case Backend::Meta:
      return DispatchKey::Meta;
    case Backend::QuantizedCPU:
      return DispatchKey::QuantizedCPU;
    case Backend::QuantizedCUDA:
      return DispatchKey::QuantizedCUDA;
    case Backend::QuantizedPrivateUse1:
      return DispatchKey::QuantizedPrivateUse1;
    case Backend::Undefined:
      return DispatchKey::Undefined;
    case Backend::MPS:
      return DispatchKey::MPS;
    case Backend::HPU:
      return DispatchKey::HPU;
    case Backend::MTIA:
      return DispatchKey::MTIA;
    case Backend::PrivateUse1:
      return DispatchKey::PrivateUse1;
    default:
      throw std::runtime_error("Unknown backend");
  }
}

inline DeviceType backendToDeviceType(Backend b) {
  switch (b) {
    case Backend::CPU:
    case Backend::MkldnnCPU:
    case Backend::SparseCPU:
    case Backend::SparseCsrCPU:
    case Backend::QuantizedCPU:
      return DeviceType::CPU;
    case Backend::CUDA:
    case Backend::SparseCUDA:
    case Backend::QuantizedCUDA:
    case Backend::SparseCsrCUDA:
      return DeviceType::CUDA;
    case Backend::HIP:
      return DeviceType::HIP;
    case Backend::VE:
      return DeviceType::VE;
    case Backend::FPGA:
      return DeviceType::FPGA;
    case Backend::MAIA:
      return DeviceType::MAIA;
    case Backend::XLA:
      return DeviceType::XLA;
    case Backend::Lazy:
      return DeviceType::Lazy;
    case Backend::SparseHIP:
      return DeviceType::HIP;
    case Backend::SparseVE:
      return DeviceType::VE;
    case Backend::SparseCsrHIP:
      return DeviceType::HIP;
    case Backend::SparseCsrVE:
      return DeviceType::VE;
    case Backend::IPU:
      return DeviceType::IPU;
    case Backend::XPU:
    case Backend::SparseXPU:
    case Backend::SparseCsrXPU:
    case Backend::QuantizedXPU:
      return DeviceType::XPU;
    case Backend::Vulkan:
      return DeviceType::Vulkan;
    case Backend::Metal:
      return DeviceType::Metal;
    case Backend::Meta:
      return DeviceType::Meta;
    case Backend::MPS:
    case Backend::SparseMPS:
    case Backend::SparseCsrMPS:
      return DeviceType::MPS;
    case Backend::HPU:
      return DeviceType::HPU;
    case Backend::MTIA:
      return DeviceType::MTIA;
    case Backend::PrivateUse1:
    case Backend::SparsePrivateUse1:
    case Backend::SparseCsrPrivateUse1:
    case Backend::QuantizedPrivateUse1:
      return DeviceType::PrivateUse1;
    case Backend::Undefined:
      TORCH_CHECK(false, "Undefined backend is not a valid device type");
    default:
      TORCH_CHECK(false, "Unknown backend");
  }
}

inline const char* toString(Backend b) {
  switch (b) {
    case Backend::CPU:
      return "CPU";
    case Backend::CUDA:
      return "CUDA";
    case Backend::HIP:
      return "HIP";
    case Backend::VE:
      return "VE";
    case Backend::FPGA:
      return "FPGA";
    case Backend::XPU:
      return "XPU";
    case Backend::IPU:
      return "IPU";
    case Backend::MAIA:
      return "MAIA";
    case Backend::XLA:
      return "XLA";
    case Backend::Lazy:
      return "Lazy";
    case Backend::MPS:
      return "MPS";
    case Backend::SparseCPU:
      return "SparseCPU";
    case Backend::SparseCUDA:
      return "SparseCUDA";
    case Backend::SparseMPS:
      return "SparseMPS";
    case Backend::SparseCsrMPS:
      return "SparseCsrMPS";
    case Backend::SparseHIP:
      return "SparseHIP";
    case Backend::SparseVE:
      return "SparseVE";
    case Backend::SparseXPU:
      return "SparseXPU";
    case Backend::SparsePrivateUse1:
      return "SparsePrivateUse1";
    case Backend::SparseCsrCPU:
      return "SparseCsrCPU";
    case Backend::SparseCsrCUDA:
      return "SparseCsrCUDA";
    case Backend::SparseCsrHIP:
      return "SparseCsrHIP";
    case Backend::SparseCsrVE:
      return "SparseCsrVE";
    case Backend::SparseCsrXPU:
      return "SparseCsrXPU";
    case Backend::SparseCsrPrivateUse1:
      return "SparseCsrPrivateUse1";
    case Backend::MkldnnCPU:
      return "MkldnnCPU";
    case Backend::Vulkan:
      return "Vulkan";
    case Backend::Metal:
      return "Metal";
    case Backend::Meta:
      return "Meta";
    case Backend::QuantizedCPU:
      return "QuantizedCPU";
    case Backend::QuantizedCUDA:
      return "QuantizedCUDA";
    case Backend::QuantizedXPU:
      return "QuantizedXPU";
    case Backend::QuantizedPrivateUse1:
      return "QuantizedPrivateUse1";
    case Backend::HPU:
      return "HPU";
    case Backend::MTIA:
      return "MTIA";
    case Backend::PrivateUse1:
      return "PrivateUseOne";
    default:
      return "UNKNOWN_BACKEND";
  }
}

inline bool isSparse(Backend b) {
  switch (b) {
    case Backend::SparseXPU:
    case Backend::SparseCPU:
    case Backend::SparseCUDA:
    case Backend::SparseMPS:
    case Backend::SparseHIP:
    case Backend::SparseVE:
    case Backend::SparsePrivateUse1:
      return true;
    default:
      return false;
  }
}

inline bool isSparseCsr(Backend b) {
  switch (b) {
    case Backend::SparseCsrXPU:
    case Backend::SparseCsrCPU:
    case Backend::SparseCsrCUDA:
    case Backend::SparseCsrHIP:
    case Backend::SparseCsrVE:
    case Backend::SparseCsrPrivateUse1:
      return true;
    default:
      return false;
  }
}

} // namespace c10
