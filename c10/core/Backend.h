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
  SparseHIP,
  SparseVE,
  SparseXPU,
  SparsePrivateUse1,
  ORT,
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

static inline Backend dispatchKeyToBackend(DispatchKey t) {
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
  } else if (t == DispatchKey::ORT) {
    return Backend::ORT;
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

static inline DispatchKey backendToDispatchKey(Backend b) {
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
    case Backend::ORT:
      return DispatchKey::ORT;
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
    case Backend::SparseCPU:
      return DispatchKey::SparseCPU;
    case Backend::SparseCUDA:
      return DispatchKey::SparseCUDA;
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

static inline DeviceType backendToDeviceType(Backend b) {
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
    case Backend::ORT:
      return DeviceType::ORT;
    case Backend::XLA:
      return DeviceType::XLA;
    case Backend::Lazy:
      return DeviceType::Lazy;
    case Backend::SparseHIP:
      return DeviceType::HIP;
    case Backend::SparseVE:
      return DeviceType::VE;
    case Backend::IPU:
      return DeviceType::IPU;
    case Backend::XPU:
    case Backend::SparseXPU:
    case Backend::QuantizedXPU:
      return DeviceType::XPU;
    case Backend::Vulkan:
      return DeviceType::Vulkan;
    case Backend::Metal:
      return DeviceType::Metal;
    case Backend::Meta:
      return DeviceType::Meta;
    case Backend::MPS:
      return DeviceType::MPS;
    case Backend::HPU:
      return DeviceType::HPU;
    case Backend::MTIA:
      return DeviceType::MTIA;
    case Backend::PrivateUse1:
    case Backend::SparsePrivateUse1:
    case Backend::QuantizedPrivateUse1:
      return DeviceType::PrivateUse1;
    case Backend::Undefined:
      TORCH_CHECK(false, "Undefined backend is not a valid device type");
    default:
      TORCH_CHECK(false, "Unknown backend");
  }
}

// TODO: This probably shouldn't actually be static inline
static inline const char* toString(Backend b) {
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
    case Backend::ORT:
      return "ORT";
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

static inline bool isSparse(Backend b) {
  switch (b) {
    case Backend::SparseXPU:
    case Backend::SparseCPU:
    case Backend::SparseCUDA:
    case Backend::SparseHIP:
    case Backend::SparseVE:
    case Backend::SparsePrivateUse1:
      return true;
    default:
      return false;
  }
}

static inline bool isSparseCsr(Backend b) {
  switch (b) {
    case Backend::SparseCsrCPU:
    case Backend::SparseCsrCUDA:
      return true;
    default:
      return false;
  }
}

} // namespace c10
