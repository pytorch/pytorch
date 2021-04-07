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
  FPGA,
  XPU,
  SparseCPU,
  SparseCUDA,
  SparseHIP,
  SparseXPU,
  MSNPU,
  XLA,
  Vulkan,
  Metal,
  QuantizedCPU,
  QuantizedCUDA,
  QuantizedXPU,
  Undefined,
  MkldnnCPU,
  MLC,
  NumOptions
};

static inline Backend dispatchKeyToBackend(DispatchKey t) {
  if (t == DispatchKey::CPU || t == DispatchKey::AutogradCPU) {
    return Backend::CPU;
  } else if (t == DispatchKey::CUDA || t == DispatchKey::AutogradCUDA) {
    return Backend::CUDA;
  } else if (t == DispatchKey::HIP) {
    return Backend::HIP;
  } else if (t == DispatchKey::FPGA) {
    return Backend::FPGA;
  } else if (t == DispatchKey::MSNPU) {
    return Backend::MSNPU;
  } else if (t == DispatchKey::XLA || t == DispatchKey::AutogradXLA) {
    return Backend::XLA;
  } else if (t == DispatchKey::MLC || t == DispatchKey::AutogradMLC) {
    return Backend::MLC;
  } else if (t == DispatchKey::Vulkan) {
    return Backend::Vulkan;
  } else if (t == DispatchKey::Metal) {
    return Backend::Metal;
  } else if (t == DispatchKey::SparseCPU) {
    return Backend::SparseCPU;
  } else if (t == DispatchKey::SparseCUDA) {
    return Backend::SparseCUDA;
  } else if (t == DispatchKey::SparseHIP) {
    return Backend::SparseHIP;
  } else if (t == DispatchKey::MkldnnCPU) {
    return Backend::MkldnnCPU;
  } else if (t == DispatchKey::QuantizedCPU) {
    return Backend::QuantizedCPU;
  } else if (t == DispatchKey::QuantizedCUDA) {
    return Backend::QuantizedCUDA;
  } else if (t == DispatchKey::XPU) {
    return Backend::XPU;
  } else if (t == DispatchKey::SparseXPU) {
    return Backend::SparseXPU;
  } else if (t == DispatchKey::QuantizedXPU) {
    return Backend::QuantizedXPU;
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
    case Backend::FPGA:
      return DispatchKey::FPGA;
    case Backend::MSNPU:
      return DispatchKey::MSNPU;
    case Backend::XLA:
      return DispatchKey::XLA;
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
    case Backend::MkldnnCPU:
      return DispatchKey::MkldnnCPU;
    case Backend::Vulkan:
      return DispatchKey::Vulkan;
    case Backend::Metal:
      return DispatchKey::Metal;
    case Backend::QuantizedCPU:
      return DispatchKey::QuantizedCPU;
    case Backend::QuantizedCUDA:
      return DispatchKey::QuantizedCUDA;
    case Backend::Undefined:
      return DispatchKey::Undefined;
    case Backend::MLC:
      return DispatchKey::MLC;
    default:
      throw std::runtime_error("Unknown backend");
  }
}

static inline DeviceType backendToDeviceType(Backend b) {
  switch (b) {
    case Backend::CPU:
      return DeviceType::CPU;
    case Backend::CUDA:
      return DeviceType::CUDA;
    case Backend::HIP:
      return DeviceType::HIP;
    case Backend::FPGA:
      return DeviceType::FPGA;
    case Backend::MSNPU:
      return DeviceType::MSNPU;
    case Backend::XLA:
      return DeviceType::XLA;
    case Backend::SparseCPU:
      return DeviceType::CPU;
    case Backend::SparseCUDA:
      return DeviceType::CUDA;
    case Backend::SparseHIP:
      return DeviceType::HIP;
    case Backend::XPU:
    case Backend::SparseXPU:
    case Backend::QuantizedXPU:
      return DeviceType::XPU;
    case Backend::MkldnnCPU:
    case Backend::QuantizedCPU:
      return DeviceType::CPU;
    case Backend::QuantizedCUDA:
      return DeviceType::CUDA;
    case Backend::Vulkan:
      return DeviceType::Vulkan;
    case Backend::Metal:
      return DeviceType::Metal;
    case Backend::MLC:
      return DeviceType::MLC;
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
    case Backend::FPGA:
      return "FPGA";
    case Backend::XPU:
      return "XPU";
    case Backend::MSNPU:
      return "MSNPU";
    case Backend::XLA:
      return "XLA";
    case Backend::MLC:
      return "MLC";
    case Backend::SparseCPU:
      return "SparseCPU";
    case Backend::SparseCUDA:
      return "SparseCUDA";
    case Backend::SparseHIP:
      return "SparseHIP";
    case Backend::SparseXPU:
      return "SparseXPU";
    case Backend::MkldnnCPU:
      return "MkldnnCPU";
    case Backend::Vulkan:
      return "Vulkan";
    case Backend::Metal:
      return "Metal";
    case Backend::QuantizedCPU:
      return "QuantizedCPU";
    case Backend::QuantizedCUDA:
      return "QuantizedCUDA";
    case Backend::QuantizedXPU:
      return "QuantizedXPU";
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
      return true;
    default:
      return false;
  }
}

} // namespace c10
