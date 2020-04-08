#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/DispatchKey.h>
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
enum class Backend { CPU, CUDA, HIP, SparseCPU, SparseCUDA, SparseHIP, MSNPU, XLA, QuantizedCPU, Undefined, MkldnnCPU, NumOptions };

static inline Backend toSparse(Backend b) {
  switch (b) {
    case Backend::CPU:
      return Backend::SparseCPU;
    case Backend::CUDA:
      return Backend::SparseCUDA;
    case Backend::HIP:
      return Backend::SparseHIP;
    case Backend::SparseCPU:
      return Backend::SparseCPU;
    case Backend::SparseCUDA:
      return Backend::SparseCUDA;
    case Backend::SparseHIP:
      return Backend::SparseHIP;
    default:
      throw std::runtime_error("Unknown backend");
  }
}

static inline Backend toDense(Backend b) {
  switch (b) {
    case Backend::CPU:
      return Backend::CPU;
    case Backend::CUDA:
      return Backend::CUDA;
    case Backend::HIP:
      return Backend::HIP;
    case Backend::MSNPU:
      return Backend::MSNPU;
    case Backend::XLA:
      return Backend::XLA;
    case Backend::SparseCPU:
      return Backend::CPU;
    case Backend::SparseCUDA:
      return Backend::CUDA;
    case Backend::SparseHIP:
      return Backend::HIP;
    case Backend::QuantizedCPU:
      return Backend::QuantizedCPU;
    default:
      throw std::runtime_error("Unknown backend");
  }
}

static inline Backend dispatchKeyToBackend(DispatchKey t) {
  if (t == DispatchKey::CPU) {
    return Backend::CPU;
  } else if (t == DispatchKey::CUDA) {
    return Backend::CUDA;
  } else if (t == DispatchKey::HIP) {
    return Backend::HIP;
  } else if (t == DispatchKey::MSNPU) {
    return Backend::MSNPU;
  } else if (t == DispatchKey::XLA || t == DispatchKey::XLAPreAutograd) {
    return Backend::XLA;
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
  } else if (t == DispatchKey::Undefined) {
    return Backend::Undefined;
  } else {
    AT_ERROR("Unrecognized tensor type ID: ", t);
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
    case Backend::MSNPU:
      return DispatchKey::MSNPU;
    case Backend::XLA:
      return DispatchKey::XLA;
    case Backend::SparseCPU:
      return DispatchKey::SparseCPU;
    case Backend::SparseCUDA:
      return DispatchKey::SparseCUDA;
    case Backend::SparseHIP:
      return DispatchKey::SparseHIP;
    case Backend::MkldnnCPU:
      return DispatchKey::MkldnnCPU;
    case Backend::QuantizedCPU:
      return DispatchKey::QuantizedCPU;
    case Backend::Undefined:
      return DispatchKey::Undefined;
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
    case Backend::MkldnnCPU:
    case Backend::QuantizedCPU:
      return DeviceType::CPU;
    case Backend::Undefined:
      AT_ERROR("Undefined backend is not a valid device type");
    default:
      AT_ERROR("Unknown backend");
  }
}

static inline Backend backendToCPU(Backend b) {
  switch (b) {
    case Backend::CPU:
      return Backend::CPU;
    case Backend::CUDA:
      return Backend::CPU;
    case Backend::HIP:
      return Backend::CPU;
    case Backend::SparseCPU:
      return Backend::SparseCPU;
    case Backend::SparseCUDA:
      return Backend::SparseCPU;
    case Backend::SparseHIP:
      return Backend::SparseCPU;
    case Backend::MSNPU:
    case Backend::XLA:
      return Backend::CPU;
    case Backend::MkldnnCPU:
      return Backend::MkldnnCPU;
    case Backend::QuantizedCPU:
      return Backend::QuantizedCPU;
    case Backend::Undefined:
      return Backend::Undefined;
    default:
      AT_ERROR("Unknown backend");
  }
}

static inline Backend backendToCUDA(Backend b) {
  switch (b) {
    case Backend::CPU:
    case Backend::CUDA:
    case Backend::HIP:
    case Backend::MSNPU:
    case Backend::XLA:
      return Backend::CUDA;
    case Backend::SparseCPU:
    case Backend::SparseCUDA:
    case Backend::SparseHIP:
      return Backend::SparseCUDA;
    case Backend::Undefined:
      return Backend::Undefined;
    default:
      AT_ERROR("Unknown backend");
  }
}

static inline Backend backendToHIP(Backend b) {
  switch (b) {
    case Backend::CPU:
    case Backend::CUDA:
    case Backend::HIP:
    case Backend::MSNPU:
    case Backend::XLA:
      return Backend::HIP;
    case Backend::SparseCPU:
    case Backend::SparseCUDA:
    case Backend::SparseHIP:
      return Backend::SparseHIP;
    case Backend::Undefined:
      return Backend::Undefined;
    default:
      AT_ERROR("Unknown backend");
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
    case Backend::MSNPU:
      return "MSNPU";
    case Backend::XLA:
      return "XLA";
    case Backend::SparseCPU:
      return "SparseCPU";
    case Backend::SparseCUDA:
      return "SparseCUDA";
    case Backend::SparseHIP:
      return "SparseHIP";
    case Backend::MkldnnCPU:
      return "MkldnnCPU";
    case Backend::QuantizedCPU:
      return "QuantizedCPU";
    default:
      return "UNKNOWN_BACKEND";
  }
}

static inline bool isSparse(Backend b) {
  switch (b) {
    case Backend::SparseCPU:
    case Backend::SparseCUDA:
    case Backend::SparseHIP:
      return true;
    default:
      return false;
  }
}

} // namespace c10
