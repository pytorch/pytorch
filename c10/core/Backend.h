#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/TensorTypeId.h>
#include <c10/core/TensorTypeIdRegistration.h>
#include <c10/util/Exception.h>

#include <stdexcept>

namespace c10 {

/**
 * This legacy enum class defines the set of backends supported by
 * old school, code generated Type-based ATen.  The reason we are
 * sunsetting this enum class is because it doesn't allow for
 * open registration of backends.  TensorTypeId is the replacement
 * for Backend which supports open registration.
 *
 * ARE YOU SURE YOU WANT TO USE THIS TYPE?  Think about if SparseCPU/SparseCUDA
 * would make sense in your use case.  If it doesn't make sense, maybe
 * you want DeviceType.
 */
enum class Backend { CPU, CUDA, HIP, SparseCPU, SparseCUDA, SparseHIP, MSNPU, XLA, Undefined, NumOptions };

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
    default:
      throw std::runtime_error("Unknown backend");
  }
}

static inline Backend tensorTypeIdToBackend(TensorTypeId t) {
  if (t == CPUTensorId()) {
    return Backend::CPU;
  } else if (t == CUDATensorId()) {
    return Backend::CUDA;
  } else if (t == HIPTensorId()) {
    return Backend::HIP;
  } else if (t == MSNPUTensorId()) {
    return Backend::MSNPU;
  } else if (t == XLATensorId()) {
    return Backend::XLA;
  } else if (t == SparseCPUTensorId()) {
    return Backend::SparseCPU;
  } else if (t == SparseCUDATensorId()) {
    return Backend::SparseCUDA;
  } else if (t == SparseHIPTensorId()) {
    return Backend::SparseHIP;
  } else if (t == UndefinedTensorId()) {
    return Backend::Undefined;
  } else {
    AT_ERROR("Unrecognized tensor type ID: ", t);
  }
}

static inline TensorTypeId backendToTensorTypeId(Backend b) {
  switch (b) {
    case Backend::CPU:
      return CPUTensorId();
    case Backend::CUDA:
      return CUDATensorId();
    case Backend::HIP:
      return HIPTensorId();
    case Backend::MSNPU:
      return MSNPUTensorId();
    case Backend::XLA:
      return XLATensorId();
    case Backend::SparseCPU:
      return SparseCPUTensorId();
    case Backend::SparseCUDA:
      return SparseCUDATensorId();
    case Backend::SparseHIP:
      return SparseHIPTensorId();
    case Backend::Undefined:
      return UndefinedTensorId();
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
    case Backend::Undefined:
      AT_ERROR("Undefined backend is not a valid device type");
    default:
      AT_ERROR("Unknown backend");
  }
}

static inline Backend deviceTypeToBackend(DeviceType d) {
  switch (d) {
    case DeviceType::CPU:
      return Backend::CPU;
    case DeviceType::CUDA:
      return Backend::CUDA;
    case DeviceType::HIP:
      return Backend::HIP;
    case DeviceType::MSNPU:
      return Backend::MSNPU;
    case DeviceType::XLA:
      return Backend::XLA;
    default:
      AT_ERROR("Unknown device type ", d);
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

constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;
constexpr DeviceType kHIP = DeviceType::HIP;
constexpr DeviceType kMSNPU = DeviceType::MSNPU;
constexpr DeviceType kXLA = DeviceType::XLA;

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
    default:
      return "UNKNOWN_BACKEND";
  }
}

} // namespace c10
