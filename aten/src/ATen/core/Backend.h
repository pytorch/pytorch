#pragma once

#include <ATen/core/TensorTypeId.h>
#include <ATen/core/TensorTypeIdRegistration.h>
#include <ATen/core/Error.h>
#include <ATen/core/DeviceType.h>

#include <stdexcept>

namespace at {

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
enum class Backend { CPU, CUDA, SparseCPU, SparseCUDA, Undefined, NumOptions };

static inline Backend toSparse(Backend b) {
  switch (b) {
    case Backend::CPU:
      return Backend::SparseCPU;
    case Backend::CUDA:
      return Backend::SparseCUDA;
    case Backend::SparseCPU:
      return Backend::SparseCPU;
    case Backend::SparseCUDA:
      return Backend::SparseCUDA;
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
    case Backend::SparseCPU:
      return Backend::CPU;
    case Backend::SparseCUDA:
      return Backend::CUDA;
    default:
      throw std::runtime_error("Unknown backend");
  }
}

static inline Backend tensorTypeIdToBackend(TensorTypeId t) {
  if (t == CPUTensorId()) {
    return Backend::CPU;
  } else if (t == CUDATensorId()) {
    return Backend::CUDA;
  } else if (t == SparseCPUTensorId()) {
    return Backend::SparseCPU;
  } else if (t == SparseCUDATensorId()) {
    return Backend::SparseCUDA;
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
    case Backend::SparseCPU:
      return SparseCPUTensorId();
    case Backend::SparseCUDA:
      return SparseCUDATensorId();
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
    case Backend::SparseCPU:
      return DeviceType::CPU;
    case Backend::SparseCUDA:
      return DeviceType::CUDA;
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
    case Backend::SparseCPU:
      return Backend::SparseCPU;
    case Backend::SparseCUDA:
      return Backend::SparseCPU;
    case Backend::Undefined:
      return Backend::Undefined;
    default:
      AT_ERROR("Unknown backend");
  }
}

static inline Backend backendToCUDA(Backend b) {
  switch (b) {
    case Backend::CPU:
      return Backend::CUDA;
    case Backend::CUDA:
      return Backend::CUDA;
    case Backend::SparseCPU:
      return Backend::SparseCUDA;
    case Backend::SparseCUDA:
      return Backend::SparseCUDA;
    case Backend::Undefined:
      return Backend::Undefined;
    default:
      AT_ERROR("Unknown backend");
  }
}

constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;

static inline const char* toString(Backend b) {
  switch (b) {
    case Backend::CPU:
      return "CPU";
    case Backend::CUDA:
      return "CUDA";
    case Backend::SparseCPU:
      return "SparseCPU";
    case Backend::SparseCUDA:
      return "SparseCUDA";
    default:
      return "UNKNOWN_BACKEND";
  }
}

} // namespace at
