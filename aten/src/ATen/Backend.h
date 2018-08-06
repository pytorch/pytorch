#pragma once
#include <stdexcept>

namespace at {

enum class Backend { CPU, CUDA, SparseCPU, SparseCUDA, Undefined, NumOptions };

constexpr Backend kCPU = Backend::CPU;
constexpr Backend kCUDA = Backend::CUDA;
constexpr Backend kSparseCPU = Backend::SparseCPU;
constexpr Backend kSparseCUDA = Backend::SparseCUDA;

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
