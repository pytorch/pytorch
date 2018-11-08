#pragma once

namespace torch {
namespace utils {

static inline bool cuda_enabled() {
#ifdef USE_CUDA
  return true;
#else
  return false;
#endif
}

}
}
