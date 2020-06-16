#pragma once

namespace torch {
namespace utils {

static inline bool cuda_enabled() {
#ifdef USE_ROCM
  return true;
#else
  return false;
#endif
}

}
}
