#pragma once

#include <mkl_dfti.h>
#include <sstream>
#include <stdexcept>
#include <string>

namespace at {
namespace native {

static inline void MKL_DFTI_CHECK(MKL_INT status) {
  if (status && !DftiErrorClass(status, DFTI_NO_ERROR)) {
    std::ostringstream ss;
    ss << "MKL FFT error: " << DftiErrorMessage(status);
    throw std::runtime_error(ss.str());
  }
}

} // namespace native
} // namespace at
