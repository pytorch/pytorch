#pragma once

#include <string>
#include <stdexcept>
#include <sstream>
#include <mkl_dfti.h>

namespace at { namespace native {

static inline void MKL_DFTI_CHECK(MKL_INT status)
{
  if (status && !DftiErrorClass(status, DFTI_NO_ERROR)) {
    std::ostringstream ss;
    ss << "MKL FFT error: " << DftiErrorMessage(status);
    throw std::runtime_error(ss.str());
  }
}

}}  // namespace at::native
