#pragma once

#include <string>
#include <stdexcept>
#include <sstream>
#include <mkl_dfti.h>
#include <mkl_spblas.h>

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

namespace at { namespace mkl { namespace sparse {
const char* _mklGetErrorString(sparse_status_t status);
}}} // namespace at::mkl::sparse

#define TORCH_MKLSPARSE_CHECK(EXPR)     \
  do {                                  \
    sparse_status_t __err = EXPR;       \
    TORCH_CHECK(                        \
        __err == SPARSE_STATUS_SUCCESS, \
        "MKL error: ",                  \
        _mklGetErrorString(__err),      \
        " when calling `" #EXPR "`");   \
  } while (0)
