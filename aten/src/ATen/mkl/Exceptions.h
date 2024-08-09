#pragma once

#include <string>
#include <stdexcept>
#include <sstream>
#include <mkl_dfti.h>
#include <mkl_spblas.h>

namespace at::native {

static inline void MKL_DFTI_CHECK(MKL_INT status)
{
  if (status && !DftiErrorClass(status, DFTI_NO_ERROR)) {
    std::ostringstream ss;
    ss << "MKL FFT error: " << DftiErrorMessage(status);
    throw std::runtime_error(ss.str());
  }
}

}  // namespace at::native

namespace at::mkl::sparse {
static inline const char* _mklGetErrorString(sparse_status_t status) {
  if (status == SPARSE_STATUS_SUCCESS) {
    return "SPARSE_STATUS_SUCCESS";
  }
  if (status == SPARSE_STATUS_NOT_INITIALIZED) {
    return "SPARSE_STATUS_NOT_INITIALIZED";
  }
  if (status == SPARSE_STATUS_ALLOC_FAILED) {
    return "SPARSE_STATUS_ALLOC_FAILED";
  }
  if (status == SPARSE_STATUS_INVALID_VALUE) {
    return "SPARSE_STATUS_INVALID_VALUE";
  }
  if (status == SPARSE_STATUS_EXECUTION_FAILED) {
    return "SPARSE_STATUS_EXECUTION_FAILED";
  }
  if (status == SPARSE_STATUS_INTERNAL_ERROR) {
    return "SPARSE_STATUS_INTERNAL_ERROR";
  }
  if (status == SPARSE_STATUS_NOT_SUPPORTED) {
    return "SPARSE_STATUS_NOT_SUPPORTED";
  }
  return "<unknown>";
}
} // namespace at::mkl::sparse

#define TORCH_MKLSPARSE_CHECK(EXPR)                 \
  do {                                              \
    sparse_status_t __err = EXPR;                   \
    TORCH_CHECK(                                    \
        __err == SPARSE_STATUS_SUCCESS,             \
        "MKL error: ",                              \
        at::mkl::sparse::_mklGetErrorString(__err), \
        " when calling `" #EXPR "`");               \
  } while (0)

#define TORCH_MKLSPARSE_CHECK_SUCCESS_OR_INVALID(status, function_name) \
  do {                                                   \
    sparse_status_t __status = (status);                 \
    TORCH_CHECK(                                         \
        __status == SPARSE_STATUS_SUCCESS ||             \
            __status == SPARSE_STATUS_INVALID_VALUE,     \
        "MKL error: ",                                   \
        at::mkl::sparse::_mklGetErrorString(__status),   \
        " when calling `" function_name "`");            \
  } while (0)
