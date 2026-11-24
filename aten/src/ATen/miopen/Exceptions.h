#pragma once

#include <ATen/miopen/miopen-wrapper.h>
#include <string>
#include <stdexcept>
#include <sstream>

namespace at { namespace native {

class miopen_exception : public std::runtime_error {
public:
  miopenStatus_t status;
  miopen_exception(miopenStatus_t status, const char* msg)
      : std::runtime_error(msg)
      , status(status) {}
  miopen_exception(miopenStatus_t status, const std::string& msg)
      : std::runtime_error(msg)
      , status(status) {}
};

inline void MIOPEN_CHECK(miopenStatus_t status)
{
  if (status != miopenStatusSuccess) {
    if (status == miopenStatusNotImplemented) {
        // This header is installed, so out-of-tree ROCm code may catch
        // miopen_exception or std::runtime_error by name. Retyping it needs a
        // BC decision, not a lint sweep.
        // @allow-raw-throw: installed type, may be caught by name downstream
        throw miopen_exception(status, std::string(miopenGetErrorString(status)) +
                ". This error may appear if you passed in a non-contiguous input.");
    }
    // @allow-raw-throw: installed type, may be caught by name downstream
    throw miopen_exception(status, miopenGetErrorString(status));
  }
}

inline void HIP_CHECK(hipError_t error)
{
  if (error != hipSuccess) {
    std::string msg("HIP error: ");
    msg += hipGetErrorString(error);
    // Same installed-header argument as above: no in-tree caller, so any
    // consumer is out of tree and may catch std::runtime_error by name.
    // @allow-raw-throw: may be caught as std::runtime_error out of tree
    throw std::runtime_error(msg);
  }
}

}} // namespace at::native
