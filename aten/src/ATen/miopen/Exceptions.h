#pragma once

#include "miopen-wrapper.h"
#include <string>
#include <stdexcept>
#include <sstream>

struct THCState;

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
    throw miopen_exception(status, "Error");	// FIXME: error string?
  }
}

inline void CUDA_CHECK(hipError_t error)
{
  if (error != hipSuccess) {
    std::string msg("HIP error: ");
    //msg += hipErrorString(error);
    throw std::runtime_error(msg);
  }
}

}}  // namespace at::miopen
