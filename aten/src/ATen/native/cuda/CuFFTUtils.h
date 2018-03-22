#pragma once

#include "ATen/ATen.h"
#include "ATen/Config.h"

#include <string>
#include <stdexcept>
#include <sstream>
#include <cufft.h>
#include <cufftXt.h>


namespace at { namespace native {

static inline std::string _cudaGetErrorEnum(cufftResult error)
{
  switch (error)
  {
    case CUFFT_SUCCESS:
      return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA";
    case CUFFT_INCOMPLETE_PARAMETER_LIST:
      return "CUFFT_INCOMPLETE_PARAMETER_LIST";
    case CUFFT_INVALID_DEVICE:
      return "CUFFT_INVALID_DEVICE";
    case CUFFT_PARSE_ERROR:
      return "CUFFT_PARSE_ERROR";
    case CUFFT_NO_WORKSPACE:
      return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED:
      return "CUFFT_NOT_IMPLEMENTED";
    case CUFFT_LICENSE_ERROR:
      return "CUFFT_LICENSE_ERROR";
    case CUFFT_NOT_SUPPORTED:
      return "CUFFT_NOT_SUPPORTED";
    default:
      std::ostringstream ss;
      ss << "unknown error " << error;
      return ss.str();
  }
}

static inline void CUFFT_CHECK(cufftResult error)
{
  if (error != CUFFT_SUCCESS) {
    std::ostringstream ss;
    ss << "cuFFT error: " << _cudaGetErrorEnum(error);
    throw std::runtime_error(ss.str());
  }
}

class CufftHandle {
public:
  explicit CufftHandle() {
    CUFFT_CHECK(cufftCreate(&raw_plan));
  }

  const cufftHandle &get() const { return raw_plan; }

  ~CufftHandle() {
    CUFFT_CHECK(cufftDestroy(raw_plan));
  }
private:
  cufftHandle raw_plan;
};

}} // at::native
