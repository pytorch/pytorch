#pragma once

#include <string>
#include <stdexcept>
#include <sstream>
#include <hipdnn_frontend.hpp>

namespace at { namespace native {

class hipdnn_exception : public std::runtime_error {
public:
  hipdnnStatus_t status;
  hipdnn_exception(hipdnnStatus_t status, const char* msg)
      : std::runtime_error(msg)
      , status(status) {}
  hipdnn_exception(hipdnnStatus_t status, const std::string& msg)
      : std::runtime_error(msg)
      , status(status) {}
};

class hipdnn_frontend_exception : public std::runtime_error {
  public:
    const hipdnn_frontend::Error status;
    hipdnn_frontend_exception(hipdnn_frontend::Error status, const char* msg)
        : std::runtime_error(msg)
        , status(status) {}
    hipdnn_frontend_exception(hipdnn_frontend::Error status, const std::string& msg)
        : std::runtime_error(msg)
        , status(status) {}
  };

inline void HIPDNN_CHECK(hipdnnStatus_t status)
{
  if (status != HIPDNN_STATUS_SUCCESS ) {
    if (status == HIPDNN_STATUS_NOT_SUPPORTED) {
        throw hipdnn_exception(status, std::string(hipdnnGetErrorString(status)) +
                ". This error may appear if you passed in a non-contiguous input.");
    }
    throw hipdnn_exception(status, hipdnnGetErrorString(status));
  }
}

inline void HIPDNN_FE_CHECK(const hipdnn_frontend::Error& status)
{
    if(!status.is_good())
    {
      throw hipdnn_frontend_exception(status, status.get_message());
    }
}

}} // namespace at::native
