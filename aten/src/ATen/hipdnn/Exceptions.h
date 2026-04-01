#pragma once

#include <c10/util/Exception.h>
#include <hipdnn_frontend.hpp>

namespace c10 {

class HipDNNError : public c10::Error {
  using Error::Error;
};

} // namespace c10

#define HIPDNN_CHECK(EXPR, ...)                                         \
  do {                                                                  \
    hipdnnStatus_t status = EXPR;                                       \
    if (status != HIPDNN_STATUS_SUCCESS) {                              \
      if (status == HIPDNN_STATUS_NOT_SUPPORTED) {                      \
        TORCH_CHECK_WITH(                                               \
            HipDNNError,                                                \
            false,                                                      \
            "hipDNN error: ",                                           \
            hipdnnGetErrorString(status),                               \
            ". This error may appear if you passed in a non-contiguous" \
            " input.",                                                  \
            ##__VA_ARGS__);                                             \
      } else {                                                          \
        TORCH_CHECK_WITH(                                               \
            HipDNNError,                                                \
            false,                                                      \
            "hipDNN error: ",                                           \
            hipdnnGetErrorString(status),                               \
            ##__VA_ARGS__);                                             \
      }                                                                 \
    }                                                                   \
  } while (0)

#define HIPDNN_FE_CHECK(EXPR)          \
  do {                                 \
    auto error_object = EXPR;          \
    if (!error_object.is_good()) {     \
      TORCH_CHECK_WITH(                \
          HipDNNError,                 \
          false,                       \
          "hipDNN Frontend error: ",   \
          error_object.get_message()); \
    }                                  \
  } while (0)
