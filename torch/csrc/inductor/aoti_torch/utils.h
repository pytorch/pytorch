#pragma once

#include <c10/util/Logging.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#define AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(...)    \
  try {                                                    \
    __VA_ARGS__                                            \
  } catch (const std::exception& e) {                      \
    LOG(ERROR) << "Exception in aoti_torch: " << e.what(); \
    return AOTI_TORCH_FAILURE;                             \
  } catch (...) {                                          \
    LOG(ERROR) << "Exception in aoti_torch: UNKNOWN";      \
    return AOTI_TORCH_FAILURE;                             \
  }                                                        \
  return AOTI_TORCH_SUCCESS;
