
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <iostream>

#define AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(...)      \
  try {                                                      \
    __VA_ARGS__                                              \
  } catch (const std::exception& e) {                        \
    std::cerr << "Error: " << e.what() << std::endl;         \
    return AOTI_TORCH_FAILURE;                               \
  } catch (...) {                                            \
    std::cerr << "Unknown exception occurred." << std::endl; \
    return AOTI_TORCH_FAILURE;                               \
  }                                                          \
  return AOTI_TORCH_SUCCESS;
