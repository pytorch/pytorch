#pragma once

#include <assert.h>
#include <stdint.h>
#include <memory>
#include <stdexcept>
#include <string>

#ifdef __GNUC__
#define AOT_INDUCTOR_EXPORT \
  __attribute__((__visibility__("default"))) __attribute__((used))
#else // !__GNUC__
#ifdef _WIN32
#define AOT_INDUCTOR_EXPORT __declspec(dllexport)
#else // !_WIN32
#define AOT_INDUCTOR_EXPORT
#endif // _WIN32
#endif // __GNUC__

#define AOT_INDUCTOR_CHECK(cond, msg) \
  { assert((cond) && (msg)); }

#define AOT_VECTOR_SIZE_CHECK(vec, expected_size)                 \
  do {                                                            \
    auto actual_size = vec.size();                                \
    std::string msg = "expected vector size to be ";              \
    msg += std::to_string(expected_size);                         \
    msg += ", but got ";                                          \
    msg += std::to_string(actual_size);                           \
    AOT_INDUCTOR_CHECK(actual_size == expected_size, msg.c_str()) \
  } while (0);

#define CUDA_ERROR_CHECK(EXPR)                 \
  do {                                         \
    cudaError_t __err = EXPR;                  \
    if (__err != cudaSuccess) {                \
      std::string msg = "CUDA driver error: "; \
      msg += std::to_string(__err);            \
      AOT_INDUCTOR_CHECK(false, msg.c_str())   \
    }                                          \
  } while (0);

#define CUDA_ERROR_CHECK(EXPR)                 \
  do {                                         \
    cudaError_t __err = EXPR;                  \
    if (__err != cudaSuccess) {                \
      std::string msg = "CUDA driver error: "; \
      msg += std::to_string(__err);            \
      AOT_INDUCTOR_CHECK(false, msg.c_str())   \
    }                                          \
  } while (0);

#define AOT_VECTOR_SIZE_CHECK(vec, expected_size)                 \
  do {                                                            \
    auto actual_size = vec.size();                                \
    std::string msg = "expected vector size to be ";              \
    msg += std::to_string(expected_size);                         \
    msg += ", but got ";                                          \
    msg += std::to_string(actual_size);                           \
    AOT_INDUCTOR_CHECK(actual_size == expected_size, msg.c_str()) \
  } while (0);
