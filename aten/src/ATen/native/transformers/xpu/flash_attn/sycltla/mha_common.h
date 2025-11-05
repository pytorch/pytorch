#pragma once
#include <cassert>

#include <ATen/xpu/XPUContext.h>

#include <ATen/native/transformers/xpu/flash_attn/flash_api.h>
#include <ATen/native/transformers/xpu/flash_attn/sycltla/flash_api.h>
#include <ATen/native/transformers/xpu/flash_attn/utils.h>

/**
 * Panic wrapper for unwinding CUTLASS errors
 */
#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_xpu(), #x " must be on XPU")

#define CHECK_SHAPE(x, ...)                        \
  TORCH_CHECK(                                     \
      x.sizes() == at::IntArrayRef({__VA_ARGS__}), \
      #x " must have shape (" #__VA_ARGS__ ")")

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define FP16_SWITCH(COND, ...)            \
  [&] {                                   \
    if (COND) {                           \
      using elem_type = cute::half_t;     \
      return __VA_ARGS__();               \
    } else {                              \
      using elem_type = cute::bfloat16_t; \
      return __VA_ARGS__();               \
    }                                     \
  }()
