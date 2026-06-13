// Copyright © 2025 Apple Inc.

#pragma once

#include <cstddef>

#include "jaccl/types.h"

namespace jaccl {

template <typename T>
struct SumOp {
  void operator()(const T* input, T* output, size_t N) const {
    for (size_t i = 0; i < N; i++) {
      output[i] = output[i] + input[i];
    }
  }
};

template <typename T>
struct MaxOp {
  void operator()(const T* input, T* output, size_t N) const {
    for (size_t i = 0; i < N; i++) {
      output[i] = (output[i] > input[i]) ? output[i] : input[i];
    }
  }
};

template <typename T>
struct MinOp {
  void operator()(const T* input, T* output, size_t N) const {
    for (size_t i = 0; i < N; i++) {
      output[i] = (output[i] < input[i]) ? output[i] : input[i];
    }
  }
};

//
// The last piece of the puzzle to use the native bf16 while compiling a single
// binary for all Macs is to compile these functions with
// target("arch=armv8.6-a").
//
// Now we can simply check in runtime and call them only when they are
// supported.
//

#if defined(__aarch64__)

__attribute__((target("arch=armv8.6-a"))) inline void
native_bf16_sum(const void* input, void* output, size_t N) {
  auto in = reinterpret_cast<const __bf16*>(input);
  auto out = reinterpret_cast<__bf16*>(output);
  for (size_t i = 0; i < N; i++) {
    out[i] = out[i] + in[i];
  }
}

__attribute__((target("arch=armv8.6-a"))) inline void
native_bf16_max(const void* input, void* output, size_t N) {
  auto in = reinterpret_cast<const __bf16*>(input);
  auto out = reinterpret_cast<__bf16*>(output);
  for (size_t i = 0; i < N; i++) {
    out[i] = (out[i] > in[i]) ? out[i] : in[i];
  }
}

__attribute__((target("arch=armv8.6-a"))) inline void
native_bf16_min(const void* input, void* output, size_t N) {
  auto in = reinterpret_cast<const __bf16*>(input);
  auto out = reinterpret_cast<__bf16*>(output);
  for (size_t i = 0; i < N; i++) {
    out[i] = (out[i] < in[i]) ? out[i] : in[i];
  }
}

template <>
struct SumOp<bfloat16_t> {
  void operator()(const bfloat16_t* input, bfloat16_t* output, size_t N) const {
    if (has_native_bf16_support()) {
      native_bf16_sum(input, output, N);
    } else {
      for (size_t i = 0; i < N; i++) {
        output[i] = output[i] + input[i];
      }
    }
  }
};

template <>
struct MaxOp<bfloat16_t> {
  void operator()(const bfloat16_t* input, bfloat16_t* output, size_t N) const {
    if (has_native_bf16_support()) {
      native_bf16_max(input, output, N);
    } else {
      for (size_t i = 0; i < N; i++) {
        output[i] = (output[i] > input[i]) ? output[i] : input[i];
      }
    }
  }
};

template <>
struct MinOp<bfloat16_t> {
  void operator()(const bfloat16_t* input, bfloat16_t* output, size_t N) const {
    if (has_native_bf16_support()) {
      native_bf16_min(input, output, N);
    } else {
      for (size_t i = 0; i < N; i++) {
        output[i] = (output[i] < input[i]) ? output[i] : input[i];
      }
    }
  }
};

#endif // defined(__aarch64__)

} // namespace jaccl
