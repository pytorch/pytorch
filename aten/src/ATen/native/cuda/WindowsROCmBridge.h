// Windows ROCm ABI Bridge - Macro-based solution
// Fixes ArrayRef/std::optional ABI mismatch between MSVC and Clang
#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/ArrayRef.h>
#include <optional>

#if defined(_WIN32) && defined(USE_ROCM)
#define WINDOWS_ROCM_BRIDGE 1

// Reconstruct ArrayRef from (ptr, size) on HIP side
#define ABI_ARRAYREF(Type, ptr, sz) c10::ArrayRef<Type>(ptr, static_cast<size_t>(sz))

// Reconstruct std::optional from pointer on HIP side  
#define ABI_OPTIONAL(ptr) (ptr ? std::make_optional(*ptr) : std::nullopt)

// Extract to raw pointers on CPU side
#define EXTRACT_ARRAYREF(arr) (arr).data(), static_cast<int64_t>((arr).size())
#define EXTRACT_OPTIONAL(opt) ((opt).has_value() ? &(*(opt)) : nullptr)

#else
#define WINDOWS_ROCM_BRIDGE 0
#endif
