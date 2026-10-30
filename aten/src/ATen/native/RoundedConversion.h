#pragma once

#include <ATen/native/DispatchStub.h>
#include <c10/core/ScalarType.h>
#include <c10/util/FloatRounding.h>

#include <string_view>

namespace at {
struct TensorIteratorBase;
} // namespace at

namespace at::native {

TORCH_API c10::RoundingMode parse_rounding_mode(std::string_view round);

// Checks that src -> dst is a supported narrowing floating-point conversion.
TORCH_API void check_rounded_conversion(ScalarType src, ScalarType dst);

using to_dtype_rounded_fn = void (*)(TensorIteratorBase&, c10::RoundingMode);
DECLARE_DISPATCH(to_dtype_rounded_fn, to_dtype_rounded_stub)

} // namespace at::native
