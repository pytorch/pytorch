#pragma once

#include <ATen/native/DispatchStub.h>
#include <c10/core/ScalarType.h>
#include <c10/util/FloatRounding.h>

#include <string_view>

namespace at {
class TensorBase;
struct TensorIteratorBase;
} // namespace at

namespace at::native {

TORCH_API c10::RoundingMode parse_rounding_mode(std::string_view round);

// Checks that src -> dst is a supported narrowing floating-point conversion.
TORCH_API void check_rounded_conversion(ScalarType src, ScalarType dst);

// Checks the arguments of round_to, in the same order as its meta function.
TORCH_API c10::RoundingMode check_round_to(
    const TensorBase& self,
    ScalarType dtype,
    std::string_view mode);

using to_dtype_rounded_fn = void (*)(TensorIteratorBase&, c10::RoundingMode);
DECLARE_DISPATCH(to_dtype_rounded_fn, to_dtype_rounded_stub)

} // namespace at::native
