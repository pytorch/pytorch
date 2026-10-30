#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/RoundedConversion.h>

#include <ATen/TensorIterator.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_to_dtype_rounded_native.h>
#endif

namespace at::native {

c10::RoundingMode parse_rounding_mode(std::string_view round) {
  if (round == "to_nearest_even") {
    return c10::RoundingMode::ToNearestEven;
  } else if (round == "toward_zero") {
    return c10::RoundingMode::TowardZero;
  } else if (round == "toward_positive") {
    return c10::RoundingMode::TowardPositive;
  } else if (round == "toward_negative") {
    return c10::RoundingMode::TowardNegative;
  }
  TORCH_CHECK_VALUE(
      false,
      "round must be one of 'to_nearest_even', 'toward_zero', ",
      "'toward_positive' or 'toward_negative', but got '",
      round,
      "'");
}

void check_rounded_conversion(ScalarType src, ScalarType dst) {
  const auto is_narrowing = [&]() {
    switch (src) {
      case kDouble:
        return dst == kFloat || dst == kHalf || dst == kBFloat16 ||
            isFloat8Type(dst);
      case kFloat:
        return dst == kHalf || dst == kBFloat16 || isFloat8Type(dst);
      case kHalf:
        return dst == kBFloat16 || isFloat8Type(dst);
      case kBFloat16:
        return dst == kHalf || isFloat8Type(dst);
      default:
        return false;
    }
  };
  TORCH_CHECK(
      is_narrowing(),
      "round is only supported when converting a float64, float32, float16 or ",
      "bfloat16 tensor to a narrower floating-point dtype (one that cannot ",
      "represent all its values exactly), but got a conversion from ",
      src,
      " to ",
      dst);
}

DEFINE_DISPATCH(to_dtype_rounded_stub);

Tensor& _to_dtype_rounded_out(
    const Tensor& self,
    ScalarType dtype,
    std::string_view round,
    std::optional<MemoryFormat> /*memory_format*/,
    Tensor& out) {
  check_rounded_conversion(self.scalar_type(), dtype);
  const auto mode = parse_rounding_mode(round);
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(true)
                  .check_all_same_dtype(false)
                  .declare_static_dtype(dtype)
                  .declare_static_device(self.device())
                  .add_output(out)
                  .add_const_input(self)
                  .build();
  if (!out.is_meta()) {
    to_dtype_rounded_stub(iter.device_type(), iter, mode);
  }
  return out;
}

} // namespace at::native
