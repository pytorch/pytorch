#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/RoundedConversion.h>

#include <ATen/Dispatch_v2.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

namespace at::native {
namespace {

// Pairs rejected by check_rounded_conversion are not instantiated, which needs
// a template: AT_DISPATCH's lambdas are not.
template <typename src_t, typename dst_t>
void to_dtype_rounded_loop(TensorIteratorBase& iter, c10::RoundingMode mode) {
  if constexpr (
      !std::is_same_v<src_t, dst_t> &&
      (std::is_same_v<src_t, double> || !std::is_same_v<dst_t, float>)) {
    // Half and BFloat16 widen to float exactly.
    using wide_t =
        std::conditional_t<std::is_same_v<src_t, double>, double, float>;
    cpu_kernel(iter, [mode](src_t x) -> dst_t {
      return c10::convert_with_rounding<dst_t>(static_cast<wide_t>(x), mode);
    });
  }
}

void to_dtype_rounded_kernel(TensorIteratorBase& iter, c10::RoundingMode mode) {
  AT_DISPATCH_V2(
      iter.input_dtype(),
      "to_dtype_rounded_cpu",
      AT_WRAP([&] {
        using src_t = scalar_t;
        AT_DISPATCH_V2(
            iter.dtype(),
            "to_dtype_rounded_cpu",
            AT_WRAP([&] { to_dtype_rounded_loop<src_t, scalar_t>(iter, mode); }),
            kFloat,
            kHalf,
            kBFloat16,
            AT_EXPAND(AT_FLOAT8_TYPES));
      }),
      kDouble,
      kFloat,
      kHalf,
      kBFloat16);
}

} // namespace

REGISTER_DISPATCH(to_dtype_rounded_stub, &to_dtype_rounded_kernel)

} // namespace at::native
