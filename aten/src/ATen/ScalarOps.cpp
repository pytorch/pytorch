#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch_v2.h>
#include <ATen/EmptyTensor.h>
#include <ATen/ScalarOps.h>
#include <ATen/core/LegacyTypeDispatch.h>

#include <unordered_set>

namespace at {
namespace {
template <typename scalar_t>
inline void fill_inplace(Tensor& self, const Scalar& value_scalar) {
  scalar_t value{};

  if constexpr (std::is_same_v<scalar_t, at::Half> ||
                std::is_same_v<scalar_t, at::BFloat16> ||
                std::is_same_v<scalar_t, at::Float8_e5m2> ||
                std::is_same_v<scalar_t, at::Float8_e5m2fnuz> ||
                std::is_same_v<scalar_t, at::Float8_e4m3fn> ||
                std::is_same_v<scalar_t, at::Float8_e4m3fnuz> ||
                std::is_same_v<scalar_t, at::Float8_e8m0fnu> ||
                std::is_same_v<scalar_t, at::Float8_e5m3fnu>) {
    // relaxed float cast: allow inf similar to the torch.tensor constructor
    //
    // without this, we had the following divergence:
    //   torch.tensor(1123581321.0, dtype=torch.float16)
    //     => tensor(inf, dtype=torch.float16)
    //   torch.ops.aten.scalar_tensor.default(1123581321, dtype=torch.float16)
    //     => RuntimeError: value cannot be converted to type at::Half without overflow

    value = static_cast<scalar_t>(value_scalar.to<double>());
  } else {
    value = value_scalar.to<scalar_t>();
  }

  scalar_t* dptr = static_cast<scalar_t*>(self.data_ptr());
  *dptr = value;
}
}

namespace detail {
Tensor& scalar_fill(Tensor& self, const Scalar& value) {
  AT_DISPATCH_V2(
      self.scalar_type(), "fill_out", AT_WRAP([&]() {
        fill_inplace<scalar_t>(self, value);
      }), kComplexHalf, kBComplex32, kHalf, kBool, kBFloat16, AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), AT_EXPAND(AT_FLOAT8_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  return self;
}

Tensor scalar_tensor_static(const Scalar& s, std::optional<ScalarType> dtype_opt, std::optional<Device> device_opt) {
  at::tracer::impl::NoTracerDispatchMode tracer_guard;
  at::AutoDispatchBelowAutograd mode;
  Tensor result = at::detail::empty_cpu(
      {}, dtype_opt, std::nullopt, device_opt, std::nullopt, std::nullopt);
  scalar_fill(result, s);
  return result;
}
} // namespace detail

// TODO: remove this. This is a temporary list of functions that allow Python
// numbers to bind to Tensors. Some binary ops have separate Tensor and Scalar
// overloads and binding to the Tensor overload with a number of a different
// type will trigger a type error.
//
// If you modify this, you will need to adjust the blocklist in
// tools/pyi/gen_pyi.py (and add hardcoded signatures for these
// functions.)
bool should_allow_numbers_as_tensors(std::string_view name) {
  static const std::unordered_set<std::string_view> allowed = {
      "add",
      "add_",
      "add_out",
      "div",
      "div_",
      "div_out",
      "divide",
      "divide_",
      "divide_out", // alias of div
      "mul",
      "mul_",
      "mul_out",
      "multiply",
      "multiply_",
      "multiply_out", // alias of mul
      "sub",
      "sub_",
      "sub_out",
      "subtract",
      "subtract_",
      "subtract_out", // alias of sub
      "true_divide",
      "true_divide_",
      "true_divide_out",
      "to",
      "_to_copy",
      "copy_",
      "copy",
      "floor_divide",
      "floor_divide_",
      "floor_divide_out",
      "_conj"}; // _conj needed because mul.Tensor backward calls it
  return allowed.contains(name);
}
} // namespace at
