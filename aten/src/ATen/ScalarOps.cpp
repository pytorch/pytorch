#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/EmptyTensor.h>
#include <ATen/ScalarOps.h>

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
                std::is_same_v<scalar_t, at::Float8_e8m0fnu>) {
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
      }), kComplexHalf, kHalf, kBool, kBFloat16, AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), AT_EXPAND(AT_FLOAT8_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
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
} // namespace at
