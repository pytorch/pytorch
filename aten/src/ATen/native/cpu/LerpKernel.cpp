#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/Lerp.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

#include <c10/util/irange.h>

namespace at {
namespace native {
namespace {

template <typename scalar_t>
Vectorized<scalar_t> is_lerp_weight_small(Vectorized<scalar_t> weight) {
  static_assert(!c10::is_complex<scalar_t>::value, "");
  return weight.abs() < Vectorized<scalar_t>(0.5);
}

// is_lerp_weight_small doesn't work for complex because z.abs() returns a
// complex vector which can't be compared. Either implement it with z.abs_2_(),
// or fallback to the scalar function.
#if !(defined(CPU_CAPABILITY_DEFAULT) || defined(_MSC_VER))
template <typename value_t>
Vectorized<c10::complex<value_t>> is_lerp_weight_small(Vectorized<c10::complex<value_t>> weight) {
  using vec_reg_t = decltype(weight.abs_2_());
  vec_reg_t mask = Vectorized<value_t>(weight.abs_2_()) < Vectorized<value_t>(0.25);
  return Vectorized<c10::complex<value_t>>(mask);
}
#else
template <typename scalar_t>
Vectorized<scalar_t> lerp_vec_map(Vectorized<scalar_t> start, Vectorized<scalar_t> end, Vectorized<scalar_t> weight) {
  using vec_t = Vectorized<scalar_t>;
  __at_align__ scalar_t start_arr[vec_t::size()];
  __at_align__ scalar_t end_arr[vec_t::size()];
  __at_align__ scalar_t weight_arr[vec_t::size()];
  __at_align__ scalar_t result_arr[vec_t::size()];

  start.store(start_arr);
  end.store(end_arr);
  weight.store(weight_arr);

  for (auto i : c10::irange(vec_t::size())) {
    result_arr[i] = lerp(start_arr[i], end_arr[i], weight_arr[i]);
  }
  return vec_t::loadu(result_arr);
}

template <typename value_t>
Vectorized<c10::complex<value_t>> lerp_vec(Vectorized<c10::complex<value_t>> start, Vectorized<c10::complex<value_t>> end, Vectorized<c10::complex<value_t>> weight) {
  return lerp_vec_map(start, end, weight);
}
#endif

template <typename scalar_t>
Vectorized<scalar_t> lerp_vec(Vectorized<scalar_t> start, Vectorized<scalar_t> end, Vectorized<scalar_t> weight) {
  using vec_t = Vectorized<scalar_t>;
  auto mask = is_lerp_weight_small(weight);
  auto coeff = vec_t::blendv(weight - vec_t(1), weight, mask);
  auto base = vec_t::blendv(end, start, mask);
  return vec::fmadd(coeff, end - start, base);
}

void lerp_scalar_kernel(at::TensorIteratorBase& iter, const Scalar& weight) {
  if (iter.common_dtype() == kBFloat16) {
    using bVec = Vectorized<BFloat16>;
    using fVec = Vectorized<float>;
    float weight_val = weight.to<float>();
    auto weight_vec = fVec(weight_val);
    at::native::cpu_kernel_vec(
      iter,
      [weight_val](BFloat16 self_val, BFloat16 end_val) -> BFloat16 {
        return lerp(self_val, end_val, weight_val);
      },
      [=](bVec self_vec, bVec end_vec) -> bVec {
          auto [self_vec0, self_vec1] = convert_bfloat16_float(self_vec);
          auto [end_vec0, end_vec1] = convert_bfloat16_float(end_vec);
          auto result0 = lerp_vec(self_vec0, end_vec0, weight_vec);
          auto result1 = lerp_vec(self_vec1, end_vec1, weight_vec);
          return convert_float_bfloat16(result0, result1);
      });
  } else if (iter.common_dtype() == kHalf) {
    using hVec = Vectorized<Half>;
    using fVec = Vectorized<float>;
    float weight_val = weight.to<float>();
    auto weight_vec = fVec(weight_val);
    at::native::cpu_kernel_vec(
      iter,
      [weight_val](Half self_val, Half end_val) -> Half {
        return lerp(self_val, end_val, weight_val);
      },
      [=](hVec self_vec, hVec end_vec) -> hVec {
          auto [self_vec0, self_vec1] = convert_half_float(self_vec);
          auto [end_vec0, end_vec1] = convert_half_float(end_vec);
          auto result0 = lerp_vec(self_vec0, end_vec0, weight_vec);
          auto result1 = lerp_vec(self_vec1, end_vec1, weight_vec);
          return convert_float_half(result0, result1);
      });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.common_dtype(), "lerp_kernel_scalar", [&] {
      auto weight_val = weight.to<scalar_t>();
      at::native::cpu_kernel_vec(
          iter,
          [weight_val](scalar_t self_val, scalar_t end_val) {
            return lerp(self_val, end_val, weight_val);
          },
          [weight_val](Vectorized<scalar_t> self, Vectorized<scalar_t> end) {
            const Vectorized<scalar_t> weight(weight_val);
            return lerp_vec(self, end, weight);
          });
    });
  }
}

void lerp_tensor_kernel(at::TensorIteratorBase& iter) {
  if (iter.common_dtype() == kBFloat16) {
    using bVec = Vectorized<BFloat16>;
    at::native::cpu_kernel_vec(
      iter,
      [=](BFloat16 self_val, BFloat16 end_val, BFloat16 weight_val) -> BFloat16 {
        return lerp(self_val, end_val, weight_val);
      },
      [=](bVec self_vec, bVec end_vec, bVec weight_vec) -> bVec {
          auto [self_vec0, self_vec1] = convert_bfloat16_float(self_vec);
          auto [end_vec0, end_vec1] = convert_bfloat16_float(end_vec);
          auto [weight_vec0, weight_vec1] = convert_bfloat16_float(weight_vec);
          auto result0 = lerp_vec(self_vec0, end_vec0, weight_vec0);
          auto result1 = lerp_vec(self_vec1, end_vec1, weight_vec1);
          return convert_float_bfloat16(result0, result1);
      });
  } else if (iter.common_dtype() == kHalf) {
    using hVec = Vectorized<Half>;
    at::native::cpu_kernel_vec(
      iter,
      [=](Half self_val, Half end_val, Half weight_val) -> Half {
        return lerp(self_val, end_val, weight_val);
      },
      [=](hVec self_vec, hVec end_vec, hVec weight_vec) -> hVec {
          auto [self_vec0, self_vec1] = convert_half_float(self_vec);
          auto [end_vec0, end_vec1] = convert_half_float(end_vec);
          auto [weight_vec0, weight_vec1] = convert_half_float(weight_vec);
          auto result0 = lerp_vec(self_vec0, end_vec0, weight_vec0);
          auto result1 = lerp_vec(self_vec1, end_vec1, weight_vec1);
          return convert_float_half(result0, result1);
      });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.common_dtype(), "lerp_kernel_tensor", [&] {
      at::native::cpu_kernel_vec(
          iter,
          [](scalar_t self_val, scalar_t end_val, scalar_t weight_val) {
            return lerp(self_val, end_val, weight_val);
          },
          [](Vectorized<scalar_t> self_val, Vectorized<scalar_t> end_val, Vectorized<scalar_t> weight_val) {
            return lerp_vec(self_val, end_val, weight_val);
          });
    });
  }
}

} // anonymous namespace

REGISTER_DISPATCH(lerp_kernel_scalar_weight, &lerp_scalar_kernel);
REGISTER_DISPATCH(lerp_kernel_tensor_weight, &lerp_tensor_kernel);

} // namespace native
} // namespace at
