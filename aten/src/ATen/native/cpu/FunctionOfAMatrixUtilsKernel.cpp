#include <ATen/native/FunctionOfAMatrixUtils.h>

#include <ATen/native/cpu/Loops.h>

#if (defined(_WIN32) || defined(_WIN64))
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

namespace at { namespace native {

namespace {

void _compute_linear_combination_cpu_kernel(
  TensorIterator& iter,
  int64_t in_stride,
  int64_t coeff_stride,
  int64_t num_summations
) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
    at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16,
    iter.dtype(),
    "_compute_linear_combination_cpu", [&] {
      auto loop = [&](char** data, const int64_t* strides, int64_t n) {
        auto* RESTRICT out_ptr = data[0];
        auto* RESTRICT in_ptr = data[1];
        auto* RESTRICT coeff_ptr = data[2];

        for (int64_t elem = 0; elem < n; ++elem) {
          auto* RESTRICT out_data = reinterpret_cast<scalar_t*>(out_ptr);
          auto* RESTRICT in_data = reinterpret_cast<scalar_t*>(in_ptr);
          using primitive_t = typename scalar_value_type<scalar_t>::type;
          auto* RESTRICT coeff_data = reinterpret_cast<primitive_t*>(coeff_ptr);

          // perform summation
          for (int32_t i = 0; i < num_summations; ++i) {
            *out_data += in_data[i * in_stride] * coeff_data[i * coeff_stride];
          }

          out_ptr += strides[0];
          in_ptr += strides[1];
          coeff_ptr += strides[2];
        }
      };
      iter.for_each(loop);
  });
}

}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(_compute_linear_combination_stub, &_compute_linear_combination_cpu_kernel);

}} // namespace at::native
