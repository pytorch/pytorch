#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/FunctionOfAMatrixUtils.h>

#include <ATen/core/Tensor.h>
#include <ATen/TensorIterator.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_compute_linear_combination_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

DEFINE_DISPATCH(_compute_linear_combination_stub);

// If `coefficients` is a [m, n] Tensor and
// `input` is a [n, ...] Tensor, then the output
// `output` is going to be a [m, ...] Tensor such that
// for i in range(m):
//    for j in range(n):
//        output[i, ...] += coefficients[i, j] * input[j, ...]
//
// Note: if input.dtype == scalar_t<T>, then coefficients.dtype == T.
// This is relevant when scalar_t<T> == complex<T>.
Tensor _compute_linear_combination(const Tensor& input, const Tensor& coefficients) {
  TORCH_CHECK(input.ndimension() > 0 && input.numel() > 0, "Empty tensor not supported");
  auto output_first_dim_size = coefficients.size(0);

  auto output_sizes = input.sizes().vec();
  output_sizes[0] = output_first_dim_size;
  auto output = at::zeros(
    output_sizes,
    input.options().memory_format(at::MemoryFormat::Contiguous)
  );

  native::_compute_linear_combination_out(input, coefficients, output);

  return output;
}

// Note: the function is implemented using the __restrict__ memory modifier,
// which means that if `output` actually is aliased by `input`, the result
// produced is undefined.
Tensor& _compute_linear_combination_out(const Tensor& input, const Tensor& coefficients, Tensor& output) {
  auto output_first_dim_size = coefficients.size(0);
  auto input_first_dim_size = coefficients.size(1);

  // Recall that `coefficients` is a [m, n] Tensor,
  // `input` is a [n, ...] Tensor, `output` is a [m, ...] Tensor.
  // We restride Tensors to the common dim == input.dim() + 1, so that
  // coefficients.sizes() = [m, 1 (instead of n), 1 repeated (input.dim() - 1) times],
  // input.sizes() = [1, 1 (instead of n), ...],
  // output.sizes() = [m, 1 (instead of n), ...].
  // The second dimension in newly restrided Tensors is traversed inside the kernels.
  // This is done to avoid synchronizations/atomic operations in the kernels
  // and also guarantees determinism, required by the autograd.

  // restride output
  auto output_to_broadcasted_dim = output.unsqueeze(1);
  auto output_restrided_sizes = output_to_broadcasted_dim.sizes().vec();
  auto output_restrided_strides = output_to_broadcasted_dim.strides().vec();
  output_restrided_sizes[1] = 1;
  output_restrided_strides[1] = 0;
  auto output_restrided = output.as_strided(
    output_restrided_sizes,
    output_restrided_strides
  );

  // restride input
  auto input_to_broadcasted_dim = input.unsqueeze(0);
  auto input_restrided_sizes = input_to_broadcasted_dim.sizes().vec();
  auto input_restrided_strides = input_to_broadcasted_dim.strides().vec();
  input_restrided_sizes[1] = 1;
  input_restrided_strides[1] = 0;
  auto input_restrided = input.as_strided(
    input_restrided_sizes,
    input_restrided_strides
  );

  // restride coefficients
  auto coefficients_restrided_sizes = std::vector<int64_t>(input.dim() + 1, 1);
  coefficients_restrided_sizes[0] = output_first_dim_size;
  coefficients_restrided_sizes[1] = 1;
  auto coefficients_restrided_strides = std::vector<int64_t>(input.dim() + 1, 0);
  coefficients_restrided_strides[0] = coefficients.stride(0);
  coefficients_restrided_strides[1] = 0;
  auto coefficients_restrided = coefficients.as_strided(
    coefficients_restrided_sizes,
    coefficients_restrided_strides
  );

  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)  // Output is intentionally 0 strided above
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .add_output(output_restrided)
    .add_input(input_restrided)
    .add_input(coefficients_restrided)
    .build();

  // The dimension of size n is traversed inside the kernels,
  // it is the first dimension of `input` and the second of `coefficients`
  auto input_stride = input.stride(0);
  auto coeff_stride = coefficients.stride(1);
  _compute_linear_combination_stub(
    iter.device_type(),
    iter,
    input_stride,
    coeff_stride,
    input_first_dim_size
  );
  return output;
}

} // namespace at::native
