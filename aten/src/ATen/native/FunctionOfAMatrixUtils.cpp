#include <ATen/native/FunctionOfAMatrixUtils.h>

namespace at { namespace native {

DEFINE_DISPATCH(_compute_linear_combination_stub);

Tensor _compute_linear_combination(const Tensor& input, const Tensor& coefficients) {
  auto output_first_dim_size = coefficients.size(0);
  auto input_first_dim_size = coefficients.size(1);

  auto output_sizes = input.sizes().vec();
  output_sizes[0] = output_first_dim_size;
  auto output = at::zeros(
    output_sizes,
    input.options().memory_format(at::MemoryFormat::Contiguous)
  );

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
  auto coefficients_sizes = std::vector<int64_t>(input.dim() + 1, 1);
  coefficients_sizes[0] = output_first_dim_size;
  coefficients_sizes[1] = 1;
  auto coefficients_strides = std::vector<int64_t>(input.dim() + 1, 0);
  coefficients_strides[0] = coefficients.stride(0);
  coefficients_strides[1] = 0;
  auto coefficients_restrided = coefficients.as_strided(
    coefficients_sizes,
    coefficients_strides
  );

  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .add_output(output_restrided)
    .add_input(input_restrided)
    .add_input(coefficients_restrided)
    .build();

  auto input_stride = input.stride(0);
  _compute_linear_combination_stub(
    iter.device_type(),
    iter,
    input.stride(0),
    coefficients.stride(1),
    input_first_dim_size
  );
  return output;
}

}} // namespace at::native
