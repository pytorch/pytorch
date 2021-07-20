#include <ATen/Aten.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>

namespace at {
namespace native {

namespace {

constexpr int64_t GRAIN_SIZE = at::internal::GRAIN_SIZE;

template <typename input_t, typename output_t>
void convert_coo_to_csr_cpu(
    Tensor& result,
    const Tensor& input,
    const Scalar& size) {
  int64_t numel_in = input.numel();
  const input_t* data_in = input.data_ptr<input_t>();
  output_t* data_out = result.data_ptr<output_t>();

  if (numel_in == 0) {
    result.zero_();
    return;
  }

  for (int64_t i = 0; i < data_in[0]; i++)
    data_out[i] = (output_t)0;

  at::parallel_for(0, numel - 1, GRAIN_SIZE, [&](int64_t start, int64_t end) {
    input_t curr_value = data_in[start], next_value;
    for (int64_t i = start; i < end; i++) {
      next_value = data_in[i + 1];
      for (; curr_value < next_value; curr_value++)
        data_out[curr_value + 1] = (output_t)(i + 1);
    }
  });

  for (int64_t i = data_in[numel - 1] + 1; i < size + 1; i++)
    data_out[i] = (output_t)numel;
}

void dispatch(
    Tensor& result,
    const Tensor& input,
    const Scalar& size,
    bool out_int32) {
  if (!out_int32) {
    AT_DISPATCH_INTEGRAL_TYPES(
        input.scalar_type(), "convert_coo_to_csr_cpu", [&] {
          convert_coo_to_csr_cpu<scalar_t, int64_t>(result, input, size);
        });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(
        input.scalar_type(), "convert_coo_to_csr_cpu", [&] {
          convert_coo_to_csr_cpu<scalar_t, int>(result, input, size);
        });
  }
}

} // namespace

Tensor& _convert_coo_to_csr_out_cpu(
    const Tensor& self,
    const Scalar& size,
    bool out_int32,
    Tensor& result) {
  dispatch(result, self, size, out_int32);
  return result;
}

Tensor _convert_coo_to_csr_cpu(
    const Tensor& self,
    const Scalar& size,
    bool out_int32) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options =
      TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({size + 1}, options, MemoryFormat::Contiguous);
  at::native::_convert_coo_to_csr_out_cpu(self, size, out_int32, result);
  return result;
}

} // namespace native
} // namespace at
