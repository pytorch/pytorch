#include <torch/torch.h>

#include <THC/THCNumerics.cuh>

template <typename T, typename U>
__global__ void half_test_kernel(const T* input, U* output) {
  if (input[0] < input[1] || input[0] >= input[1]) {
    output[0] = 123;
  }
}

at::Tensor half_test(at::Tensor input) {
  auto output = at::empty(1, input.options().dtype(at::kFloat));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "half_test", [&] {
    half_test_kernel<scalar_t>
        <<<1, 1>>>(input.data<scalar_t>(), output.data<float>());
  });
  return output;
}
