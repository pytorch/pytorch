#include <tuple>
#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"

namespace at {
namespace native {


template <typename scalar_t>
at::Tensor MaxUnpooling2d_forward_cpu_out_(
    const Tensor& output,
    const Tensor& input,
    const Tensor& indices,
    int outputHeight,
    int outputWidth) {
  // TODO: replicate is_empty() cbeck in SpatialMaxUnpooling.c
  AT_CHECK(
      input.ndimension() == 4,
      "Input to MaxUnpooling2d should be a NCHW Tensor");
  AT_CHECK(
      input.sizes() == indices.sizes(),
      "Shape of indices should match shape of input");
  AT_CHECK(input.is_contiguous(), "input must be contiguous");
  AT_CHECK(indices.is_contiguous(), "indices must be contiguous");

  auto numBatch = input.size(0);
  auto numChannels = input.size(1);
  AT_CHECK(output.sizes() == IntList({numBatch, numChannels, outputHeight, outputWidth}),
      "The first two dimensions of output should match those of input, and last two dimensions should match output height and width");
  AT_CHECK(output.is_contiguous(), "output must be contiguous");

  auto inputHeight = input.size(2);
  auto inputWidth = input.size(3);

  auto* rawInput = input.data<scalar_t>();
  auto* rawIndices = indices.data<int>();
  auto* rawOutput = output.data<scalar_t>();

  int maxp;
  for (auto n = 0; n < numBatch; n++) {
    auto nOffset = n * numChannels * outputWidth * outputHeight;
    for (auto k = 0; k < numChannels; k++) {
      auto finalOffset = nOffset + k * outputWidth * outputHeight;
      auto* output_p_k = rawOutput + finalOffset;
      auto* input_p_k = rawInput + finalOffset;
      auto* ind_p_k = rawIndices + finalOffset;

      for (auto i = 0; i < inputHeight; i++) {
        for (auto j = 0; j < inputWidth; j++) {
          maxp = ind_p_k[i * inputWidth + j];
          if (maxp < 0 || maxp >= outputWidth * outputHeight) {
            AT_ERROR("Invalid index");
          } else {
            output_p_k[maxp] = input_p_k[i * inputWidth + j];
          }
        }
      }
    }
  }
  return output;
}


at::Tensor MaxUnpooling2d_forward_cpu_out(const Tensor& output,
  const Tensor& self,
  const Tensor& indices,
  IntList output_size) {
  AT_CHECK(
      output_size.size() == 2,
      "There should be exactly two elements (height, width) in output_size");
  AT_DISPATCH_FLOATING_TYPES(self.type(),
    "MaxUnpooling2d_forward_cpu_out_",
  ([&] {
    MaxUnpooling2d_forward_cpu_out_<scalar_t>(output, self, indices, output_size[0], output_size[1]);
  }));
  return output;
};

at::Tensor MaxUnpooling2d_forward_cpu(const Tensor& self,
  const Tensor& indices,
  IntList output_size) {
  AT_CHECK(
      self.ndimension() == 4,
      "Input to MaxUnpooling2d should be a NCHW Tensor",
      self.sizes());
  AT_CHECK(
      output_size.size() == 2,
      "There should be exactly two elements (height, width) in output_size");
  auto output = at::zeros({self.size(0), self.size(1), output_size[0], output_size[1]}, self.options());
  MaxUnpooling2d_forward_cpu_out(output, self, indices, output_size);
  return output;
};

Tensor MaxUnpooling2d_backward_cpu(
    const Tensor& self,
    const Tensor& indices,
    IntList output_size) {
  AT_ERROR("not implemented");
}

// stopgap until GPU version is implemented
at::Tensor MaxUnpooling2d_forward_gpu(const Tensor& self,
  const Tensor& indices,
  IntList output_size) {
    return at::max_unpool2d(self, indices, output_size);
};

} // namespace native
} // namespace at
