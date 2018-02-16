#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include <tuple>

namespace at {
namespace native {

Tensor conv_tbc(const Tensor& self, const Tensor& weight, const Tensor& bias, int64_t pad) {
  AT_ASSERT(self.dim() == 3, "Input must have 3 dims: time, batch, "
      "in_channel");
  AT_ASSERT(weight.dim() == 3, "Weight tensor must have 3 dims: kernel_width,"
      " in_channels, out_channels.");
  AT_ASSERT(bias.dim() == 1, "Bias must be 1-D");

  auto input_size = self.sizes();
  auto weight_size = weight.sizes();

  auto ilen = input_size[0];
  auto batchSize = input_size[1];
  auto inputPlanes = input_size[2];
  auto outputPlanes = weight_size[2];
  auto kw = weight_size[0];
  auto olen = input_size[0] - kw + 1 + pad * 2;
  auto real_pad = (olen - ilen + kw - 1) / 2;

  // Make sure shapes are correct.
  // Input = (time, batch, in_channels)
  // Weight = (kernel_width, in_channels, out_channels)
  // Bias = (out_channels)
  AT_ASSERT(inputPlanes == weight_size[1], "Input dim 2 (input channels) "
      "is not == dim 1 in the weight tensor");
  AT_ASSERT(weight_size[2] == bias.sizes()[0], "Bias size must equal dim 2 in "
      "the weight tensor (output channels).");

  // input * weights + bias -> output_features
  Tensor output = self.type().tensor({
    olen,
    input_size[1],
    weight_size[2],
  });
  output.copy_(bias.expand(output.sizes()));
  for (int k = 0; k < kw; k++) {
    int iShift = std::max(0, static_cast<int>(k - real_pad));
    int oShift = std::max(0, static_cast<int>(real_pad - k));
    int t = std::min(ilen + real_pad - k, olen) - oShift;
    // Note: gemm assumes column-major matrices
    // input    is l*m (row-major)
    // weight   is m*r (row-major)
    // output   is l*r (row-major)
    if (t > 0) {
      auto W = weight[k];
      auto I = self.narrow(0, iShift, t).view({t * batchSize, inputPlanes});
      auto O = output.narrow(0, oShift, t).view({t * batchSize, outputPlanes});
      O.addmm_(I, W);
    }
  }
  return output;
}

std::tuple<Tensor, Tensor, Tensor> conv_tbc_backward(const Tensor& dOutput, const Tensor& input, const Tensor& weight, const Tensor& bias, int64_t pad) {
  auto input_size = input.sizes();
  auto weight_size = weight.sizes();

  auto ilen = input_size[0];
  auto batchSize = input_size[1];
  auto inputPlanes = input_size[2];
  auto outputPlanes = weight_size[2];
  auto kw = weight.sizes()[0];
  auto olen = input_size[0] - kw + 1 + pad * 2;
  int real_pad = (olen - ilen + kw - 1) / 2;

  Tensor dInput = at::zeros_like(input);
  for (int k = 0; k < kw; k++) {
    int iShift = std::max(0, k - real_pad);
    int oShift = std::max(0, real_pad - k);
    int t = std::min(ilen + real_pad - k, olen) - oShift;
    // dOutput * T(weight) -> dInput
    if (t > 0) {
      auto dO = dOutput.narrow(0, oShift, t).view({t * batchSize, outputPlanes});
      auto dI = dInput.narrow(0, iShift, t).view({t * batchSize, inputPlanes});
      dI.addmm_(dO, weight[k].t());
    }
  }

  Tensor dWeight = at::zeros_like(weight);
  for (int k = 0; k < kw; k++) {
    int iShift = std::max(0, k - real_pad);
    int oShift = std::max(0, real_pad - k);
    int t = std::min(ilen + real_pad - k, olen) - oShift;
    // T(input) * dOutput -> dWeight
    if (t > 0) {
      auto dW = dWeight[k];
      auto dO = dOutput.narrow(0, oShift, t).view({t * batchSize, outputPlanes});
      auto I = input.narrow(0, iShift, t).view({t * batchSize, inputPlanes}).t();
      dW.addmm_(I, dO);
    }
  }

  Tensor dBias = at::zeros_like(bias);
  auto tmp = dOutput.sum(0, false);
  dBias.copy_(tmp.sum(0));

  return std::make_tuple(dInput, dWeight, dBias);
}

}
}
