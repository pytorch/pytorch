#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include <tuple>

namespace at {
namespace native {

static Tensor view4d(const at::Tensor& tensor) {
  if (tensor.ndimension() != 3) throw std::runtime_error("expected 3D tensor");
  return tensor.unsqueeze(1);
}

// We currently only have depthwise support for the case where groups ==
// nInputPlane and nInputPlane == nOutputPlane
static bool is_depthwise(const at::Tensor& input, const at::Tensor& weight, int64_t groups) {
  return input.type().is_cuda() &&
         input.ndimension() == 4 &&
         input.size(3) == groups &&
         weight.size(3) % input.size(3) == 0; // output channels must be a multiple of input channels
}

Tensor conv_tbc_group(const Tensor& self, const Tensor& weight, const Tensor& bias, int64_t pad, int64_t groups) {
  AT_CHECK(self.dim() == 3, "Input must have 3 dims: time, batch, in_channel");
  AT_CHECK(weight.dim() == 3, "Weight tensor must have 3 dims: kernel_width,"
      " in_channels / groups, out_channels.");
  AT_CHECK(bias.dim() == 1, "Bias must be 1-D");

  if (!self.is_cuda()) {
    AT_ERROR("NYI: conv_tbc_group only supports CUDA tensors. "
             "Please file a feature request.");
  }

  auto input = view4d(self);
  auto weights = view4d(weight);

  if (groups <= 0) {
    AT_ERROR("Groups must be positive, got ", groups);
  }

  if (is_depthwise(input, weights, groups)) {
    // NB: thnn_convtbc_depthwise2d, also known as SpatialDepthwiseConvolutionTBC,
    // has only been tested for the input_channels = output_channels = groups 1D case.
    // It probably works for the general output_channels = input_channels * depth case,
    // as well as the 2D case, but that requires further testing.
    if (weights.size(3) != input.size(3)) {
      AT_ERROR("Input channels (", input.size(3), ") must be equal to output channels (",
               weights.size(3), ")");
    }
    auto sizes = weights.sizes();
    auto kernel_size = std::vector<int64_t>(sizes.begin(), sizes.begin() + 2);

    // NB: This is not necessarily faster than performing a (transpose, conv1d, transpose).
    auto output = at::thnn_convtbc_depthwise2d(
        input, weights, kernel_size, bias,
        /*stride=*/{1, 1},
        /*padding=*/{pad, 0},
        /*dilation=*/{1, 1});
    return output.squeeze(1);
  }

  AT_ERROR("NYI: Only groups == input_channels == output_channels is supported. "
           "Please file a feature request.");
}

Tensor conv_tbc(const Tensor& self, const Tensor& weight, const Tensor& bias, int64_t pad) {
  AT_CHECK(self.dim() == 3, "Input must have 3 dims: time, batch, "
      "in_channel");
  AT_CHECK(weight.dim() == 3, "Weight tensor must have 3 dims: kernel_width,"
      " in_channels, out_channels.");
  AT_CHECK(bias.dim() == 1, "Bias must be 1-D");

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
  AT_CHECK(inputPlanes == weight_size[1], "Input dim 2 (input channels) "
      "is not == dim 1 in the weight tensor");
  AT_CHECK(weight_size[2] == bias.sizes()[0], "Bias size must equal dim 2 in "
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
