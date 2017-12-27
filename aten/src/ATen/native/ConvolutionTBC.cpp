#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include <tuple>

namespace at {
namespace native {

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

  Tensor dInput = zeros_like(input);
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

  Tensor dWeight = zeros_like(weight);
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

  Tensor dBias = zeros_like(bias); 
  auto tmp = dOutput.sum(0, false);
  dBias.assign_(tmp.sum(0));

  return std::make_tuple(dInput, dWeight, dBias);
}

}
}
