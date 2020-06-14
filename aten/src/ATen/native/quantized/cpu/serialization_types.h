#pragma once

#include <tuple>

using ConvPackedParamsSerializationType = std::tuple<
  at::Tensor /*weight*/,
  c10::optional<at::Tensor> /*bias*/,
  // these are meant to be torch::List<int64_t> but
  // it's not supported by onnx, so we'll use Tensor as
  // a workaround
  torch::List<at::Tensor> /*stride*/,
  torch::List<at::Tensor> /*padding*/,
  torch::List<at::Tensor> /*dilation*/,
  at::Tensor /*groups*/
>;
