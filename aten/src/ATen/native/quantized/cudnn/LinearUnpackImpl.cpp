#ifdef USE_CUDA
#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()

#include <ATen/ATen.h>
#include <ATen/native/quantized/cudnn/utils.h>
#include <ATen/native/quantized/PackedParams.h>
#include <torch/library.h>

#include <tuple>

std::tuple<at::Tensor, std::optional<at::Tensor>> PackedLinearWeightCudnn::unpack() {
  return std::tuple<at::Tensor, std::optional<at::Tensor>>{orig_weight, bias_};
}

#endif  // AT_CUDNN_ENABLED
#endif  // USE_CUDA
