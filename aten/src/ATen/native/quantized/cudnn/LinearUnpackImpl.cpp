#ifdef USE_CUDA
#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()

#include <ATen/native/cudnn/Macros.h>

#if HAS_CUDNN_V8()

#include <ATen/ATen.h>
#include <ATen/native/quantized/cudnn/utils.h>
#include <ATen/native/quantized/packed_params.h>
#include <torch/library.h>

#include <tuple>

std::tuple<at::Tensor, c10::optional<at::Tensor>> PackedLinearWeightCudnn::unpack() {
  return std::tuple<at::Tensor, c10::optional<at::Tensor>>{orig_weight, bias_};
}

#endif  // HAS_CUDNN_V8
#endif  // AT_CUDNN_ENABLED
#endif  // USE_CUDA
