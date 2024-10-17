#ifdef USE_CUDA
#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()

#include <ATen/ATen.h>
#include <ATen/native/quantized/cudnn/utils.h>
#include <ATen/native/quantized/PackedParams.h>
#include <torch/library.h>

#include <tuple>

template <int kSpatialDim>
std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeightCudnn<
    kSpatialDim>::unpack() {
  return std::tuple<at::Tensor, std::optional<at::Tensor>>{maybe_padded_weight_, bias_};
}

template std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeightCudnn<
    2>::unpack();

#endif  // AT_CUDNN_ENABLED
#endif  // USE_CUDA
