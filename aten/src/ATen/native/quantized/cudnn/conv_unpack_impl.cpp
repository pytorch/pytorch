#ifdef USE_CUDA
#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()

#include <ATen/native/cudnn/Macros.h>

#if HAS_CUDNN_V8()

#include <ATen/ATen.h>
#include <ATen/native/quantized/cudnn/cudnnpack_utils.h>
#include <ATen/native/quantized/packed_params.h>
#include <torch/library.h>

#include <tuple>

template <int kSpatialDim>
std::tuple<at::Tensor, c10::optional<at::Tensor>> PackedConvWeightCudnn<
    kSpatialDim>::unpack() {
  return std::tuple<at::Tensor, c10::optional<at::Tensor>>{orig_weight_, bias_};
}

template std::tuple<at::Tensor, c10::optional<at::Tensor>> PackedConvWeightCudnn<
    2>::unpack();

#endif  // HAS_CUDNN_V8
#endif  // AT_CUDNN_ENABLED
#endif  // USE_CUDA
