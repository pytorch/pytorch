#include <c10/util/Exception.h>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()
#include <ATen/cuda/Exceptions.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/cudnn/Types.h>
#endif // AT_CUDNN_ENABLED
#endif // USE_CUDA

#include <ATen/ATen.h>
#include <ATen/native/Pool.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/QScheme.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <torch/library.h>

#include <vector>

namespace at {
namespace native {
namespace {
// TODO: This function is the same as that of Pooling.cpp. We should refactor this into quantized directory
// so that we don't need to duplicate the function
#ifdef USE_CUDA
#if AT_CUDNN_ENABLED()
void check_maxpool2d_params(
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
              "Expected 1d or 2d kernel size, got ", kernel_size.size());
  TORCH_CHECK(stride.empty() || stride.size() == 2,
              "Expected no strides or 2d strides, got", stride.size());
  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
              "Expected 1d or 2d padding, got ", padding.size());
  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
              "Expected 1d or 2d dilation, got ", dilation.size());
}
#endif
#endif
}

// The current implementation of quantized cuda adaptive average pooling uses the following:
// dequant -> fp32 adaptive average pooling -> quant. This is the same numerically as
// quantized adaptive average pooling. This is not the ideal implementation, as we desire to
// operate on the quantized values directly.
// However, we are currently blocked on this as we are waiting for cudnn's 8.5.0 release, which is anticipated
// to support adaptive average pooling. When that support is made available, we will use it directly. TODO
Tensor adaptive_avg_pool2d_quantized_cuda(
    const at::Tensor& input,
    IntArrayRef output_size) {
// TODO: renable these cudnn preprocessors like quantized_max_pool2d_cudnn below when we implement this function with cudnn
#ifdef USE_CUDA
// #if AT_CUDNN_ENABLED()
    // TODO: limit this to per tensor quantized tensors for now, though should be easy to adapt
    // to per channel quantized tensors
    TORCH_CHECK(input.qscheme() == at::kPerTensorAffine, "adaptive_avg_pool2d_quantized_cuda oonly supports per tensor quantized tensors");
    auto input_fp32 = at::dequantize(input);
    auto result_fp32 = at::adaptive_avg_pool2d(input_fp32, output_size);
    return at::quantize_per_tensor(result_fp32, input.q_scale(), input.q_zero_point(), input.scalar_type());
#else // USE_CUDA
  AT_ERROR("at::native::adaptive_avg_pool2d_quantized_cuda: ATen not compiled with USE_CUDA support");
  return Tensor{}; // never reached, placates the compiler
#endif
}

// Currently we support 4D and 3D input (qx) tensors, the latter of which is supported for
// legacy reasons. The first dimension of a 4D input tensor is the batch size.
// For a 3D tensor, there is no batch size dimension -- it can be viewed as a single batch.
// cudnn's 2D pooling operation requires the input and output to be 4D tensors, so we must cast
// any 3D tensors to 4D prior to using cudnn
// This implementation currently uses the v7 cudnn APIs as v8 cudnn APIs are not yet available for
// pooling operations.
// Consult https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnPoolingForward for
// documentation on the APIs
// Currently, it appears there is no cudnn support for dilated pooling -- we will
// submit a feature request for this with cudnn
// TODO: ideally, we would like to use structured kernel support here so we do not have to repeat
// the input checks, however, that would require us to implement max_pool2d_with_indices_out_quantized_cuda
// based on how the dispatch table is currently constructed in native_functions.yaml. currently,
// there is no support for producing indices with cudnn max pooling, so until that becomes available, this cannot be done.
Tensor quantized_max_pool2d_cudnn(
    const Tensor& qx,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
#ifdef USE_CUDA
#if AT_CUDNN_ENABLED()
  check_maxpool2d_params(
      kernel_size,
      stride,
      padding,
      dilation);
  if (stride.empty()) {
    stride = kernel_size;
  }
  auto ndim = qx.dim();
  TORCH_CHECK(
      ndim == 3 || ndim == 4, "Expecting the input tensor of rank 3 or 4.");
  TORCH_CHECK(
      kernel_size.size() == 2,
      "quantized_max_pool2d_cudnn(): Expected kernel_size to be 2-dimensional: got ",
      kernel_size.size());
  TORCH_CHECK(
      stride.size() == 2,
      "quantized_max_pool2d_cudnn(): Expected stride to be 2-dimensional: got ",
      stride.size());
  TORCH_CHECK(
      dilation.size() == 2,
      "quantized_max_pool2d_cudnn(): Expected dilation to be 2-dimensional: got ",
      dilation.size());
  TORCH_CHECK(
      dilation[0] == 1 && dilation[1] == 1,
      "quantized_max_pool2d_cudnn(): Expected dilation=[1, 1] (cudnn does not currently support dilation[i] != 1), got",
      dilation);
  TORCH_CHECK(
      padding.size() == 2,
      "quantized_max_pool2d_cudnn(): Expected padding to be 2-dimensional: got ",
      padding.size());

  auto input = qx;
  if (ndim == 4) {
    input = qx.to(MemoryFormat::ChannelsLast);
  } else { // 3D
    std::vector<int64_t> new_sizes{1, qx.size(0), qx.size(1), qx.size(2)};
    input = qx.view(new_sizes);
  }
  int batch_size = input.size(0);
  int64_t inC = input.size(1);
  int64_t inH = input.size(2);
  int64_t inW = input.size(3);
  // Check output dimensions.
  int64_t padH = padding[0];
  int64_t padW = padding[1];
  int64_t kH = kernel_size[0];
  int64_t kW = kernel_size[1];
  int64_t strideH = stride[0];
  int64_t strideW = stride[1];
  TORCH_CHECK(
      kH > 0 && kW > 0,
      "qnnpack_maxpool2d(): kernel_size should be greater than zero.");
  TORCH_CHECK(
      strideH > 0 && strideW > 0,
      "qnnpack_maxpool2d(): strides should be greater than zero.");
  int64_t dilationH = dilation[0];
  int64_t dilationW = dilation[1];
  int64_t outC = inC;
  int64_t outH = pooling_output_shape(inH, kH, padH, strideH, dilationH, ceil_mode);
  int64_t outW = pooling_output_shape(inW, kW, padW, strideW, dilationW, ceil_mode);
  TORCH_CHECK(outH > 0 && outW > 0,
              "Given input size: (",
              inC, "x", inH, "x", inW,
              "). Calculated output size: (",
              outC, "x", outH, "x", outW,
              "). Output size is too small.");

  std::vector<int64_t> output_shape;
  if (ndim == 3) {
    // cudnn requires 4D input and output for 2D pooling, so we prepend a dummy dimension
    // whose size represents the batch size (1)
    output_shape = {1, outC, outH, outW};
  } else {
    output_shape = {batch_size, outC, outH, outW};
  }
  auto qy = at::_empty_affine_quantized(
      output_shape,
      at::device(at::kCUDA).dtype(at::ScalarType::QInt8),
      input.q_scale(),
      input.q_zero_point(),
      (ndim == 4 ? MemoryFormat::ChannelsLast : MemoryFormat::Contiguous));

  cudnnHandle_t handle = getCudnnHandle();
  cudnnPoolingDescriptor_t poolingDesc;
  AT_CUDNN_CHECK_WITH_SHAPES(cudnnCreatePoolingDescriptor(&poolingDesc));
  AT_CUDNN_CHECK_WITH_SHAPES(cudnnSetPooling2dDescriptor(
      poolingDesc,
      CUDNN_POOLING_MAX_DETERMINISTIC,
      CUDNN_NOT_PROPAGATE_NAN,
      kernel_size[0], // kernel height
      kernel_size[1], // kernel width
      padding[0], // vertical padding
      padding[1], // horizontal padding
      stride[0], // vertical stride
      stride[1])); // horizontal stride

  float one{1};
  float zero{0.0};
  TensorDescriptor xDesc;
  at::MemoryFormat memory_format = (ndim == 4 ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous);
  xDesc.set(input, memory_format);
  TensorDescriptor yDesc;
  yDesc.set(qy, memory_format);
  cudnnPoolingForward(handle,
                      poolingDesc,
                      &one,
                      xDesc.desc(),
                      input.data_ptr<int8_t>(),
                      &zero,
                      yDesc.desc(),
                      qy.data_ptr<int8_t>());

  // recall we casted our input and output to 4D if qx was 3D, so we recast it back to 3D prior to returning
  return (ndim == 3 ? qy.view(std::vector<int64_t>(output_shape.begin() + 1, output_shape.end())) : qy);
#else // AT_CUDNN_ENABLED()
  AT_ERROR("at::native::quantized_max_pool2d_cudnn: ATen not compiled with cuDNN support");
  return Tensor{}; // never reached, placates the compiler
#endif // AT_CUDNN_ENABLED()
#else // USE_CUDA
  AT_ERROR("at::native::quantized_max_pool2d_cudnn: ATen not compiled with USE_CUDA support");
  return Tensor{}; // never reached, placates the compiler
#endif
}

// Keep the registry in the anonymous namespace.
namespace {
template <uint32_t kSpatialDim>
class QMaxPool_arr_args final {
 public:
  static Tensor run(
      const Tensor& qx,
      std::vector<int64_t> kernel_size,
      std::vector<int64_t> stride,
      std::vector<int64_t> padding,
      std::vector<int64_t> dilation,
      bool ceil_mode) {
    TORCH_CHECK(kSpatialDim == 2, "quantized max pool is only valid for 2D")
    return quantized_max_pool2d_cudnn(qx, kernel_size, stride, padding,
                                    dilation, ceil_mode);
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::max_pool2d"), TORCH_FN(QMaxPool_arr_args<2>::run));
}

} // namespace
} // namespace native
} // namespace at
