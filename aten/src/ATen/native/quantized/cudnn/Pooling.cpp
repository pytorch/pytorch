#include <ATen/ATen.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/cudnn/Types.h>
#include <ATen/native/Pool.h>
#include <ATen/native/TensorIterator.h>
#include <torch/library.h>

#include <vector>


namespace at {
namespace native {

// DEFINE_DISPATCH(qmaxpool_2d_nhwc_stub);

namespace {
// TODO: same as that of qpool.cpp. should refactor this into quantized directory
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
}

// at::native functions for the native_functions.yaml
Tensor quantized_max_pool2d(
    const Tensor& qx,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  check_maxpool2d_params(
      kernel_size,
      stride,
      padding,
      dilation);
  if (stride.empty()) {
    stride = kernel_size;
  }
  auto ndim = qx.dim();
  // qnnpack can only handle 4D but other one can be 3D/4D. what about cudnn?
  TORCH_CHECK(
      ndim == 3 || ndim == 4, "Expecting the input tensor of rank 3 or 4.");
  TORCH_CHECK(
      kernel_size.size() == 2,
      "qnnpack_maxpool2d(): Expected kernel_size to be 2-dimensional: got ",
      kernel_size.size());
  TORCH_CHECK(
      stride.size() == 2,
      "qnnpack_maxpool2d(): Expected stride to be 2-dimensional: got ",
      stride.size());
  TORCH_CHECK(
      dilation.size() == 2,
      "qnnpack_maxpool2d(): Expected dilation to be 2-dimensional: got ",
      dilation.size());
  TORCH_CHECK(
      padding.size() == 2,
      "qnnpack_maxpool2d(): Expected padding to be 2-dimensional: got ",
      padding.size());
  auto input = qx.contiguous(MemoryFormat::ChannelsLast);
  int64_t batch_size = input.size(0);
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
    output_shape = {outC, outH, outW};
  } else {
    output_shape = {batch_size, outC, outH, outW};
  }
  auto qy = at::_empty_affine_quantized(
      output_shape,
      at::device(at::kCUDA).dtype(at::ScalarType::QInt8),
      input.q_scale(),
      input.q_zero_point(),
      MemoryFormat::ChannelsLast);

  // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnHandle_t
  // cudnnHandle_t is a pointer to an opaque structure holding the cuDNN library context.
  // The cuDNN library context must be created using cudnnCreate() and the returned handle
  // must be passed to all subsequent library function calls.
  // The context should be destroyed at the end using cudnnDestroy().
  // The context is associated with only one GPU device,
  // the current device at the time of the call to cudnnCreate().
  // However, multiple contexts can be created on the same GPU device.
  cudnnHandle_t handle = getCudnnHandle();

  // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnPoolingDescriptor_t
  // cudnnPoolingDescriptor_t is a pointer to an opaque structure holding the description of a pooling operation.
  // cudnnCreatePoolingDescriptor() is used to create one instance, and cudnnSetPoolingNdDescriptor() or cudnnSetPooling2dDescriptor()
  // must be used to initialize this instance.
  cudnnPoolingDescriptor_t poolingDesc;
  AT_CUDNN_CHECK_WITH_SHAPES(cudnnCreatePoolingDescriptor(&poolingDesc));
  AT_CUDNN_CHECK_WITH_SHAPES(cudnnSetPooling2dDescriptor(
      poolingDesc,
      CUDNN_POOLING_MAX_DETERMINISTIC, // we also have CUDNN_POOLING_MAX which is not detemrinistic i think
      CUDNN_NOT_PROPAGATE_NAN,
      kernel_size[0], // kernel height
      kernel_size[1], // kernel width
      padding[0], // vertical padding
      padding[1], // horizontal padding
      stride[0], // vertical stride
      stride[1])); // vertical stride

  auto dataType = getCudnnDataType(input);
  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  // cudnnTensorDescriptor_t xDesc;
  TensorDescriptor xDesc;
  at::MemoryFormat memory_format = at::MemoryFormat::ChannelsLast; // I'm not sure what to put here? NHWC or NCHW or...?
  xDesc.set(input, memory_format); // qint8->int8 dtype is already taken care of i think
  // cudnnTensorDescriptor_t yDesc;
  TensorDescriptor yDesc;
  yDesc.set(qy, memory_format); // qint8->int8 dtype is already taken care of i think

  cudnnPoolingForward(handle,
                      poolingDesc,
                      &one,
                      xDesc.desc(),
                      reinterpret_cast<int8_t*>(input.data_ptr()),
                      &zero,
                      yDesc.desc(),
                      reinterpret_cast<int8_t*>(qy.data_ptr()));
  return qy;
}

// Keep the registry in the anonymous namespace.
namespace {
template <uint32_t kSpatialDim>
class QMaxPool_arr_args final {
 public:
  static Tensor run(
      Tensor qx,
      std::vector<int64_t> kernel_size,
      std::vector<int64_t> stride,
      std::vector<int64_t> padding,
      std::vector<int64_t> dilation,
      bool ceil_mode) {
    TORCH_CHECK(kSpatialDim == 2, "quantized max pool is only valid for 2D")
    return at::quantized_max_pool2d(qx, kernel_size, stride, padding,
                                    dilation, ceil_mode);
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::max_pool2d"), TORCH_FN(QMaxPool_arr_args<2>::run));
}

} // namespace
} // namespace native
} // namespace at
