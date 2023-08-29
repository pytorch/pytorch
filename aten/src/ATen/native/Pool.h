#include <ATen/core/Tensor.h>
#include <ATen/div_rtn.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/DispatchStub.h>
#include <c10/util/irange.h>

#include <utility>

#pragma once

namespace at::native {

using max_pool2d_fn = void(*)(const Tensor& output, const Tensor& indices, const Tensor& input,
    int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH);
using max_pool2d_backward_fn = void(*)(const Tensor& grad_input, const Tensor& grad_output, const Tensor& indices);

DECLARE_DISPATCH(max_pool2d_fn, max_pool2d_kernel);
DECLARE_DISPATCH(max_pool2d_backward_fn, max_pool2d_backward_kernel);

// averge pooling has same signature for forward and backward
using avg_pool2d_fn = void(*)(const Tensor& output, const Tensor& input, int64_t kW, int64_t kH,
    int64_t dW, int64_t dH, int64_t padW, int64_t padH, bool count_include_pad, c10::optional<int64_t> divisor_override);
using avg_pool2d_backward_fn = void(*)(const Tensor& output, const Tensor& input, int kW, int kH,
    int dW, int dH, int padW, int padH, bool count_include_pad, c10::optional<int64_t> divisor_override);

DECLARE_DISPATCH(avg_pool2d_fn, avg_pool2d_kernel);
DECLARE_DISPATCH(avg_pool2d_backward_fn, avg_pool2d_backward_kernel);

using max_pool3d_fn = void(*)(Tensor& output, Tensor& indices, const Tensor& input,
    int kW, int kH, int kD, int dW, int dH, int dD, int pW, int pH, int pD, int dilationW, int dilationH, int dilationD);
using max_pool3d_backward_fn = void(*)(Tensor& grad_input, const Tensor& grad_output, const Tensor& indices);

DECLARE_DISPATCH(max_pool3d_fn, max_pool3d_kernel);
DECLARE_DISPATCH(max_pool3d_backward_fn, max_pool3d_backward_kernel);
namespace {

template <typename dest_t, typename src_t>
static inline dest_t
safe_downcast(src_t v)
{
  TORCH_CHECK(std::numeric_limits<dest_t>::min() <= v && v <= std::numeric_limits<dest_t>::max(),
              "integer out of range");

  return static_cast<dest_t>(v);
}

template<typename T>
static inline T pooling_output_shape_pad_lr(
        T inputSize, T kernelSize, T pad_l, T pad_r, T stride, T dilation,
        bool ceil_mode) {
    T outputSize = div_rtn<T>(
        inputSize + pad_l + pad_r - dilation * (kernelSize - 1) - 1 +
        (ceil_mode ? stride - 1 : 0), stride) + 1;
    if (ceil_mode) {
        // ensure that the last pooling starts inside the image
        // needed to avoid problems in ceil mode
        if ((outputSize - 1) * stride >= inputSize + pad_l) {
          --outputSize;
        }
    }
    return outputSize;
}

template<typename T>
static inline T pooling_output_shape(
      T inputSize, T kernelSize, T pad, T stride, T dilation, bool ceil_mode) {
    TORCH_CHECK(stride != 0, "stride should not be zero");
    TORCH_CHECK(pad >= 0,
                "pad must be non-negative, but got pad: ", pad);
    TORCH_CHECK(pad <= kernelSize / 2,
                "pad should be at most half of kernel size, but got pad=",
                pad, " and kernel_size=", kernelSize)
    return pooling_output_shape_pad_lr(
        inputSize, kernelSize, pad, pad, stride, dilation, ceil_mode);
}

template <typename T>
std::pair<T, T> _pooling_same_mode_padding_lr(
    T inputSize, T kernelSize, int64_t stride, int64_t dilation) {
  // NOTE: with strides, the output shape is ceil(inputSize/stride)
  auto total_padding = T(dilation) * (kernelSize - 1);

  // Prefer symmetric padding if possible
  if (stride > 2 && (total_padding % 2 == 1)) {
    // The floor in the output size calculation gives us a little wiggle room
    auto wiggle_room = inputSize % stride - 1;
    if (wiggle_room > 0) {
      total_padding = total_padding - 1;
    }
  }

  auto left = total_padding / 2;
  return {left, total_padding - left};
}

inline std::pair<int64_t, int64_t> pooling_same_mode_padding_lr(
    int64_t inputSize, int64_t kernelSize, int64_t stride, int64_t dilation) {
  return _pooling_same_mode_padding_lr(inputSize, kernelSize, stride, dilation);
}

inline std::pair<c10::SymInt, c10::SymInt> pooling_same_mode_padding_lr(
    c10::SymInt inputSize, c10::SymInt kernelSize, int64_t stride, int64_t dilation) {
  return _pooling_same_mode_padding_lr(std::move(inputSize), std::move(kernelSize), stride, dilation);
}

// AveragePool2d/DilatedMaxPool2d (forward)
static inline void
pool2d_shape_check(
  const Tensor& input,
  int kH, int kW, int dH, int dW, int padH, int padW, int dilationH, int dilationW,
  int64_t nInputPlane,
  int64_t inputHeight, int64_t inputWidth,
  int64_t outputHeight, int64_t outputWidth, MemoryFormat memory_format)
{
  const int64_t ndim = input.ndimension();
  const int64_t nOutputPlane = nInputPlane;

  TORCH_CHECK(kW > 0 && kH > 0,
              "kernel size should be greater than zero, but got ",
              "kH: ", kH, " kW: ", kW);
  TORCH_CHECK(dW > 0 && dH > 0,
              "stride should be greater than zero, but got "
              "dH: ", dH, " dW: ", dW);
  TORCH_CHECK(dilationH > 0 && dilationW > 0,
              "dilation should be greater than zero, but got ",
              "dilationH: ", dilationH, " dilationW: ", dilationW);

  bool valid_dims = input.size(1) != 0 && input.size(2) != 0;
  if (memory_format == at::MemoryFormat::ChannelsLast){
    // Expect tensor in NHWC format and allow 0-dim only for N.
    TORCH_CHECK((ndim == 4 && valid_dims && input.size(3) != 0),
      "Expected 4D (batch mode) tensor expected for input with channels_last layout"
      " with optional 0 dim batch size for input, but got: ", input.sizes());
  } else {
    TORCH_CHECK((ndim == 3 && input.size(0) != 0 && valid_dims) ||
      (ndim == 4 && valid_dims && input.size(3) != 0),
      "Expected 3D or 4D (batch mode) tensor with optional 0 dim batch size for input, but got:",
      input.sizes());
  }

  TORCH_CHECK(kW/2 >= padW && kH/2 >= padH,
              "pad should be smaller than or equal to half of kernel size, but got ",
              "padW = ", padW, ", padH = ", padH, ", kW = ", kW, ", kH = ", kH);

  TORCH_CHECK(outputWidth >= 1 && outputHeight >= 1,
              "Given input size: (",
              nInputPlane, "x", inputHeight, "x", inputWidth, "). ",
              "Calculated output size: (",
              nOutputPlane, "x", outputHeight, "x", outputWidth, "). ",
              "Output size is too small");
}

// DilatedMaxPool2d (backward)
static inline void
max_pool2d_backward_shape_check(
  const Tensor& input,
  const Tensor& gradOutput,
  const Tensor& indices,
  int kH, int kW, int dH, int dW, int padH, int padW, int dilationH, int dilationW,
  int64_t nInputPlane,
  int64_t inputHeight, int64_t inputWidth,
  int64_t outputHeight, int64_t outputWidth, MemoryFormat memory_format)
{
  pool2d_shape_check(
    input,
    kH, kW, dH, dW, padH, padW, dilationH, dilationW,
    nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth, memory_format);

  const int64_t ndim = input.ndimension();
  const int64_t nOutputPlane = nInputPlane;

  check_dim_size(gradOutput, ndim, ndim-3, nOutputPlane);
  check_dim_size(gradOutput, ndim, ndim-2, outputHeight);
  check_dim_size(gradOutput, ndim, ndim-1, outputWidth);

  check_dim_size(indices, ndim, ndim-3, nOutputPlane);
  check_dim_size(indices, ndim, ndim-2, outputHeight);
  check_dim_size(indices, ndim, ndim-1, outputWidth);
}

// AveragePool2d (backward)
static inline void
avg_pool2d_backward_shape_check(
  const Tensor& input,
  const Tensor& gradOutput,
  int64_t /*nbatch*/,
  int kH, int kW, int dH, int dW, int padH, int padW,
  int64_t nInputPlane,
  int64_t inputHeight, int64_t inputWidth,
  int64_t outputHeight, int64_t outputWidth,
  MemoryFormat memory_format)
{
  pool2d_shape_check(
    input,
    kH, kW, dH, dW, padH, padW, 1, 1,
    nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
    memory_format);

  const int64_t ndim = input.ndimension();
  const int64_t nOutputPlane = nInputPlane;

  check_dim_size(gradOutput, ndim, ndim-3, nOutputPlane);
  check_dim_size(gradOutput, ndim, ndim-2, outputHeight);
  check_dim_size(gradOutput, ndim, ndim-1, outputWidth);
}

// AveragePool3d/DilatedMaxPool3d (forward)
static inline void
pool3d_shape_check(
  const Tensor& input,
  int64_t nslices,
  int kT, int kH, int kW,
  int dT, int dH, int dW,
  int pT, int pH, int pW,
  int dilationT, int dilationH, int dilationW,
  int64_t itime, int64_t iheight, int64_t iwidth,
  int64_t otime, int64_t oheight, int64_t owidth,
  const char *fn_name,
  bool check_input_size=false)
{
  const int64_t ndim = input.ndimension();

  TORCH_CHECK(kT > 0 && kW > 0 && kH > 0,
              "kernel size should be greater than zero, but got ",
              "kT: ", kT, " kH: ", kH, " kW: ", kW);
  TORCH_CHECK(dT > 0 && dW > 0 && dH > 0,
              "stride should be greater than zero, but got ",
              "dT: ", dT, " dH: ", dH, " dW: ", dW);
  TORCH_CHECK(dilationT > 0 && dilationW > 0 && dilationH > 0,
              "dilation should be greater than zero, but got ",
              "dilationT: ", dilationT, " dilationH: ", dilationH, " dilationW: ", dilationW);

  TORCH_CHECK(ndim == 4 || ndim == 5,
              fn_name, ": Expected 4D or 5D tensor for input, but got: ", input.sizes());

  for (const auto i : c10::irange(ndim)) {
    if (ndim == 5 && i == 0) {
      // size of batch-dim can be 0.
      continue;
    }
    TORCH_CHECK(
        input.size(i) > 0,
        fn_name,
        ": Expected input's non-batch dimensions to have positive length,"
        " but input has a shape of ",
        input.sizes(),
        " and non-batch dimension ",
        input.size(i),
        " has length zero!")
  }

  if (check_input_size) { // AveragePool3d
    TORCH_CHECK(itime >= kT && iheight >= kH && iwidth >= kW,
                "input image ", "(T: ", itime, " H: ", iheight, " W: ", iwidth, ") smaller than ",
                "kernel size ", "(kT: ", kT, " kH: ", kH, " kW: ", kW, ")");
  }

  TORCH_CHECK(kT/2 >= pT && kW/2 >= pW && kH/2 >= pH,
              "pad should be smaller than or equal to half of kernel size, but got "
              "kT: ", kT, " kW: ", kW, " kH: ", kH, " padT: ", pT, " padW: ", pW, " padH: ", pH);

  TORCH_CHECK(otime >= 1 && owidth >= 1 && oheight >= 1,
              "Given input size: (",
              nslices,"x", itime, "x", iheight, "x", iwidth, "). ",
              "Calculated output size: (",
              nslices, "x", otime, "x", oheight, "x", owidth, "). ",
              "Output size is too small");
}

static inline void
max_pool3d_backward_shape_check(
  const Tensor& input,
  const Tensor& gradOutput,
  const Tensor& indices,
  int64_t nslices,
  int kT, int kH, int kW,
  int dT, int dH, int dW,
  int pT, int pH, int pW,
  int dilationT, int dilationH, int dilationW,
  int64_t itime, int64_t iheight, int64_t iwidth,
  int64_t otime, int64_t oheight, int64_t owidth,
  const char* fn_name)
{
  const int64_t ndim = input.ndimension();

  pool3d_shape_check(
    input,
    nslices,
    kT, kH, kW,
    dT, dH, dW,
    pT, pH, pW,
    dilationT, dilationH, dilationW,
    itime, iheight, iwidth,
    otime, oheight, owidth, fn_name);

  check_dim_size(gradOutput, ndim, ndim-4, nslices);
  check_dim_size(gradOutput, ndim, ndim-3, otime);
  check_dim_size(gradOutput, ndim, ndim-2, oheight);
  check_dim_size(gradOutput, ndim, ndim-1, owidth);

  check_dim_size(indices, ndim, ndim-4, nslices);
  check_dim_size(indices, ndim, ndim-3, otime);
  check_dim_size(indices, ndim, ndim-2, oheight);
  check_dim_size(indices, ndim, ndim-1, owidth);
}

static inline void
avg_pool3d_backward_shape_check(
  const Tensor& input,
  const Tensor& gradOutput,
  int64_t nslices,
  int kT, int kH, int kW,
  int dT, int dH, int dW,
  int pT, int pH, int pW,
  int64_t itime, int64_t iheight, int64_t iwidth,
  int64_t otime, int64_t oheight, int64_t owidth,
  const char *fn_name)
{
  const int64_t ndim = input.ndimension();

  pool3d_shape_check(
    input,
    nslices,
    kT, kH, kW,
    dT, dH, dW,
    pT, pH, pW,
    1, 1, 1,
    itime, iheight, iwidth,
    otime, oheight, owidth,
    fn_name, true);

  check_dim_size(gradOutput, ndim, ndim-4, nslices);
  check_dim_size(gradOutput, ndim, ndim-3, otime);
  check_dim_size(gradOutput, ndim, ndim-2, oheight);
  check_dim_size(gradOutput, ndim, ndim-1, owidth);
}

} // anonymous namespace

} // namespace at::native
