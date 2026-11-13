#pragma once

#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/util/irange.h>

// Shape checks shared by the pooling operator family: max/avg pooling, max
// unpooling, adaptive pooling and fractional max pooling. Keeping them here
// lets the CPU, CUDA and MPS implementations of an op report identical errors.

namespace at::native {

// Every pooling family checks that the non-batch dimensions of a tensor are
// non-empty; `first_dim` is 0 when the tensor is unbatched.
inline void check_non_empty_dims(
    const Tensor& t,
    int64_t first_dim,
    const char* fn_name,
    const char* arg_name) {
  for (const auto i : c10::irange(first_dim, t.ndimension())) {
    TORCH_CHECK(t.size(i) > 0, fn_name,
                ": Expected ", arg_name, " to have non-zero size for non-batch dimensions, but ",
                arg_name, " has sizes ", t.sizes(), " with dimension ", i, " being empty");
  }
}

// Checks the trailing `sizes.size()` dimensions of `t`, as the backward passes
// do for gradOutput and indices.
inline void check_trailing_dim_sizes(const Tensor& t, int64_t ndim, IntArrayRef sizes) {
  const auto n = static_cast<int64_t>(sizes.size());
  for (const auto i : c10::irange(n)) {
    check_dim_size(t, ndim, ndim - n + i, sizes[i]);
  }
}

// AveragePool2d/DilatedMaxPool2d (forward)
inline void
pool2d_shape_check(
  const Tensor& input,
  int64_t kH, int64_t kW, int64_t dH, int64_t dW, int64_t padH, int64_t padW, int64_t dilationH, int64_t dilationW,
  int64_t nInputPlane,
  int64_t inputHeight, int64_t inputWidth,
  int64_t outputHeight, int64_t outputWidth, MemoryFormat memory_format)
{
  const int64_t ndim = input.ndimension();
#ifndef STRIP_ERROR_MESSAGES
  const int64_t nOutputPlane = nInputPlane;
#endif

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

// DilatedMaxPool2d/AveragePool2d (backward); `indices` is only produced by max pooling
inline void
pool2d_backward_shape_check(
  const Tensor& input,
  const Tensor& gradOutput,
  const std::optional<Tensor>& indices,
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
  const int64_t batchSize = ndim == 4 ? input.size(0) : 1;

  check_trailing_dim_sizes(gradOutput, ndim, {nInputPlane, outputHeight, outputWidth});
  if (ndim == 4) {
    check_dim_size(gradOutput, ndim, 0, batchSize);
  }

  if (indices.has_value()) {
    check_trailing_dim_sizes(*indices, ndim, {nInputPlane, outputHeight, outputWidth});
    if (ndim == 4) {
      check_dim_size(*indices, ndim, 0, batchSize);
    }
  }
}

// AveragePool3d/DilatedMaxPool3d (forward)
inline void
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

  // size of batch-dim can be 0.
  check_non_empty_dims(input, /*first_dim=*/ndim == 5 ? 1 : 0, fn_name, "input");

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

// AveragePool3d/DilatedMaxPool3d (backward); `indices` is only produced by max pooling
inline void
pool3d_backward_shape_check(
  const Tensor& input,
  const Tensor& gradOutput,
  const std::optional<Tensor>& indices,
  int64_t nslices,
  int kT, int kH, int kW,
  int dT, int dH, int dW,
  int pT, int pH, int pW,
  int dilationT, int dilationH, int dilationW,
  int64_t itime, int64_t iheight, int64_t iwidth,
  int64_t otime, int64_t oheight, int64_t owidth,
  const char* fn_name,
  bool check_input_size=false)
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
    otime, oheight, owidth, fn_name, check_input_size);

  check_trailing_dim_sizes(gradOutput, ndim, {nslices, otime, oheight, owidth});
  if (indices.has_value()) {
    check_trailing_dim_sizes(*indices, ndim, {nslices, otime, oheight, owidth});
  }
}

// TODO(#196230): remove this alias once the torch-xpu-ops pin in
// third_party/xpu.txt calls pool3d_backward_shape_check directly. Its
// DilatedMaxPool3d kernel still uses the pre-merge name, with the same argument
// order (Tensor converts to the optional indices). torch-xpu-ops compiles every
// TU with -DUSE_XPU (see its cmake/BuildFlags.cmake), while in-tree USE_XPU is
// PRIVATE to torch_xpu, so this stays out of torch_cpu.
#ifdef USE_XPU
#define max_pool3d_backward_shape_check pool3d_backward_shape_check
#endif

// MaxUnpool2d/MaxUnpool3d. Only max_unpool3d takes stride/padding, which it
// validates but never uses; 2d callers leave them empty.
inline void max_unpooling_shape_check(
    const Tensor& input,
    const Tensor& indices,
    IntArrayRef output_size,
    int64_t pooling_dims,
    const char* fn_name,
    IntArrayRef stride = {},
    IntArrayRef padding = {},
    const std::optional<Tensor>& gradOutput = std::nullopt) {
  const int64_t ndim = input.ndimension();
  const char* dims_desc =
      pooling_dims == 2 ? "two elements (height, width)" : "three elements (depth, height, width)";

  TORCH_CHECK(
      indices.scalar_type() == at::ScalarType::Long,
      "elements in indices should be type int64 but got: ", indices.scalar_type());
  TORCH_CHECK(
      ndim == pooling_dims + 1 || ndim == pooling_dims + 2,
      "Input to max_unpooling", pooling_dims, "d should be a ", pooling_dims + 1, "d or ", pooling_dims + 2,
      "d Tensor, but got a tensor with ", ndim, " dimensions.");
  TORCH_CHECK(
      static_cast<int64_t>(output_size.size()) == pooling_dims,
      "There should be exactly ", dims_desc, " in output_size, but got ", output_size.size(), " elements.");
  TORCH_CHECK(
      input.sizes() == indices.sizes(),
      "Expected shape of indices to be same as that of the input tensor (", input.sizes(),
      ") but got indices tensor with shape: ", indices.sizes());

  check_non_empty_dims(input, /*first_dim=*/1, fn_name, "input");

  if (pooling_dims == 3) {
    TORCH_CHECK(
        static_cast<int64_t>(stride.size()) == pooling_dims,
        "There should be exactly ", dims_desc, " in stride, but got: ", stride.size(), " elements.");
    TORCH_CHECK(
        static_cast<int64_t>(padding.size()) == pooling_dims,
        "There should be exactly ", dims_desc, " in padding, but got: ", padding.size(), " elements.");
    TORCH_CHECK(
        stride[0] > 0 && stride[1] > 0 && stride[2] > 0,
        "strides should be greater than zero, but got stride: ", stride);
  }

  for (const auto i : c10::irange(pooling_dims)) {
    TORCH_CHECK(
        output_size[i] >= 0,
        "max_unpooling", pooling_dims,
        "d(): output_size must contain non-negative spatial dimensions, but got output_size=", output_size);
  }

  if (gradOutput.has_value()) {
    const int64_t leading_dims = ndim - pooling_dims;
    TORCH_CHECK(
        gradOutput->ndimension() == ndim && gradOutput->size(leading_dims - 1) == input.size(leading_dims - 1),
        "gradOutput and input Tensors should have same number of dimensions and also the same number of channels/slices");
    TORCH_CHECK(
        output_size == gradOutput->sizes().slice(leading_dims, pooling_dims),
        "Inconsistent gradOutput size. Expected output_size ", output_size,
        ", but gradOutput has ", gradOutput->sizes().slice(leading_dims, pooling_dims));
  }
}

// AdaptiveAvgPool/AdaptiveMaxPool (backward)
inline void adaptive_pool_empty_output_check(const Tensor& gradOutput_, const char* arg_name) {
  const auto fn_name = c10::str(arg_name, "()");
  check_non_empty_dims(gradOutput_, /*first_dim=*/1, fn_name.c_str(), "grad_output");
}

// FractionalMaxPool2d/FractionalMaxPool3d
template <int64_t ndim>
inline void fractional_max_pool_check_shape(
    const Tensor& input,
    const Tensor& randomSamples) {

  TORCH_CHECK(
      input.scalar_type() == randomSamples.scalar_type(),
      "Expect _random_samples to have the same dtype as input");

  int64_t ndimension = randomSamples.ndimension();
  TORCH_CHECK(
      ndimension == 3,
      "Expect _random_samples to have 3 dimensions, got ", ndimension);

  int64_t N = randomSamples.size(0);
  int64_t C = randomSamples.size(1);
  int64_t D = randomSamples.size(2);

  int64_t input_batch = 0, input_channel = 0;
  if (ndim == 2) {
    // fractional_max_pool2d
    if (input.ndimension() == 3) {
      input_batch = 1;
      input_channel = input.size(0);
    } else {
      input_batch = input.size(0);
      input_channel = input.size(1);
    }
  } else {
    // factional_max_pool3d
    if (input.ndimension() == 4) {
      input_batch = 1;
      input_channel = input.size(0);
    } else {
      input_batch = input.size(0);
      input_channel = input.size(1);
    }
  }

  TORCH_CHECK(
      N >= input_batch,
      "Expect _random_samples.size(0) no less then input batch size.");
  TORCH_CHECK(
      C == input_channel,
      "Expect _random_samples.size(1) equals to input channel size.");
  TORCH_CHECK(
      D == ndim,
      "Expect _random_samples.size(2) equals to ", ndim, "; got ", D, ".");
}

} // namespace at::native
