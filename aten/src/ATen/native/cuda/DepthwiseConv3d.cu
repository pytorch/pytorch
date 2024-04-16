#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/AccumulateType.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/ConvUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/conv_depthwise3d_native.h>
#endif

#include <algorithm>
#include <tuple>
#include <limits>

namespace at::native {
namespace {

template <typename scalar_t, typename accscalar_t,
    int kKnownKernelT, int kKnownKernelH, int kKnownKernelW,
    int kKnownDilationT, int kKnownDilationH, int kKnownDilationW>
__global__ void conv_depthwise3d_cuda_kernel(
    const PackedTensorAccessor32<const scalar_t, 5> input,
    PackedTensorAccessor32<scalar_t, 5> output,
    const PackedTensorAccessor32<const scalar_t, 5> kernel,
    const scalar_t* bias,
    int strideT, int strideH, int strideW,
    int paddingT, int paddingH, int paddingW,
    int dilationT_, int dilationH_, int dilationW_)
{
  const int kT = kKnownKernelT > 0 ? kKnownKernelT : kernel.size(2);
  const int kH = kKnownKernelH > 0 ? kKnownKernelH : kernel.size(3);
  const int kW = kKnownKernelW > 0 ? kKnownKernelW : kernel.size(4);
  const int oC = output.size(1);
  const int oT = output.size(2);
  const int oH = output.size(3);
  const int oW = output.size(4);
  const int iC = input.size(1);
  const int iT = input.size(2);
  const int iH = input.size(3);
  const int iW = input.size(4);
  const int channel_multiplier = oC / iC;
  const int dilationT = kKnownDilationT > 0 ? kKnownDilationT : dilationT_;
  const int dilationH = kKnownDilationH > 0 ? kKnownDilationH : dilationH_;
  const int dilationW = kKnownDilationW > 0 ? kKnownDilationW : dilationW_;
  const int num_output = output.size(0) * output.stride(0);

  CUDA_KERNEL_LOOP(index, num_output) {
    const int out_col = index % oW;
    const int out_row = (index / oW) % oH;
    const int out_frame = (index / oW / oH) % oT;
    const int out_channel = (index / oW / oH / oT) % oC;
    const int batch = index / oW / oH / oT / oC;

    const int in_channel = out_channel / channel_multiplier;

    const int in_col_start = out_col * strideW - paddingW;
    const int in_row_start = out_row * strideH - paddingH;
    const int in_frame_start = out_frame * strideT - paddingT;

    accscalar_t sum = 0;
    const scalar_t *kernel_ptr = kernel[out_channel].data();
    const scalar_t *input_ptr =
        &input[batch][in_channel][in_frame_start][in_row_start][in_col_start];
    for (int k_frame = 0; k_frame < kT; ++k_frame) {
      const int in_frame = in_frame_start + k_frame * dilationT;
      for (int k_row = 0; k_row < kH; ++k_row) {
        const int in_row = in_row_start + k_row * dilationH;
        for (int k_col = 0; k_col < kW; ++k_col) {
          const accscalar_t op1 = *(kernel_ptr++);
          const int in_col = in_col_start + k_col * dilationW;
          if (in_frame >= 0 && in_row >= 0 && in_col >= 0 &&
              in_frame < iT && in_row < iH && in_col < iW) {
            sum += op1 * *(input_ptr);
          }
          input_ptr += dilationW;
        }
        input_ptr += iW * dilationH - kW * dilationW;
      }
      input_ptr += iW * (iH * dilationT - kH * dilationH);
    }
    if (bias != NULL) {
      sum += bias[out_channel];
    }

    output[batch][out_channel][out_frame][out_row][out_col] = sum;
  }
}

template <typename scalar_t, typename accscalar_t,
    int kKnownKernelT, int kKnownKernelH, int kKnownKernelW,
    int kKnownDilationT, int kKnownDilationH, int kKnownDilationW,
    int kKnownStrideT, int kKnownStrideH, int kKnownStrideW>
__global__ void
conv_depthwise3d_cuda_backward_input_kernel(
    const PackedTensorAccessor32<const scalar_t, 5> grad_output,
    PackedTensorAccessor32<scalar_t, 5> grad_input,
    const PackedTensorAccessor32<const scalar_t, 5> kernel,
    int strideT_, int strideH_, int strideW_,
    int paddingT, int paddingH, int paddingW,
    int dilationT_, int dilationH_, int dilationW_) {
  const int kT = kKnownKernelT > 0 ? kKnownKernelT : kernel.size(2);
  const int kH = kKnownKernelH > 0 ? kKnownKernelH : kernel.size(3);
  const int kW = kKnownKernelW > 0 ? kKnownKernelW : kernel.size(4);
  const int oC = grad_output.size(1);
  const int oT = grad_output.size(2);
  const int oH = grad_output.size(3);
  const int oW = grad_output.size(4);
  const int iC = grad_input.size(1);
  const int iT = grad_input.size(2);
  const int iH = grad_input.size(3);
  const int iW = grad_input.size(4);
  const int channel_multiplier = oC / iC;
  const int dilationT = kKnownDilationT > 0 ? kKnownDilationT : dilationT_;
  const int dilationH = kKnownDilationH > 0 ? kKnownDilationH : dilationH_;
  const int dilationW = kKnownDilationW > 0 ? kKnownDilationW : dilationW_;
  const int strideT = kKnownStrideT > 0 ? kKnownStrideT : strideT_;
  const int strideH = kKnownStrideH > 0 ? kKnownStrideH : strideH_;
  const int strideW = kKnownStrideW > 0 ? kKnownStrideW : strideW_;
  const int num_input = grad_input.size(0) * grad_input.stride(0);

  CUDA_KERNEL_LOOP(index, num_input) {
    const int in_col = index % iW;
    const int in_row = (index / iW) % iH;
    const int in_frame = (index / iW / iH) % iT;
    const int in_channel = (index / iW / iH / iT) % iC;
    const int batch = index / iW / iH / iT / iC;

    const int out_col_end = in_col + paddingW;
    const int out_row_end = in_row + paddingH;
    const int out_frame_end = in_frame + paddingT;

    const scalar_t* kernel_ptr = kernel[in_channel * channel_multiplier].data();
    accscalar_t sum = 0;

    for (int k_chn = in_channel * channel_multiplier;
         k_chn < (in_channel + 1) * channel_multiplier;
         ++k_chn) {
      const scalar_t* gout_ptr = grad_output[batch][k_chn].data();

      for (int k_frame = 0; k_frame < kT; ++k_frame) {
        const int out_frame_raw = out_frame_end - k_frame * dilationT;
        const int out_frame = out_frame_raw / strideT;
        for (int k_row = 0; k_row < kH; ++k_row) {
          const int out_row_raw = out_row_end - k_row * dilationH;
          const int out_row = out_row_raw / strideH;
          for (int k_col = 0; k_col < kW; ++k_col) {
            const accscalar_t op1 = *(kernel_ptr++);
            const int out_col_raw = out_col_end - k_col * dilationW;
            const int out_col = out_col_raw / strideW;

            const int out_offs = (out_frame * oH + out_row) * oW + out_col;

            accscalar_t op2 = (accscalar_t)0;
            if (out_col >= 0 && out_row >= 0 && out_frame >= 0 &&
                out_col < oW && out_row < oH && out_frame < oT) {
              op2 = *(gout_ptr + out_offs);
            }
            if (out_frame * strideT == out_frame_raw &&
                out_row * strideH == out_row_raw &&
                out_col * strideW == out_col_raw) {
              sum += op1 * op2;
            }
          }
        }
      }
    }

    grad_input[batch][in_channel][in_frame][in_row][in_col] = sum;
  }
}

template <typename scalar_t, typename accscalar_t,
    int kKnownStrideH, int kKnownStrideW>
__global__ void
conv_depthwise3d_cuda_backward_weight_kernel(
    const PackedTensorAccessor32<const scalar_t, 5> grad_output,
    const PackedTensorAccessor32<const scalar_t, 5> input,
    PackedTensorAccessor32<scalar_t, 5> grad_kernel,
    int strideT, int strideH_, int strideW_,
    int paddingT, int paddingH, int paddingW,
    int dilationT, int dilationH, int dilationW) {
  const int kC = grad_kernel.size(0);
  const int kT = grad_kernel.size(2);
  const int kH = grad_kernel.size(3);
  const int kW = grad_kernel.size(4);

  const int strideH = kKnownStrideH > 0 ? kKnownStrideH : strideH_;
  const int strideW = kKnownStrideW > 0 ? kKnownStrideW : strideW_;

  const int k_col = blockIdx.x % kW;
  const int k_row = (blockIdx.x / kW) % kH;
  const int k_frame = (blockIdx.x / kW / kH) % kT;
  const int k_channel = blockIdx.x / kW / kH / kT;
  scalar_t *result = &grad_kernel[k_channel][0][k_frame][k_row][k_col];

  const int oT = grad_output.size(2);
  const int oH = grad_output.size(3);
  const int oW = grad_output.size(4);
  const int iT = input.size(2);
  const int iH = input.size(3);
  const int iW = input.size(4);
  const int channel_multiplier = grad_output.size(1) / input.size(1);
  const int in_channel = k_channel / channel_multiplier;

  extern __shared__ int sdata_raw[];
  scalar_t* sdata = reinterpret_cast<scalar_t*>(sdata_raw);

  if (k_channel >= kC) {
    return;
  }

  const int laneid = threadIdx.x % C10_WARP_SIZE;
  const int warpid = threadIdx.x / C10_WARP_SIZE;
  const int nwarps = blockDim.x / C10_WARP_SIZE;

  accscalar_t grad = 0;
  int batch = warpid / oT;
  int gout_frame = warpid - batch * oT;
  for (int outer_pos = warpid; outer_pos < input.size(0) * oT;
       outer_pos += nwarps, gout_frame += nwarps) {
    while (gout_frame >= oT) { gout_frame -= oT; batch ++; }

    const int in_frame = (gout_frame * strideT) + (k_frame * dilationT) - paddingT;

    if (in_frame < 0 || in_frame >= iT) {
      continue;
    }

    const scalar_t* gout_ptr = grad_output[batch][k_channel][gout_frame].data() + laneid;
    const scalar_t* input_ptr = input[batch][in_channel][in_frame].data();

    int gout_row = laneid / oW;
    int gout_col = laneid - gout_row * oW;

    for (; gout_row < oH; ) {
      const accscalar_t op1 = *(gout_ptr);
      gout_ptr += C10_WARP_SIZE;

      const int in_col = (gout_col * strideW) + (k_col * dilationW) - paddingW;
      const int in_row = (gout_row * strideH) + (k_row * dilationH) - paddingH;
      const int in_pos = in_row * iW + in_col;

      accscalar_t op2 = (accscalar_t)0;
      if (in_col >= 0 && in_col < iW && in_row >= 0 && in_row < iH) {
        op2 = *(input_ptr + in_pos);
      }

      gout_col += C10_WARP_SIZE;
      while (gout_col >= oW) {
        gout_col -= oW; gout_row ++;
      }

      grad += op1 * op2;
    }
  }

  sdata[threadIdx.x] = grad;
  __syncthreads();

  CUDA_KERNEL_ASSERT(__popc(blockDim.x) == 1);
#pragma unroll
  for (int i = blockDim.x / 2; i >= 1; i >>= 1) {
    if (threadIdx.x < i) {
      sdata[threadIdx.x] += sdata[threadIdx.x + i];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *result = sdata[0];
  }
}

template <int dim>
void conv_depthwise_shape_check(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& grad_output,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  TORCH_CHECK(kernel_size.size() == dim,
              "kernel size length should be ", dim, ", but got ", kernel_size.size());
  TORCH_CHECK(stride.size() == dim,
              "stride length should be ", dim, ", but got ", stride.size());
  TORCH_CHECK(padding.size() == dim,
              "padding length should be ", dim, ", but got ", padding.size());
  TORCH_CHECK(dilation.size() == dim,
              "dilation length should be ", dim, ", but got ", dilation.size());

  TORCH_CHECK(weight.defined(),
              "Weight must be defined.");
  TORCH_CHECK(input.dim() == dim + 1 || input.dim() == dim + 2,
              "Input dimension should be ",
              dim + 1, "D or ", dim + 2, "D, got ",
              input.dim(), "D");
  TORCH_CHECK(weight.dim() == dim + 2,
              "Weight dimension should be ", dim + 2, "D, got ", weight.dim(), "D");
  TORCH_CHECK(weight.size(1) == 1,
              "Depthwise weight should have in_channels=1, got ", weight.size(1));
  TORCH_CHECK(weight.size(0) % input.size(-dim - 1) == 0,
              "Depthwise out channels should be a multiple of in channels, got ",
              weight.size(0), " and ", input.size(-dim - 1));
  for (int i = 0; i < dim; ++i) {
    TORCH_CHECK(weight.size(i + 2) == kernel_size[i],
                "kernel size and weight size mismatch, got ",
                kernel_size, " and ", weight.sizes());
    TORCH_CHECK(stride[i] >= 1,
                "stride should be at least 1, got ", stride);
    TORCH_CHECK(padding[i] >= 0,
                "padding should be non-negative, got ", padding);
    TORCH_CHECK(dilation[i] >= 1,
                "dilation should be at least 1, got ", dilation);
  }

  if (bias.defined()) {
    TORCH_CHECK(bias.dim() == 1,
                "Bias should be 1D tensor, got ", bias.dim(), "D");
    TORCH_CHECK(bias.size(0) == weight.size(0),
                "Bias length should be equal to out_channels, got ",
                bias.size(0), " and ", weight.size(0));
  }

  if (grad_output.defined()) {
    auto expected_output_size = conv_output_size(input.sizes(), weight.sizes(),
                                                 padding, stride, dilation);
    TORCH_CHECK(static_cast<size_t>(grad_output.dim()) == expected_output_size.size(),
                "Expect grad_output to be ",
                expected_output_size.size(), "D, got ",
                grad_output.dim(), "D.");
    for (int i = 0; i < grad_output.dim(); ++i) {
      TORCH_CHECK(grad_output.size(i) == expected_output_size[i],
                  "Expect grad_output to be of same shape as output, got ",
                  grad_output.size(i), " and ", expected_output_size[i],
                  " at dimension ", i);
    }
  }
}

}

#define NODEF_OR_EQUAL(x, y) ((y) < 0 || (x) == (y))
#define NODEF_OR_EQUAL_3(x, y1, y2, y3) \
  (NODEF_OR_EQUAL(x[0], y1) && \
   NODEF_OR_EQUAL(x[1], y2) && \
   NODEF_OR_EQUAL(x[2], y3))

#define DWCONV3D_FORWARD_DISPATCH_SPECIALIZATION(kt, kh, kw, dilt, dilh, dilw) \
  if (NODEF_OR_EQUAL_3(kernel_size, (kt), (kh), (kw)) &&                    \
      NODEF_OR_EQUAL_3(dilation, (dilt), (dilh), (dilw))) {                 \
    using accscalar_t = acc_type<scalar_t, true>;                           \
    conv_depthwise3d_cuda_kernel                                            \
    <scalar_t, accscalar_t, (kt), (kh), (kw), (dilt), (dilh), (dilw)>       \
      <<<grid, block, (smem), at::cuda::getCurrentCUDAStream()>>>(          \
        input_.packed_accessor32<const scalar_t, 5>(),                      \
        output_.packed_accessor32<scalar_t, 5>(),                           \
        weight_.packed_accessor32<const scalar_t, 5>(),                     \
        bias_ptr,                                                           \
        stride[0], stride[1], stride[2],                                    \
        padding[0], padding[1], padding[2],                                 \
        dilation[0], dilation[1], dilation[2]);                             \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                         \
  } else

#define DWCONV3D_FORWARD_DISPATCH_OTHERS \
  {                                      \
    using accscalar_t = acc_type<scalar_t, true>;                           \
    conv_depthwise3d_cuda_kernel                                            \
    <scalar_t,accscalar_t, -1, -1, -1, -1, -1, -1>                          \
      <<<grid, block, (smem), at::cuda::getCurrentCUDAStream()>>>(          \
        input_.packed_accessor32<const scalar_t, 5>(),                      \
        output_.packed_accessor32<scalar_t, 5>(),                           \
        weight_.packed_accessor32<const scalar_t, 5>(),                     \
        bias_ptr,                                                           \
        stride[0], stride[1], stride[2],                                    \
        padding[0], padding[1], padding[2],                                 \
        dilation[0], dilation[1], dilation[2]);                             \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                         \
  }

Tensor conv_depthwise3d_cuda(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size, const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  TORCH_CHECK(input.device() == weight.device(), "expects input and weight tensors to be on the same device.");
  if (bias.defined()) {
    TORCH_CHECK(input.device() == bias.device(), "expects input and bias tensors to be on the same device.");
  }

  conv_depthwise_shape_check<3>(input, weight, bias, Tensor() /* undefined */,
                                kernel_size, stride, padding, dilation);

  Tensor input_ = input.contiguous();

  if (input.dim() == 4 /* no batch */) {
    input_ = input.unsqueeze(0);
  }

  auto output_size = conv_output_size(input_.sizes(), weight.sizes(),
                                      padding, stride, dilation);
  for (size_t i = 0; i < output_size.size(); ++i) {
    TORCH_CHECK(output_size[i] > 0,
                "Output size should be positive, got ", output_size[i], " at dim ", i);
  }
  Tensor output = at::empty(output_size, input.options());
  Tensor output_ = output;
  Tensor weight_ = weight.contiguous();
  Tensor bias_ = bias.defined() ? bias.contiguous() : bias;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      input.scalar_type(),
      "conv_depthwise3d",
      [&]{
        int64_t num_outputs = output_.numel();
        int64_t block = 256;
        int64_t grid = std::min((num_outputs - 1) / block + 1, (int64_t)65536);
        int64_t smem = 0;

        const scalar_t* bias_ptr =
            bias_.defined() ? bias_.const_data_ptr<scalar_t>() : NULL;

        // Range check to avoid overflow in CUDA kernels.
        TORCH_CHECK(input_.numel() <= std::numeric_limits<int32_t>::max(),
                    "Input tensor is too large.");
        TORCH_CHECK(output_.numel() <= std::numeric_limits<int32_t>::max(),
                    "Output tensor is too large.");
        TORCH_CHECK(weight_.numel() <= std::numeric_limits<int32_t>::max(),
                    "Weight tensor is too large.");
        for (int i = 0; i < 3; ++i) {
          TORCH_CHECK(padding[i] * 2 + input.size(i + 2) <= std::numeric_limits<int32_t>::max(),
                      "Padded input tensor is too large.");
        }

        DWCONV3D_FORWARD_DISPATCH_SPECIALIZATION(3, 3, 3, 1, 1, 1)
        DWCONV3D_FORWARD_DISPATCH_SPECIALIZATION(-1, -1, -1, 1, 1, 1)
        DWCONV3D_FORWARD_DISPATCH_OTHERS
      }
  );

  return output;
}

#undef DWCONV3D_FORWARD_DISPATCH_SPECIALIZATION
#undef DWCONV3D_FORWARD_DISPATCH_OTHERS

#define DWCONV3D_BACKWARD_INPUT_DISPATCH_SPECIALIZATION(                    \
    kt, kh, kw, dilt, dilh, dilw, dt, dh, dw)                               \
  if (NODEF_OR_EQUAL_3(kernel_size, (kt), (kh), (kw)) &&                    \
      NODEF_OR_EQUAL_3(dilation, (dilt), (dilh), (dilw)) &&                 \
      NODEF_OR_EQUAL_3(stride, (dt), (dh), (dw))) {                         \
    using accscalar_t = acc_type<scalar_t, true>;                           \
    conv_depthwise3d_cuda_backward_input_kernel                             \
    <scalar_t, accscalar_t, (kt), (kh), (kw), (dilt), (dilh), (dilw), (dt), (dh), (dw)>  \
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(               \
        grad_output_.packed_accessor32<const scalar_t, 5>(),                \
        grad_input_.packed_accessor32<scalar_t, 5>(),                       \
        weight_.packed_accessor32<const scalar_t, 5>(),                     \
        stride[0], stride[1], stride[2],                                    \
        padding[0], padding[1], padding[2],                                 \
        dilation[0], dilation[1], dilation[2]);                             \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                         \
  } else

#define DWCONV3D_BACKWARD_INPUT_DISPATCH_OTHERS                             \
  {                                                                         \
    using accscalar_t = acc_type<scalar_t, true>;                           \
    conv_depthwise3d_cuda_backward_input_kernel                             \
    <scalar_t, accscalar_t, -1, -1, -1, -1, -1, -1, -1, -1, -1>             \
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(               \
        grad_output_.packed_accessor32<const scalar_t, 5>(),                \
        grad_input_.packed_accessor32<scalar_t, 5>(),                       \
        weight_.packed_accessor32<const scalar_t, 5>(),                     \
        stride[0], stride[1], stride[2],                                    \
        padding[0], padding[1], padding[2],                                 \
        dilation[0], dilation[1], dilation[2]);                             \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                         \
  }

#define DWCONV3D_BACKWARD_WEIGHT_DISPATCH_SPECIALIZATION(dh, dw)            \
  if (NODEF_OR_EQUAL_3(stride, -1, (dh), (dw))) {                           \
    using accscalar_t = acc_type<scalar_t, true>;                           \
    conv_depthwise3d_cuda_backward_weight_kernel                            \
    <scalar_t, accscalar_t, (dh), (dw)>                                     \
      <<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(            \
        grad_output_.packed_accessor32<const scalar_t, 5>(),                \
        input_.packed_accessor32<const scalar_t, 5>(),                      \
        grad_weight.packed_accessor32<scalar_t, 5>(),                       \
        stride[0], stride[1], stride[2],                                    \
        padding[0], padding[1], padding[2],                                 \
        dilation[0], dilation[1], dilation[2]);                             \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                         \
  } else

#define DWCONV3D_BACKWARD_WEIGHT_DISPATCH_OTHERS                            \
  {                                                                         \
    using accscalar_t = acc_type<scalar_t, true>;                           \
    conv_depthwise3d_cuda_backward_weight_kernel                            \
    <scalar_t, accscalar_t, -1, -1>                                         \
      <<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(            \
        grad_output_.packed_accessor32<const scalar_t, 5>(),                \
        input_.packed_accessor32<const scalar_t, 5>(),                      \
        grad_weight.packed_accessor32<scalar_t, 5>(),                       \
        stride[0], stride[1], stride[2],                                    \
        padding[0], padding[1], padding[2],                                 \
        dilation[0], dilation[1], dilation[2]);                             \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                         \
  }

std::tuple<Tensor&, Tensor&, Tensor&> _depthwise_3d_backward_cuda_out(
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    const std::array<bool, 3> output_mask)
{

  TORCH_CHECK(grad_output.device() == input.device() &&
              input.device() == weight.device(),
              "expects input, weight and grad_output to be on the same device.");
  conv_depthwise_shape_check<3>(
      input, weight, Tensor() /* undefined */, grad_output,
      kernel_size, stride, padding, dilation);

  const Tensor grad_output_ = grad_output.contiguous();

  Tensor grad_input_ =
      (output_mask[0] ?  grad_input
                      : Tensor());

  if (output_mask[0]) {
    const Tensor weight_ = weight.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        grad_output.scalar_type(),
        "conv_depthwise3d",
        [&] {
          int64_t num_inputs = grad_input_.numel();
          int64_t block = 256;
          int64_t grid = std::min((num_inputs - 1) / block + 1, (int64_t)65536);

          // Range check to avoid overflow in CUDA kernels.
          TORCH_CHECK(grad_input_.numel() <= std::numeric_limits<int32_t>::max(),
                      "Input tensor is too large.");
          TORCH_CHECK(grad_output_.numel() <= std::numeric_limits<int32_t>::max(),
                      "Output tensor is too large.");
          TORCH_CHECK(weight_.numel() <= std::numeric_limits<int32_t>::max(),
                      "Weight tensor is too large.");
          for (int i = 0; i < 3; ++i) {
            TORCH_CHECK(padding[i] * 2 + input.size(i + 2) <= std::numeric_limits<int32_t>::max(),
                        "Padded input tensor is too large.");
          }

          DWCONV3D_BACKWARD_INPUT_DISPATCH_SPECIALIZATION(
              3, 3, 3, 1, 1, 1, 1, 1, 1)
          DWCONV3D_BACKWARD_INPUT_DISPATCH_SPECIALIZATION(
              3, 3, 3, 1, 1, 1, -1, -1, -1)
          DWCONV3D_BACKWARD_INPUT_DISPATCH_SPECIALIZATION(
              3, 3, 3, -1, -1, -1, 1, 1, 1)
          DWCONV3D_BACKWARD_INPUT_DISPATCH_SPECIALIZATION(
              3, 3, 3, -1, -1, -1, -1, -1, -1)
          DWCONV3D_BACKWARD_INPUT_DISPATCH_OTHERS
        }
    );
  }

  if (output_mask[1]) {
    const Tensor input_ = input.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        grad_output.scalar_type(),
        "conv_depthwise3d",
        [&] {
          int64_t grid = grad_weight.numel();
          int64_t block = 256;
          int64_t smem = sizeof(scalar_t) * block;

          const int64_t int_max = std::numeric_limits<int32_t>::max();
          TORCH_CHECK(grad_input_.numel() <= int_max,
                      "Input tensor is too large.");
          TORCH_CHECK(grad_output_.numel() <= int_max,
                      "Output tensor is too large.");
          TORCH_CHECK(weight.numel() <= int_max,
                      "Weight tensor is too large.");
          for (int i = 0; i < 3; ++i) {
            TORCH_CHECK(padding[i] * 2 + input.size(i + 2) <= int_max,
                        "Padded input tensor is too large.");
          }
          int64_t warp_size = at::cuda::warp_size();
          TORCH_CHECK(grad_output_.size(0) * grad_output_.size(2) < int_max - block / warp_size &&
                      grad_output_.size(3) <= int_max - warp_size &&
                      grad_output_.size(4) <= int_max - warp_size,
                      "Output size is too large.");

          DWCONV3D_BACKWARD_WEIGHT_DISPATCH_SPECIALIZATION(1, 1)
          DWCONV3D_BACKWARD_WEIGHT_DISPATCH_SPECIALIZATION(2, 2)
          DWCONV3D_BACKWARD_WEIGHT_DISPATCH_OTHERS
        }
    );
  }

  if (output_mask[2]) {
    grad_bias = grad_output.sum({0, 2, 3, 4});
  }

  return std::tie(grad_input, grad_weight, grad_bias);

}


std::tuple<Tensor&, Tensor&, Tensor&> conv_depthwise3d_backward_cuda_out(const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias) {
  if (grad_weight.defined()) {
    grad_weight.resize_(weight.sizes());
    grad_weight.zero_();
  }

  return _depthwise_3d_backward_cuda_out(
      grad_input,
      grad_weight,
      grad_bias,
      grad_output,
      input,
      weight,
      kernel_size,
      stride,
      padding,
      dilation,
      {true,true,true});
}

std::tuple<Tensor, Tensor, Tensor> conv_depthwise3d_backward_cuda(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    const std::array<bool, 3> output_mask) {

  auto options = grad_output.options();
  Tensor grad_input =
      (output_mask[0] ? at::empty(input.sizes(), options) : Tensor());
  Tensor grad_weight =
      (output_mask[1] ? at::empty(weight.sizes(), options) : Tensor());
  Tensor grad_bias; /* undefined temporarily */

  return _depthwise_3d_backward_cuda_out(
      grad_input,
      grad_weight,
      grad_bias,
      grad_output,
      input,
      weight,
      kernel_size,
      stride,
      padding,
      dilation,
      output_mask
  );

}

REGISTER_CUDA_DISPATCH(conv_depthwise3d_backward_stub, &conv_depthwise3d_backward_cuda);

#undef DWCONV3D_BACKWARD_INPUT_DISPATCH_SPECIALIZATION
#undef DWCONV3D_BACKWARD_INPUT_DISPATCH_OTHERS

#undef NODEF_OR_EQUAL_3
#undef NODEF_OR_EQUAL

} // namespace at::native
