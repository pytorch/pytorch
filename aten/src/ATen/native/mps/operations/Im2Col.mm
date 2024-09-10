#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_like.h>
#include <ATen/ops/col2im_native.h>
#include <ATen/ops/im2col_native.h>
#endif

namespace at::native {
using namespace mps;
static MetalShaderLibrary lib(R"IM2COL_METAL(
template<typename T>
kernel void im2col(
    constant T               * inputData       [[buffer(0)]],
    device   T               * outputData      [[buffer(1)]],
    constant uint4           & kernel_dilation [[buffer(2)]],
    constant int4            & padding_stride  [[buffer(3)]],
    constant ulong4          & input_strides   [[buffer(4)]],
    constant ulong4          & output_strides  [[buffer(5)]],
    uint3                      thread_index    [[thread_position_in_grid]]) {
    const auto N = thread_index.z;
    const auto C = thread_index.y;
    const auto L = thread_index.x;
    const auto output_width = output_strides.w;
    const auto o_x = L % output_width;
    const auto o_y = L / output_width;
    auto i_x = o_x * padding_stride.z - padding_stride.x;
    auto i_y = o_y * padding_stride.w - padding_stride.y;
    ulong kernel_size = kernel_dilation.x * kernel_dilation.y;
    ulong I_C = C / (kernel_size);
    ulong inp_offs = C % kernel_size;
    const auto i = inp_offs / kernel_dilation.x;
    const auto j = inp_offs % kernel_dilation.x;
    i_x += i * kernel_dilation.z;
    i_y += j * kernel_dilation.w;
    const auto val = inputData[N*input_strides.w + I_C*input_strides.z + i_y * input_strides.y + i_x * input_strides.x];
    outputData[L * output_strides.x + C * output_strides.y + N * output_strides.z ] = val;
}

#define INSTANTIATE_IM2COL(DTYPE)                                          \
template                                                                   \
[[host_name("im2col_" #DTYPE)]]                                            \
kernel void im2col<DTYPE>(                                                 \
    constant DTYPE           * inputData       [[buffer(0)]],              \
    device   DTYPE           * outputData      [[buffer(1)]],              \
    constant uint4           & kernel_dilation [[buffer(2)]],              \
    constant int4            & padding_stride  [[buffer(3)]],              \
    constant ulong4          & input_strides   [[buffer(4)]],              \
    constant ulong4          & output_strides  [[buffer(5)]],              \
    uint3                      thread_index  [[thread_position_in_grid]])

INSTANTIATE_IM2COL(float);
)IM2COL_METAL");

namespace {
static void im2col_out_mps_template(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  TORCH_CHECK(
      kernel_size.size() == 2,
      "It is expected kernel_size equals to 2, but got size ",
      kernel_size.size());

  TORCH_CHECK(
      dilation.size() == 2,
      "It is expected dilation equals to 2, but got size ",
      dilation.size());

  TORCH_CHECK(
      padding.size() == 2,
      "It is expected padding equals to 2, but got size ",
      padding.size());

  TORCH_CHECK(
      stride.size() == 2,
      "It is expected stride equals to 2, but got size ",
      stride.size());

  const auto kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t dilation_height = dilation[0];
  int64_t dilation_width = dilation[1];
  int64_t pad_height = padding[0];
  int64_t pad_width = padding[1];
  int64_t stride_height = stride[0];
  int64_t stride_width = stride[1];

  Tensor input = input_.contiguous();

  bool batched_input = true;

  if (input.dim() == 3) {
    batched_input = false;
    input = input.view({1, input.size(0), input.size(1), input.size(2)});
  }

  int64_t batch_size = input.size(0);
  int64_t n_input_plane = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);

  int64_t output_height = (input_height + 2 * pad_height -
                           (dilation_height * (kernel_height - 1) + 1)) /
          stride_height + 1;
  int64_t output_width = (input_width + 2 * pad_width -
                          (dilation_width * (kernel_width - 1) + 1)) /
          stride_width + 1;
  int64_t n_output_plane = n_input_plane * kernel_width * kernel_height;
  int64_t output_length = output_height * output_width;

  output.resize_({batch_size, n_output_plane, output_length});
  auto stream = getCurrentMPSStream();
  auto device = MPSDevice::getInstance()->device();
  auto im2colPSO = lib.getPipelineStateForFunc("im2col_" + mps::scalarToMetalTypeString(input));
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      std::array<int32_t, 4> kernel_dilation = {static_cast<int32_t>(kernel_width), static_cast<int32_t>(kernel_height), static_cast<int32_t>(dilation_width), static_cast<int32_t>(dilation_height)};
      std::array<int32_t, 4> padding_stride = {static_cast<int32_t>(pad_width), static_cast<int32_t>(pad_height), static_cast<int32_t>(stride_width), static_cast<int32_t>(stride_height)};
      std::array<int64_t, 4> input_strides = {input.stride(3), input.stride(2), input.stride(1), input.stride(0)};
      std::array<int64_t, 4> output_strides = {output.stride(2), output.stride(1), output.stride(0), output_width};
      getMPSProfiler().beginProfileKernel(im2colPSO, "im2col", {input});

      if (getMPSProfiler().isCaptureEnabled()) {
        getMPSProfiler().startCapture("im2col", stream);
      }
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:im2colPSO];
      mtl_setBuffer(computeEncoder, input, 0);
      mtl_setBuffer(computeEncoder, output, 1);
      mtl_setBytes(computeEncoder, kernel_dilation, 2);
      mtl_setBytes(computeEncoder, padding_stride, 3);
      mtl_setBytes(computeEncoder, input_strides, 4);
      mtl_setBytes(computeEncoder, output_strides, 5);
      [computeEncoder dispatchThreads:MTLSizeMake(output_length, n_output_plane, batch_size)
                      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      if (getMPSProfiler().isCapturing()) {
        getMPSProfiler().stopCapture(stream);
      }
      getMPSProfiler().endProfileKernel(im2colPSO);
    }
  });
  if (!batched_input) {
    output.resize_({n_output_plane, output_length});
  }
}

} // anonymous namespace
Tensor& im2col_out_mps(const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    Tensor& output) {
  im2col_out_mps_template(
      output, input, kernel_size, dilation, padding, stride);
  return output;
}

Tensor im2col_mps(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  Tensor output = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  im2col_out_mps_template(
      output, input, kernel_size, dilation, padding, stride);
  return output;
}
} // namespace at::native
